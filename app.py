# app.py
# Offline visual search UI (Gradio)
# - Embedding-only with score threshold filtering
# - Optional reranker toggle (if installed), applied to top-N only
# - Heuristic Priors applied (time/date/recency)

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import gradio as gr


import os
import warnings

# Silence noisy tokenizer warnings coming from the reranker tokenizer plumbing
warnings.filterwarnings("ignore", message=r"You're using a Qwen2TokenizerFast tokenizer.*")
warnings.filterwarnings("ignore", message=r"`max_length` is ignored when `padding`=`True`.*")

try:
    from transformers.utils import logging as hf_logging  # type: ignore
    hf_logging.set_verbosity_error()
except Exception:
    pass
# Local modules
from priors import parse_query_intent, compute_prior_score
from offline_store import OfflineStore

# Make src import work without PYTHONPATH hassles
QWEN_REPO = Path("/workspace/Qwen3-VL-Embedding")
if QWEN_REPO.exists():
    import sys
    sys.path.insert(0, str(QWEN_REPO))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

try:
    from src.models.qwen3_vl_reranker import Qwen3VLReranker
    HAS_RERANKER = True
except Exception:
    HAS_RERANKER = False

DB_PATH = "/workspace/offline/index.sqlite"
FRAMES_FAISS = "/workspace/offline/frames.faiss"
TX_FAISS = "/workspace/offline/transcripts.faiss"

INSTRUCTION = "Retrieve images relevant to the user's query."
DIM = 2048

# Filtering to avoid "show everything" on tiny corpora:
MIN_SCORE = 0.22         # absolute cutoff
KEEP_RATIO = 0.75        # keep results with score >= best_score * KEEP_RATIO
MAX_SHOW = 20            # gallery size

store = OfflineStore(DB_PATH, FRAMES_FAISS, TX_FAISS, dim=DIM)
embedder = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")




# Reranker is heavy; initialize lazily once (first time user toggles it)
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Qwen3VLReranker(model_name_or_path="Qwen/Qwen3-VL-Reranker-2B")
    return _reranker
def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _embed_query(q: str) -> np.ndarray:
    e = _to_numpy(embedder.process([{"text": q, "instruction": INSTRUCTION}])).astype("float32")
    return e[0]


def _label(row: Dict[str, Any]) -> str:
    kind = row.get("kind", "item")
    raw_score = float(row.get("score", 0.0))
    final_score = float(row.get("final_score", raw_score))
    
    s = f"{kind}"

    if kind == "video_frame":
        s += f" | {Path(row.get('video_path','')).name} @ {float(row.get('t_sec',0.0)):.1f}s"
    else:
        s += f" | {Path(row.get('path','')).name}"

    # Display final score. If prior modified it, show raw in parens.
    if abs(final_score - raw_score) > 1e-5:
        s += f" | score={final_score:.3f} (emb={raw_score:.3f})"
    else:
        s += f" | score={final_score:.3f}"

    if row.get("rerank_score") is not None:
        s += f" | rerank={float(row['rerank_score']):.3f}"

    return s


def search(q: str, topk: int, use_rerank: bool):
    q = (q or "").strip()
    if not q:
        return [], "Type a query."

    frames_path = Path(FRAMES_FAISS)
    if not frames_path.exists():
        return [], f"Missing {FRAMES_FAISS}. Run: python ingest.py"

    frames_index = store.load_index(frames_path)
    qv = _embed_query(q)

    # 1. Raw FAISS Search
    hits = store.search_with_scores(frames_index, qv, int(topk))
    if not hits:
        return [], "No hits (unexpected)."

    best = hits[0][1]
    keep_min = max(MIN_SCORE, best * KEEP_RATIO)

    # 2. Hard threshold filtering
    hits = [(i, s) for (i, s) in hits if s >= keep_min]

    ids = [i for (i, _) in hits]
    rows = store.get_frames_by_ids(ids)

    score_by_id = {i: s for (i, s) in hits}
    for r in rows:
        r["score"] = score_by_id.get(int(r["id"]), 0.0)

    # 3. Apply Priors (Logic Insertion)
    intent = parse_query_intent(q)

    for r in rows:
        # r acts as the item dict. 
        # r["score"] is the raw embedding score
        base_score = r.get("score", 0.0)
        prior_boost = compute_prior_score(r, intent)
        r["final_score"] = base_score + prior_boost

    # 4. Final Sort (incorporating priors)
    rows.sort(key=lambda x: x["final_score"], reverse=True)

    
# 5. Optional reranker on top N only (expensive)
# IMPORTANT: If reranker is enabled, treat rerank_score as the final ordering signal for the top-N.
# We keep priors for thresholding + tie-breaks, but we DO NOT mix rerank_score into the embedding threshold.
if use_rerank and HAS_RERANKER and rows:
    reranker = get_reranker()
    N = min(12, len(rows))

    # Build reranker docs from the current top-N (already sorted by priors)
    docs = [{"image": r["path"], "text": r.get("name") or ""} for r in rows[:N]]

    inputs = {
        "instruction": INSTRUCTION,
        "query": {"text": q},
        "documents": docs,
        "fps": 1.0,
        "max_frames": 64,
    }

    # Some wrappers return torch tensors / lists; coerce to python floats
    scores = reranker.process(inputs)
    try:
        scores = list(scores)
    except Exception:
        pass

    for i, s in enumerate(scores[:N]):
        rows[i]["rerank_score"] = float(_to_numpy(s) if hasattr(s, "detach") else s)

    # Final: sort top-N by rerank score first, then prior-adjusted score as tie-breaker
    rows[:N] = sorted(
        rows[:N],
        key=lambda x: (x.get("rerank_score", -1e9), x.get("final_score", x.get("score", 0.0))),
        reverse=True,
    )

    gallery = []
    debug_lines = []
    for r in rows[:MAX_SHOW]:
        gallery.append((r["path"], _label(r)))
        debug_lines.append(_label(r))

    debug = (
        f"Query: {q}\n"
        f"Intent: {intent}\n"
        f"Raw best score: {best:.3f}\n"
        + (f"Best rerank (topN): {max([r.get('rerank_score', -1e9) for r in rows[:min(12,len(rows))]]):.3f}\n" if (use_rerank and HAS_RERANKER and any(r.get("rerank_score") is not None for r in rows[:min(12,len(rows))])) else "")
        f"Threshold used: {keep_min:.3f} (MIN_SCORE={MIN_SCORE}, KEEP_RATIO={KEEP_RATIO})\n\n"
        + "\n".join(debug_lines)
    )

    return gallery, debug


with gr.Blocks(title="Offline Media Search") as demo:
    gr.Markdown(
        "## Offline Media Search (FAISS + SQLite)\n"
        "- **Visual-first**: photos + video frames\n"
        "- **Score filtering**: avoids showing irrelevant items on small datasets\n"
        "- **Smart Priors**: Time/Date logic applied to scores\n"
        "- Optional reranker toggle (if installed)\n"
    )
    q = gr.Textbox(label="Query", placeholder="e.g., dog, receipt, passport, beach sunset")
    with gr.Row():
        topk = gr.Slider(5, 200, value=40, step=1, label="TopK candidates")
        use_rerank = gr.Checkbox(value=False, label="Use reranker (top ~12)")

    if not HAS_RERANKER:
        gr.Markdown("ℹ️ Reranker not available (embedding-only mode).")

    btn = gr.Button("Search")
    gallery = gr.Gallery(label="Results", columns=2, height=520)
    debug = gr.Textbox(label="Debug", lines=14)

    btn.click(search, [q, topk, use_rerank], [gallery, debug])

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    allowed_paths=["/workspace/media", "/workspace/cache", "/workspace/offline"]
)
