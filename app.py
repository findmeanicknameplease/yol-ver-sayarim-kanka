# app.py
# Offline visual search UI (Gradio)
# - Embedding-only with score threshold filtering (feels real immediately)
# - Optional reranker toggle (if installed), applied to top-N only
# - No transcripts required

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import gradio as gr

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
TX_FAISS = "/workspace/offline/transcripts.faiss"  # optional, not used in this visual-first demo

INSTRUCTION = "Retrieve images relevant to the user's query."
DIM = 2048

# Filtering to avoid "show everything" on tiny corpora:
MIN_SCORE = 0.22         # absolute cutoff
KEEP_RATIO = 0.75        # keep results with score >= best_score * KEEP_RATIO
MAX_SHOW = 20            # gallery size

store = OfflineStore(DB_PATH, FRAMES_FAISS, TX_FAISS, dim=DIM)
embedder = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _embed_query(q: str) -> np.ndarray:
    e = _to_numpy(embedder.process([{"text": q, "instruction": INSTRUCTION}])).astype("float32")
    return e[0]


def _label(row: Dict[str, Any]) -> str:
    kind = row.get("kind", "item")
    score = row.get("score", None)
    s = f"{kind}"

    if kind == "video_frame":
        s += f" | {Path(row.get('video_path','')).name} @ {float(row.get('t_sec',0.0)):.1f}s"
    else:
        s += f" | {Path(row.get('path','')).name}"

    if score is not None:
        s += f" | score={float(score):.3f}"

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

    hits = store.search_with_scores(frames_index, qv, int(topk))
    if not hits:
        return [], "No hits (unexpected)."

    best = hits[0][1]
    keep_min = max(MIN_SCORE, best * KEEP_RATIO)

    # filter
    hits = [(i, s) for (i, s) in hits if s >= keep_min]

    ids = [i for (i, _) in hits]
    rows = store.get_frames_by_ids(ids)

    score_by_id = {i: s for (i, s) in hits}
    for r in rows:
        r["score"] = score_by_id.get(int(r["id"]), None)

    # Optional reranker on top N only (expensive)
    if use_rerank and HAS_RERANKER and rows:
        reranker = Qwen3VLReranker(model_name_or_path="Qwen/Qwen3-VL-Reranker-2B")
        N = min(12, len(rows))

        docs = []
        for r in rows[:N]:
            docs.append({"image": r["path"], "text": r.get("name") or ""})

        inputs = {
            "instruction": INSTRUCTION,
            "query": {"text": q},
            "documents": docs,
            "fps": 1.0,
            "max_frames": 64,
        }
        scores = reranker.process(inputs)
        for i, s in enumerate(scores):
            rows[i]["rerank_score"] = float(s)

        rows[:N] = sorted(rows[:N], key=lambda x: x.get("rerank_score", -1e9), reverse=True)

    gallery = []
    debug_lines = []
    for r in rows[:MAX_SHOW]:
        gallery.append((r["path"], _label(r)))
        debug_lines.append(_label(r))

    debug = (
        f"Query: {q}\n"
        f"Raw best score: {best:.3f}\n"
        f"Threshold used: {keep_min:.3f} (MIN_SCORE={MIN_SCORE}, KEEP_RATIO={KEEP_RATIO})\n\n"
        + "\n".join(debug_lines)
    )

    return gallery, debug


with gr.Blocks(title="Offline Media Search") as demo:
    gr.Markdown(
        "## Offline Media Search (FAISS + SQLite)\n"
        "- **Visual-first**: photos + video frames\n"
        "- **Score filtering**: avoids showing irrelevant items on small datasets\n"
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
