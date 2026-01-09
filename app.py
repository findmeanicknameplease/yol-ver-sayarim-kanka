# app.py
"""
Offline visual search UI (Gradio)

Features
- Embedding-only retrieval with score threshold filtering
- Optional reranker toggle (if installed), applied to top-N only
- Heuristic priors (time/date/recency) applied to scores
- Defensive error handling + production-friendly configuration knobs

Run:
  python app.py

Env vars (optional):
  DB_PATH=/workspace/offline/index.sqlite
  FRAMES_FAISS=/workspace/offline/frames.faiss
  TX_FAISS=/workspace/offline/transcripts.faiss
  SERVER_NAME=0.0.0.0
  SERVER_PORT=7860
  MAX_SHOW=20
  MIN_SCORE=0.22
  KEEP_RATIO=0.75
  TOPN_RERANK=12
  MODEL_EMBED=Qwen/Qwen3-VL-Embedding-2B
  MODEL_RERANK=Qwen/Qwen3-VL-Reranker-2B
"""

from __future__ import annotations

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gradio as gr

# ---- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("offline-media-search")

# ---- Silence noisy tokenizer warnings (optional) ------------------------------
warnings.filterwarnings("ignore", message=r"You're using a Qwen2TokenizerFast tokenizer.*")
warnings.filterwarnings("ignore", message=r"`max_length` is ignored when `padding`=`True`.*")
try:
    from transformers.utils import logging as hf_logging  # type: ignore

    hf_logging.set_verbosity_error()
except Exception:
    pass

# ---- Local modules ------------------------------------------------------------
from priors import parse_query_intent, compute_prior_score
from offline_store import OfflineStore

# ---- Make src import work without PYTHONPATH hassles --------------------------
QWEN_REPO = Path("/workspace/Qwen3-VL-Embedding")
if QWEN_REPO.exists():
    sys.path.insert(0, str(QWEN_REPO))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

try:
    from src.models.qwen3_vl_reranker import Qwen3VLReranker

    HAS_RERANKER = True
except Exception:
    HAS_RERANKER = False

# ---- Configuration ------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "/workspace/offline/index.sqlite")
FRAMES_FAISS = os.getenv("FRAMES_FAISS", "/workspace/offline/frames.faiss")
TX_FAISS = os.getenv("TX_FAISS", "/workspace/offline/transcripts.faiss")

MODEL_EMBED = os.getenv("MODEL_EMBED", "Qwen/Qwen3-VL-Embedding-2B")
MODEL_RERANK = os.getenv("MODEL_RERANK", "Qwen/Qwen3-VL-Reranker-2B")

INSTRUCTION = "Retrieve images relevant to the user's query."
DIM = int(os.getenv("DIM", "2048"))

# Filtering to avoid "show everything" on tiny corpora:
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.22"))      # absolute cutoff
KEEP_RATIO = float(os.getenv("KEEP_RATIO", "0.75"))    # keep >= best * KEEP_RATIO
MAX_SHOW = int(os.getenv("MAX_SHOW", "20"))            # gallery size
TOPN_RERANK = int(os.getenv("TOPN_RERANK", "12"))      # rerank top-N only

SERVER_NAME = os.getenv("SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))

ALLOWED_PATHS = [
    "/workspace/media",
    "/workspace/cache",
    "/workspace/offline",
]

# ---- Core objects -------------------------------------------------------------
store = OfflineStore(DB_PATH, FRAMES_FAISS, TX_FAISS, dim=DIM)
embedder = Qwen3VLEmbedder(model_name_or_path=MODEL_EMBED)

# Reranker is heavy; initialize lazily once (first time user toggles it)
_reranker: Optional["Qwen3VLReranker"] = None

# Cache index in memory so repeated searches don't reload it every click
_frames_index_cached = None
_frames_index_path_cached: Optional[Path] = None


def get_reranker() -> "Qwen3VLReranker":
    global _reranker
    if _reranker is None:
        if not HAS_RERANKER:
            raise RuntimeError("Reranker requested but not available in this environment.")
        log.info("Initializing reranker model: %s", MODEL_RERANK)
        _reranker = Qwen3VLReranker(model_name_or_path=MODEL_RERANK)
    return _reranker


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _embed_query(q: str) -> np.ndarray:
    e = _to_numpy(embedder.process([{"text": q, "instruction": INSTRUCTION}])).astype("float32")
    if e.ndim != 2 or e.shape[0] < 1:
        raise ValueError(f"Unexpected embedding shape: {getattr(e, 'shape', None)}")
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


def _load_frames_index() -> Any:
    """
    Loads and caches the FAISS index.
    If the file path changes (unlikely), cache is invalidated.
    """
    global _frames_index_cached, _frames_index_path_cached

    frames_path = Path(FRAMES_FAISS)
    if not frames_path.exists():
        raise FileNotFoundError(f"Missing {FRAMES_FAISS}. Run: python ingest.py")

    if _frames_index_cached is None or _frames_index_path_cached != frames_path:
        log.info("Loading FAISS index: %s", frames_path)
        _frames_index_cached = store.load_index(frames_path)
        _frames_index_path_cached = frames_path

    return _frames_index_cached


def search(q: str, topk: int, use_rerank: bool) -> Tuple[List[Tuple[str, str]], str]:
    """
    Returns:
      - gallery: list[(image_path, label)]
      - debug: multiline string
    """
    try:
        q = (q or "").strip()
        if not q:
            return [], "Type a query."

        frames_index = _load_frames_index()
        qv = _embed_query(q)

        # 1) Raw FAISS Search
        hits = store.search_with_scores(frames_index, qv, int(topk))
        if not hits:
            return [], "No hits (unexpected)."

        best = float(hits[0][1])
        keep_min = max(MIN_SCORE, best * KEEP_RATIO)

        # 2) Hard threshold filtering
        hits = [(i, float(s)) for (i, s) in hits if float(s) >= keep_min]
        if not hits:
            return [], f"No results above threshold ({keep_min:.3f}). Try a broader query."

        ids = [int(i) for (i, _) in hits]
        rows = store.get_frames_by_ids(ids)

        score_by_id = {int(i): float(s) for (i, s) in hits}
        for r in rows:
            r["score"] = score_by_id.get(int(r["id"]), 0.0)

        # 3) Apply Priors (Logic Insertion)
        intent = parse_query_intent(q)
        for r in rows:
            base_score = float(r.get("score", 0.0))
            prior_boost = float(compute_prior_score(r, intent))
            r["final_score"] = base_score + prior_boost

        # 4) Final sort incorporating priors
        rows.sort(key=lambda x: float(x.get("final_score", x.get("score", 0.0))), reverse=True)

        # 5) Optional reranker on top N only (expensive)
        rerank_applied = False
        if use_rerank and HAS_RERANKER and rows:
            reranker = get_reranker()
            N = min(TOPN_RERANK, len(rows))

            # Build reranker docs from current top-N (already sorted by priors)
            docs = [{"image": r["path"], "text": r.get("name") or ""} for r in rows[:N]]

            inputs = {
                "instruction": INSTRUCTION,
                "query": {"text": q},
                "documents": docs,
                "fps": 1.0,
                "max_frames": 64,
            }

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
                key=lambda x: (
                    float(x.get("rerank_score", -1e9)),
                    float(x.get("final_score", x.get("score", 0.0))),
                ),
                reverse=True,
            )
            rerank_applied = True

        # Build outputs
        gallery: List[Tuple[str, str]] = []
        debug_lines: List[str] = []

        for r in rows[:MAX_SHOW]:
            gallery.append((r["path"], _label(r)))
            debug_lines.append(_label(r))

        best_rerank_line = ""
        if rerank_applied:
            topn = rows[: min(TOPN_RERANK, len(rows))]
            if any(r.get("rerank_score") is not None for r in topn):
                best_rerank = max(float(r.get("rerank_score", -1e9)) for r in topn)
                best_rerank_line = f"Best rerank (topN): {best_rerank:.3f}\n"

        debug = (
            f"Query: {q}\n"
            f"Intent: {intent}\n"
            f"Raw best score: {best:.3f}\n"
            f"{best_rerank_line}"
            f"Threshold used: {keep_min:.3f} (MIN_SCORE={MIN_SCORE}, KEEP_RATIO={KEEP_RATIO})\n"
            f"Candidates (post-threshold): {len(rows)} | Showing: {min(MAX_SHOW, len(rows))}\n\n"
            + "\n".join(debug_lines)
        )

        return gallery, debug

    except FileNotFoundError as e:
        return [], str(e)
    except Exception as e:
        log.exception("Search failed")
        return [], f"Search failed: {type(e).__name__}: {e}"


# ---- UI ----------------------------------------------------------------------
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
        use_rerank = gr.Checkbox(value=False, label=f"Use reranker (top ~{TOPN_RERANK})")

    if not HAS_RERANKER:
        gr.Markdown("ℹ️ Reranker not available (embedding-only mode).")

    btn = gr.Button("Search")
    gallery = gr.Gallery(label="Results", columns=2, height=520)
    debug = gr.Textbox(label="Debug", lines=14)

    btn.click(search, [q, topk, use_rerank], [gallery, debug])

# Production-ish defaults:
# - queue() helps under concurrency
# - show_error=False keeps UI cleaner; we still return a readable error string
demo.queue()
demo.launch(
    server_name=SERVER_NAME,
    server_port=SERVER_PORT,
    allowed_paths=ALLOWED_PATHS,
    show_error=False,
)
