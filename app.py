# /workspace/demo/app.py
# Offline Media RAG Demo (FAISS + SQLite)
# - Works even if transcripts.faiss DOES NOT exist (visual-first mode)
# - Optional reranker toggle (only runs on top frames)

from pathlib import Path
import numpy as np
import gradio as gr

from offline_store import OfflineStore

# Qwen3-VL-Embedding repo must be on PYTHONPATH or installed editable
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
try:
    from src.models.qwen3_vl_reranker import Qwen3VLReranker
    HAS_RERANKER = True
except Exception:
    HAS_RERANKER = False

DB_PATH = "/workspace/offline/index.sqlite"
FRAMES_FAISS = "/workspace/offline/frames.faiss"
TX_FAISS = "/workspace/offline/transcripts.faiss"

INSTRUCTION = "Retrieve images or text relevant to the user's query."
DIM = 2048

store = OfflineStore(DB_PATH, FRAMES_FAISS, TX_FAISS, dim=DIM)
embedder = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _embed_query(q: str) -> np.ndarray:
    e = _to_numpy(embedder.process([{"text": q, "instruction": INSTRUCTION}])).astype("float32")
    return e[0]


def search(q: str, topk_frames: int, topk_tx: int, use_rerank: bool):
    q = (q or "").strip()
    if not q:
        return [], "Type a query."

    # Ensure frames index exists
    frames_path = Path(FRAMES_FAISS)
    if not frames_path.exists():
        return [], f"Missing frames index: {FRAMES_FAISS}. Run: python ingest.py"

    # Load indexes
    frames_index = store.load_index(frames_path)

    # Embed query
    qv = _embed_query(q)

    # Search frames/photos
    frame_ids = store.search(frames_index, qv, int(topk_frames))
    frames = store.get_frames_by_ids(frame_ids)

    # Optional transcripts (only if transcripts.faiss exists)
    tx = []
    tx_path = Path(TX_FAISS)
    if tx_path.exists():
        tx_index = store.load_index(tx_path)
        tx_ids = store.search(tx_index, qv, int(topk_tx))
        tx = store.get_tx_by_ids(tx_ids)

    # Optional rerank on top 20 frames
    if use_rerank and HAS_RERANKER and frames:
        reranker = Qwen3VLReranker(model_name_or_path="Qwen/Qwen3-VL-Reranker-2B")
        limit = min(20, len(frames))
        docs = []
        for p in frames[:limit]:
            docs.append({"image": p["path"], "text": p.get("name") or p.get("text") or ""})

        inputs = {
            "instruction": INSTRUCTION,
            "query": {"text": q},
            "documents": docs,
            "fps": 1.0,
            "max_frames": 64,
        }
        scores = reranker.process(inputs)
        for i, s in enumerate(scores):
            frames[i]["rerank_score"] = float(s)

        frames[:limit] = sorted(frames[:limit], key=lambda x: x.get("rerank_score", -1e9), reverse=True)

    # Build UI outputs
    gallery = []
    lines = []

    for p in frames[:20]:
        kind = p.get("kind", "item")
        label = kind
        if kind == "video_frame":
            label += f" | {Path(p['video_path']).name} @ {float(p['t_sec']):.1f}s"
        else:
            label += f" | {Path(p['path']).name}"

        if "rerank_score" in p:
            label += f" | rerank={p['rerank_score']:.3f}"

        gallery.append((p["path"], label))
        lines.append(label)

    tx_lines = []
    for p in tx[:10]:
        clip = f"{Path(p['video_path']).name} [{float(p['start']):.1f}-{float(p['end']):.1f}]"
        tx_lines.append(f"{clip}: {p['text'][:180]}")

    debug = "FRAMES:\n" + "\n".join(lines)
    if tx_path.exists():
        debug += "\n\nTRANSCRIPTS:\n" + ("\n".join(tx_lines) if tx_lines else "(none)")
    else:
        debug += "\n\nTRANSCRIPTS:\n(index not built yet — visual-only mode)"

    return gallery, debug


with gr.Blocks(title="Offline Media RAG Demo") as demo:
    gr.Markdown(
        "## Offline Media Search (FAISS + SQLite)\n"
        "- Visual-first: works without transcripts index\n"
        "- Optional reranker toggle (if available)\n"
    )
    q = gr.Textbox(label="Query", placeholder="e.g., 'receipt', 'sunset beach', 'passport', 'laughing'")
    with gr.Row():
        topk_frames = gr.Slider(5, 200, value=40, step=1, label="TopK frames/photos")
        topk_tx = gr.Slider(5, 200, value=20, step=1, label="TopK transcript segments (if built)")
        use_rerank = gr.Checkbox(value=False, label="Use reranker (top 20 frames)")

    if not HAS_RERANKER:
        gr.Markdown("⚠️ Reranker not available in this environment. (Embedding-only mode.)")

    btn = gr.Button("Search")
    gallery = gr.Gallery(label="Frames / Photos", columns=2, height=520)
    debug = gr.Textbox(label="Debug", lines=16)

    btn.click(search, [q, topk_frames, topk_tx, use_rerank], [gallery, debug])

demo.launch(server_name="0.0.0.0", server_port=7860)
