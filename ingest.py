# /workspace/demo/ingest.py
# Production-grade (for a demo) offline ingest:
# - Rebuilds index from scratch every run (no FAISS<->DB mismatch)
# - Loud, deterministic logging (you always see what's happening)
# - Incremental embedding + FAISS add (low memory spikes)
# - Photos + optional video frame sampling (KISS 1 fps) with cache folder
# - Writes:
#     /workspace/offline/index.sqlite
#     /workspace/offline/frames.faiss
#   (No transcripts in this visual-first ingest)

from __future__ import annotations

import sys
import math
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable

import numpy as np
from PIL import Image
import faiss

from offline_store import OfflineStore, l2_normalize

# ---------- Paths ----------
MEDIA_DIR = Path("/workspace/media")
CACHE_DIR = Path("/workspace/cache/frames")
OFFLINE_DIR = Path("/workspace/offline")

DB_PATH = str(OFFLINE_DIR / "index.sqlite")
FRAMES_FAISS = str(OFFLINE_DIR / "frames.faiss")
TX_FAISS = str(OFFLINE_DIR / "transcripts.faiss")  # unused in visual-first

# Qwen repo import (either PYTHONPATH or direct sys.path insert)
QWEN_REPO = Path("/workspace/Qwen3-VL-Embedding")
if QWEN_REPO.exists():
    sys.path.insert(0, str(QWEN_REPO))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

# ---------- Config ----------
DIM = 2048
INSTRUCTION = "Retrieve images relevant to the user's query."

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

# Video sampling (KISS). Upgrade later to scene-change keyframes if needed.
FPS = 1.0
MAX_FRAMES_PER_VIDEO = 180

# Embedding batching (small to avoid spikes)
BATCH = 2

# Rebuild behavior (demo-safe)
REBUILD_EACH_RUN = True
WIPE_FRAME_CACHE = False  # keep extracted frames between runs unless you want full reset

# ---------- Helpers ----------
def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VID_EXT


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        text=True
    ).strip()
    try:
        return float(out)
    except Exception:
        return 0.0


def extract_video_frames(video_path: Path, fps: float, max_frames: int) -> List[Tuple[Path, float]]:
    """
    Extract frames to CACHE_DIR/<video_stem>/frame_000001.jpg ...
    Returns list of (frame_path, t_sec).
    """
    dur = ffprobe_duration(video_path)
    if dur <= 0:
        return []

    n_frames = min(max_frames, max(1, int(math.floor(dur * fps))))
    out_dir = CACHE_DIR / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(out_dir / "frame_%06d.jpg")
    run(["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}", "-q:v", "2", pattern])

    frames = sorted(out_dir.glob("frame_*.jpg"))[:n_frames]
    return [(fp, float(i) / float(fps)) for i, fp in enumerate(frames)]


def pil_load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def iter_media_items() -> List[Dict[str, Any]]:
    """
    Returns ordered list of items:
      - photos: {"kind":"photo","path":..., "name":...}
      - video frames: {"kind":"video_frame","path":..., "name":..., "video_path":..., "t_sec":...}
    """
    items: List[Dict[str, Any]] = []

    # Photos first
    for p in sorted(MEDIA_DIR.rglob("*")):
        if p.is_file() and is_image(p):
            items.append({"kind": "photo", "path": str(p), "name": p.name})

    # Videos -> frames
    for v in sorted(MEDIA_DIR.rglob("*")):
        if v.is_file() and is_video(v):
            frames = extract_video_frames(v, FPS, MAX_FRAMES_PER_VIDEO)
            for fp, t in frames:
                items.append({
                    "kind": "video_frame",
                    "path": str(fp),
                    "name": fp.name,
                    "video_path": str(v),
                    "t_sec": float(t),
                })

    return items


def chunked(xs: List[Dict[str, Any]], n: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def main():
    t0 = time.time()
    log("=== OFFLINE INGEST (visual-first) ===")
    log(f"MEDIA_DIR:   {MEDIA_DIR} (exists={MEDIA_DIR.exists()})")
    log(f"CACHE_DIR:   {CACHE_DIR}")
    log(f"OFFLINE_DIR: {OFFLINE_DIR}")
    log(f"FPS={FPS} | MAX_FRAMES_PER_VIDEO={MAX_FRAMES_PER_VIDEO} | BATCH={BATCH}")

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OFFLINE_DIR.mkdir(parents=True, exist_ok=True)

    if REBUILD_EACH_RUN:
        log("Rebuild mode: ON (wiping DB + FAISS indexes)")
        for p in [DB_PATH, FRAMES_FAISS, TX_FAISS]:
            try:
                Path(p).unlink()
            except FileNotFoundError:
                pass

    if WIPE_FRAME_CACHE:
        log("Wiping extracted frame cache...")
        shutil.rmtree(str(CACHE_DIR), ignore_errors=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Collect items (this will also extract frames for videos)
    log("Scanning media + extracting video frames (if any)...")
    items = iter_media_items()
    log(f"Found {len(items)} total items (photos + video frames).")

    if len(items) == 0:
        log("⚠️ No media found. Put jpg/png/webp/mp4 files into /workspace/media")
        return

    # Init DB
    store = OfflineStore(DB_PATH, FRAMES_FAISS, TX_FAISS, dim=DIM)
    store.init_db()

    # Init index
    index = store.new_index()

    # Load embedder
    log("Loading Qwen3-VL-Embedding-2B (this can take a bit on first run)...")
    embedder = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")
    log("Embedder loaded.")

    done = 0
    for batch in chunked(items, BATCH):
        # 1) insert rows into DB in the same order we add vectors to FAISS
        for it in batch:
            store.add_frame(
                kind=it["kind"],
                path=it["path"],
                name=it.get("name", ""),
                video_path=it.get("video_path", ""),
                t_sec=float(it.get("t_sec", 0.0)),
            )

        # 2) embed batch
        batch_inputs = []
        for it in batch:
            batch_inputs.append({
                "image": pil_load(Path(it["path"])),
                "text": it.get("name", ""),
                "instruction": INSTRUCTION,
            })

        emb = np.asarray(embedder.process(batch_inputs), dtype="float32")
        emb = l2_normalize(emb)

        # 3) add to FAISS
        index.add(emb)

        done += len(batch)
        log(f"  ✅ embedded+indexed {done}/{len(items)}")

    # Save index
    store.save_index(index, FRAMES_FAISS)

    log("✅ Offline indexing complete.")
    log(f"   DB:         {DB_PATH}")
    log(f"   Frames:     {FRAMES_FAISS}")
    log(f"   Transcripts:(not built in visual-first mode)")
    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
