# ingest.py
# Visual-first ingest:
# - Photos -> 1 embedding each
# - Videos -> sampled frames -> embeddings + timestamps
# - Writes:
#   /workspace/offline/index.sqlite
#   /workspace/offline/frames.faiss
#   (no transcripts.faiss unless you add ASR later)

from __future__ import annotations

import os
import math
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image

from offline_store import OfflineStore, l2_normalize

# Make src import work without PYTHONPATH headaches
QWEN_REPO = Path("/workspace/Qwen3-VL-Embedding")
if QWEN_REPO.exists():
    import sys
    sys.path.insert(0, str(QWEN_REPO))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

MEDIA_DIR = Path("/workspace/media")
CACHE_DIR = Path("/workspace/cache/frames")
OFFLINE_DIR = Path("/workspace/offline")

DB_PATH = str(OFFLINE_DIR / "index.sqlite")
FRAMES_FAISS = str(OFFLINE_DIR / "frames.faiss")
TX_FAISS = str(OFFLINE_DIR / "transcripts.faiss")  # not created in visual-first

DIM = 2048
INSTRUCTION = "Retrieve images relevant to the user's query."

# Video sampling
FPS = 1.0            # 1 frame per second (KISS)
MAX_FRAMES = 180     # cap per video to prevent runaway indexing


IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VID_EXT


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def ffprobe_duration(path: Path) -> float:
    # duration in seconds
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        text=True
    ).strip()
    try:
        return float(out)
    except Exception:
        return 0.0


def extract_video_frames(video_path: Path, fps: float, max_frames: int) -> List[Tuple[Path, float]]:
    """
    Extract frames to CACHE_DIR/<video_stem>/frame_000001.jpg
    Returns list of (frame_path, t_sec).
    """
    duration = ffprobe_duration(video_path)
    if duration <= 0:
        return []

    n_frames = min(max_frames, max(1, int(math.floor(duration * fps))))
    out_dir = CACHE_DIR / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames at fixed fps
    # Note: This is KISS sampling. Better later: scene-change keyframes.
    pattern = str(out_dir / "frame_%06d.jpg")
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern
    ])

    frames = sorted(out_dir.glob("frame_*.jpg"))
    frames = frames[:n_frames]

    out: List[Tuple[Path, float]] = []
    for i, fp in enumerate(frames):
        t = float(i) / float(fps)
        out.append((fp, t))
    return out


def pil_load(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def main():
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OFFLINE_DIR.mkdir(parents=True, exist_ok=True)

    store = OfflineStore(DB_PATH, FRAMES_FAISS, TX_FAISS, dim=DIM)
    store.init_db()

    embedder = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")

    items: List[Dict[str, Any]] = []

    # Collect images
    for p in sorted(MEDIA_DIR.rglob("*")):
        if p.is_file() and is_image(p):
            items.append({"kind": "photo", "path": str(p), "name": p.name})

    # Collect videos -> frames
    for v in sorted(MEDIA_DIR.rglob("*")):
        if v.is_file() and is_video(v):
            frame_list = extract_video_frames(v, FPS, MAX_FRAMES)
            for fp, t in frame_list:
                items.append({
                    "kind": "video_frame",
                    "path": str(fp),
                    "name": fp.name,
                    "video_path": str(v),
                    "t_sec": float(t),
                })

    if not items:
        print(f"⚠️ No media found in {MEDIA_DIR}. Put a few jpg/png/mp4 files there.")
        return

    # Insert metadata in DB in the same order we will add vectors to FAISS
