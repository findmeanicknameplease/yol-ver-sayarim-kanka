# offline_store.py
# Offline store: SQLite metadata + FAISS indexes (frames, optional transcripts)
# - Uses cosine similarity via normalized vectors + IndexFlatIP
# - Score-aware search (returns ids + scores)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import faiss


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype("float32", copy=False)
    if x.ndim == 1:
        denom = np.linalg.norm(x) + eps
        return x / denom
    denom = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / denom


@dataclass
class OfflineStore:
    db_path: str
    frames_index_path: str
    tx_index_path: str
    dim: int = 2048

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,              -- 'photo' or 'video_frame'
                    path TEXT NOT NULL,              -- local filepath to image
                    name TEXT,                       -- filename or label
                    video_path TEXT,                 -- for video_frame
                    t_sec REAL                       -- timestamp for video_frame
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_path TEXT NOT NULL,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    text TEXT NOT NULL
                );
                """
            )
            conn.commit()

    # ---------- FAISS ----------
    def new_index(self) -> faiss.Index:
        # cosine similarity via inner product on L2-normalized vectors
        return faiss.IndexFlatIP(self.dim)

    def save_index(self, index: faiss.Index, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))

    def load_index(self, path: Path) -> faiss.Index:
        return faiss.read_index(str(path))

    # ---------- Insert metadata ----------
    def add_frame(self, kind: str, path: str, name: str = "", video_path: str = "", t_sec: float = 0.0) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO frames(kind, path, name, video_path, t_sec) VALUES(?,?,?,?,?)",
                (kind, path, name, video_path if video_path else None, t_sec if kind == "video_frame" else None),
            )
            conn.commit()
            return int(cur.lastrowid)

    def add_transcript(self, video_path: str, start: float, end: float, text: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO transcripts(video_path, start, end, text) VALUES(?,?,?,?)",
                (video_path, float(start), float(end), text),
            )
            conn.commit()
            return int(cur.lastrowid)

    # ---------- Fetch metadata by ids ----------
    def get_frames_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        q = "SELECT * FROM frames WHERE id IN (%s)" % ",".join(["?"] * len(ids))
        with self._conn() as conn:
            rows = conn.execute(q, ids).fetchall()
        by_id = {int(r["id"]): dict(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def get_tx_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        q = "SELECT * FROM transcripts WHERE id IN (%s)" % ",".join(["?"] * len(ids))
        with self._conn() as conn:
            rows = conn.execute(q, ids).fetchall()
        by_id = {int(r["id"]): dict(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    # ---------- Search ----------
    def search_with_scores(self, index: faiss.Index, qvec: np.ndarray, topk: int) -> List[Tuple[int, float]]:
        """
        Returns list of (db_id, score). db_id assumes insertion order == vector order.
        We enforce that by always adding vectors in the same order as DB inserts.
        """
        q = l2_normalize(qvec.reshape(1, -1))
        scores, pos = index.search(q, int(topk))
        scores = scores[0].tolist()
        pos = pos[0].tolist()

        out: List[Tuple[int, float]] = []
        for p, s in zip(pos, scores):
            if p is None or p < 0:
                continue
            db_id = int(p) + 1  # vector position 0 -> db id 1 (because AUTOINCREMENT)
            out.append((db_id, float(s)))
        return out
