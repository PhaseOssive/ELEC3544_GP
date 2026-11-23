"""BGE text embeddings with a deterministic disk cache.

Embeddings are computed from text only, so they can be precomputed for the
whole corpus without label leakage.  Any dimensionality reduction/PCA is fitted
later on training rows only.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class EmbeddingStore:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        cache_dir: str | Path = "cache",
        batch_size: int = 64,
        device: Optional[str] = None,
        max_seq_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.batch_size = int(batch_size)
        self.device = device
        self.max_seq_length = int(max_seq_length)
        self.matrix: Optional[np.ndarray] = None
        self.ids: Optional[list[str]] = None
        self._id_to_row: Optional[dict[str, int]] = None

    def _cache_key(self, frame: pd.DataFrame) -> str:
        digest = hashlib.sha256()
        digest.update(self.model_name.encode("utf-8"))
        for paper_id, text in zip(frame["paper_id"], frame["text_all"]):
            digest.update(str(paper_id).encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(str(text).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()[:32]

    def _paths(self, key: str) -> tuple[Path, Path]:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.model_name).strip("_")[:60]
        return (
            self.cache_dir / f"bge_{slug}_{key}.npy",
            self.cache_dir / f"bge_{slug}_{key}.ids.json",
        )

    def load_or_compute(self, frame: pd.DataFrame) -> "EmbeddingStore":
        key = self._cache_key(frame)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        matrix_path, ids_path = self._paths(key)
        expected_ids = [str(value) for value in frame["paper_id"].tolist()]

        if matrix_path.is_file() and ids_path.is_file():
            try:
                matrix = np.load(matrix_path, mmap_mode="r")
                with ids_path.open("r", encoding="utf-8") as handle:
                    cached_ids = json.load(handle)
                if cached_ids == expected_ids and matrix.shape[0] == len(expected_ids):
                    self.matrix = np.asarray(matrix, dtype=np.float32)
                    self.ids = cached_ids
                    self._id_to_row = {pid: index for index, pid in enumerate(cached_ids)}
                    return self
            except Exception:
                # Corrupt/incompatible cache: recompute below.
                pass

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "sentence-transformers is required for BGE embeddings. Install "
                "requirements.txt, or run with --no-bge."
            ) from exc

        device = self.device
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        model = SentenceTransformer(self.model_name, device=device)
        model.max_seq_length = self.max_seq_length
        texts = frame["text_all"].astype(str).tolist()
        matrix = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32, copy=False)
        np.save(matrix_path, matrix)
        with ids_path.open("w", encoding="utf-8") as handle:
            json.dump(expected_ids, handle)
        self.matrix = matrix
        self.ids = expected_ids
        self._id_to_row = {pid: index for index, pid in enumerate(expected_ids)}
        return self

    def get(self, frame: pd.DataFrame) -> np.ndarray:
        if self.matrix is None or self._id_to_row is None:
            raise RuntimeError("EmbeddingStore not loaded/computed.")
        rows = [self._id_to_row[str(pid)] for pid in frame["paper_id"]]
        return self.matrix[rows]
