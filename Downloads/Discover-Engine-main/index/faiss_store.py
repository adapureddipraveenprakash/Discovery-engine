"""FAISS index wrapper for online retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from index.build_index import load_index, save_index


class FaissStore:
    """Thin wrapper around IndexFlatIP for search + persistence."""

    def __init__(self, index: faiss.Index | None = None, dim: int = 128):
        self.dim = dim
        self._index = index

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            raise RuntimeError("FAISS index not loaded")
        return self._index

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int,
        exclude: Iterable[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Inner-product search with optional post-filtering of excluded item ids.

        Returns:
            scores (N, top_k), indices (N, top_k)
        """
        queries = query_vectors.astype("float32")
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        fetch_k = top_k * 5 if exclude else top_k
        scores, indices = self.index.search(queries, min(fetch_k, self.ntotal))

        if exclude:
            exclude_set = set(exclude)
            filtered_scores = []
            filtered_indices = []
            for row_scores, row_indices in zip(scores, indices):
                kept_s, kept_i = [], []
                for s, i in zip(row_scores, row_indices):
                    if i < 0 or i in exclude_set:
                        continue
                    kept_s.append(s)
                    kept_i.append(i)
                    if len(kept_i) >= top_k:
                        break
                filtered_scores.append(kept_s)
                filtered_indices.append(kept_i)
            return np.array(filtered_scores, dtype="float32"), np.array(filtered_indices, dtype="int64")

        return scores, indices

    def save(self, path: Path | str) -> None:
        save_index(self.index, path)

    @classmethod
    def load(cls, path: Path | str) -> "FaissStore":
        index = load_index(path)
        return cls(index=index, dim=index.d)
