"""
Search agent: text and/or image query → FashionCLIP → project → FAISS.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from index.faiss_store import FaissStore
from models.item_tower import ItemTower


class SearchAgent:
    def __init__(
        self,
        item_tower: ItemTower,
        faiss_store: FaissStore,
        device: torch.device,
    ):
        self.item_tower = item_tower
        self.faiss_store = faiss_store
        self.device = device

    def search(
        self,
        query_text: str | None = None,
        query_image: str | None = None,
        top_k: int = 10,
        explain: bool = True,
    ) -> dict[str, Any]:
        start = time.time()

        if not query_text and not query_image:
            raise ValueError("Provide query_text and/or query_image")

        texts = [query_text or "fashion item"]
        images = [query_image] if query_image else None

        with torch.no_grad():
            query_emb = self.item_tower.encode_from_raw(
                images=images,
                texts=texts,
                device=self.device,
            )

        query_np = query_emb.cpu().numpy().astype("float32")
        scores, indices = self.faiss_store.search(query_np, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            row: dict[str, Any] = {"item_idx": int(idx), "score": float(score)}
            if explain:
                row["reason"] = "FashionCLIP query embedding inner-product match"
            results.append(row)

        latency_ms = round((time.time() - start) * 1000, 2)
        return {
            "query_text": query_text,
            "query_image": query_image,
            "results": results,
            "latency_ms": latency_ms,
        }
