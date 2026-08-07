"""
Complete-the-look agent: seed item → complementary categories + diversity rules.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from guardrails.diversity import DEFAULT_MAX_CATEGORY_FRACTION, apply_category_cap
from index.faiss_store import FaissStore

# Category complementarity map for outfit building
COMPLEMENTARY_CATEGORIES: dict[str, list[str]] = {
    "Topwear": ["Bottomwear", "Footwear", "Accessories", "Outerwear"],
    "Bottomwear": ["Topwear", "Footwear", "Accessories", "Outerwear"],
    "Dress": ["Footwear", "Accessories", "Outerwear"],
    "Footwear": ["Topwear", "Bottomwear", "Dress", "Accessories"],
    "Accessories": ["Topwear", "Bottomwear", "Dress", "Footwear"],
    "Outerwear": ["Topwear", "Bottomwear", "Dress", "Footwear"],
}

MAX_CATEGORY_FRACTION = DEFAULT_MAX_CATEGORY_FRACTION


class CompleteTheLookAgent:
    def __init__(
        self,
        faiss_store: FaissStore,
        item_embeddings: np.ndarray,
        item_meta: list[dict],
    ):
        self.faiss_store = faiss_store
        self.item_embeddings = item_embeddings.astype("float32")
        self.item_meta = item_meta
        self.idx_to_category = {m["item_idx"]: m.get("category", "Unknown") for m in item_meta}

    def _seed_vector(self, seed_item_idx: int) -> np.ndarray:
        return self.item_embeddings[seed_item_idx].reshape(1, -1)

    def _is_complementary(self, seed_category: str, candidate_category: str) -> bool:
        if seed_category == candidate_category:
            return False
        allowed = COMPLEMENTARY_CATEGORIES.get(seed_category, [])
        return candidate_category in allowed or seed_category in COMPLEMENTARY_CATEGORIES.get(
            candidate_category, []
        )

    def complete(
        self,
        seed_item_idx: int,
        top_k: int = 10,
        fetch_multiplier: int = 10,
        explain: bool = True,
    ) -> dict[str, Any]:
        start = time.time()

        if seed_item_idx < 0 or seed_item_idx >= len(self.item_embeddings):
            raise ValueError(f"Invalid seed_item_idx: {seed_item_idx}")

        seed_category = self.idx_to_category.get(seed_item_idx, "Unknown")
        seed_vec = self._seed_vector(seed_item_idx)

        fetch_k = min(top_k * fetch_multiplier, self.faiss_store.ntotal)
        _, indices = self.faiss_store.search(seed_vec, fetch_k, exclude={seed_item_idx})

        filtered = [
            int(i)
            for i in indices[0]
            if i >= 0 and self._is_complementary(seed_category, self.idx_to_category.get(i, ""))
        ]

        diversity = apply_category_cap(
            filtered,
            top_k,
            self.idx_to_category,
            max_category_fraction=MAX_CATEGORY_FRACTION,
        )
        recommendations = diversity.selected

        items = []
        if explain:
            for idx in recommendations:
                cat = self.idx_to_category.get(idx, "Unknown")
                items.append(
                    {
                        "item_idx": idx,
                        "category": cat,
                        "reason": f"complementary to {seed_category}; within 35% category cap",
                    }
                )

        latency_ms = round((time.time() - start) * 1000, 2)
        return {
            "seed_item_idx": seed_item_idx,
            "seed_category": seed_category,
            "recommendations": recommendations,
            "items": items,
            "diversity_audit": diversity.to_audit_dict(),
            "latency_ms": latency_ms,
        }
