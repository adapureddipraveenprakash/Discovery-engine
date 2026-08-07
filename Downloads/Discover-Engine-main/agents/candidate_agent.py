"""
Candidate generation agent: session history → user tower → FAISS → filter seen.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from index.faiss_store import FaissStore
from models.discovery_model import DiscoveryModel


class CandidateAgent:
    def __init__(
        self,
        model: DiscoveryModel,
        faiss_store: FaissStore,
        user_histories: dict[int, list[int]],
        user_interactions: dict[int, list[int]],
        reverse_item_map: dict[int, str],
        device: torch.device,
        max_history: int = 10,
    ):
        self.model = model
        self.faiss_store = faiss_store
        self.user_histories = user_histories
        self.user_interactions = user_interactions
        self.reverse_item_map = reverse_item_map
        self.device = device
        self.max_history = max_history

    def recommend(
        self,
        user_id: int | str,
        user_map: dict,
        top_k: int = 10,
        explain: bool = True,
    ) -> dict[str, Any]:
        start = time.time()

        if user_id not in user_map:
            raise KeyError(f"User {user_id} not found")

        user_idx = user_map[user_id]
        hist = self.user_histories.get(user_idx, [])
        seen = set(self.user_interactions.get(user_idx, []))

        pad = self.max_history - len(hist)
        hist_ids = [0] * pad + hist[-self.max_history :]
        mask = [0.0] * pad + [1.0] * len(hist[-self.max_history :])

        history_tensor = torch.tensor([hist_ids], dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor([mask], dtype=torch.float, device=self.device)

        self.model.eval()
        with torch.no_grad():
            user_emb = self.model.encode_users(history_tensor, mask_tensor)

        user_np = user_emb.cpu().numpy().astype("float32")
        scores, indices = self.faiss_store.search(user_np, top_k, exclude=seen)

        rec_indices = indices[0].tolist() if len(indices) else []
        rec_scores = scores[0].tolist() if len(scores) else []
        recommendations = [int(i) for i in rec_indices[:top_k]]

        items = []
        if explain:
            for idx, score in zip(recommendations, rec_scores):
                items.append(
                    {
                        "item_idx": idx,
                        "score": float(score),
                        "reason": "session history embedding similarity (excludes seen items)",
                    }
                )

        latency_ms = round((time.time() - start) * 1000, 2)
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "items": items,
            "latency_ms": latency_ms,
        }
