"""
Evaluation metrics: Recall@k, NDCG@k, cold-start slice.

Adapted from external/two-tower-retrieval-system/src/evaluation/metrics.py
with FAISS-based retrieval for scaled eval.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def _build_ground_truth(test_df: pd.DataFrame) -> dict[int, set[int]]:
    gt: dict[int, set[int]] = {}
    for row in test_df.itertuples():
        gt.setdefault(int(row.user_idx), set()).add(int(row.item_idx))
    return gt


def recall_at_k(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    ground_truth: dict[int, set[int]],
    k: int = 10,
    train_mask: dict[int, set[int]] | None = None,
) -> float:
    """Fraction of users with at least one relevant item in top-k."""
    hits = 0
    users = list(ground_truth.keys())
    for user_idx in users:
        query = user_embeddings[user_idx : user_idx + 1]
        scores = query @ item_embeddings.T
        if train_mask and user_idx in train_mask:
            for ti in train_mask[user_idx]:
                scores[0, ti] = -1e9
        top_k = np.argpartition(-scores[0], min(k, len(scores[0]) - 1))[:k]
        if ground_truth[user_idx] & set(top_k.tolist()):
            hits += 1
    return hits / max(len(users), 1)


def ndcg_at_k(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    ground_truth: dict[int, set[int]],
    k: int = 10,
    train_mask: dict[int, set[int]] | None = None,
) -> float:
    """NDCG@k assuming binary relevance."""
    ndcg_total = 0.0
    users = list(ground_truth.keys())
    for user_idx in users:
        query = user_embeddings[user_idx : user_idx + 1]
        scores = query @ item_embeddings.T
        if train_mask and user_idx in train_mask:
            for ti in train_mask[user_idx]:
                scores[0, ti] = -1e9
        top_k = np.argsort(-scores[0])[:k]
        relevant = ground_truth[user_idx]
        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, item in enumerate(top_k)
            if item in relevant
        )
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        ndcg_total += dcg / idcg if idcg > 0 else 0.0
    return ndcg_total / max(len(users), 1)


def cold_start_recall_at_k(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    ground_truth: dict[int, set[int]],
    history_lengths: dict[int, int],
    k: int = 10,
    max_history_for_cold: int = 2,
    train_mask: dict[int, set[int]] | None = None,
) -> float:
    """Recall@k restricted to users with <= max_history_for_cold interactions."""
    cold_users = {u for u, h in history_lengths.items() if h <= max_history_for_cold}
    if not cold_users:
        return 0.0
    subset_gt = {u: ground_truth[u] for u in cold_users if u in ground_truth}
    if not subset_gt:
        return 0.0

    hits = 0
    for user_idx in subset_gt:
        query = user_embeddings[user_idx : user_idx + 1]
        scores = query @ item_embeddings.T
        if train_mask and user_idx in train_mask:
            for ti in train_mask[user_idx]:
                scores[0, ti] = -1e9
        top_k = np.argpartition(-scores[0], min(k, len(scores[0]) - 1))[:k]
        if subset_gt[user_idx] & set(top_k.tolist()):
            hits += 1
    return hits / len(subset_gt)


def evaluate_model(
    model,
    test_df: pd.DataFrame,
    item_embeddings: np.ndarray,
    user_histories: dict[int, list[int]],
    device: torch.device,
    max_history: int = 10,
    train_interactions: dict[int, list[int]] | None = None,
) -> dict[str, float]:
    """Run full metric suite on a trained DiscoveryModel."""
    model.eval()
    num_users = max(test_df["user_idx"].max() + 1, max(user_histories.keys()) + 1)

    user_emb_list = []
    with torch.no_grad():
        for user_idx in range(num_users):
            hist = user_histories.get(user_idx, [])
            pad = max_history - len(hist)
            hist_ids = [0] * pad + hist[-max_history:]
            mask = [0.0] * pad + [1.0] * len(hist[-max_history:])
            h = torch.tensor([hist_ids], dtype=torch.long, device=device)
            m = torch.tensor([mask], dtype=torch.float, device=device)
            u = model.encode_users(h, m)
            user_emb_list.append(u.cpu().numpy()[0])

    user_embeddings = np.vstack(user_emb_list)
    gt = _build_ground_truth(test_df)
    train_mask = {u: set(items) for u, items in (train_interactions or {}).items()}
    history_lengths = {u: len(h) for u, h in user_histories.items()}

    return {
        "recall@10": recall_at_k(user_embeddings, item_embeddings, gt, k=10, train_mask=train_mask),
        "recall@20": recall_at_k(user_embeddings, item_embeddings, gt, k=20, train_mask=train_mask),
        "ndcg@10": ndcg_at_k(user_embeddings, item_embeddings, gt, k=10, train_mask=train_mask),
        "cold_start_recall@10": cold_start_recall_at_k(
            user_embeddings,
            item_embeddings,
            gt,
            history_lengths,
            k=10,
            train_mask=train_mask,
        ),
    }
