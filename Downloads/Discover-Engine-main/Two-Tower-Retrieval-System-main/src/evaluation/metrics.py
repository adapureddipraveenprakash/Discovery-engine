import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _score_user(model, user_idx, train_df, test_df, num_items, k, device):
    """
    Return (top_k_indices, true_items) for one user.
    Computes scores through the full towers and masks training positives.
    """
    user_tensor = torch.tensor([user_idx]).to(device)
    all_items = torch.arange(num_items).to(device)

    user_emb = model.encode_users(user_tensor)          # (1, D)
    item_emb = model.encode_items(all_items)             # (num_items, D)

    scores = torch.matmul(user_emb, item_emb.T)         # (1, num_items)

    train_items = train_df[train_df["user_idx"] == user_idx]["item_idx"].values.copy()
    scores[0, train_items] = -1e9

    top_k = torch.topk(scores, k=k).indices.cpu().numpy()[0]
    true_items = test_df[test_df["user_idx"] == user_idx]["item_idx"].values
    return top_k, true_items


def recall_at_k(model, train_df, test_df, num_items, k=10, device="cpu"):
    """Fraction of users with at least one relevant item in their top-k recommendations."""
    model.eval()
    hits = 0

    with torch.no_grad():
        users = test_df["user_idx"].unique()
        for user in users:
            top_k, true_items = _score_user(model, user, train_df, test_df, num_items, k, device)
            if any(item in top_k for item in true_items):
                hits += 1

    return hits / len(users)


def mrr_at_k(model, train_df, test_df, num_items, k=10, device="cpu"):
    """Mean reciprocal rank of the first relevant item in the top-k list."""
    model.eval()
    rr_total = 0.0

    with torch.no_grad():
        users = test_df["user_idx"].unique()
        for user in users:
            top_k, true_items = _score_user(model, user, train_df, test_df, num_items, k, device)
            for rank, item in enumerate(top_k):
                if item in true_items:
                    rr_total += 1 / (rank + 1)
                    break

    return rr_total / len(users)


def ndcg_at_k(model, train_df, test_df, num_items, k=10, device="cpu"):
    """Normalized discounted cumulative gain at k (ideal DCG = 1 hit)."""
    model.eval()
    ndcg_total = 0.0

    with torch.no_grad():
        users = test_df["user_idx"].unique()
        for user in users:
            top_k, true_items = _score_user(model, user, train_df, test_df, num_items, k, device)
            dcg = sum(
                1 / np.log2(rank + 2)
                for rank, item in enumerate(top_k)
                if item in true_items
            )
            ndcg_total += dcg  # IDCG = 1.0 (one relevant item per user)

    return ndcg_total / len(users)
