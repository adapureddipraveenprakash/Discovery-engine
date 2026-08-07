"""
Convenience wrapper that computes all three ranking metrics in a single pass,
avoiding redundant embedding computation that occurs when calling each metric separately.
"""
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_all_metrics(model, train_df, test_df, num_items: int, k: int = 10, device="cpu") -> dict:
    """
    Compute Recall@k, MRR@k, and NDCG@k in one forward pass per user.

    Returns:
        dict with keys "recall", "mrr", "ndcg"
    """
    model.eval()
    hits = 0
    rr_total = 0.0
    ndcg_total = 0.0

    users = test_df["user_idx"].unique()

    with torch.no_grad():
        # Pre-compute all item embeddings once
        all_items = torch.arange(num_items).to(device)
        item_emb = model.encode_items(all_items)  # (num_items, D)

        for user in users:
            user_tensor = torch.tensor([user]).to(device)
            user_emb = model.encode_users(user_tensor)              # (1, D)
            scores = torch.matmul(user_emb, item_emb.T)             # (1, num_items)

            train_items = train_df[train_df["user_idx"] == user]["item_idx"].values.copy()
            scores[0, train_items] = -1e9

            top_k = torch.topk(scores, k=k).indices.cpu().numpy()[0]
            true_items = set(test_df[test_df["user_idx"] == user]["item_idx"].values)

            hit = False
            for rank, item in enumerate(top_k):
                if item in true_items:
                    if not hit:
                        hits += 1
                        rr_total += 1 / (rank + 1)
                        hit = True
                    ndcg_total += 1 / np.log2(rank + 2)

    n = len(users)
    results = {"recall": hits / n, "mrr": rr_total / n, "ndcg": ndcg_total / n}
    logger.info("Recall@%d=%.4f  MRR@%d=%.4f  NDCG@%d=%.4f", k, results["recall"], k, results["mrr"], k, results["ndcg"])
    return results
