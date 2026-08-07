"""
User tower: session/clickstream features instead of user_id embedding.

Pools recent item embeddings (128-d, from item tower) + optional demographics → MLP → 128-d.
Adapted from external/two-tower-retrieval-system/src/models/two_tower.py MLP pattern.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_DIM = 128


class UserTower(nn.Module):
    """
    Encodes a user from their recent interaction history (item embeddings) and
    optional demographic features — no user_id lookup table.
    """

    def __init__(
        self,
        embedding_dim: int = OUTPUT_DIM,
        hidden_dim: int = 256,
        max_history: int = 10,
        demo_dim: int = 0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_history = max_history
        self.demo_dim = demo_dim

        mlp_in = embedding_dim + demo_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def _pool_history(self, history_embeddings: torch.Tensor, history_mask: torch.Tensor | None) -> torch.Tensor:
        """
        Mean-pool recent item embeddings.

        Args:
            history_embeddings: (B, H, D)
            history_mask: (B, H) bool, True = valid history slot
        """
        if history_mask is None:
            return history_embeddings.mean(dim=1)

        mask = history_mask.float().unsqueeze(-1)
        summed = (history_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        demographics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            history_embeddings: (B, H, D) recent item 128-d vectors
            history_mask: optional (B, H) validity mask
            demographics: optional (B, demo_dim) normalized demo features

        Returns:
            L2-normalized (B, D) user vectors
        """
        pooled = self._pool_history(history_embeddings, history_mask)

        if demographics is not None and self.demo_dim > 0:
            pooled = torch.cat([pooled, demographics], dim=-1)

        emb = self.mlp(pooled)
        return F.normalize(emb, dim=-1)

    def encode_from_item_table(
        self,
        history_item_ids: torch.Tensor,
        item_embedding_table: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        demographics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Lookup recent items in a precomputed embedding table then encode user.

        Args:
            history_item_ids: (B, H) integer item indices
            item_embedding_table: (num_items, D) frozen or trainable table
        """
        history_embeddings = item_embedding_table[history_item_ids]
        return self.forward(history_embeddings, history_mask, demographics)
