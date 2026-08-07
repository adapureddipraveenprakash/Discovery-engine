"""
Combined discovery model for InfoNCE training.

Item side uses precomputed/projected FashionCLIP embeddings; user side uses
session history pooling (no user_id embedding).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.item_tower import ItemTower
from models.user_tower import UserTower


class DiscoveryModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_embedding_table: torch.Tensor | None = None,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        max_history: int = 10,
        demo_dim: int = 0,
        temperature: float = 0.05,
        item_tower: ItemTower | None = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_history = max_history
        self.temperature = temperature

        self.item_tower = item_tower or ItemTower(output_dim=embedding_dim)
        self.user_tower = UserTower(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            max_history=max_history,
            demo_dim=demo_dim,
        )

        if item_embedding_table is not None:
            self.register_buffer("item_embeddings", item_embedding_table)
        else:
            self.register_buffer(
                "item_embeddings",
                torch.zeros(num_items, embedding_dim),
            )

    def set_item_embeddings(self, embeddings: torch.Tensor) -> None:
        self.item_embeddings = embeddings

    def set_fused_item_vectors(self, fused_vectors: torch.Tensor) -> None:
        """Store raw 512-d FashionCLIP fused vectors for trainable projection."""
        self.register_buffer("fused_item_vectors", fused_vectors)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized 128-d item vectors."""
        if hasattr(self, "fused_item_vectors") and self.fused_item_vectors is not None:
            fused = self.fused_item_vectors[item_ids]
            return self.item_tower.encode_fclip_vectors(fused)
        return F.normalize(self.item_embeddings[item_ids], dim=-1)

    def encode_users(
        self,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        demographics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hasattr(self, "fused_item_vectors") and self.fused_item_vectors is not None:
            history_embeddings = self.item_tower.encode_fclip_vectors(
                self.fused_item_vectors[history_item_ids]
            )
            return self.user_tower.forward(history_embeddings, history_mask, demographics)
        return self.user_tower.encode_from_item_table(
            history_item_ids,
            self.item_embeddings,
            history_mask,
            demographics,
        )

    def forward(
        self,
        history_item_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        demographics: torch.Tensor | None = None,
    ):
        user_emb = self.encode_users(history_item_ids, history_mask, demographics)
        item_emb = self.encode_items(pos_item_ids)
        return user_emb, item_emb
