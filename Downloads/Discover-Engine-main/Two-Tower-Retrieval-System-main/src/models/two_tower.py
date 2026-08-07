import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        temperature: float = 0.05,
    ):
        super().__init__()

        # Store temperature
        self.temperature = temperature

        # User embedding + MLP
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Item embedding + MLP
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Pass user IDs through the full user tower and return L2-normalized embeddings."""
        emb = self.user_embedding(user_ids)
        emb = self.user_mlp(emb)
        return F.normalize(emb, dim=-1)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Pass item IDs through the full item tower and return L2-normalized embeddings."""
        emb = self.item_embedding(item_ids)
        emb = self.item_mlp(emb)
        return F.normalize(emb, dim=-1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor):
        """Return normalized (user_emb, item_emb) through both towers."""
        return self.encode_users(user_ids), self.encode_items(item_ids)