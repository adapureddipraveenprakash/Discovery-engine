import torch
import torch.nn as nn
import torch.nn.functional as F


class MFBaseline(nn.Module):
    """Simple matrix factorization baseline: dot product of L2-normalized user and item embeddings."""

    def __init__(self, num_users, num_items, embedding_dim=256):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def encode_users(self, user_ids):
        return F.normalize(self.user_embedding(user_ids), dim=1)

    def encode_items(self, item_ids):
        return F.normalize(self.item_embedding(item_ids), dim=1)

    def forward(self, users, items):
        return (self.encode_users(users) * self.encode_items(items)).sum(dim=1)