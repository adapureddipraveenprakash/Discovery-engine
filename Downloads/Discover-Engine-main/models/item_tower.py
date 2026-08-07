"""
Item tower: FashionCLIP fusion → trainable Linear(512→128) → L2 norm.

Replaces ID-based item embedding lookup from the two-tower reference.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from models.fusion import FCLIP_DIM, FashionCLIPEncoder, fuse_embeddings

OUTPUT_DIM = 128


class ItemTower(nn.Module):
    """
    Projects FashionCLIP fused 512-d vectors into the shared 128-d retrieval space.
    """

    def __init__(
        self,
        input_dim: int = FCLIP_DIM,
        output_dim: int = OUTPUT_DIM,
        encoder: FashionCLIPEncoder | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = nn.Linear(input_dim, output_dim)
        self._encoder = encoder

    @property
    def encoder(self) -> FashionCLIPEncoder:
        if self._encoder is None:
            self._encoder = FashionCLIPEncoder()
        return self._encoder

    def encode_fclip_vectors(self, fused_vectors: torch.Tensor) -> torch.Tensor:
        """Project precomputed or on-the-fly fused 512-d vectors to 128-d."""
        projected = self.projection(fused_vectors)
        return F.normalize(projected, dim=-1)

    def forward(self, fused_vectors: torch.Tensor) -> torch.Tensor:
        return self.encode_fclip_vectors(fused_vectors)

    @torch.no_grad()
    def encode_from_raw(
        self,
        images: Union[List[str], List[Image.Image], None],
        texts: List[str],
        batch_size: int = 32,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Full pipeline: images + text → fusion → projection → L2 norm."""
        fused_np = self.encoder.encode_fused(images, texts, batch_size=batch_size)
        fused = torch.from_numpy(fused_np).float()
        if device is not None:
            fused = fused.to(device)
        self.eval()
        return self.encode_fclip_vectors(fused)

    def encode_fused_numpy(self, fused_np: np.ndarray, device: torch.device | None = None) -> np.ndarray:
        """Project numpy fused vectors and return numpy 128-d embeddings."""
        fused = torch.from_numpy(fused_np.astype(np.float32))
        if device is not None:
            self.to(device)
            fused = fused.to(device)
        self.eval()
        with torch.no_grad():
            out = self.encode_fclip_vectors(fused)
        return out.cpu().numpy()

    @staticmethod
    def fuse_batch(
        image_vectors: np.ndarray,
        text_vectors: np.ndarray,
    ) -> np.ndarray:
        return fuse_embeddings(image_vectors, text_vectors)
