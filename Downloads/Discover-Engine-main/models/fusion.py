"""
FashionCLIP image + text fusion.

Wraps the HuggingFace `patrickjohncyh/fashion-clip` model (same backbone as
external/fashion-clip/fashion_clip/fashion_clip.py) without S3/Annoy dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

# Allow importing reference FashionCLIP utilities if needed (read-only external/)
_EXTERNAL_FCLIP = Path(__file__).resolve().parent.parent / "external" / "fashion-clip"
if _EXTERNAL_FCLIP.exists() and str(_EXTERNAL_FCLIP) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_FCLIP))

FASHION_CLIP_MODEL = "patrickjohncyh/fashion-clip"
FCLIP_DIM = 512
IMAGE_WEIGHT = 0.6
TEXT_WEIGHT = 0.4


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, ord=2, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def _feature_tensor(feats) -> torch.Tensor:
    """Normalize CLIP feature outputs across transformers versions."""
    if isinstance(feats, torch.Tensor):
        return feats
    if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
        return feats.pooler_output
    if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
        return feats.image_embeds
    if hasattr(feats, "text_embeds") and feats.text_embeds is not None:
        return feats.text_embeds
    raise TypeError(f"Unexpected CLIP feature output type: {type(feats)}")


def fuse_embeddings(
    image_vectors: np.ndarray,
    text_vectors: np.ndarray,
    image_weight: float = IMAGE_WEIGHT,
    text_weight: float = TEXT_WEIGHT,
) -> np.ndarray:
    """
    Weighted average fusion with L2 normalization before and after.

    TODO(stage2): Replace with learned fusion MLP (concat → Linear → ReLU → Linear).
    """
    image_norm = _l2_normalize(image_vectors)
    text_norm = _l2_normalize(text_vectors)
    fused = image_weight * image_norm + text_weight * text_norm
    return _l2_normalize(fused)


class FashionCLIPEncoder:
    """Lightweight FashionCLIP wrapper matching external encode_images/encode_text API."""

    def __init__(self, model_name: str = FASHION_CLIP_MODEL, device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_model(model_name: str, retries: int = 3) -> CLIPModel:
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return CLIPModel.from_pretrained(model_name)
            except OSError as exc:
                last_err = exc
                if attempt < retries:
                    import time

                    time.sleep(2 * attempt)
        raise OSError(
            f"Failed to load FashionCLIP model '{model_name}' after {retries} attempts. "
            "Check network/HuggingFace access or pass --synthetic-embeddings for offline demo."
        ) from last_err

    def encode_images(
        self,
        images: Union[List[str], List[Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Batch image paths or PIL images → (N, 512) float32 vectors."""
        from datasets import Dataset as HFDataset

        def transform_fn(batch):
            if isinstance(batch["image"][0], Image.Image):
                imgs = batch["image"]
            else:
                imgs = [Image.open(p).convert("RGB") for p in batch["image"]]
            return self.processor(images=imgs, return_tensors="pt")

        if isinstance(images[0], str):
            dataset = HFDataset.from_dict({"image": images})
        else:
            dataset = HFDataset.from_dict({"image": images})

        dataset.set_format("torch")
        dataset.set_transform(transform_fn)
        loader = DataLoader(dataset, batch_size=batch_size)

        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                feats = _feature_tensor(self.model.get_image_features(**batch))
                embeddings.append(feats.detach().cpu().numpy())

        return np.vstack(embeddings).astype(np.float32)

    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch text strings → (N, 512) float32 vectors."""
        from datasets import Dataset as HFDataset

        dataset = HFDataset.from_dict({"text": texts})
        dataset = dataset.map(
            lambda el: self.processor(
                text=el["text"],
                return_tensors="pt",
                max_length=77,
                padding="max_length",
                truncation=True,
            ),
            batched=True,
            remove_columns=["text"],
        )
        dataset.set_format("torch")
        loader = DataLoader(dataset, batch_size=batch_size)

        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                feats = _feature_tensor(self.model.get_text_features(**batch))
                embeddings.append(feats.detach().cpu().numpy())

        return np.vstack(embeddings).astype(np.float32)

    def encode_fused(
        self,
        images: Union[List[str], List[Image.Image], None],
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode and fuse image + text modalities."""
        if images is None:
            text_vecs = self.encode_text(texts, batch_size=batch_size)
            return _l2_normalize(text_vecs)
        image_vecs = self.encode_images(images, batch_size=batch_size)
        text_vecs = self.encode_text(texts, batch_size=batch_size)
        return fuse_embeddings(image_vecs, text_vecs)
