"""
Verify item and user embedding pipelines (Step 4 checkpoint).

Usage: python scripts/verify_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.fusion import fuse_embeddings
from models.item_tower import ItemTower
from models.user_tower import UserTower


def _sample_image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (224, 224), (80, 120, 200))
    draw = ImageDraw.Draw(img)
    draw.text((40, 100), "sample", fill=(255, 255, 255))
    img.save(path)
    return str(path)


def main() -> None:
    print("=" * 60)
    print("Discovery Engine — Embedding Verification")
    print("=" * 60)

    device = torch.device("cpu")
    sample_img = _sample_image(ROOT / "data" / "samples" / "verify_item.jpg")
    sample_text = "navy cotton t-shirt casual topwear"

    # --- Item path ---
    print("\n[1] Item embedding (FashionCLIP fusion -> Linear(512->128))")
    item_tower = ItemTower().to(device)
    item_tower.eval()

    with torch.no_grad():
        item_emb = item_tower.encode_from_raw(
            images=[sample_img],
            texts=[sample_text],
            device=device,
        )

    item_np = item_emb.cpu().numpy()
    item_norm = float(np.linalg.norm(item_np[0]))
    print(f"  Input image : {sample_img}")
    print(f"  Input text  : {sample_text}")
    print(f"  Output shape: {item_np.shape}")
    print(f"  L2 norm     : {item_norm:.6f} (expect ~1.0)")
    assert item_np.shape == (1, 128), f"Expected (1, 128), got {item_np.shape}"
    assert abs(item_norm - 1.0) < 0.01, f"Item vector not unit-normalized: {item_norm}"

    # --- User path ---
    print("\n[2] User embedding (history pool -> MLP -> 128-d)")
    user_tower = UserTower(embedding_dim=128, max_history=3).to(device)
    user_tower.eval()

    # Simulate 3 recent item embeddings (reuse item vector with small perturbations)
    history = torch.stack([item_emb[0], item_emb[0] * 0.9, item_emb[0] * 0.8]).unsqueeze(0)
    mask = torch.ones(1, 3)

    with torch.no_grad():
        user_emb = user_tower(history, mask)

    user_np = user_emb.cpu().numpy()
    user_norm = float(np.linalg.norm(user_np[0]))
    print(f"  History len : 3 (pooled recent item embeddings)")
    print(f"  Output shape: {user_np.shape}")
    print(f"  L2 norm     : {user_norm:.6f} (expect ~1.0)")
    assert user_np.shape == (1, 128), f"Expected (1, 128), got {user_np.shape}"
    assert abs(user_norm - 1.0) < 0.01, f"User vector not unit-normalized: {user_norm}"

    # --- Fusion sanity ---
    print("\n[3] Fusion weights (0.6 image + 0.4 text)")
    rng = np.random.default_rng(0)
    img_vec = rng.standard_normal((1, 512)).astype(np.float32)
    txt_vec = rng.standard_normal((1, 512)).astype(np.float32)
    fused = fuse_embeddings(img_vec, txt_vec)
    print(f"  Fused shape : {fused.shape}")
    print(f"  Fused norm  : {np.linalg.norm(fused[0]):.6f}")

    print("\n" + "=" * 60)
    print("PASS: Item and user embeddings verified successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
