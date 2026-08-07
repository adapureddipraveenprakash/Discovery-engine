"""
InfoNCE training for user tower against FashionCLIP-projected item embeddings.

Training loop adapted from external/two-tower-retrieval-system/src/training/trainer.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.discovery_model import DiscoveryModel
from models.item_tower import ItemTower
from models.fusion import FashionCLIPEncoder, _l2_normalize

logger = logging.getLogger(__name__)

ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"


class SessionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_history: int = 10):
        self.max_history = max_history
        self.rows = df.to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        hist = list(row["history"])[-self.max_history :]
        pad = max(0, self.max_history - len(hist))
        hist = [0] * pad + hist
        mask = [0.0] * pad + [1.0] * (self.max_history - pad)
        return (
            torch.tensor(hist, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float),
            torch.tensor(int(row["pos_item_idx"]), dtype=torch.long),
        )


def pad_collate(batch):
    hists, masks, pos = zip(*batch)
    return (
        torch.stack(hists),
        torch.stack(masks),
        torch.stack(pos),
    )


def infonce_loss(model, history_ids, history_mask, pos_items, temperature: float):
    user_emb = model.encode_users(history_ids, history_mask)
    item_emb = model.encode_items(pos_items)
    logits = (user_emb @ item_emb.T) / temperature
    log_probs = F.log_softmax(logits, dim=1)
    B = user_emb.size(0)
    diag = torch.arange(B, device=user_emb.device)
    return -log_probs[diag, diag].mean()


def precompute_fused_vectors(
    data_dir: Path,
    item_meta_path: Path,
    encoder: FashionCLIPEncoder | None,
    batch_size: int = 32,
    synthetic: bool = False,
) -> np.ndarray:
    """Encode all catalog items through FashionCLIP (512-d fused vectors)."""
    meta = pd.read_json(item_meta_path)
    all_fused = []

    for start in range(0, len(meta), batch_size):
        batch = meta.iloc[start : start + batch_size]
        texts = batch["description"].fillna("fashion item").tolist()

        if synthetic or encoder is None:
            rng = np.random.default_rng(hash(tuple(texts)) & 0xFFFFFFFF)
            fused = rng.standard_normal((len(batch), 512)).astype(np.float32)
            fused = fused / np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)
            all_fused.append(fused)
            continue

        images = []
        for p in batch["image_path"]:
            full = data_dir / str(p) if p else None
            images.append(str(full) if full and full.exists() else None)

        if all(i is not None for i in images):
            fused = encoder.encode_fused(images, texts, batch_size=batch_size)
        else:
            fused = _l2_normalize(encoder.encode_text(texts, batch_size=batch_size))

        all_fused.append(fused)

    return np.vstack(all_fused).astype("float32")


def project_all_items(model: DiscoveryModel, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        ids = torch.arange(model.fused_item_vectors.size(0), device=device)
        projected = model.encode_items(ids)
    return projected.cpu().numpy().astype("float32")


def train(
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.05,
    max_history: int = 10,
    device: str | None = None,
    synthetic_embeddings: bool = False,
    reuse_cached_fused: bool = True,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    train_path = ARTIFACTS / "train_pairs.parquet"
    if not train_path.exists():
        from training.prepare_dataset import build_processed_artifacts, generate_synthetic_dataset

        generate_synthetic_dataset(DATA_DIR)
        build_processed_artifacts(DATA_DIR, ARTIFACTS, max_history=max_history)

    df = pd.read_parquet(train_path)
    num_items = len(pd.read_json(ARTIFACTS / "item_meta.json"))

    item_tower = ItemTower().to(dev)
    fused_cache = ARTIFACTS / "fused_item_vectors.npy"

    if reuse_cached_fused and fused_cache.exists() and not synthetic_embeddings:
        logger.info("Loading cached fused vectors from %s", fused_cache)
        fused_np = np.load(fused_cache)
    else:
        encoder = None
        if not synthetic_embeddings:
            try:
                encoder = FashionCLIPEncoder(device=str(dev))
            except OSError as exc:
                logger.warning("FashionCLIP unavailable (%s); using synthetic 512-d vectors", exc)
                synthetic_embeddings = True

        if synthetic_embeddings:
            logger.info("Precomputing synthetic 512-d fused vectors (offline demo mode)...")
        else:
            logger.info("Precomputing FashionCLIP fused 512-d vectors...")

        fused_np = precompute_fused_vectors(
            DATA_DIR,
            ARTIFACTS / "item_meta.json",
            encoder,
            synthetic=synthetic_embeddings,
        )
        np.save(fused_cache, fused_np)

    fused_tensor = torch.from_numpy(fused_np).to(dev)
    model = DiscoveryModel(
        num_items=num_items,
        item_embedding_table=torch.zeros(num_items, 128),
        max_history=max_history,
        temperature=temperature,
        item_tower=item_tower,
    ).to(dev)
    model.set_fused_item_vectors(fused_tensor)

    dataset = SessionDataset(df, max_history=max_history)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    optimizer = torch.optim.Adam(
        list(model.user_tower.parameters()) + list(model.item_tower.parameters()),
        lr=lr,
    )

    logger.info("Training with InfoNCE (epochs=%d, batch=%d)", epochs, batch_size)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for hists, masks, pos in loader:
            hists = hists.to(dev)
            masks = masks.to(dev)
            pos = pos.to(dev)

            optimizer.zero_grad()
            loss = infonce_loss(model, hists, masks, pos, temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / max(len(loader), 1)
        logger.info("Epoch %d/%d  infonce_loss=%.4f", epoch + 1, epochs, avg)

    item_emb_np = project_all_items(model, dev)
    model.set_item_embeddings(torch.from_numpy(item_emb_np).to(dev))
    np.save(ARTIFACTS / "item_embeddings.npy", item_emb_np)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "item_tower_state": item_tower.state_dict(),
            "user_tower_state": model.user_tower.state_dict(),
            "config": {
                "embedding_dim": 128,
                "max_history": max_history,
                "temperature": temperature,
            },
        },
        ARTIFACTS / "discovery_model.pt",
    )

    with open(ARTIFACTS / "train_meta.json", "w") as f:
        json.dump({"epochs": epochs, "loss": "infonce", "temperature": temperature}, f)

    logger.info("Saved model to %s", ARTIFACTS / "discovery_model.pt")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument(
        "--synthetic-embeddings",
        action="store_true",
        help="Use deterministic random 512-d vectors instead of FashionCLIP (offline demo)",
    )
    parser.add_argument(
        "--no-reuse-cached-fused",
        action="store_true",
        help="Re-encode items even if artifacts/fused_item_vectors.npy exists",
    )
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        max_history=args.max_history,
        synthetic_embeddings=args.synthetic_embeddings,
        reuse_cached_fused=not args.no_reuse_cached_fused,
    )


if __name__ == "__main__":
    main()
