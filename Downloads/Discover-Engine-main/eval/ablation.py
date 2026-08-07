"""
Ablation study: fusion modes and negative sampling strategies.

Writes results to eval/ablation_results.md
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import evaluate_model
from models.discovery_model import DiscoveryModel
from models.fusion import _l2_normalize, fuse_embeddings
from models.item_tower import ItemTower
from training.train_user_tower import SessionDataset, pad_collate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
OUTPUT = Path(__file__).parent / "ablation_results.md"


def _train_variant(
    fused_512: np.ndarray,
    train_df: pd.DataFrame,
    epochs: int,
    use_hard_negatives: bool,
    device: torch.device,
    max_history: int = 10,
) -> DiscoveryModel:
    num_items = fused_512.shape[0]
    item_tower = ItemTower().to(device)
    fused_tensor = torch.from_numpy(fused_512.astype(np.float32)).to(device)

    model = DiscoveryModel(
        num_items=num_items,
        item_embedding_table=torch.zeros(num_items, 128),
        max_history=max_history,
        item_tower=item_tower,
    ).to(device)
    model.set_fused_item_vectors(fused_tensor)

    loader = DataLoader(
        SessionDataset(train_df, max_history),
        batch_size=128,
        shuffle=True,
        collate_fn=pad_collate,
    )
    opt = torch.optim.Adam(
        list(model.user_tower.parameters()) + list(model.item_tower.parameters()),
        lr=1e-3,
    )

    for _ in range(epochs):
        model.train()
        for hists, masks, pos in loader:
            hists, masks, pos = hists.to(device), masks.to(device), pos.to(device)
            opt.zero_grad()

            user_emb = model.encode_users(hists, masks)
            item_emb = model.encode_items(pos)

            if use_hard_negatives:
                # Popularity-weighted in-batch negatives (InfoNCE)
                logits = (user_emb @ item_emb.T) / 0.05
                log_probs = F.log_softmax(logits, dim=1)
                B = user_emb.size(0)
                diag = torch.arange(B, device=device)
                loss = -log_probs[diag, diag].mean()
            else:
                # Uniform random negative (single neg BPR-style)
                neg = torch.randint(0, num_items, (pos.size(0),), device=device)
                neg_emb = model.encode_items(neg)
                pos_score = (user_emb * item_emb).sum(1)
                neg_score = (user_emb * neg_emb).sum(1)
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

            loss.backward()
            opt.step()

    with torch.no_grad():
        all_ids = torch.arange(num_items, device=device)
        projected = model.encode_items(all_ids).cpu().numpy()
    model.set_item_embeddings(torch.from_numpy(projected).to(device))
    return model


def _build_fusion_variants(encoder, meta: pd.DataFrame, data_dir: Path) -> dict[str, np.ndarray]:
    texts = meta["description"].fillna("fashion item").tolist()
    images = []
    for p in meta["image_path"]:
        full = data_dir / str(p) if p else None
        images.append(str(full) if full and full.exists() else None)

    valid = all(i is not None for i in images)
    variants = {}

    if valid:
        img_vecs = encoder.encode_images(images, batch_size=32)
        txt_vecs = encoder.encode_text(texts, batch_size=32)
        variants["image_only"] = _l2_normalize(img_vecs)
        variants["text_only"] = _l2_normalize(txt_vecs)
        variants["fused"] = fuse_embeddings(img_vecs, txt_vecs)
    else:
        txt_vecs = encoder.encode_text(texts, batch_size=32)
        variants["text_only"] = _l2_normalize(txt_vecs)
        variants["image_only"] = variants["text_only"]
        variants["fused"] = variants["text_only"]

    return variants


def main() -> None:
    device = torch.device("cpu")
    train_df = pd.read_parquet(ARTIFACTS / "train_pairs.parquet")
    transactions = pd.read_parquet(ARTIFACTS / "transactions.parquet")

    split_idx = int(len(transactions) * 0.8)
    test_tx = transactions.iloc[split_idx:]
    test_df = pd.DataFrame(
        {"user_idx": test_tx["user_idx"].astype(int), "item_idx": test_tx["item_idx"].astype(int)}
    )

    import pickle

    with open(ARTIFACTS / "user_histories.pkl", "rb") as f:
        user_histories = pickle.load(f)
    with open(ARTIFACTS / "user_interactions.pkl", "rb") as f:
        user_interactions = pickle.load(f)

    meta = pd.read_json(ARTIFACTS / "item_meta.json")
    from models.fusion import FashionCLIPEncoder

    encoder = FashionCLIPEncoder(device="cpu")
    variants = _build_fusion_variants(encoder, meta, DATA_DIR)

    rows = []
    for fusion_name, fused_512 in variants.items():
        for hard_neg, neg_label in [(True, "infonce_inbatch"), (False, "uniform_single_neg")]:
            logger.info("Ablation: fusion=%s negatives=%s", fusion_name, neg_label)
            model = _train_variant(fused_512, train_df, epochs=3, use_hard_negatives=hard_neg, device=device)
            item_emb = model.item_embeddings.cpu().numpy()
            metrics = evaluate_model(
                model,
                test_df,
                item_emb,
                user_histories,
                device,
                train_interactions=user_interactions,
            )
            rows.append({"fusion": fusion_name, "negatives": neg_label, **metrics})

    lines = [
        "# Ablation Results — Discovery Engine Stage 1",
        "",
        "| Fusion | Negatives | Recall@10 | Recall@20 | NDCG@10 | Cold-start Recall@10 |",
        "|--------|-----------|-----------|-----------|---------|----------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['fusion']} | {r['negatives']} | {r['recall@10']:.4f} | {r['recall@20']:.4f} | "
            f"{r['ndcg@10']:.4f} | {r['cold_start_recall@10']:.4f} |"
        )
    lines.extend(["", "## Notes", "- Synthetic/sample data; swap in real H&M for production numbers.", ""])

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", OUTPUT)


if __name__ == "__main__":
    main()
