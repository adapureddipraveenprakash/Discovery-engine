"""
Ablation study runner.

Trains the two-tower model with different configurations and reports
Recall@10 / MRR@10 / NDCG@10 for each. Designed to answer:
  - Does the MLP help vs. plain embeddings?
  - Does InfoNCE beat BPR?
  - Do popularity-weighted negatives help vs. uniform?
  - How does embedding dim trade off accuracy vs. compute?

Usage:
    python ablation.py            # runs all experiments, writes ablation_results.md
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.encoder import encode_ids
from src.data.loader import InteractionDataset
from src.data.split import time_based_split
from src.evaluation.recall import compute_all_metrics
from src.models.two_tower import TwoTowerModel
from src.training.trainer import train_two_tower

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_data():
    with open(ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    sep = cfg.get("data_sep", "\t")
    df = pd.read_csv(
        ROOT / cfg["data_path"], sep=sep,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python" if len(sep) > 1 else "c",
    )
    df = df[df["rating"] >= 4].copy()
    train_df, test_df = time_based_split(df, test_ratio=0.2)
    train_enc, user_map, item_map = encode_ids(train_df)

    test_df = test_df[
        test_df["user_id"].isin(user_map) & test_df["item_id"].isin(item_map)
    ].copy()
    test_df["user_idx"] = test_df["user_id"].map(user_map)
    test_df["item_idx"] = test_df["item_id"].map(item_map)

    return train_enc, test_df, len(user_map), len(item_map)


def run_experiment(name, train_enc, test_df, num_users, num_items, **overrides):
    logger.info("=" * 70)
    logger.info("Running: %s", name)
    logger.info("Config: %s", overrides)

    cfg = {
        "embedding_dim": 128, "hidden_dim": 256,
        "loss": "infonce", "temperature": 0.07,
        "negative_sampling": "popularity", "num_negatives": 4,
        "epochs": 5, "batch_size": 1024, "learning_rate": 0.001,
        "use_mlp": True,
    }
    cfg.update(overrides)

    dataset = InteractionDataset(
        train_enc, num_items,
        num_negatives=cfg["num_negatives"],
        sampling=cfg["negative_sampling"],
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

    model = TwoTowerModel(
        num_users=num_users, num_items=num_items,
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        temperature=cfg["temperature"],
    )

    # Ablation: disable MLP by setting it to identity
    if not cfg["use_mlp"]:
        model.user_mlp = torch.nn.Identity()
        model.item_mlp = torch.nn.Identity()

    optim = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    t0 = time.time()
    train_two_tower(model, loader, optim, DEVICE, epochs=cfg["epochs"],
                    loss_type=cfg["loss"], temperature=cfg["temperature"])
    train_time = time.time() - t0

    metrics = compute_all_metrics(model, train_enc, test_df, num_items, k=10, device=DEVICE)
    metrics["train_time_s"] = round(train_time, 1)
    metrics["name"] = name
    return metrics


def main():
    train_enc, test_df, num_users, num_items = load_data()
    logger.info("Dataset: %d users, %d items, %d train interactions",
                num_users, num_items, len(train_enc))

    experiments = [
        # ----- baseline -----
        ("Baseline (InfoNCE + MLP + popularity negs, D=128)", {}),

        # ----- loss ablation -----
        ("BPR loss (instead of InfoNCE)", {"loss": "bpr"}),

        # ----- negative sampling ablation -----
        ("Uniform negatives (instead of popularity)", {"negative_sampling": "uniform"}),

        # ----- architecture ablation -----
        ("No MLP (plain embeddings)", {"use_mlp": False}),

        # ----- dimension ablation -----
        ("Embedding dim = 64", {"embedding_dim": 64, "hidden_dim": 128}),
        ("Embedding dim = 256", {"embedding_dim": 256, "hidden_dim": 512}),
    ]

    results = []
    for name, overrides in experiments:
        try:
            r = run_experiment(name, train_enc, test_df, num_users, num_items, **overrides)
            results.append(r)
        except Exception as e:
            logger.error("Experiment %s failed: %s", name, e)

    # Write markdown table
    out = ["# Ablation Results\n", "| Configuration | Recall@10 | MRR@10 | NDCG@10 | Train (s) |",
           "|---|---|---|---|---|"]
    for r in results:
        out.append(
            f"| {r['name']} | {r['recall']:.4f} | {r['mrr']:.4f} | {r['ndcg']:.4f} | {r['train_time_s']} |"
        )
    text = "\n".join(out) + "\n"
    (ROOT / "ablation_results.md").write_text(text)
    logger.info("Results written to ablation_results.md")
    print("\n" + text)


if __name__ == "__main__":
    main()
