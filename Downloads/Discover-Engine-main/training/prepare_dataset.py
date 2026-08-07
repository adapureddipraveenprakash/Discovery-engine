"""
Prepare H&M-style dataset or generate synthetic sample data.

Real H&M Personalised Fashion Recommendations structure:
  - articles.csv
  - customers.csv
  - transactions_train.csv
  - images/{article_id}.jpg

Set DATA_ROOT env var or pass --data-root to point at the real dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

CATEGORIES = [
    "Topwear",
    "Bottomwear",
    "Footwear",
    "Accessories",
    "Dress",
    "Outerwear",
]

PRODUCT_TYPES = {
    "Topwear": ["T-shirt", "Blouse", "Shirt", "Sweater"],
    "Bottomwear": ["Jeans", "Trousers", "Skirt", "Shorts"],
    "Footwear": ["Sneakers", "Boots", "Sandals", "Heels"],
    "Accessories": ["Bag", "Belt", "Scarf", "Hat"],
    "Dress": ["Midi dress", "Maxi dress", "Mini dress"],
    "Outerwear": ["Jacket", "Coat", "Blazer"],
}

COLORS = ["black", "white", "navy", "beige", "red", "green", "blue", "grey"]


def _make_placeholder_image(path: Path, color: tuple[int, int, int], label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (224, 224), color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 100), label[:20], fill=(255, 255, 255))
    img.save(path)


def generate_synthetic_dataset(
    data_dir: Path,
    num_items: int = 200,
    num_users: int = 500,
    num_transactions: int = 5000,
    seed: int = 42,
) -> dict:
    """Create a minimal H&M-shaped dataset for pipeline demos."""
    random.seed(seed)
    np.random.seed(seed)

    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    articles = []
    for i in range(num_items):
        article_id = f"{1000000 + i}"
        category = CATEGORIES[i % len(CATEGORIES)]
        product_type = random.choice(PRODUCT_TYPES[category])
        color = random.choice(COLORS)
        desc = f"{color} {product_type.lower()} for everyday wear"
        img_path = images_dir / f"{article_id}.jpg"
        hue = (i * 37) % 256
        _make_placeholder_image(img_path, (hue, 100, 200), article_id)
        articles.append(
            {
                "article_id": article_id,
                "product_code": f"P{i:04d}",
                "prod_name": f"{color.title()} {product_type}",
                "product_type_name": product_type,
                "product_group_name": category,
                "colour_group_name": color,
                "detail_desc": desc,
                "image_path": str(img_path.relative_to(data_dir)),
            }
        )

    customers = []
    for u in range(num_users):
        customers.append(
            {
                "customer_id": f"U{u:05d}",
                "age": random.randint(18, 65),
                "club_member_status": random.choice(["ACTIVE", "PRE-CREATE", "LEFT CLUB"]),
                "fashion_news_frequency": random.choice(["Regularly", "None", "Monthly"]),
            }
        )

    article_ids = [a["article_id"] for a in articles]
    user_ids = [c["customer_id"] for c in customers]
    transactions = []
    for _ in range(num_transactions):
        transactions.append(
            {
                "t_dat": f"2020-{(random.randint(1,12)):02d}-{(random.randint(1,28)):02d}",
                "customer_id": random.choice(user_ids),
                "article_id": random.choice(article_ids),
                "price": round(random.uniform(5.0, 99.0), 2),
            }
        )

    articles_df = pd.DataFrame(articles)
    customers_df = pd.DataFrame(customers)
    transactions_df = pd.DataFrame(transactions)

    data_dir.mkdir(parents=True, exist_ok=True)
    articles_df.to_csv(data_dir / "articles.csv", index=False)
    customers_df.to_csv(data_dir / "customers.csv", index=False)
    transactions_df.to_csv(data_dir / "transactions_train.csv", index=False)

    meta = {
        "source": "synthetic",
        "num_items": num_items,
        "num_users": num_users,
        "num_transactions": num_transactions,
    }
    with open(data_dir / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Synthetic dataset written to %s", data_dir)
    return meta


def load_hm_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load H&M CSV files; raises FileNotFoundError if missing."""
    articles = pd.read_csv(data_dir / "articles.csv")
    customers = pd.read_csv(data_dir / "customers.csv")
    transactions = pd.read_csv(data_dir / "transactions_train.csv")
    return articles, customers, transactions


def build_processed_artifacts(data_dir: Path, artifacts_dir: Path, max_history: int = 10) -> None:
    """
    Build id maps, interaction history, and item metadata for training/serving.
    """
    if not (data_dir / "articles.csv").exists():
        generate_synthetic_dataset(data_dir)

    articles, customers, transactions = load_hm_dataset(data_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Stable integer maps
    item_ids = articles["article_id"].astype(str).tolist()
    user_ids = sorted(customers["customer_id"].astype(str).unique().tolist())

    item_map = {aid: idx for idx, aid in enumerate(item_ids)}
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    reverse_item_map = {v: k for k, v in item_map.items()}

    # Item metadata for agents
    item_meta = []
    for _, row in articles.iterrows():
        aid = str(row["article_id"])
        idx = item_map[aid]
        img = row.get("image_path")
        if pd.isna(img):
            candidate = data_dir / "images" / f"{aid}.jpg"
            img = str(candidate.relative_to(data_dir)) if candidate.exists() else ""
        item_meta.append(
            {
                "item_idx": idx,
                "article_id": aid,
                "category": row.get("product_group_name", "Unknown"),
                "product_type": row.get("product_type_name", ""),
                "description": row.get("detail_desc") or row.get("prod_name", ""),
                "image_path": str(img),
            }
        )

    # Sort transactions by date for history building
    transactions = transactions.sort_values("t_dat")
    transactions["user_idx"] = transactions["customer_id"].map(user_map)
    transactions["item_idx"] = transactions["article_id"].astype(str).map(item_map)
    transactions = transactions.dropna(subset=["user_idx", "item_idx"])
    transactions["user_idx"] = transactions["user_idx"].astype(int)
    transactions["item_idx"] = transactions["item_idx"].astype(int)

    # Per-user interaction sets and histories
    user_interactions: dict[int, list[int]] = {}
    user_histories: dict[int, list[int]] = {}
    for uid, group in transactions.groupby("user_idx"):
        items = group["item_idx"].tolist()
        user_interactions[int(uid)] = items
        user_histories[int(uid)] = items[-max_history:]

    # Training pairs: (user, history, positive item)
    train_rows = []
    for uid, items in user_interactions.items():
        if len(items) < 2:
            continue
        hist = items[:-1][-max_history:]
        pos = items[-1]
        train_rows.append({"user_idx": uid, "history": hist, "pos_item_idx": pos})

    train_df = pd.DataFrame(train_rows)

    import pickle

    with open(artifacts_dir / "item_map.pkl", "wb") as f:
        pickle.dump(item_map, f)
    with open(artifacts_dir / "user_map.pkl", "wb") as f:
        pickle.dump(user_map, f)
    with open(artifacts_dir / "reverse_item_map.pkl", "wb") as f:
        pickle.dump(reverse_item_map, f)
    with open(artifacts_dir / "user_interactions.pkl", "wb") as f:
        pickle.dump(user_interactions, f)
    with open(artifacts_dir / "user_histories.pkl", "wb") as f:
        pickle.dump(user_histories, f)

    pd.DataFrame(item_meta).to_json(artifacts_dir / "item_meta.json", orient="records", indent=2)
    train_df.to_parquet(artifacts_dir / "train_pairs.parquet", index=False)
    transactions.to_parquet(artifacts_dir / "transactions.parquet", index=False)

    logger.info(
        "Artifacts: %d items, %d users, %d train pairs",
        len(item_map),
        len(user_map),
        len(train_df),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Prepare H&M or synthetic dataset")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--num-items", type=int, default=200)
    parser.add_argument("--num-users", type=int, default=500)
    args = parser.parse_args()

    if args.synthetic_only or not (args.data_dir / "articles.csv").exists():
        generate_synthetic_dataset(
            args.data_dir,
            num_items=args.num_items,
            num_users=args.num_users,
        )
    build_processed_artifacts(args.data_dir, args.artifacts_dir)


if __name__ == "__main__":
    main()
