import logging
import pickle
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.encoder import encode_ids
from src.data.loader import InteractionDataset
from src.data.split import time_based_split
from src.evaluation.metrics import ndcg_at_k, mrr_at_k, recall_at_k
from src.models.mf_baseline import MFBaseline
from src.models.two_tower import TwoTowerModel
from src.training.trainer import train_two_tower

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# ---- Config ----
with open(ROOT / "configs" / "config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info("Using device: %s | dataset: %s", device, config["dataset"])

# ---- Load Data (handles ML-100K and ML-1M formats) ----
sep = config.get("data_sep", "\t")
df = pd.read_csv(
    ROOT / config["data_path"],
    sep=sep,
    names=["user_id", "item_id", "rating", "timestamp"],
    engine="python" if len(sep) > 1 else "c",
)
logger.info("Loaded %d interactions", len(df))

# Implicit feedback: treat ratings >= 4 as positive
df = df[df["rating"] >= 4].copy()
logger.info("After filtering rating>=4: %d interactions", len(df))

train_df, test_df = time_based_split(df, test_ratio=0.2)

# ---- Encode IDs ----
train_encoded, user_map, item_map = encode_ids(train_df)

with open(ARTIFACTS / "user_map.pkl", "wb") as f:
    pickle.dump(user_map, f)
with open(ARTIFACTS / "item_map.pkl", "wb") as f:
    pickle.dump(item_map, f)

logger.info("Users: %d  Items: %d", len(user_map), len(item_map))

test_df = test_df[test_df["user_id"].isin(user_map) & test_df["item_id"].isin(item_map)].copy()
test_df["user_idx"] = test_df["user_id"].map(user_map)
test_df["item_idx"] = test_df["item_id"].map(item_map)

num_users = len(user_map)
num_items = len(item_map)

# ---- Dataset ----
dataset = InteractionDataset(
    train_encoded,
    num_items,
    num_negatives=config.get("num_negatives", 4),
    sampling=config.get("negative_sampling", "uniform"),
)
dataloader = DataLoader(
    dataset, batch_size=config["batch_size"], shuffle=True,
    num_workers=0,
)

# ---- Two-Tower Model ----
model = TwoTowerModel(
    num_users=num_users,
    num_items=num_items,
    embedding_dim=config.get("embedding_dim", 256),
    hidden_dim=config.get("hidden_dim", 512),
    temperature=config.get("temperature", 0.07),
)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

best_recall = 0.0
loss_type = config.get("loss", "infonce")
temperature = config.get("temperature", 0.07)

for epoch in range(config["epochs"]):
    train_two_tower(model, dataloader, optimizer, device, epochs=1,
                    loss_type=loss_type, temperature=temperature)

    recall = recall_at_k(model, train_encoded, test_df, num_items, k=10, device=device)
    logger.info("Epoch %d/%d  Recall@10=%.4f", epoch + 1, config["epochs"], recall)

    if recall > best_recall:
        best_recall = recall
        torch.save(model.state_dict(), ARTIFACTS / "best_model.pt")
        logger.info("New best model saved (Recall@10=%.4f)", best_recall)

# ---- Final Evaluation (load best checkpoint) ----
model.load_state_dict(torch.load(ARTIFACTS / "best_model.pt", map_location=device))
recall = recall_at_k(model, train_encoded, test_df, num_items, k=10, device=device)
mrr = mrr_at_k(model, train_encoded, test_df, num_items, k=10, device=device)
ndcg = ndcg_at_k(model, train_encoded, test_df, num_items, k=10, device=device)

logger.info("===== TWO-TOWER (best) =====")
logger.info("Recall@10 : %.4f  MRR@10 : %.4f  NDCG@10 : %.4f", recall, mrr, ndcg)

# ---- MF Baseline ----
logger.info("Training MF baseline...")
baseline = MFBaseline(num_users, num_items, embedding_dim=config.get("embedding_dim", 256)).to(device)
baseline_opt = torch.optim.Adam(baseline.parameters(), lr=config["learning_rate"])
bce_loss = torch.nn.BCEWithLogitsLoss()

for epoch in range(5):
    total_loss = 0
    for users, pos_items, _ in dataloader:
        users, pos_items = users.to(device), pos_items.to(device)
        baseline_opt.zero_grad()
        scores = baseline(users, pos_items)
        loss = bce_loss(scores, torch.ones_like(scores))
        loss.backward()
        baseline_opt.step()
        total_loss += loss.item()
    logger.info("Baseline epoch %d  loss=%.4f", epoch + 1, total_loss / len(dataloader))

baseline_recall = recall_at_k(baseline, train_encoded, test_df, num_items, k=10, device=device)
baseline_mrr = mrr_at_k(baseline, train_encoded, test_df, num_items, k=10, device=device)
baseline_ndcg = ndcg_at_k(baseline, train_encoded, test_df, num_items, k=10, device=device)

logger.info("===== MF BASELINE =====")
logger.info("Recall@10 : %.4f  MRR@10 : %.4f  NDCG@10 : %.4f",
            baseline_recall, baseline_mrr, baseline_ndcg)
