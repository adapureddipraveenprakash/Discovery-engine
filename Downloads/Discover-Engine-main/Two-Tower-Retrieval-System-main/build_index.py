import logging
import pickle
from pathlib import Path

import faiss
import torch
import yaml

from src.indexing.build_index import build_faiss_index
from src.models.two_tower import TwoTowerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

with open(ROOT / "configs" / "config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info("Using device: %s", device)

with open(ARTIFACTS / "user_map.pkl", "rb") as f:
    user_map = pickle.load(f)
with open(ARTIFACTS / "item_map.pkl", "rb") as f:
    item_map = pickle.load(f)

num_users, num_items = len(user_map), len(item_map)

model = TwoTowerModel(
    num_users=num_users,
    num_items=num_items,
    embedding_dim=config.get("embedding_dim", 128),
    hidden_dim=config.get("hidden_dim", 256),
    temperature=config.get("temperature", 0.07),
)
model.load_state_dict(torch.load(ARTIFACTS / "best_model.pt", map_location=device))
model.to(device)
logger.info("Model loaded from %s", ARTIFACTS / "best_model.pt")

index = build_faiss_index(model, num_items, device)
faiss.write_index(index, str(ARTIFACTS / "faiss.index"))
logger.info("FAISS index saved to %s  (%d vectors)", ARTIFACTS / "faiss.index", num_items)
