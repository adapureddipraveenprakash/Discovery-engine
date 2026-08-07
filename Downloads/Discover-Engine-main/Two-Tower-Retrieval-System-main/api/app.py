import logging
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.two_tower import TwoTowerModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"

app = FastAPI(title="Two-Tower Retrieval API", version="1.0.0")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info("Using device: %s", device)

# ---- Config ----
with open(ROOT / "configs" / "config.yaml") as f:
    config = yaml.safe_load(f)

# ---- ID Mappings ----
with open(ARTIFACTS / "user_map.pkl", "rb") as f:
    user_map: dict = pickle.load(f)
with open(ARTIFACTS / "item_map.pkl", "rb") as f:
    item_map: dict = pickle.load(f)

num_users = len(user_map)
num_items = len(item_map)
logger.info("Loaded mappings: %d users, %d items", num_users, num_items)

# ---- Build interaction set for filtering ----
df = pd.read_csv(
    ROOT / "data" / "raw" / "ml-100k" / "u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
)

# Map to encoded indices for O(1) filtering at inference
user_interactions: dict[int, set] = defaultdict(set)
for row in df.itertuples():
    if row.user_id in user_map and row.item_id in item_map:
        user_interactions[user_map[row.user_id]].add(item_map[row.item_id])

# ---- Model ----
# Pick variant: LOSS=bpr (default) or LOSS=infonce. Falls back to legacy best_model.pt.
LOSS_VARIANT = os.environ.get("LOSS", "bpr")
variant_model = ARTIFACTS / f"model_{LOSS_VARIANT}.pt"
variant_index = ARTIFACTS / f"faiss_{LOSS_VARIANT}.index"
model_path = variant_model if variant_model.exists() else ARTIFACTS / "best_model.pt"
index_path = variant_index if variant_index.exists() else ARTIFACTS / "faiss.index"

model = TwoTowerModel(
    num_users=num_users,
    num_items=num_items,
    embedding_dim=config.get("embedding_dim", 128),
    hidden_dim=config.get("hidden_dim", 256),
    temperature=config.get("temperature", 0.05),
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
logger.info("Loaded variant=%s  model=%s", LOSS_VARIANT, model_path.name)

# ---- FAISS Index ----
index = faiss.read_index(str(index_path))
logger.info("FAISS index loaded (%d vectors)", index.ntotal)


# ---- Schema ----
class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 10


@app.get("/info")
def info():
    """Return which model variant is currently loaded."""
    return {
        "variant": LOSS_VARIANT,
        "model_file": model_path.name,
        "index_file": index_path.name,
        "num_users": num_users,
        "num_items": num_items,
    }


# ---- Endpoint ----
@app.post("/recommend")
def recommend(request: RecommendationRequest):
    """Return top-k item recommendations for a user with filtering of seen items."""
    start = time.time()

    if request.user_id not in user_map:
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")

    encoded_user = user_map[request.user_id]
    interacted = user_interactions.get(encoded_user, set())

    user_tensor = torch.tensor([encoded_user]).to(device)

    with torch.no_grad():
        user_emb = model.encode_users(user_tensor)  # full tower: embedding + MLP + L2

    user_emb_np = user_emb.cpu().numpy().astype("float32")

    _, indices = index.search(user_emb_np, request.top_k * 5)

    recommendations = [
        int(item) for item in indices[0] if item not in interacted
    ][:request.top_k]

    latency_ms = round((time.time() - start) * 1000, 2)
    logger.info("user=%d  top_k=%d  latency=%.2fms", request.user_id, request.top_k, latency_ms)

    return {
        "user_id": request.user_id,
        "recommendations": recommendations,
        "latency_ms": latency_ms,
    }
