"""
Quick inference timing script — measures embedding + FAISS search latency for one user.
Run after training and building the index:
    python train.py && python build_index.py && python serve.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import time
from pathlib import Path

import faiss
import torch
import yaml

from src.models.two_tower import TwoTowerModel

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

with open(ROOT / "configs" / "config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open(ARTIFACTS / "user_map.pkl", "rb") as f:
    user_map = pickle.load(f)
with open(ARTIFACTS / "item_map.pkl", "rb") as f:
    item_map = pickle.load(f)

model = TwoTowerModel(
    len(user_map), len(item_map),
    embedding_dim=config.get("embedding_dim", 128),
    hidden_dim=config.get("hidden_dim", 256),
)
model.load_state_dict(torch.load(ARTIFACTS / "best_model.pt", map_location=device))
model.to(device)
model.eval()

index = faiss.read_index(str(ARTIFACTS / "faiss.index"))

user_tensor = torch.tensor([0]).to(device)

# Warmup
with torch.no_grad():
    for _ in range(10):
        emb = model.encode_users(user_tensor).cpu().numpy().astype("float32")
        index.search(emb, 10)

# Measure
with torch.no_grad():
    t0 = time.perf_counter()
    emb = model.encode_users(user_tensor)
    t1 = time.perf_counter()
    emb_np = emb.cpu().numpy().astype("float32")
    _, recommendations = index.search(emb_np, 10)
    t2 = time.perf_counter()

embed_ms = (t1 - t0) * 1000
search_ms = (t2 - t1) * 1000

print(f"Embedding time : {embed_ms:.3f} ms")
print(f"FAISS search   : {search_ms:.3f} ms")
print(f"Total latency  : {embed_ms + search_ms:.3f} ms")
print(f"Top-10 items   : {recommendations[0].tolist()}")
