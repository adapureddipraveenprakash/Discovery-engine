import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import faiss
import numpy as np
import pickle
from src.models.two_tower import TwoTowerModel

# ----------------------------
# Setup
# ----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open("artifacts/user_map.pkl", "rb") as f:
    user_map = pickle.load(f)

with open("artifacts/item_map.pkl", "rb") as f:
    item_map = pickle.load(f)

num_users = len(user_map)
num_items = len(item_map)
embedding_dim = 256

# ----------------------------
# Load Model
# ----------------------------
model = TwoTowerModel(num_users, num_items, embedding_dim)
model.load_state_dict(torch.load("artifacts/best_model.pt", map_location=device))
model.to(device)
model.eval()

# ----------------------------
# Load FAISS Index
# ----------------------------
index = faiss.read_index("artifacts/faiss.index")

# ----------------------------
# Warmup (important)
# ----------------------------
with torch.no_grad():
    for _ in range(10):
        user = torch.tensor([0]).to(device)
        emb = model.user_embedding(user)
        emb = torch.nn.functional.normalize(emb, dim=1)
        emb = emb.cpu().numpy().astype("float32")
        index.search(emb, 10)

# ----------------------------
# Single Query Latency
# ----------------------------
user = torch.tensor([0]).to(device)

start = time.time()

with torch.no_grad():
    emb = model.user_embedding(user)
    emb = torch.nn.functional.normalize(emb, dim=1)
    emb = emb.cpu().numpy().astype("float32")
    D, I = index.search(emb, 10)

end = time.time()

print("Single Query Latency (ms):", (end - start) * 1000)

# ----------------------------
# Batch Benchmark (1000 queries)
# ----------------------------
latencies = []

for user_id in range(1000):
    user = torch.tensor([user_id % num_users]).to(device)

    start = time.time()

    with torch.no_grad():
        emb = model.user_embedding(user)
        emb = torch.nn.functional.normalize(emb, dim=1)
        emb = emb.cpu().numpy().astype("float32")
        index.search(emb, 10)

    end = time.time()
    latencies.append((end - start) * 1000)

latencies = np.array(latencies)

print("Average Latency (ms):", latencies.mean())
print("P95 Latency (ms):", np.percentile(latencies, 95))
print("Max Latency (ms):", latencies.max())