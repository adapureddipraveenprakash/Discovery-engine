import numpy as np
import torch

from agents.complete_the_look_agent import CompleteTheLookAgent
from index.build_index import build_index_from_embeddings
from index.faiss_store import FaissStore


def test_complete_the_look_diversity():
    rng = np.random.default_rng(1)
    n = 30
    emb = rng.standard_normal((n, 128)).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    categories = ["Topwear"] * 10 + ["Bottomwear"] * 10 + ["Footwear"] * 10
    meta = [{"item_idx": i, "category": categories[i]} for i in range(n)]

    store = FaissStore(index=build_index_from_embeddings(emb))
    agent = CompleteTheLookAgent(store, emb, meta)

    result = agent.complete(seed_item_idx=0, top_k=6)
    assert len(result["recommendations"]) <= 6
    assert 0 not in result["recommendations"]

    rec_cats = [categories[i] for i in result["recommendations"]]
    from collections import Counter
    counts = Counter(rec_cats)
    for c, cnt in counts.items():
        assert cnt <= max(1, int(6 * 0.35 + 0.999))
