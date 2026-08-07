import numpy as np

from index.build_index import build_index_from_embeddings
from index.faiss_store import FaissStore


def test_faiss_build_and_search():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((50, 128)).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    index = build_index_from_embeddings(emb)
    store = FaissStore(index=index, dim=128)
    assert store.ntotal == 50

    query = emb[0:1]
    scores, indices = store.search(query, top_k=5)
    assert indices.shape[1] == 5
    assert indices[0, 0] == 0


def test_faiss_exclude_filter():
    emb = np.eye(10, 128, dtype=np.float32)
    store = FaissStore(index=build_index_from_embeddings(emb))
    _, indices = store.search(emb[0:1], top_k=3, exclude={0, 1})
    flat = indices[0].tolist()
    assert 0 not in flat
    assert 1 not in flat
