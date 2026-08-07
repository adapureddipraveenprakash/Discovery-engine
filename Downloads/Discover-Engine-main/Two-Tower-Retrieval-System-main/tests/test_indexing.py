import torch

from src.indexing.build_index import build_faiss_index
from src.models.two_tower import TwoTowerModel


def test_faiss_index_returns_correct_shape():
    model = TwoTowerModel(num_users=10, num_items=50, embedding_dim=8, hidden_dim=16)
    index = build_faiss_index(model, num_items=50, device=torch.device("cpu"))
    assert index.ntotal == 50
    assert index.d == 8


def test_faiss_search_returns_top_k():
    model = TwoTowerModel(num_users=5, num_items=20, embedding_dim=8, hidden_dim=16)
    index = build_faiss_index(model, num_items=20, device=torch.device("cpu"))

    query = model.encode_users(torch.tensor([0])).detach().numpy().astype("float32")
    scores, ids = index.search(query, 5)
    assert ids.shape == (1, 5)
    assert scores.shape == (1, 5)
    # All returned IDs must be valid item indices
    assert (ids[0] >= 0).all() and (ids[0] < 20).all()
