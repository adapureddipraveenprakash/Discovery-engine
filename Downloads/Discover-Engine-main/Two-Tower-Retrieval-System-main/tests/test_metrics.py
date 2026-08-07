import pandas as pd
import torch

from src.evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from src.evaluation.recall import compute_all_metrics
from src.models.two_tower import TwoTowerModel


def _toy_data():
    train = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [0, 1, 2]})
    test = pd.DataFrame({"user_idx": [0, 1], "item_idx": [3, 0]})
    return train, test


def test_metrics_run_end_to_end():
    train, test = _toy_data()
    model = TwoTowerModel(num_users=2, num_items=4, embedding_dim=4, hidden_dim=8)
    r = recall_at_k(model, train, test, num_items=4, k=2, device="cpu")
    m = mrr_at_k(model, train, test, num_items=4, k=2, device="cpu")
    n = ndcg_at_k(model, train, test, num_items=4, k=2, device="cpu")
    for v in [r, m, n]:
        assert 0.0 <= v <= 1.0


def test_compute_all_metrics_matches_individual():
    """The fused compute_all_metrics should match individual metric calls."""
    torch.manual_seed(42)
    train, test = _toy_data()
    model = TwoTowerModel(num_users=2, num_items=4, embedding_dim=4, hidden_dim=8)
    model.eval()

    fused = compute_all_metrics(model, train, test, num_items=4, k=2, device="cpu")
    r = recall_at_k(model, train, test, num_items=4, k=2, device="cpu")
    m = mrr_at_k(model, train, test, num_items=4, k=2, device="cpu")
    n = ndcg_at_k(model, train, test, num_items=4, k=2, device="cpu")

    assert abs(fused["recall"] - r) < 1e-6
    assert abs(fused["mrr"] - m) < 1e-6
    assert abs(fused["ndcg"] - n) < 1e-6


def test_recall_perfect_when_train_items_masked_correctly():
    """If a user's only test item is item 3, and train items 0,1 are masked,
    a model that always picks the lowest-index unmasked item should still hit it
    given enough k."""
    train = pd.DataFrame({"user_idx": [0], "item_idx": [0]})
    test = pd.DataFrame({"user_idx": [0], "item_idx": [3]})

    model = TwoTowerModel(num_users=1, num_items=4, embedding_dim=4, hidden_dim=8)
    # k=3 means top-3 must contain item 3 (out of items 1,2,3 since 0 is masked)
    r = recall_at_k(model, train, test, num_items=4, k=3, device="cpu")
    assert r == 1.0
