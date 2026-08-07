import torch

from src.models.two_tower import TwoTowerModel
from src.models.mf_baseline import MFBaseline


def test_two_tower_output_shape():
    model = TwoTowerModel(num_users=100, num_items=50, embedding_dim=16, hidden_dim=32)
    users = torch.randint(0, 100, (8,))
    items = torch.randint(0, 50, (8,))
    user_emb, item_emb = model(users, items)
    assert user_emb.shape == (8, 16)
    assert item_emb.shape == (8, 16)


def test_two_tower_embeddings_are_normalized():
    """L2 norm of every output embedding must be ~1.0."""
    model = TwoTowerModel(num_users=10, num_items=10, embedding_dim=8, hidden_dim=16)
    users = torch.arange(10)
    items = torch.arange(10)
    user_emb, item_emb = model(users, items)

    user_norms = torch.linalg.norm(user_emb, dim=1)
    item_norms = torch.linalg.norm(item_emb, dim=1)
    assert torch.allclose(user_norms, torch.ones(10), atol=1e-5)
    assert torch.allclose(item_norms, torch.ones(10), atol=1e-5)


def test_encode_users_uses_full_tower():
    """Regression test: encode_users must apply the MLP, not just the embedding lookup.
    This catches the bug where the MLP was bypassed during training."""
    model = TwoTowerModel(num_users=5, num_items=5, embedding_dim=8, hidden_dim=16)
    users = torch.tensor([0, 1, 2])

    # Through full tower
    full = model.encode_users(users)
    # Just embedding lookup, normalized — should differ
    raw = torch.nn.functional.normalize(model.user_embedding(users), dim=-1)

    assert not torch.allclose(full, raw), "encode_users should pass through MLP, not just embedding"


def test_mf_baseline_score_shape():
    model = MFBaseline(num_users=20, num_items=30, embedding_dim=8)
    users = torch.randint(0, 20, (5,))
    items = torch.randint(0, 30, (5,))
    scores = model(users, items)
    assert scores.shape == (5,)


def test_mf_baseline_has_encoder_methods():
    """MF baseline must expose encode_users/encode_items so it can reuse shared metrics code."""
    model = MFBaseline(num_users=10, num_items=10, embedding_dim=4)
    assert hasattr(model, "encode_users")
    assert hasattr(model, "encode_items")
