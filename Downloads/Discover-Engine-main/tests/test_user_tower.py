import torch

from models.user_tower import UserTower


def test_user_tower_output_shape_and_norm():
    tower = UserTower(embedding_dim=128, max_history=5)
    history = torch.randn(2, 5, 128)
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.float)
    out = tower(history, mask)
    assert out.shape == (2, 128)
    norms = torch.norm(out, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_user_tower_mean_pooling():
    tower = UserTower(embedding_dim=8, max_history=2)
    e = torch.ones(1, 2, 8)
    e[0, 1] = 3.0
    mask = torch.ones(1, 2)
    out = tower(e, mask)
    assert out.shape == (1, 8)
