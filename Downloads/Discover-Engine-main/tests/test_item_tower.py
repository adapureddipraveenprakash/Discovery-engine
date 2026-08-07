import torch

from models.item_tower import ItemTower


def test_item_tower_projection():
    tower = ItemTower(input_dim=512, output_dim=128)
    fused = torch.randn(4, 512)
    out = tower(fused)
    assert out.shape == (4, 128)
    assert torch.allclose(torch.norm(out, dim=1), torch.ones(4), atol=1e-5)
