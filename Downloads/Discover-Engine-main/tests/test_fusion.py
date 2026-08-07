import numpy as np

from models.fusion import fuse_embeddings, _l2_normalize


def test_fuse_embeddings_unit_norm():
    rng = np.random.default_rng(42)
    img = rng.standard_normal((4, 512)).astype(np.float32)
    txt = rng.standard_normal((4, 512)).astype(np.float32)
    fused = fuse_embeddings(img, txt)
    norms = np.linalg.norm(fused, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_l2_normalize():
    v = np.array([[3.0, 4.0]], dtype=np.float32)
    out = _l2_normalize(v)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-6
