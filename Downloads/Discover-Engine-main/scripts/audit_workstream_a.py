"""
Workstream A audit: fusion, projection, InfoNCE, FAISS self-retrieval, diversity cap.
Run: python scripts/audit_workstream_a.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.complete_the_look_agent import MAX_CATEGORY_FRACTION, CompleteTheLookAgent
from index.build_index import build_index_from_embeddings
from index.faiss_store import FaissStore
from models.discovery_model import DiscoveryModel
from models.fusion import _l2_normalize, fuse_embeddings
from models.item_tower import ItemTower
from training.train_user_tower import infonce_loss


def audit_fusion() -> dict:
    rng = np.random.default_rng(42)
    img = rng.standard_normal((100, 512)).astype(np.float32)
    txt = rng.standard_normal((100, 512)).astype(np.float32)

    fused = fuse_embeddings(img, txt)
    fused_norms = np.linalg.norm(fused, axis=1)

    img_norm = _l2_normalize(img)
    txt_norm = _l2_normalize(txt)
    expected = _l2_normalize(0.6 * img_norm + 0.4 * txt_norm)

    passed = bool(
        np.allclose(fused_norms, 1.0, atol=1e-5) and np.allclose(fused, expected, atol=1e-6)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "fused_norm_max_deviation": float(np.abs(fused_norms - 1.0).max()),
        "manual_recompute_match": bool(np.allclose(fused, expected, atol=1e-6)),
        "weights": "0.6 image + 0.4 text",
    }


def audit_projection() -> dict:
    tower = ItemTower(input_dim=512, output_dim=128)
    rng = np.random.default_rng(42)
    fused = torch.from_numpy(rng.standard_normal((10, 512)).astype(np.float32))

    out = tower(fused)
    norms = torch.norm(out, dim=1)
    with torch.no_grad():
        raw_proj = tower.projection(fused)
        raw_norms = torch.norm(raw_proj, dim=1)

    passed = bool(
        torch.allclose(norms, torch.ones(10), atol=1e-5)
        and not torch.allclose(raw_norms, torch.ones(10), atol=0.1)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "output_norms": [float(x) for x in norms.tolist()],
        "pre_normalize_norm_range": [float(raw_norms.min()), float(raw_norms.max())],
        "normalize_after_linear": True,
    }


def audit_infonce() -> dict:
    num_items = 50
    model = DiscoveryModel(num_items=num_items, max_history=5)
    model.set_fused_item_vectors(torch.randn(num_items, 512))

    B = 8
    hists = torch.randint(0, num_items, (B, 5))
    masks = torch.ones(B, 5)
    pos = torch.randint(0, num_items, (B,))
    temp = 0.05

    model.eval()
    with torch.no_grad():
        user_emb = model.encode_users(hists, masks)
        item_emb = model.encode_items(pos)
        logits = (user_emb @ item_emb.T) / temp
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        diag = torch.arange(B)
        manual_loss = -log_probs[diag, diag].mean()

    model.train()
    loss = infonce_loss(model, hists, masks, pos, temp)

    passed = abs(manual_loss.item() - loss.item()) < 1e-5
    return {
        "status": "PASS" if passed else "FAIL",
        "temperature": temp,
        "logits_shape": f"({B}, {B})",
        "manual_loss": float(manual_loss.item()),
        "function_loss": float(loss.item()),
        "user_emb_unit_norm": bool(torch.allclose(torch.norm(user_emb, dim=1), torch.ones(B), atol=1e-5)),
        "item_emb_unit_norm": bool(torch.allclose(torch.norm(item_emb, dim=1), torch.ones(B), atol=1e-5)),
    }


def audit_faiss_self_retrieval(n_items: int = 200) -> dict:
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n_items, 128)).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    store = FaissStore(index=build_index_from_embeddings(emb))

    failures = []
    for i in range(n_items):
        _, indices = store.search(emb[i : i + 1], top_k=1)
        if indices[0, 0] != i:
            failures.append(i)

    # Projected embeddings path
    tower = ItemTower()
    fused512 = rng.standard_normal((n_items, 512)).astype(np.float32)
    fused512 = fused512 / np.linalg.norm(fused512, axis=1, keepdims=True)
    proj_emb = tower.encode_fused_numpy(fused512).astype(np.float32)
    store2 = FaissStore(index=build_index_from_embeddings(proj_emb))
    proj_failures = []
    for i in range(n_items):
        _, indices = store2.search(proj_emb[i : i + 1], top_k=1)
        if indices[0, 0] != i:
            proj_failures.append(i)

    passed = len(failures) == 0 and len(proj_failures) == 0
    return {
        "status": "PASS" if passed else "FAIL",
        "index_size": n_items,
        "random_unit_vector_failures": len(failures),
        "projected_embedding_failures": len(proj_failures),
    }


def audit_diversity() -> dict:
    rng = np.random.default_rng(1)
    n = 50
    emb = rng.standard_normal((n, 128)).astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    meta = [{"item_idx": 0, "category": "Topwear"}]
    meta += [{"item_idx": i, "category": "Bottomwear"} for i in range(1, n)]
    store = FaissStore(index=build_index_from_embeddings(emb))
    agent = CompleteTheLookAgent(store, emb, meta)

    top_k = 10
    result = agent.complete(seed_item_idx=0, top_k=top_k)
    max_allowed = max(1, int(top_k * MAX_CATEGORY_FRACTION + 0.999))
    counts = Counter(
        m["category"] for m in meta if m["item_idx"] in result["recommendations"]
    )

    passed = all(c <= max_allowed for c in counts.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "max_category_fraction": MAX_CATEGORY_FRACTION,
        "enforcement": "hard (greedy skip when category count >= cap)",
        "top_k": top_k,
        "max_per_category": max_allowed,
        "category_counts": dict(counts),
        "recommendations_returned": len(result["recommendations"]),
    }


def main() -> None:
    sections = [
        ("1_fusion_normalization", audit_fusion),
        ("2_projection_layer", audit_projection),
        ("3_infonce_temperature", audit_infonce),
        ("4_faiss_self_retrieval", audit_faiss_self_retrieval),
        ("5_diversity_cap", audit_diversity),
    ]

    print("Workstream A Audit Results")
    print("=" * 60)
    for name, fn in sections:
        result = fn()
        print(f"\n[{name}] {result['status']}")
        for k, v in result.items():
            if k != "status":
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
