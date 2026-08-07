"""Guardrail unit tests (no artifacts required)."""

import numpy as np
import pytest
from pydantic import ValidationError

from guardrails.diversity import apply_category_cap
from guardrails.pii import redact_pii
from guardrails.rate_limit import SlidingWindowRateLimiter
from index.build_index import assert_unit_norm, build_index_from_embeddings
from serving.schemas import SearchRequest


def test_pii_redacts_email_and_phone():
    text = "Contact me at user@example.com or 9876543210 for jacket"
    result = redact_pii(text)
    assert result.had_pii
    assert "user@example.com" not in result.redacted_text
    assert "9876543210" not in result.redacted_text
    assert "email" in result.types_found


def test_diversity_cap_hard_skip():
    meta = {i: "Bottomwear" for i in range(20)}
    candidates = list(range(20))
    out = apply_category_cap(candidates, top_k=10, idx_to_category=meta)
    assert len(out.selected) == 4  # 35% of 10 -> max 3 per cat... max(1, int(3.5))=4
    assert out.underfill
    audit = out.to_audit_dict()
    assert audit["skipped_diversity_cap"] > 0


def test_rate_limiter_blocks_burst():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_sec=60.0)
    assert limiter.check("client-a").allowed
    assert limiter.check("client-a").allowed
    blocked = limiter.check("client-a")
    assert not blocked.allowed
    assert limiter.check("client-b").allowed


def test_search_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        SearchRequest(query_text=None, query_image=None)


def test_search_request_rejects_unsafe_image_path():
    with pytest.raises(ValidationError):
        SearchRequest(query_text="dress", query_image="../../../etc/passwd")


def test_index_rejects_unnormalized_vectors():
    emb = np.random.randn(5, 128).astype("float32")
    with pytest.raises(ValueError, match="L2-normalized"):
        build_index_from_embeddings(emb)

    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    assert_unit_norm(emb)
    build_index_from_embeddings(emb)
