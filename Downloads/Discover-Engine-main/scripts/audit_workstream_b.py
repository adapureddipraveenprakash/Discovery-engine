"""
Workstream B audit: validation, PII, diversity module, index norm gate, config knobs.
Run: python scripts/audit_workstream_b.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from pydantic import ValidationError

from guardrails.diversity import apply_category_cap
from guardrails.pii import redact_pii
from guardrails.rate_limit import SlidingWindowRateLimiter
from index.build_index import build_index_from_embeddings
from serving.schemas import RecommendRequest, SearchRequest


def audit_validation() -> dict:
    ok = True
    details = {}
    try:
        RecommendRequest(user_id="U00001", top_k=10)
        SearchRequest(query_text="blue jacket", top_k=5)
        details["valid_samples"] = True
    except ValidationError as e:
        ok = False
        details["valid_samples_error"] = str(e)

    try:
        SearchRequest(query_text=None, query_image=None)
        ok = False
        details["empty_search_should_fail"] = False
    except ValidationError:
        details["empty_search_rejected"] = True

    return {"pass": ok, **details}


def audit_pii() -> dict:
    r = redact_pii("email test@corp.in phone 9123456789")
    return {
        "pass": r.had_pii and "test@corp.in" not in r.redacted_text,
        "types_found": list(r.types_found),
    }


def audit_diversity_module() -> dict:
    idx_to_cat = {i: "A" for i in range(50)}
    res = apply_category_cap(list(range(50)), top_k=10, idx_to_category=idx_to_cat)
    audit = res.to_audit_dict()
    return {
        "pass": len(res.selected) <= 10 and audit["max_per_category"] == 4,
        "selected": len(res.selected),
        "audit_keys": sorted(audit.keys()),
    }


def audit_index_norm_gate() -> dict:
    bad = np.ones((3, 8), dtype="float32") * 2.0
    try:
        build_index_from_embeddings(bad)
        return {"pass": False, "reason": "unnormalized vectors accepted"}
    except ValueError:
        good = bad / np.linalg.norm(bad, axis=1, keepdims=True)
        build_index_from_embeddings(good)
        return {"pass": True}


def audit_config() -> dict:
    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    g = cfg.get("guardrails") or {}
    required = {"latency_sla_ms", "rate_limit_per_minute", "inference_cost_usd"}
    missing = required - set(g.keys())
    return {"pass": not missing, "missing": sorted(missing), "guardrails": g}


def audit_rate_limit() -> dict:
    lim = SlidingWindowRateLimiter(3, 60.0)
    allowed = [lim.check("x").allowed for _ in range(4)]
    return {"pass": allowed == [True, True, True, False]}


def main() -> int:
    sections = [
        ("1_request_validation", audit_validation),
        ("2_pii_redaction", audit_pii),
        ("3_auditable_diversity", audit_diversity_module),
        ("4_index_norm_gate", audit_index_norm_gate),
        ("5_guardrails_config", audit_config),
        ("6_rate_limiting", audit_rate_limit),
    ]
    print("Workstream B Audit Results")
    print("=" * 40)
    all_pass = True
    for name, fn in sections:
        result = fn()
        status = "PASS" if result.get("pass") else "FAIL"
        if not result.get("pass"):
            all_pass = False
        print(f"[{name}] {status}")
        for k, v in result.items():
            if k != "pass":
                print(f"  {k}: {v}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
