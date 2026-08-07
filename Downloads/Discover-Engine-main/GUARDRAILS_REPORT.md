# Workstream B — Serving Guardrails Report

**Date:** 2026-08-08  
**Prerequisite:** Workstream A **PASS** (`VERIFICATION_REPORT.md`)  
**API version:** 2.0.0  

---

## Executive Summary

| # | Guardrail | Status | Location |
|---|-----------|--------|----------|
| 1 | Request validation | **PASS** | `serving/schemas.py`, FastAPI 422 handling |
| 2 | Latency SLA + load test | **PASS** | `guardrails` meta on responses; `configs/config.yaml`; `scripts/load_test_sla.py` |
| 3 | Auditable diversity module | **PASS** | `guardrails/diversity.py`; `diversity_audit` on `/complete-the-look` |
| 4 | DPDP-style PII handling | **PASS** | `guardrails/pii.py`; redact before search + audit logs |
| 5 | Explainability fields | **PASS** | `items[]` / `reason` on agents; `explain` request flag |
| 6 | Audit logging | **PASS** | `guardrails/audit.py`; JSON lines on logger `discovery.audit` |
| 7 | Rate limiting | **PASS** | `guardrails/rate_limit.py`; 429 + `Retry-After` |
| 8 | Cost-per-inference | **PASS** | `guardrails/cost.py`; `guardrails.inference_cost_usd` on responses |

**W3 (Workstream A):** Index build now asserts unit-norm vectors in `index/build_index.py`.

---

## Response envelope

All inference endpoints return a `guardrails` object:

```json
{
  "request_id": "uuid",
  "latency_ms": 12.5,
  "sla_met": true,
  "inference_cost_usd": 0.00012,
  "pii_redacted": false
}
```

`/complete-the-look` additionally returns `diversity_audit` (category counts, skips, underfill flag).

---

## Configuration

See `configs/config.yaml` → `guardrails:`

- `latency_sla_ms` — compared to agent `latency_ms` (not client RTT)
- `rate_limit_per_minute` — per client IP (sliding window)
- `pii_redact_responses` — echo redacted query text when PII detected
- `inference_cost_usd` — per-endpoint static estimates (billing hook)

---

## Verification

```powershell
python -m pytest tests/ -v
python scripts/audit_workstream_b.py
# Optional (requires running API + artifacts):
python scripts/load_test_sla.py --url http://127.0.0.1:8000 --requests 50
```

---

## Deferred (explicit TODOs removed from Stage 1 placeholders)

- Toxicity / brand safety filters on `/recommend`
- Style coherence + inventory-aware filtering beyond category complementarity
- Learned fusion MLP (`models/fusion.py`)

These are product-policy features beyond the Stage 2 guardrail baseline.
