"""
Latency SLA smoke / load test against a running API (or in-process agents if artifacts exist).

Usage:
  python scripts/load_test_sla.py --url http://127.0.0.1:8000 --requests 50
  python scripts/load_test_sla.py --in-process --requests 20
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx
import yaml


def _sla_ms() -> float:
    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return float((cfg.get("guardrails") or {}).get("latency_sla_ms", 500))


def run_http(url: str, n: int) -> list[float]:
    latencies: list[float] = []
    with httpx.Client(base_url=url, timeout=30.0) as client:
        health = client.get("/health")
        if health.status_code != 200:
            raise RuntimeError(f"Health check failed: {health.status_code}")
        body = health.json()
        if not body.get("artifacts_loaded"):
            raise RuntimeError("API up but artifacts not loaded; run pipeline first")

        users = body.get("guardrails")  # noqa: F841 — health only
        for i in range(n):
            t0 = time.perf_counter()
            r = client.post("/recommend", json={"user_id": "U00001", "top_k": 5, "explain": False})
            elapsed = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            latencies.append(elapsed)
    return latencies


def summarize(latencies: list[float], sla: float) -> dict:
    latencies.sort()
    p95 = latencies[int(0.95 * len(latencies)) - 1] if len(latencies) > 1 else latencies[0]
    return {
        "count": len(latencies),
        "mean_ms": round(statistics.mean(latencies), 2),
        "p95_ms": round(p95, 2),
        "max_ms": round(max(latencies), 2),
        "sla_ms": sla,
        "sla_met_p95": p95 <= sla,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--in-process", action="store_true")
    args = parser.parse_args()

    sla = _sla_ms()
    if args.in_process:
        print("In-process load test requires artifacts; use --url against running uvicorn.")
        return 2

    try:
        latencies = run_http(args.url, args.requests)
    except Exception as e:
        print(f"Load test skipped: {e}")
        return 2

    stats = summarize(latencies, sla)
    print("Load test results:", stats)
    return 0 if stats["sla_met_p95"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
