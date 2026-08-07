"""Cost-per-inference estimates (USD) for observability / billing hooks."""

from __future__ import annotations

from typing import Mapping

ENDPOINT_COST_USD: Mapping[str, float] = {
    "recommend": 0.00012,
    "search": 0.0018,
    "complete_the_look": 0.00008,
}


def estimate_inference_cost_usd(endpoint: str, overrides: Mapping[str, float] | None = None) -> float:
    table = dict(ENDPOINT_COST_USD)
    if overrides:
        table.update(overrides)
    return float(table.get(endpoint, 0.0))
