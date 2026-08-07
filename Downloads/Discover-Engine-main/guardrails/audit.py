"""Structured audit logging for inference requests (PII-safe)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

audit_logger = logging.getLogger("discovery.audit")


def new_request_id() -> str:
    return str(uuid.uuid4())


def log_inference_event(
    *,
    request_id: str,
    endpoint: str,
    client_ip: str,
    outcome: str,
    latency_ms: float,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "endpoint": endpoint,
        "client_ip": client_ip,
        "outcome": outcome,
        "latency_ms": latency_ms,
        **(extra or {}),
    }
    audit_logger.info(json.dumps(payload, default=str))
