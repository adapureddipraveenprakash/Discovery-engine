"""
DPDP-style PII handling: detect and redact common identifiers in free-text queries.

Redaction applies before audit logs and optional response echo (search).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# India-centric + generic patterns (conservative — prefer over-redaction in logs)
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b")),
    ("aadhaar", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
]


@dataclass(frozen=True)
class PiiScanResult:
    redacted_text: str
    had_pii: bool
    types_found: tuple[str, ...]


def redact_pii(text: str | None) -> PiiScanResult:
    if not text:
        return PiiScanResult(redacted_text=text or "", had_pii=False, types_found=())

    found: list[str] = []
    out = text
    for name, pattern in _PATTERNS:
        if pattern.search(out):
            found.append(name)
            out = pattern.sub(f"[REDACTED_{name.upper()}]", out)

    return PiiScanResult(
        redacted_text=out,
        had_pii=bool(found),
        types_found=tuple(found),
    )
