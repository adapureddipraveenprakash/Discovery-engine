"""In-memory sliding-window rate limiter (per client IP)."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after_sec: float


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_sec: float = 60.0):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> RateLimitResult:
        now = time.monotonic()
        window_start = now - self.window_sec
        q = self._hits[key]
        while q and q[0] <= window_start:
            q.popleft()

        if len(q) >= self.max_requests:
            retry = max(0.0, q[0] + self.window_sec - now)
            return RateLimitResult(
                allowed=False,
                limit=self.max_requests,
                remaining=0,
                retry_after_sec=round(retry, 2),
            )

        q.append(now)
        remaining = self.max_requests - len(q)
        return RateLimitResult(
            allowed=True,
            limit=self.max_requests,
            remaining=remaining,
            retry_after_sec=0.0,
        )
