"""Groww trade-API client wiring for Indian market DATA (RL-2026-07-10).

READ-ONLY. This module fetches market data only. It NEVER places, modifies, or
cancels orders - those SDK methods are deliberately not wrapped or called.

Secrets: API_KEY / API_SECRET are read from the project `.env` (git-ignored) into
the process environment and never logged, printed, or returned. The access token
is held in memory only.

Rate limit: Groww allows 10 req/s for live data (300/min) and 20 req/s for
non-trading (500/min). We self-throttle to 6 req/s (owner directive 2026-07-11:
stay 4 req/s below the documented ceiling) and 280/min, via a token-bucket
limiter shared by all callers.
"""

from collections import deque
from pathlib import Path
import os
import threading
import time

_ORDER_METHODS = frozenset({  # tripwire: never invoke these through this module
    "place_order", "modify_order", "cancel_order",
    "create_smart_order", "modify_smart_order", "cancel_smart_order",
})


def load_env(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from `.env` into os.environ. Values are never printed."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


class RateLimiter:
    """Thread-safe token bucket: at most `per_sec`/s and `per_min`/min calls."""

    def __init__(self, per_sec: int = 6, per_min: int = 280):
        self.per_sec = per_sec
        self.per_min = per_min
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > 60.0:
                    self._calls.popleft()
                in_last_sec = sum(1 for t in self._calls if now - t < 1.0)
                if len(self._calls) < self.per_min and in_last_sec < self.per_sec:
                    self._calls.append(now)
                    return
                if in_last_sec >= self.per_sec:
                    oldest_in_sec = next(t for t in self._calls if now - t < 1.0)
                    time.sleep(max(0.0, 1.0 - (now - oldest_in_sec)) + 0.001)
                else:  # per-minute cap hit
                    time.sleep(max(0.0, 60.0 - (now - self._calls[0])) + 0.001)


RATE = RateLimiter()

_client = None


def client():
    """Authenticated GrowwAPI client (singleton). Token stays in memory only."""
    global _client
    if _client is not None:
        return _client
    from growwapi import GrowwAPI

    load_env()
    api_key = os.environ.get("API_KEY")
    secret = os.environ.get("API_SECRET")
    if not api_key or not secret:
        raise RuntimeError("API_KEY / API_SECRET not found in environment or .env")
    RATE.acquire()
    token = GrowwAPI.get_access_token(api_key=api_key, secret=secret)
    _client = GrowwAPI(token)
    return _client


def call(method: str, *args, **kwargs):
    """Rate-limited call to a read-only GrowwAPI method. Refuses order methods."""
    if method in _ORDER_METHODS or any(k in method for k in ("place", "cancel", "modify_order")):
        raise PermissionError(f"refusing to call order method: {method}")
    fn = getattr(client(), method)
    RATE.acquire()
    return fn(*args, **kwargs)
