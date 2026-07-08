"""Guardrail + plumbing tests for the Groww client. No network, no real auth."""

import time

import pytest

from quantlab import groww_client as gc


@pytest.mark.parametrize("method", [
    "place_order", "modify_order", "cancel_order",
    "create_smart_order", "modify_smart_order", "cancel_smart_order",
    "place_order_v2",  # caught by substring guard, not just the exact set
])
def test_call_refuses_order_methods(method):
    """The 'never trade' rule as code: order methods raise before any auth/network."""
    with pytest.raises(PermissionError):
        gc.call(method)


def test_rate_limiter_enforces_per_second():
    rl = gc.RateLimiter(per_sec=5, per_min=10_000)
    t0 = time.monotonic()
    for _ in range(10):          # 10 calls capped at 5/s must take >= ~1s
        rl.acquire()
    assert time.monotonic() - t0 >= 0.9


def test_rate_limiter_allows_burst_under_cap():
    rl = gc.RateLimiter(per_sec=50, per_min=10_000)
    t0 = time.monotonic()
    for _ in range(5):
        rl.acquire()
    assert time.monotonic() - t0 < 0.5   # under the cap: no throttling


def test_load_env_sets_names_without_clobbering(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('QL_TEST_TOKEN="xyz"\n# comment\nQL_TEST_OTHER=abc\n')
    monkeypatch.delenv("QL_TEST_TOKEN", raising=False)
    monkeypatch.setenv("QL_TEST_OTHER", "preexisting")
    gc.load_env(env)
    import os
    assert os.environ["QL_TEST_TOKEN"] == "xyz"      # quotes stripped
    assert os.environ["QL_TEST_OTHER"] == "preexisting"  # setdefault: not overwritten
