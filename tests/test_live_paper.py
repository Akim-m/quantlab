"""Tests for the REGIME live paper harness. No network, no real auth.

The load-bearing safety property: this harness fetches market data only and can
NEVER reach an order method. We prove it by spying on the groww_client dispatcher
and asserting every method the harness routes through it is read-only.
"""

import json

import pandas as pd
import pytest

from quantlab import groww_client as gc
from quantlab import live_paper as lp


class Spy:
    """Records every method name passed to groww_client.call and serves canned LTP."""

    PRICES = {"NSE_AAA": 110.0, "NSE_BBB": 50.0, "NSE_NIFTY": 20200.0}

    def __init__(self):
        self.methods = []

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:            # the spy honors the real guard
            raise PermissionError(method)
        syms = kwargs.get("exchange_trading_symbols", ())
        return {s: self.PRICES[s] for s in syms if s in self.PRICES}


def _synthetic_book():
    return lp.Book(
        weights=pd.Series({"AAA.NS": 0.3, "BBB.NS": 0.2}),
        regime_on=True, cash_frac=0.5,
        latest_date=pd.Timestamp("2026-07-08"),
        prev_close=pd.Series({"AAA.NS": 100.0, "BBB.NS": 50.0}),
        nsei_prev_close=20000.0,
    )


def test_to_groww_symbol_format():
    assert lp.to_groww("RELIANCE.NS") == "NSE_RELIANCE"
    assert lp.to_groww("m&m.ns") == "NSE_M&M"


def test_price_parser_tolerates_shapes():
    assert lp._price(1308.4) == 1308.4
    assert lp._price({"ltp": 1308.4}) == 1308.4
    assert lp._price({"last_price": 5.0}) == 5.0
    assert lp._price({"nope": 1}) is None
    assert lp._price(None) is None


def test_run_never_calls_order_method_and_computes_pnl(tmp_path, monkeypatch):
    spy = Spy()
    monkeypatch.setattr(gc, "call", spy)
    monkeypatch.setattr(lp, "current_book", lambda **kw: _synthetic_book())

    out = tmp_path / "paper_trades.jsonl"
    row = lp.run(path=str(out), write=True)

    # SAFETY: only read-only methods ever dispatched, none of them order methods.
    assert spy.methods, "harness must have fetched at least one quote"
    assert set(spy.methods) <= set(lp.READ_METHODS)
    assert not any(m in gc._ORDER_METHODS for m in spy.methods)

    # book_ret = 0.3*(110/100-1) + 0.2*(50/50-1) = 0.03 ; nifty = 20200/20000-1 = 0.01
    assert row["book_intraday_ret"] == pytest.approx(0.03)
    assert row["nifty_intraday_ret"] == pytest.approx(0.01)
    assert row["nifty_proxy"] == "NSE_NIFTY"
    assert row["regime_state"] == "risk_on"
    assert row["n_quotes_ok"] == 2 and row["n_names"] == 2
    assert row["groww_ok"] is True

    # snapshot was appended and is valid JSON with the required schema.
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    for k in ("timestamp", "regime_state", "cash_frac", "book_intraday_ret",
              "nifty_intraday_ret", "n_names", "n_quotes_ok"):
        assert k in rec


def test_fetch_ltp_degrades_gracefully_on_failure(monkeypatch):
    def boom(method, *a, **k):
        raise RuntimeError("auth blocked")
    monkeypatch.setattr(gc, "call", boom)
    prices, err = lp.fetch_ltp(["NSE_AAA", "NSE_BBB"])
    assert prices == {} and "auth blocked" in err


def test_run_records_book_only_when_quotes_unavailable(tmp_path, monkeypatch):
    def boom(method, *a, **k):
        raise RuntimeError("no entitlement")
    monkeypatch.setattr(gc, "call", boom)
    monkeypatch.setattr(lp, "current_book", lambda **kw: _synthetic_book())

    out = tmp_path / "p.jsonl"
    row = lp.run(path=str(out), write=True)
    assert row["book_intraday_ret"] is None      # no live P&L invented
    assert row["n_quotes_ok"] == 0 and row["groww_ok"] is False
    assert row["regime_state"] == "risk_on"      # book/regime still recorded
    assert out.read_text().strip()               # snapshot still written


def test_order_methods_refused_by_dispatcher():
    """The harness's only channel to Groww refuses order methods before any network."""
    with pytest.raises(PermissionError):
        gc.call("place_order")
