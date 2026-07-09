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
    assert rec["weights"] == pytest.approx({"AAA.NS": 0.3, "BBB.NS": 0.2})


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


# ---- forward_track: fully synthetic, monkeypatched price loader, no network ----

def _use_prices(monkeypatch, prices):
    monkeypatch.setattr(lp, "load_yahoo_ohlcv", lambda syms, refresh=False: {})
    monkeypatch.setattr(lp, "close_prices", lambda data: prices)


def _write_rows(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_forward_track_book_returns_hand_computed(tmp_path, monkeypatch):
    spy = Spy()
    monkeypatch.setattr(gc, "call", spy)               # prove Groww is never touched
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])
    prices = pd.DataFrame({
        "AAA.NS": [100.0, 110.0, 121.0],
        "BBB.NS": [100.0, 100.0, 100.0],
        "^NSEI":  [200.0, 202.0, 202.0],
    }, index=idx)
    _use_prices(monkeypatch, prices)
    p = tmp_path / "pt.jsonl"
    _write_rows(p, [
        {"panel_date": "2026-01-01", "weights": {"AAA.NS": 0.5, "BBB.NS": 0.5}},
        {"panel_date": "2026-01-02", "weights": {"AAA.NS": 1.0}},
        {"panel_date": "2026-01-05", "weights": {"BBB.NS": 1.0}},
    ])
    daily = lp.forward_track(path=str(p), cost_bps=0.0)

    assert spy.methods == []                            # the report path never calls Groww
    # row D1 = establishment (cost 0) -> 0 ; row D2 = W[D1].r(D1->D2) ; row D3 = W[D2].r(D2->D3)
    assert daily["book"].iloc[0] == pytest.approx(0.0)
    assert daily["book"].iloc[1] == pytest.approx(0.5 * 0.10 + 0.5 * 0.0)
    assert daily["book"].iloc[2] == pytest.approx(121 / 110 - 1)
    assert daily["nsei"].iloc[1] == pytest.approx(202 / 200 - 1)
    assert daily["active"].iloc[1] == pytest.approx(0.05 - (202 / 200 - 1))


def test_forward_track_dedupes_last_row_per_date(tmp_path, monkeypatch):
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])
    prices = pd.DataFrame({
        "AAA.NS": [100.0, 100.0, 100.0],
        "BBB.NS": [100.0, 100.0, 110.0],               # BBB moves only D2->D3
        "^NSEI":  [200.0, 200.0, 200.0],
    }, index=idx)
    _use_prices(monkeypatch, prices)
    p = tmp_path / "pt.jsonl"
    _write_rows(p, [
        {"panel_date": "2026-01-01", "weights": {"AAA.NS": 1.0}},
        {"panel_date": "2026-01-02", "weights": {"AAA.NS": 1.0}},   # stale, overwritten
        {"panel_date": "2026-01-02", "weights": {"BBB.NS": 1.0}},   # final D2 book
        {"panel_date": "2026-01-05", "weights": {"AAA.NS": 1.0}},
    ])
    daily = lp.forward_track(path=str(p), cost_bps=0.0)
    # D2->D3 return uses the LAST D2 book (BBB=1.0): BBB 100->110 = +0.10, not AAA's 0.0
    assert daily["book"].iloc[2] == pytest.approx(0.10)


def test_forward_track_no_lookahead(tmp_path, monkeypatch):
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])
    prices = pd.DataFrame({
        "AAA.NS": [100.0, 110.0, 121.0],
        "BBB.NS": [100.0, 105.0, 100.0],
        "^NSEI":  [200.0, 202.0, 205.0],
    }, index=idx)
    _use_prices(monkeypatch, prices)
    p = tmp_path / "pt.jsonl"
    base = {"panel_date": "2026-01-01", "weights": {"AAA.NS": 0.5, "BBB.NS": 0.5}}
    _write_rows(p, [base, {"panel_date": "2026-01-02", "weights": {"AAA.NS": 1.0}}])
    r0 = lp.forward_track(path=str(p), cost_bps=0.0)["book"].iloc[1]
    # perturbing the D+1 (D2) book must not move the realized D1->D2 return
    _write_rows(p, [base, {"panel_date": "2026-01-02", "weights": {"BBB.NS": 1.0}}])
    r1 = lp.forward_track(path=str(p), cost_bps=0.0)["book"].iloc[1]
    assert r0 == pytest.approx(r1)


def test_forward_track_needs_two_days_and_ignores_legacy_rows(tmp_path, capsys):
    p = tmp_path / "pt.jsonl"
    _write_rows(p, [
        {"panel_date": "2026-01-01", "weights": {"AAA.NS": 1.0}},
        {"panel_date": "2026-01-02"},                  # legacy row, no weights -> ignored
    ])
    assert lp.forward_track(path=str(p)) is None
    assert "need >= 2 snapshot days" in capsys.readouterr().out


def test_forward_track_missing_file_is_clean(tmp_path, capsys):
    assert lp.forward_track(path=str(tmp_path / "absent.jsonl")) is None
    assert "need >= 2 snapshot days" in capsys.readouterr().out


def test_forward_track_empty_book_is_cash_not_drift(tmp_path, monkeypatch):
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])
    prices = pd.DataFrame({
        "AAA.NS": [100.0, 110.0, 121.0],               # +10% each step
        "^NSEI":  [200.0, 200.0, 200.0],
    }, index=idx)
    _use_prices(monkeypatch, prices)
    p = tmp_path / "pt.jsonl"
    _write_rows(p, [
        {"panel_date": "2026-01-01", "weights": {"AAA.NS": 1.0}},
        {"panel_date": "2026-01-01"},                  # legacy: no key -> ignored, no clobber
        {"panel_date": "2026-01-02", "weights": {}},   # deliberate all-cash book
        {"panel_date": "2026-01-05", "weights": {"AAA.NS": 1.0}},
    ])
    daily = lp.forward_track(path=str(p), cost_bps=0.0)
    # D1->D2 earned by D1's book (legacy row ignored, else this would be 0)
    assert daily["book"].iloc[1] == pytest.approx(0.10)
    # D2 book is CASH: D2->D3 return is 0 despite AAA +10% (not the drifted D1 book)
    assert daily["book"].iloc[2] == pytest.approx(0.0)
    # with costs, the D2 row charges the exit turnover (full book out = 1.0 traded)
    daily = lp.forward_track(path=str(p), cost_bps=20.0)
    assert daily["book"].iloc[1] == pytest.approx(0.10 - 1.0 * 20 / 10_000)


def test_forward_track_drops_unpriced_symbol_without_renormalizing(tmp_path, monkeypatch):
    idx = pd.to_datetime(["2026-01-01", "2026-01-02"])
    prices = pd.DataFrame({          # ZZZ.NS is absent from the price panel
        "AAA.NS": [100.0, 110.0],
        "^NSEI":  [200.0, 200.0],
    }, index=idx)
    _use_prices(monkeypatch, prices)
    p = tmp_path / "pt.jsonl"
    _write_rows(p, [
        {"panel_date": "2026-01-01", "weights": {"AAA.NS": 0.5, "ZZZ.NS": 0.5}},
        {"panel_date": "2026-01-02", "weights": {"AAA.NS": 1.0}},
    ])
    daily = lp.forward_track(path=str(p), cost_bps=0.0)
    # ZZZ dropped -> its 0.5 becomes cash (earns 0), NOT renormalized onto AAA:
    # book D1->D2 = 0.5 * (110/100 - 1) = 0.05, not the renormalized 0.10.
    assert daily["book"].iloc[1] == pytest.approx(0.05)
