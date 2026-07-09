"""Tests for the F&O forward-collector. No network, no real auth.

Load-bearing safety property: this collector fetches market data only and can
NEVER reach an order method. We prove it by spying on the groww_client dispatcher
and asserting every method it routes through is read-only. The rest pins the
annualized-basis arithmetic and the PCR/ATM-IV/skew extraction to hand-computed
numbers, and proves a missing quote / down API degrades to a partial row.
"""

import json
from datetime import date

import pandas as pd
import pytest

from quantlab import fno_collect as fc
from quantlab import groww_client as gc

TODAY = date(2026, 7, 9)


# ---- annualized basis arithmetic (hand-computed, incl. dte/None edge cases) ----

def test_annualized_basis_hand_computed():
    # (105/100 - 1) * 365/30 = 0.05 * 12.16667 = 0.6083333
    assert fc.annualized_basis(105.0, 100.0, 30) == pytest.approx(0.05 * 365 / 30)
    # backwardation: fut below cash -> negative basis
    assert fc.annualized_basis(99.0, 100.0, 73) == pytest.approx(-0.01 * 365 / 73)


def test_annualized_basis_null_safe_edges():
    assert fc.annualized_basis(None, 100.0, 30) is None      # missing future
    assert fc.annualized_basis(105.0, None, 30) is None      # missing cash
    assert fc.annualized_basis(105.0, 0.0, 30) is None       # cash<=0 (no div-by-zero)
    assert fc.annualized_basis(105.0, 100.0, 0) is None      # expiry day, dte=0
    assert fc.annualized_basis(105.0, 100.0, -5) is None     # already expired
    assert fc.annualized_basis(105.0, 100.0, None) is None   # no expiry known


# ---- PCR / ATM IV / skew extraction from a canned chain payload ----

def _canned_chain():
    """spot 100, 5 strikes step 5. String keys mirror the JSON wire format.
    PCR = 105/110; ATM(100) IV = mean(15,15)=15; skew = PE.iv(95) - CE.iv(105) = 26-13 = 13."""
    def leg(oi, iv):
        return {"open_interest": oi, "volume": 0, "trading_symbol": "X",
                "greeks": {"iv": iv, "delta": 0.5}}
    return {
        "underlying_ltp": 100.0,
        "strikes": {
            "90":  {"CE": leg(10, 20), "PE": leg(40, 30)},
            "95":  {"CE": leg(10, 18), "PE": leg(30, 26)},
            "100": {"CE": leg(20, 15), "PE": leg(20, 15)},
            "105": {"CE": leg(30, 13), "PE": leg(10, 17)},
            "110": {"CE": leg(40, 12), "PE": leg(5, 20)},
        },
    }


def test_chain_metrics_hand_computed():
    m, note = fc.chain_metrics(_canned_chain())
    assert note is None
    assert m["spot"] == 100.0 and m["n_strikes"] == 5
    assert m["pcr"] == pytest.approx(105 / 110)
    assert m["atm_strike"] == 100.0 and m["atm_iv"] == pytest.approx(15.0)
    assert m["skew"] == pytest.approx(13.0)          # put(95)=26 minus call(105)=13


def test_chain_metrics_empty_and_zero_oi_null_safe():
    m, note = fc.chain_metrics({"underlying_ltp": 100.0, "strikes": {}})
    assert note == "chain_empty" and m["pcr"] is None and m["atm_iv"] is None
    # all-CE-OI-zero -> PCR null (no div-by-zero), IV of 0 treated as missing
    zero = {"underlying_ltp": 100.0, "strikes": {
        "100": {"CE": {"open_interest": 0, "greeks": {"iv": 0}},
                "PE": {"open_interest": 5, "greeks": {"iv": 0}}}}}
    m, note = fc.chain_metrics(zero)
    assert m["pcr"] is None and m["atm_iv"] is None and note is None


# ---- read-only spy: only declared read methods dispatched, degradation is partial ----

class Spy:
    """Records every method routed through groww_client.call; serves canned data."""

    CASH = {"NSE_AAA": 100.0, "NSE_BBB": 200.0}
    FNO = {"NSE_AAA26JULFUT": 101.0, "NSE_AAA26AUGFUT": 102.0,
           "NSE_BBB26JULFUT": 199.0}          # BBB next-month quote deliberately absent

    def __init__(self, fail_ltp=False):
        self.methods = []
        self.fail_ltp = fail_ltp

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:        # the spy honors the real guard
            raise PermissionError(method)
        if method == "get_ltp":
            if self.fail_ltp:
                raise RuntimeError("auth blocked")
            seg = kwargs["segment"]
            table = self.CASH if seg == "CASH" else self.FNO
            return {s: table[s] for s in kwargs["exchange_trading_symbols"] if s in table}
        if method == "get_expiries":
            return {"expiries": ["2026-06-30", "2026-07-14", "2026-08-25"]}
        if method == "get_option_chain":
            return _canned_chain()
        raise AssertionError(f"unexpected method: {method}")


def _canned_instruments():
    rows = [
        ("AAA", "AAA26JULFUT", "2026-07-28"), ("AAA", "AAA26AUGFUT", "2026-08-25"),
        ("BBB", "BBB26JULFUT", "2026-07-28"), ("BBB", "BBB26AUGFUT", "2026-08-25"),
    ]
    return pd.DataFrame([
        {"instrument_type": "FUT", "segment": "FNO", "exchange": "NSE",
         "underlying_symbol": u, "trading_symbol": ts, "expiry_date": ex}
        for u, ts, ex in rows
    ])


def _patch_universe(monkeypatch):
    monkeypatch.setattr(fc, "groww_instruments", lambda refresh=False: _canned_instruments())
    monkeypatch.setattr(fc, "fno_shortable", lambda instruments: {"AAA.NS", "BBB.NS"})


def test_collect_read_only_partial_row(tmp_path, monkeypatch):
    spy = Spy()
    monkeypatch.setattr(gc, "call", spy)
    _patch_universe(monkeypatch)

    out = tmp_path / "fno_daily.jsonl"
    row = fc.collect(today=TODAY, path=str(out), write=True, verbose=False)

    # SAFETY: only read-only methods dispatched, none of them order methods.
    assert spy.methods and set(spy.methods) <= set(fc.READ_METHODS)
    assert not any(m in gc._ORDER_METHODS for m in spy.methods)

    # coverage counters: 2 underlyings, both cash, both fut1; only AAA has fut2 quote.
    assert row["n_underlyings"] == 2 and row["n_cash_ok"] == 2
    assert row["n_fut1_ok"] == 2 and row["n_fut2_ok"] == 1

    # basis arithmetic wired end-to-end (AAA: fut1 101 / cash 100, dte = 28 Jul - 9 Jul = 19d)
    aaa = row["basis"]["AAA"]
    assert aaa["cash"] == 100.0 and aaa["fut1"] == 101.0 and aaa["dte1"] == 19
    assert aaa["b1"] == pytest.approx(round((101 / 100 - 1) * 365 / 19, 4))
    # BBB's missing next-month quote -> partial, not a crash: fut2/b2 null, fut1 present.
    bbb = row["basis"]["BBB"]
    assert bbb["fut2"] is None and bbb["b2"] is None and bbb["fut1"] == 199.0

    # NIFTY chain block computed; nearest future expiry chosen (2026-06-30 is past).
    assert row["chain_ok"] is True and row["nifty_expiry"] == "2026-07-14"
    assert row["pcr"] == pytest.approx(round(105 / 110, 4)) and row["skew"] == pytest.approx(13.0)

    # snapshot appended, valid JSON, required schema present.
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    for k in ("timestamp", "hypothesis_ref", "collect_date", "n_underlyings",
              "n_cash_ok", "n_fut1_ok", "chain_ok", "pcr", "atm_iv", "skew", "basis"):
        assert k in rec
    assert rec["hypothesis_ref"] == "RL-2026-07-15"


def test_collect_degrades_when_groww_down(tmp_path, monkeypatch):
    spy = Spy(fail_ltp=True)                    # LTP fails; chain also raises below
    monkeypatch.setattr(gc, "call", spy)
    _patch_universe(monkeypatch)

    out = tmp_path / "fno_daily.jsonl"
    row = fc.collect(today=TODAY, path=str(out), write=True, verbose=False)

    # no quotes invented, no crash: counters zero, every PRICE/basis field null
    # (dte comes from the static instrument master, so it stays populated).
    assert row["n_cash_ok"] == 0 and row["n_fut1_ok"] == 0
    aaa = row["basis"]["AAA"]
    assert all(aaa[k] is None for k in ("cash", "fut1", "fut2", "b1", "b2"))
    assert row["note"] != "ok" and "auth blocked" in row["note"]
    assert out.read_text().strip()             # snapshot still written


def test_dry_run_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "call", Spy())
    _patch_universe(monkeypatch)
    out = tmp_path / "fno_daily.jsonl"
    fc.collect(today=TODAY, path=str(out), write=False, verbose=False)
    assert not out.exists()


def test_order_methods_refused_by_dispatcher():
    """The collector's only channel to Groww refuses order methods before any network."""
    with pytest.raises(PermissionError):
        gc.call("place_order")
