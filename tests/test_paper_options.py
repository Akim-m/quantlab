"""Tests for the forward-only PAPER short-straddle harness. No network, no auth.

Load-bearing safety property: the harness fetches market data only and can NEVER
reach an order method. We prove it by spying on the groww_client dispatcher and
asserting every routed method is read-only. The rest pins the short-straddle
mark-to-market arithmetic (incl. a hand-computed two-day P&L sequence), expiry
settlement at intrinsic (ITM-put and ITM-call), roll behaviour, and that a data
gap carries the position unchanged instead of crashing.
"""

import json
from datetime import date

import pandas as pd
import pytest

from quantlab import groww_client as gc
from quantlab import paper_options as po

TODAY = date(2026, 7, 9)


def _chain(spot, ce, pe, iv, center=20000, step=50):
    """Small NIFTY chain: only the center strike carries the controlled premiums."""
    def leg(ltp, iv):
        return {"ltp": ltp, "open_interest": 100, "volume": 10,
                "greeks": {"iv": iv, "delta": 0.5}}
    strikes = {}
    for i in range(-3, 4):
        k = center + i * step
        legs = (leg(ce, iv), leg(pe, iv)) if k == center else (leg(5.0, iv), leg(5.0, iv))
        strikes[str(k)] = {"CE": legs[0], "PE": legs[1]}
    return {"underlying_ltp": spot, "strikes": strikes}


def _canned_instruments():
    return pd.DataFrame([
        {"underlying_symbol": "NIFTY", "instrument_type": "CE", "lot_size": 50},
        {"underlying_symbol": "NIFTY", "instrument_type": "PE", "lot_size": 50},
        {"underlying_symbol": "RELIANCE", "instrument_type": "EQ", "lot_size": 1},
    ])


class Spy:
    """Records every method routed through groww_client.call; serves a canned chain.
    Mutate spot / ce / pe / center between snapshot() calls to drive a sequence."""

    def __init__(self):
        self.methods = []
        self.expiries = ["2026-07-14", "2026-07-21"]
        self.spot = 20000.0
        self.ce, self.pe, self.iv = 100.0, 110.0, 12.0
        self.center = 20000

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:              # spy honors the real guard
            raise PermissionError(method)
        if method == "get_expiries":
            return {"expiries": list(self.expiries)}
        if method == "get_option_chain":
            return _chain(self.spot, self.ce, self.pe, self.iv, center=self.center)
        if method == "get_ltp":
            return {po.NIFTY_SPOT_SYM: self.spot}
        raise AssertionError(f"unexpected method: {method}")


class FailChainSpy(Spy):
    """Expiries resolve but every quote path is down (API blocked / market shut)."""

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:
            raise PermissionError(method)
        if method == "get_expiries":
            return {"expiries": list(self.expiries)}
        if method in ("get_option_chain", "get_ltp"):
            raise RuntimeError("quotes down")
        raise AssertionError(f"unexpected method: {method}")


def _patch(monkeypatch, spy):
    monkeypatch.setattr(gc, "call", spy)
    monkeypatch.setattr(po, "groww_instruments", lambda refresh=False: _canned_instruments())
    monkeypatch.setattr(po, "realized_vol_5d", lambda refresh=False: 11.0)


# ---- open: nearest >=2-dte expiry, ATM strike, short at chain LTPs -----------------

def test_open_selects_nearest_expiry_and_atm(tmp_path, monkeypatch):
    spy = Spy()
    _patch(monkeypatch, spy)
    out = tmp_path / "po.jsonl"
    row = po.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    assert row["action"] == "open"
    assert row["expiry"] == "2026-07-14" and row["dte"] == 5   # 07-14 (dte5), not 07-21
    assert row["strike"] == 20000.0                            # nearest to spot 20000
    assert row["ce_entry"] == 100.0 and row["pe_entry"] == 110.0
    assert row["entry_credit"] == 210.0 and row["lot_size"] == 50
    assert row["mark_value"] == 210.0 and row["atm_iv"] == 12.0
    assert row["daily_pnl"] == 0.0 and row["cumulative_pnl"] == 0.0
    assert row["mark_basis"] == "ltp" and row["realized_vol_5d"] == 11.0

    # SAFETY: only read-only methods dispatched, none of them order methods.
    assert spy.methods and set(spy.methods) <= set(po.READ_METHODS)
    assert not any(m in gc._ORDER_METHODS for m in spy.methods)

    rec = json.loads(out.read_text().strip())
    assert rec["hypothesis_ref"] == "RL-2026-07-18" and "timestamp" in rec


# ---- mark-to-market: hand-computed two-day P&L sequence (lot 50) -------------------

def test_mark_two_day_pnl_sequence(tmp_path, monkeypatch):
    spy = Spy()
    _patch(monkeypatch, spy)
    out = tmp_path / "po.jsonl"
    po.snapshot(today=date(2026, 7, 9), path=str(out), write=True, verbose=False)  # open @210

    spy.ce, spy.pe = 90.0, 100.0        # decay to 190 -> (210-190)*50 = +1000
    r1 = po.snapshot(today=date(2026, 7, 10), path=str(out), write=True, verbose=False)
    assert r1["action"] == "mark" and r1["mark_value"] == 190.0
    assert r1["daily_pnl"] == 1000.0 and r1["cumulative_pnl"] == 1000.0

    spy.ce, spy.pe = 120.0, 130.0       # spike to 250 -> (190-250)*50 = -3000 ; cum -2000
    r2 = po.snapshot(today=date(2026, 7, 11), path=str(out), write=True, verbose=False)
    assert r2["mark_value"] == 250.0 and r2["daily_pnl"] == -3000.0
    assert r2["cumulative_pnl"] == -2000.0


# ---- expiry settlement at intrinsic: ITM-call and ITM-put -------------------------

def test_expiry_settlement_itm_call(tmp_path, monkeypatch):
    spy = Spy()
    _patch(monkeypatch, spy)
    out = tmp_path / "po.jsonl"
    po.snapshot(today=date(2026, 7, 9), path=str(out), write=True, verbose=False)  # open 07-14 @210

    spy.spot, spy.center = 20300.0, 20300      # call ITM by 300 at expiry
    r = po.snapshot(today=date(2026, 7, 14), path=str(out), write=True, verbose=False)
    assert r["action"] == "roll_settle"
    s = r["settled"]
    assert s["basis"] == "intrinsic" and s["close_value"] == 300.0 and s["prev_mark"] == 210.0
    assert s["realized_pnl"] == (210.0 - 300.0) * 50 == -4500.0
    assert r["daily_pnl"] == -4500.0 and r["cumulative_pnl"] == -4500.0
    # rolled into the next weekly at the new ATM
    assert r["expiry"] == "2026-07-21" and r["strike"] == 20300.0 and r["entry_credit"] == 210.0


def test_expiry_settlement_itm_put(tmp_path, monkeypatch):
    spy = Spy()
    _patch(monkeypatch, spy)
    out = tmp_path / "po.jsonl"
    po.snapshot(today=date(2026, 7, 9), path=str(out), write=True, verbose=False)  # open 07-14 @210

    spy.spot, spy.center = 19800.0, 19800      # put ITM by 200 at expiry
    r = po.snapshot(today=date(2026, 7, 14), path=str(out), write=True, verbose=False)
    assert r["action"] == "roll_settle"
    s = r["settled"]
    assert s["basis"] == "intrinsic" and s["close_value"] == 200.0
    assert s["realized_pnl"] == (210.0 - 200.0) * 50 == 500.0
    assert r["daily_pnl"] == 500.0 and r["cumulative_pnl"] == 500.0
    assert r["expiry"] == "2026-07-21" and r["strike"] == 19800.0


# ---- roll one day before expiry: close at LTP, then re-open ------------------------

def test_roll_close_dte1(tmp_path, monkeypatch):
    spy = Spy()
    _patch(monkeypatch, spy)
    out = tmp_path / "po.jsonl"
    po.snapshot(today=date(2026, 7, 9), path=str(out), write=True, verbose=False)  # open 07-14 @210

    spy.ce, spy.pe = 50.0, 40.0        # close straddle at 90 on dte=1
    r = po.snapshot(today=date(2026, 7, 13), path=str(out), write=True, verbose=False)
    assert r["action"] == "roll_close"
    assert r["settled"]["basis"] == "ltp" and r["settled"]["close_value"] == 90.0
    assert r["daily_pnl"] == (210.0 - 90.0) * 50 == 6000.0 and r["cumulative_pnl"] == 6000.0
    assert r["expiry"] == "2026-07-21"          # rolled into the next weekly


# ---- degradation: a data gap carries the position unchanged, no crash -------------

def test_degradation_carries_position(tmp_path, monkeypatch):
    spy = Spy()
    _patch(monkeypatch, spy)
    out = tmp_path / "po.jsonl"
    opened = po.snapshot(today=date(2026, 7, 9), path=str(out), write=True, verbose=False)

    fail = FailChainSpy()
    monkeypatch.setattr(gc, "call", fail)       # chain goes dark on a normal mark day
    r = po.snapshot(today=date(2026, 7, 10), path=str(out), write=True, verbose=False)

    assert r["action"] == "carry" and r["daily_pnl"] is None
    for k in ("expiry", "strike", "lot_size", "entry_credit", "mark_value", "cumulative_pnl"):
        assert r[k] == opened[k]                 # position + last mark untouched
    assert r["spot"] is None and r["ce_ltp"] is None
    assert out.read_text().strip()               # still wrote a row, never crashed
    assert set(fail.methods) <= set(po.READ_METHODS)


def test_open_degrades_when_chain_down(tmp_path, monkeypatch):
    fail = FailChainSpy()
    _patch(monkeypatch, fail)                    # no prior position + quotes down
    out = tmp_path / "po.jsonl"
    r = po.snapshot(today=TODAY, path=str(out), write=True, verbose=False)
    assert r["action"] == "degraded" and r["strike"] is None
    assert r["daily_pnl"] is None and r["cumulative_pnl"] == 0.0
    assert out.read_text().strip()               # degraded row still written


# ---- dry-run and the dispatcher order-method guard --------------------------------

def test_dry_run_does_not_write(tmp_path, monkeypatch):
    _patch(monkeypatch, Spy())
    out = tmp_path / "po.jsonl"
    po.snapshot(today=TODAY, path=str(out), write=False, verbose=False)
    assert not out.exists()


def test_order_methods_refused_by_dispatcher():
    """The harness's only channel to Groww refuses order methods before any network."""
    with pytest.raises(PermissionError):
        gc.call("place_order")
