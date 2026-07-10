"""Tests for the forward-only PAPER cash-secured put-write harness. No network.

Covers what this book adds over / changes from RL-18's straddle:
  - strike selection: the listed strike nearest 0.98*spot (~2% OTM), by geometry;
  - entry / mark / settle P&L arithmetic on a single short put (hand-computed);
  - European put settlement close_value = max(0, strike - settle_spot);
  - roll at DTE < 2 (close at LTP, reopen);
  - the ledger row schema (RL-18 block + strike_ratio / otm_pct / notional);
  - determinism: identical ledger + inputs -> identical row;
  - the load-bearing safety property (RL-18): only read-only Groww methods dispatched.
"""

import json
from datetime import date

import pandas as pd

from quantlab import groww_client as gc
from quantlab import paper_options as po
from quantlab import paper_options_putw as pw

TODAY = date(2026, 7, 10)          # 4 dte before the 2026-07-14 held expiry (normal mark)
EXPIRY_DAY = date(2026, 7, 14)     # >= held expiry -> settle
ROLL_DAY = date(2026, 7, 13)       # 1 dte before held expiry -> roll (DTE < 2)


def _chain(spot, ce, pe, iv, step=50, span=60, fail=False):
    """Flat grid centered on spot (always brackets 0.98*spot): every listed strike
    carries the same CE/PE/IV, so strike selection is pure geometry and marks read a
    controllable put premium."""
    if fail:
        raise RuntimeError("chain unavailable")

    def leg(ltp):
        return {"ltp": ltp, "open_interest": 100, "volume": 10,
                "greeks": {"iv": iv, "delta": -0.3}}
    center = round(spot / step) * step
    strikes = {str(center + i * step): {"CE": leg(ce), "PE": leg(pe)}
               for i in range(-span, span + 1)}
    return {"underlying_ltp": spot, "strikes": strikes}


def _canned_instruments():
    return pd.DataFrame([
        {"underlying_symbol": "NIFTY", "instrument_type": "CE", "lot_size": 50},
        {"underlying_symbol": "NIFTY", "instrument_type": "PE", "lot_size": 50},
    ])


class Spy:
    """Serves a canned chain; `pe` drives the put mark, `spot` the settlement spot."""

    def __init__(self):
        self.methods = []
        self.expiries = ["2026-07-14", "2026-07-21"]
        self.spot = 24000.0
        self.ce, self.pe, self.iv = 60.0, 100.0, 15.0
        self.fail_chain = False

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:
            raise PermissionError(method)
        if method == "get_expiries":
            return {"expiries": list(self.expiries)}
        if method == "get_option_chain":
            return _chain(self.spot, self.ce, self.pe, self.iv, fail=self.fail_chain)
        if method == "get_ltp":
            return {po.NIFTY_SPOT_SYM: self.spot}
        raise AssertionError(f"unexpected method: {method}")


def _patch(monkeypatch, spy, rv5d=11.0):
    monkeypatch.setattr(gc, "call", spy)
    monkeypatch.setattr(po, "groww_instruments", lambda refresh=False: _canned_instruments())
    monkeypatch.setattr(po, "realized_vol_5d", lambda refresh=False: rv5d)


# a held short put going into today: strike 24000, entry/last mark 100, lot 50
_HELD = {"expiry": "2026-07-14", "strike": 24000.0, "lot_size": 50,
         "entry_date": "2026-07-09", "ce_entry": None, "pe_entry": 100.0,
         "entry_credit": 100.0, "spot": 24500.0, "dte": 5, "ce_ltp": None,
         "pe_ltp": 100.0, "mark_value": 100.0, "atm_iv": 15.0, "action": "open",
         "daily_pnl": 0.0, "cumulative_pnl": 0.0, "settled": None,
         "strike_ratio": 0.98, "otm_pct": 2.0, "cash_secured_notional": 1200000.0}


def _seed(path, position):
    row = {"hypothesis_ref": pw.HYPOTHESIS, "book": pw.BOOK, **position}
    path.write_text(json.dumps(row) + "\n")


# ---- strike selection + entry -------------------------------------------------------

def test_strike_nearest_098_and_entry(tmp_path, monkeypatch):
    spy = Spy()
    spy.spot, spy.pe = 24000.0, 100.0    # 0.98*24000 = 23520 -> nearest listed 23500
    _patch(monkeypatch, spy)
    out = tmp_path / "pw.jsonl"          # empty ledger -> fresh open
    row = pw.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    assert row["action"] == "open" and row["strike"] == 23500.0
    assert row["ce_entry"] is None and row["ce_ltp"] is None
    assert row["pe_entry"] == 100.0 and row["entry_credit"] == 100.0
    assert row["mark_value"] == 100.0 and row["daily_pnl"] == 0.0
    assert row["cumulative_pnl"] == 0.0
    # 23500/24000 = 0.979167 ; (1-0.979167)*100 = 2.0833 ; 23500*50 = 1_175_000
    assert row["strike_ratio"] == 0.979167 and row["otm_pct"] == 2.0833
    assert row["cash_secured_notional"] == 1175000.0

    # SAFETY (RL-18): only read-only methods, none of them order methods.
    assert spy.methods and set(spy.methods) <= set(pw.READ_METHODS)
    assert not any(m in gc._ORDER_METHODS for m in spy.methods)


def test_strike_selection_rounds_to_nearer_grid_point(tmp_path, monkeypatch):
    spy = Spy()
    spy.spot = 20000.0                   # 0.98*20000 = 19600 -> exact listed strike
    _patch(monkeypatch, spy)
    out = tmp_path / "pw.jsonl"
    row = pw.snapshot(today=TODAY, path=str(out), write=True, verbose=False)
    assert row["strike"] == 19600.0


# ---- mark ---------------------------------------------------------------------------

def test_mark_pnl_and_carries_entry_constants(tmp_path, monkeypatch):
    spy = Spy()
    spy.pe = 80.0                        # mark 80 -> (100-80)*50 = +1000
    _patch(monkeypatch, spy)
    out = tmp_path / "pw.jsonl"
    _seed(out, _HELD)
    row = pw.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    assert row["action"] == "mark" and row["strike"] == 24000.0
    assert row["mark_value"] == 80.0 and row["ce_ltp"] is None and row["pe_ltp"] == 80.0
    assert row["daily_pnl"] == 1000.0 and row["cumulative_pnl"] == 1000.0
    # entry-time constants carried, not recomputed
    assert row["strike_ratio"] == 0.98 and row["otm_pct"] == 2.0
    assert row["cash_secured_notional"] == 1200000.0


# ---- settlement: European put, close_value = max(0, strike - settle_spot) -----------

def test_settle_itm_hand_computed(tmp_path, monkeypatch):
    spy = Spy()
    spy.spot = 23800.0                   # ITM: intrinsic = 24000-23800 = 200
    _patch(monkeypatch, spy)
    out = tmp_path / "pw.jsonl"
    _seed(out, _HELD)
    row = pw.snapshot(today=EXPIRY_DAY, path=str(out), write=True, verbose=False)

    s = row["settled"]
    assert row["action"] == "roll_settle" and s["basis"] == "intrinsic"
    assert s["settle_spot"] == 23800.0 and s["close_value"] == 200.0
    # P&L = credit*lot - intrinsic*lot = (100 - 200)*50 = -5000
    assert s["realized_pnl"] == (100 - 200) * 50 == -5000.0
    assert row["daily_pnl"] == -5000.0 and row["cumulative_pnl"] == -5000.0
    assert row["strike"] is not None           # rolled into the next weekly put


def test_settle_otm_expires_worthless(tmp_path, monkeypatch):
    spy = Spy()
    spy.spot = 24100.0                   # OTM: intrinsic = max(0, 24000-24100) = 0
    _patch(monkeypatch, spy)
    out = tmp_path / "pw.jsonl"
    _seed(out, _HELD)
    row = pw.snapshot(today=EXPIRY_DAY, path=str(out), write=True, verbose=False)

    s = row["settled"]
    assert s["close_value"] == 0.0                      # put expires worthless
    assert s["realized_pnl"] == 100 * 50 == 5000.0      # full credit kept
    assert row["daily_pnl"] == 5000.0 and row["cumulative_pnl"] == 5000.0


# ---- roll at DTE < 2 (before expiry): close at LTP, reopen --------------------------

def test_roll_at_dte_lt_2(tmp_path, monkeypatch):
    spy = Spy()
    spy.pe = 90.0                        # close at LTP 90 -> (100-90)*50 = +500
    _patch(monkeypatch, spy)
    out = tmp_path / "pw.jsonl"
    _seed(out, _HELD)                    # held expiry 07-14; ROLL_DAY 07-13 -> dte 1
    row = pw.snapshot(today=ROLL_DAY, path=str(out), write=True, verbose=False)

    s = row["settled"]
    assert row["action"] == "roll_close" and s["basis"] == "ltp"
    assert s["close_value"] == 90.0 and s["realized_pnl"] == 500.0
    assert row["daily_pnl"] == 500.0 and row["cumulative_pnl"] == 500.0
    assert row["expiry"] == "2026-07-21"        # rolled to the next weekly


# ---- ledger row schema --------------------------------------------------------------

def test_ledger_schema(tmp_path, monkeypatch):
    _patch(monkeypatch, Spy())
    out = tmp_path / "pw.jsonl"
    pw.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    rec = json.loads(out.read_text().splitlines()[-1])
    expected = {
        "hypothesis_ref", "book", "kind", "asof_ist", "run_date", "underlying",
        "mark_basis", "realized_vol_5d", "expiry", "strike", "lot_size",
        "entry_date", "ce_entry", "pe_entry", "entry_credit", "spot", "dte",
        "ce_ltp", "pe_ltp", "mark_value", "atm_iv", "strike_ratio", "otm_pct",
        "cash_secured_notional", "action", "daily_pnl", "cumulative_pnl",
        "settled", "note", "timestamp", "git_commit", "git_dirty",
    }
    assert expected <= set(rec)
    assert rec["book"] == "cash_secured_putwrite"
    assert rec["hypothesis_ref"] == "RL-2026-07-26-03"


# ---- determinism --------------------------------------------------------------------

def test_determinism(tmp_path, monkeypatch):
    _patch(monkeypatch, Spy())
    out = tmp_path / "pw.jsonl"
    _seed(out, _HELD)
    r1 = pw.snapshot(today=TODAY, path=str(out), write=False, verbose=False)
    r2 = pw.snapshot(today=TODAY, path=str(out), write=False, verbose=False)
    drop = {"asof_ist"}                  # wall-clock only; write=False adds no git/ts
    assert {k: v for k, v in r1.items() if k not in drop} == \
           {k: v for k, v in r2.items() if k not in drop}


# ---- data gap: mark fails -> carry the held put, never crash ------------------------

def test_carry_on_chain_failure(tmp_path, monkeypatch):
    spy = Spy()
    spy.fail_chain = True                # get_option_chain raises during the mark
    _patch(monkeypatch, spy)
    out = tmp_path / "pw.jsonl"
    _seed(out, _HELD)
    row = pw.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    assert row["action"] == "carry" and row["strike"] == 24000.0   # held, not dropped
    assert row["daily_pnl"] is None and row["cumulative_pnl"] == 0.0
    assert row["strike_ratio"] == 0.98                             # constants carried
    assert not any(m in gc._ORDER_METHODS for m in spy.methods)
