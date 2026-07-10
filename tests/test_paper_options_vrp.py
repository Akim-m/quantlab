"""Tests for the forward-only VRP-GATED PAPER short-straddle harness. No network.

Covers the one thing this book adds over RL-18 - the variance-risk-premium gate:
  - gate ON (VRP > trailing median) opens / holds; gate OFF stays flat / flattens;
  - warm-up: zero prior history defaults the gate OFF but still logs VRP + decision;
  - no-look-ahead: the median uses ONLY prior rows, never today's own VRP;
  - the ledger row schema (RL-18 block + VRP/gate fields);
  - determinism: identical ledger + inputs -> identical gate/mark row;
  - the load-bearing safety property (RL-18): only read-only Groww methods dispatched.
Position/mark arithmetic itself is reused from `paper_options` and pinned there.
"""

import json
from datetime import date

import pandas as pd

from quantlab import groww_client as gc
from quantlab import paper_options as po
from quantlab import paper_options_vrp as pov

TODAY = date(2026, 7, 10)


def _chain(spot, ce, pe, iv, center=20000, step=50):
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
    ])


class Spy:
    """Serves a canned chain; `iv` drives the gate's ATM_IV, `ce`/`pe` the legs."""

    def __init__(self):
        self.methods = []
        self.expiries = ["2026-07-14", "2026-07-21"]
        self.spot = 20000.0
        self.ce, self.pe, self.iv = 100.0, 110.0, 20.0
        self.center = 20000

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:
            raise PermissionError(method)
        if method == "get_expiries":
            return {"expiries": list(self.expiries)}
        if method == "get_option_chain":
            return _chain(self.spot, self.ce, self.pe, self.iv, center=self.center)
        if method == "get_ltp":
            return {po.NIFTY_SPOT_SYM: self.spot}
        raise AssertionError(f"unexpected method: {method}")


def _patch(monkeypatch, spy, rv5d=11.0):
    monkeypatch.setattr(gc, "call", spy)
    monkeypatch.setattr(po, "groww_instruments", lambda refresh=False: _canned_instruments())
    monkeypatch.setattr(po, "realized_vol_5d", lambda refresh=False: rv5d)


def _seed(path, vrps, position=None):
    """Write prior ledger rows carrying the given VRPs (chronological). If `position`
    is given it is merged into the LAST row so has_pos is True going into today."""
    rows = []
    for i, v in enumerate(vrps):
        row = {"hypothesis_ref": "RL-2026-07-26-06", "book": pov.BOOK, "vrp": v,
               "strike": None, "cumulative_pnl": 0.0, "run_date": f"2026-06-{i + 1:02d}"}
        rows.append(row)
    if position and rows:
        rows[-1].update(position)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


_HELD = {"expiry": "2026-07-14", "strike": 20000.0, "lot_size": 50, "entry_date": "2026-07-09",
         "ce_entry": 100.0, "pe_entry": 110.0, "entry_credit": 210.0, "mark_value": 210.0,
         "cumulative_pnl": 0.0}


# ---- gate ON: VRP above trailing median -> open when flat ---------------------------

def test_gate_on_opens_when_flat(tmp_path, monkeypatch):
    spy = Spy()
    spy.iv = 20.0                        # VRP = 20 - 11 = 9 > median(5,6,7)=6
    _patch(monkeypatch, spy)
    out = tmp_path / "vrp.jsonl"
    _seed(out, [5.0, 6.0, 7.0])
    row = pov.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    assert row["gate_on"] is True and row["action"] == "open"
    assert row["vrp"] == 9.0 and row["vrp_median"] == 6.0 and row["n_vrp_hist"] == 3
    assert row["strike"] == 20000.0 and row["entry_credit"] == 210.0
    assert row["atm_iv_gate"] == 20.0 and row["realized_vol_5d"] == 11.0

    # SAFETY (RL-18): only read-only methods, none of them order methods.
    assert spy.methods and set(spy.methods) <= set(pov.READ_METHODS)
    assert not any(m in gc._ORDER_METHODS for m in spy.methods)


def test_gate_on_holds_and_marks_existing(tmp_path, monkeypatch):
    spy = Spy()
    spy.iv = 20.0                        # VRP = 9 > median(5,5)=5 -> gate on
    spy.ce, spy.pe = 90.0, 100.0         # mark 190 -> (210-190)*50 = +1000
    _patch(monkeypatch, spy)
    out = tmp_path / "vrp.jsonl"
    _seed(out, [5.0, 5.0], position=_HELD)
    row = pov.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    assert row["gate_on"] is True and row["action"] == "mark"
    assert row["mark_value"] == 190.0 and row["daily_pnl"] == 1000.0
    assert row["cumulative_pnl"] == 1000.0


# ---- gate OFF: VRP at/below trailing median ----------------------------------------

def test_gate_off_stays_flat(tmp_path, monkeypatch):
    spy = Spy()
    spy.iv = 14.0                        # VRP = 14 - 11 = 3 < median(5,6,7)=6 -> off
    _patch(monkeypatch, spy)
    out = tmp_path / "vrp.jsonl"
    _seed(out, [5.0, 6.0, 7.0])
    row = pov.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    assert row["gate_on"] is False and row["action"] == "flat"
    assert row["vrp"] == 3.0 and row["strike"] is None
    assert row["daily_pnl"] == 0.0 and row["cumulative_pnl"] == 0.0


def test_gate_off_flattens_open_position(tmp_path, monkeypatch):
    spy = Spy()
    spy.iv = 14.0                        # VRP = 3 < median(8,8)=8 -> gate off
    spy.ce, spy.pe = 50.0, 40.0          # flatten at 90 -> (210-90)*50 = +6000
    _patch(monkeypatch, spy)
    out = tmp_path / "vrp.jsonl"
    _seed(out, [8.0, 8.0], position=_HELD)
    row = pov.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    assert row["gate_on"] is False and row["action"] == "gate_close"
    assert row["strike"] is None                          # position flattened
    assert row["settled"]["basis"] == "ltp" and row["settled"]["close_value"] == 90.0
    assert row["daily_pnl"] == 6000.0 and row["cumulative_pnl"] == 6000.0


# ---- warm-up: zero prior history defaults gate OFF, still logs VRP ------------------

def test_warmup_zero_history_defaults_off(tmp_path, monkeypatch):
    spy = Spy()
    spy.iv = 20.0                        # VRP = 9, but no prior -> median undefined
    _patch(monkeypatch, spy)
    out = tmp_path / "vrp.jsonl"         # no file yet: empty history
    row = pov.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    assert row["n_vrp_hist"] == 0 and row["vrp_median"] is None
    assert row["gate_on"] is False and row["warmup"] is True
    assert row["vrp"] == 9.0 and row["action"] == "flat" and row["strike"] is None


def test_warmup_single_prior_uses_that_value(tmp_path, monkeypatch):
    spy = Spy()
    spy.iv = 20.0                        # VRP = 9 > median([4]) = 4 -> gate on
    _patch(monkeypatch, spy)
    out = tmp_path / "vrp.jsonl"
    _seed(out, [4.0])
    row = pov.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    assert row["n_vrp_hist"] == 1 and row["vrp_median"] == 4.0
    assert row["gate_on"] is True and row["warmup"] is True and row["action"] == "open"


# ---- no-look-ahead: the median excludes today's own VRP ----------------------------

def test_no_lookahead_median_uses_prior_only(tmp_path, monkeypatch):
    spy = Spy()
    spy.iv = 111.0                       # VRP = 100; if it leaked in, median would shift
    _patch(monkeypatch, spy)
    out = tmp_path / "vrp.jsonl"
    _seed(out, [1.0, 2.0])               # prior-only median = 1.5; with today = 2.0
    row = pov.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    assert row["vrp"] == 100.0
    assert row["vrp_median"] == 1.5 and row["n_vrp_hist"] == 2   # today NOT included


# ---- ledger row schema --------------------------------------------------------------

def test_ledger_schema(tmp_path, monkeypatch):
    _patch(monkeypatch, Spy())
    out = tmp_path / "vrp.jsonl"
    _seed(out, [5.0, 6.0, 7.0])
    pov.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    rec = json.loads(out.read_text().splitlines()[-1])
    expected = {
        "hypothesis_ref", "book", "kind", "asof_ist", "run_date", "underlying",
        "mark_basis", "realized_vol_5d", "atm_iv_gate", "vrp", "vrp_median",
        "n_vrp_hist", "gate_on", "warmup", "expiry", "strike", "lot_size",
        "entry_date", "ce_entry", "pe_entry", "entry_credit", "spot", "dte",
        "ce_ltp", "pe_ltp", "mark_value", "atm_iv", "action", "daily_pnl",
        "cumulative_pnl", "settled", "note", "timestamp", "git_commit", "git_dirty",
    }
    assert expected <= set(rec)
    assert rec["book"] == "vrp_gated_straddle"
    assert rec["hypothesis_ref"] == "RL-2026-07-26-06"


# ---- determinism: same ledger + inputs -> same gate/mark row -----------------------

def test_determinism(tmp_path, monkeypatch):
    _patch(monkeypatch, Spy())
    out = tmp_path / "vrp.jsonl"
    _seed(out, [5.0, 6.0, 7.0])
    r1 = pov.snapshot(today=TODAY, path=str(out), write=False, verbose=False)
    r2 = pov.snapshot(today=TODAY, path=str(out), write=False, verbose=False)
    drop = {"asof_ist"}                  # wall-clock only; write=False adds no git/ts
    assert {k: v for k, v in r1.items() if k not in drop} == \
           {k: v for k, v in r2.items() if k not in drop}


# ---- data gap: gate undecidable carries a held position, never crashes -------------

def test_vrp_gap_carries_position(tmp_path, monkeypatch):
    spy = Spy()
    _patch(monkeypatch, spy, rv5d=None)  # no realized vol -> VRP undecidable
    out = tmp_path / "vrp.jsonl"
    _seed(out, [5.0, 5.0], position=_HELD)
    row = pov.snapshot(today=TODAY, path=str(out), write=True, verbose=False)

    assert row["vrp"] is None and row["gate_on"] is None
    assert row["action"] == "carry" and row["strike"] == 20000.0   # held, not flattened
