"""Tests for the NIFTY IV term-structure extension (RL-2026-07-26-07).

The collector now logs the next-monthly ('far') ATM IV alongside the near ATM IV
and their slope. These tests pin four things offline (no network, no auth):
  1. the far-expiry SELECTION rule on synthetic expiry lists (weekly vs monthly),
  2. the slope arithmetic (iv_slope = atm_iv_far - atm_iv) wired end-to-end,
  3. FAILURE ISOLATION - a far-chain fetch that raises still writes the row with
     atm_iv_far/iv_slope None and an intact near leg,
  4. SCHEMA BACKWARD-COMPAT - every pre-existing field keeps its value and only new
     fields are added.
"""

import json
from datetime import date

import pandas as pd
import pytest

from quantlab import fno_collect as fc
from quantlab import groww_client as gc

TODAY = date(2026, 7, 10)


# ---- far-expiry selection rule: weekly vs monthly discrimination -----------------

def test_far_monthly_picks_this_months_monthly_when_near_is_weekly():
    # Weeklies 07-16, monthly 07-30, next monthly 08-27. Near = 07-16 weekly ->
    # far is July's LAST expiry (07-30, the monthly), not the next weekly.
    exp = ["2026-07-16", "2026-07-23", "2026-07-30", "2026-08-06", "2026-08-27"]
    assert fc.far_monthly_expiry(exp, "2026-07-16") == "2026-07-30"


def test_far_monthly_rolls_to_next_month_when_near_is_the_monthly():
    # Near IS July's monthly (07-30) -> far must be August's monthly (08-27),
    # skipping the intervening August weeklies.
    exp = ["2026-07-30", "2026-08-06", "2026-08-13", "2026-08-27", "2026-09-24"]
    assert fc.far_monthly_expiry(exp, "2026-07-30") == "2026-08-27"


def test_far_monthly_skips_a_weekly_that_outnumbers_the_monthly():
    # Only two expiries listed in August (06, 27) -> 08-27 is August's monthly.
    # A lone far expiry in a month is still that month's 'last', hence its monthly.
    exp = ["2026-07-16", "2026-07-30", "2026-08-27"]
    assert fc.far_monthly_expiry(exp, "2026-07-16") == "2026-07-30"
    assert fc.far_monthly_expiry(exp, "2026-07-30") == "2026-08-27"


def test_far_monthly_none_when_no_monthly_beyond_near():
    assert fc.far_monthly_expiry(["2026-07-30"], "2026-07-30") is None   # near only
    assert fc.far_monthly_expiry([], "2026-07-30") is None               # empty list
    assert fc.far_monthly_expiry(["2026-07-16"], "bad-date") is None     # unparseable near


def test_monthly_expiries_one_per_month():
    dates = [date(2026, 7, 16), date(2026, 7, 30), date(2026, 8, 6), date(2026, 8, 27)]
    assert fc._monthly_expiries(dates) == [date(2026, 7, 30), date(2026, 8, 27)]


# ---- synthetic chain + spy -------------------------------------------------------

def _chain(atm_iv: float) -> dict:
    """spot 100, ATM strike 100. CE.iv == PE.iv == atm_iv at the money, so the
    extracted ATM IV equals `atm_iv` exactly (mean of the two)."""
    def leg(oi, iv):
        return {"open_interest": oi, "greeks": {"iv": iv, "delta": 0.5}}
    return {
        "underlying_ltp": 100.0,
        "strikes": {
            "95":  {"CE": leg(10, atm_iv + 3), "PE": leg(30, atm_iv + 11)},
            "100": {"CE": leg(20, atm_iv), "PE": leg(20, atm_iv)},
            "105": {"CE": leg(30, atm_iv - 2), "PE": leg(10, atm_iv + 2)},
        },
    }


class ChainSpy:
    """Serves canned expiries + per-expiry chains; records dispatched methods.
    Near expiry (07-16) IV differs from far (07-30) IV so the slope is non-trivial."""

    EXPIRIES = ["2026-06-30", "2026-07-09", "2026-07-16", "2026-07-30", "2026-08-27"]
    NEAR = "2026-07-16"
    FAR = "2026-07-30"

    def __init__(self, fail_far=False, near_iv=15.0, far_iv=18.0):
        self.methods = []
        self.chain_expiries = []
        self.fail_far = fail_far
        self.near_iv = near_iv
        self.far_iv = far_iv

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:
            raise PermissionError(method)
        if method == "get_ltp":
            seg = kwargs["segment"]
            table = {"NSE_AAA": 100.0} if seg == "CASH" else {"NSE_AAA26JULFUT": 101.0}
            return {s: table[s] for s in kwargs["exchange_trading_symbols"] if s in table}
        if method == "get_expiries":
            return {"expiries": self.EXPIRIES}
        if method == "get_option_chain":
            exp = kwargs["expiry_date"]
            self.chain_expiries.append(exp)
            if exp == self.FAR:
                if self.fail_far:
                    raise RuntimeError("far chain 500")
                return _chain(self.far_iv)
            return _chain(self.near_iv)
        raise AssertionError(f"unexpected method: {method}")


# ---- slope arithmetic + failure isolation on nifty_chain_block -------------------

def test_iv_slope_arithmetic_end_to_end(monkeypatch):
    spy = ChainSpy(near_iv=15.0, far_iv=18.0)
    monkeypatch.setattr(gc, "call", spy)

    m, ok, note = fc.nifty_chain_block(TODAY)

    assert m["expiry"] == "2026-07-16" and m["far_expiry"] == "2026-07-30"
    assert m["atm_iv"] == pytest.approx(15.0) and m["atm_iv_far"] == pytest.approx(18.0)
    assert m["iv_slope"] == pytest.approx(3.0)      # 18 - 15
    assert ok is True and note is None
    # exactly ONE extra chain fetch (near + far), both read-only.
    assert spy.chain_expiries == ["2026-07-16", "2026-07-30"]
    assert set(spy.methods) <= set(fc.READ_METHODS)


def test_inverted_term_structure_gives_negative_slope(monkeypatch):
    # Stress shape: near IV above far IV -> slope < 0 (the signal RL-26-07 gates on).
    monkeypatch.setattr(gc, "call", ChainSpy(near_iv=28.0, far_iv=20.0))
    m, ok, note = fc.nifty_chain_block(TODAY)
    assert m["iv_slope"] == pytest.approx(-8.0) and ok is True


def test_far_fetch_failure_isolated_near_leg_intact(monkeypatch):
    spy = ChainSpy(fail_far=True)
    monkeypatch.setattr(gc, "call", spy)

    m, ok, note = fc.nifty_chain_block(TODAY)

    # near leg fully intact and chain_ok unaffected by the far failure
    assert ok is True
    assert m["expiry"] == "2026-07-16" and m["pcr"] is not None
    assert m["atm_iv"] == pytest.approx(15.0)
    # far leg degraded, not crashed: expiry still resolved, IV/slope null, note set
    assert m["far_expiry"] == "2026-07-30"
    assert m["atm_iv_far"] is None and m["iv_slope"] is None
    assert note is not None and "far" in note


def test_no_far_monthly_notes_but_does_not_crash(monkeypatch):
    class OneExpiry(ChainSpy):
        EXPIRIES = ["2026-07-16"]        # near only, no monthly beyond it
    monkeypatch.setattr(gc, "call", OneExpiry())
    m, ok, note = fc.nifty_chain_block(TODAY)
    assert m["far_expiry"] is None and m["atm_iv_far"] is None and m["iv_slope"] is None
    assert ok is True and note == "no_far_monthly"


# ---- full collect(): new fields present + old schema byte-compatible -------------

def _canned_instruments():
    return pd.DataFrame([
        {"instrument_type": "FUT", "segment": "FNO", "exchange": "NSE",
         "underlying_symbol": "AAA", "trading_symbol": "AAA26JULFUT",
         "expiry_date": "2026-07-30"},
    ])


def _patch_universe(monkeypatch):
    monkeypatch.setattr(fc, "groww_instruments", lambda refresh=False: _canned_instruments())
    monkeypatch.setattr(fc, "fno_shortable", lambda instruments: {"AAA.NS"})


def test_collect_row_new_fields_and_old_schema_unchanged(tmp_path, monkeypatch):
    spy = ChainSpy(near_iv=15.0, far_iv=18.0)
    monkeypatch.setattr(gc, "call", spy)
    _patch_universe(monkeypatch)

    out = tmp_path / "fno_daily.jsonl"
    row = fc.collect(today=TODAY, path=str(out), write=True, verbose=False)

    # NEW fields present with expected values
    assert row["far_expiry"] == "2026-07-30"
    assert row["atm_iv_far"] == pytest.approx(18.0)
    assert row["iv_slope"] == pytest.approx(3.0)

    # OLD fields unchanged: same identity/meaning as before the extension
    assert row["hypothesis_ref"] == "RL-2026-07-15"
    assert row["kind"] == "fno_daily_snapshot"
    assert row["collect_date"] == "2026-07-10"
    assert row["nifty_expiry"] == "2026-07-16" and row["nifty_spot"] == 100.0
    assert row["chain_ok"] is True
    assert row["atm_iv"] == pytest.approx(15.0)
    assert row["n_underlyings"] == 1 and row["n_cash_ok"] == 1 and row["n_fut1_ok"] == 1
    assert row["note"] == "ok"

    # persisted, valid JSON, carries both old and new keys
    rec = json.loads(out.read_text().strip())
    for k in ("timestamp", "hypothesis_ref", "collect_date", "n_underlyings",
              "chain_ok", "pcr", "atm_iv", "skew", "atm_strike", "basis"):
        assert k in rec                                  # backward-compat keys
    for k in ("far_expiry", "atm_iv_far", "iv_slope"):
        assert k in rec                                  # new keys


def test_collect_writes_row_even_when_far_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "call", ChainSpy(fail_far=True))
    _patch_universe(monkeypatch)

    out = tmp_path / "fno_daily.jsonl"
    row = fc.collect(today=TODAY, path=str(out), write=True, verbose=False)

    assert row["atm_iv_far"] is None and row["iv_slope"] is None
    assert row["far_expiry"] == "2026-07-30"        # expiry resolved, only the fetch failed
    assert row["atm_iv"] == pytest.approx(15.0)     # near leg still logged
    assert "far" in row["note"]
    assert out.read_text().strip()                  # row still persisted
