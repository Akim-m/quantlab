"""Tests for the RL-2026-07-26-01 dual-momentum rotation (DUAL-ROT) sleeve.

FORWARD-ONLY construction: no test-window backtest is exercised here. What matters is
that the frozen weights are deterministic, use only prior-day data, respect the
top-K / equal-weight / absolute-gate rules, and that the design-freeze selection reads
the TRAIN window ALONE. The live snapshot leg mirrors the trend sleeve: read-only and
forward_track-compatible. Fully synthetic prices and a Groww method-spy - no network.
"""

import json

import numpy as np
import pandas as pd
import pytest

from quantlab import dualrot
from quantlab import groww_client as gc
from quantlab import live_paper as lp
from quantlab.xasset_trend import ETFS

FK, FG = dualrot.FROZEN_TOP_K, dualrot.FROZEN_GATE


def _bidx(n, start="2009-01-01"):
    return pd.bdate_range(start=start, periods=n)


def _rw_panel(n, seed, start="2009-01-01"):
    """Random-walk-with-drift price panel for the five ETFs (distinct drifts)."""
    rng = np.random.default_rng(seed)
    idx = _bidx(n, start)
    drifts = [0.0004, 0.0003, 0.0002, 0.00015, 0.0001]
    cols = {s: 100.0 * np.exp(np.cumsum(rng.normal(drifts[j], 0.01, size=n)))
            for j, s in enumerate(ETFS)}
    return pd.DataFrame(cols, index=idx)


def _trending_panel(n, growth):
    """Deterministic exponential-growth panel: px_i = 100 * exp(growth_i * t)."""
    idx = _bidx(n)
    return pd.DataFrame({s: 100.0 * np.exp(growth[j] * np.arange(n))
                         for j, s in enumerate(ETFS)}, index=idx)


# ---- construction: determinism, causality, gating, freeze isolation ----

def test_weights_reproduce_deterministically():
    px = _rw_panel(600, seed=1)
    pd.testing.assert_frame_equal(dualrot.weights_history(px, FK, FG),
                                  dualrot.weights_history(px, FK, FG))
    pd.testing.assert_series_equal(dualrot.latest_weights(px), dualrot.latest_weights(px))


@pytest.mark.parametrize("g", dualrot.GATES)
@pytest.mark.parametrize("k", dualrot.TOP_KS)
def test_weights_use_prior_day_data_only(k, g):
    px = _rw_panel(600, seed=2)
    w = dualrot.weights_history(px, k, g)

    # Perturbing the LAST bar moves NO weight in the frame: weight[t] never reads px[t].
    px_last = px.copy()
    px_last.iloc[-1] *= 3.0
    pd.testing.assert_frame_equal(w, dualrot.weights_history(px_last, k, g))

    # Perturbing an interior bar t leaves every weight on/before t unchanged (data <= t-1).
    t = 400
    px_mid = px.copy()
    px_mid.iloc[t] *= 3.0
    pd.testing.assert_frame_equal(w.iloc[:t + 1],
                                  dualrot.weights_history(px_mid, k, g).iloc[:t + 1])


@pytest.mark.parametrize("g", dualrot.GATES)
@pytest.mark.parametrize("k", dualrot.TOP_KS)
def test_all_declining_is_all_cash(k, g):
    # Monotone decline: no sleeve passes either absolute gate, so every top-K pick is
    # gated out and the whole book is cash - the crash-protection property.
    idx = _bidx(500)
    decay = [0.999, 0.9985, 0.998, 0.9975, 0.997]
    px = pd.DataFrame({s: 100.0 * np.power(decay[j], np.arange(500))
                       for j, s in enumerate(ETFS)}, index=idx)
    assert (dualrot.weights_history(px, k, g) == 0.0).all().all()


def test_top_k_selection_and_equal_weight():
    px = _trending_panel(500, [0.0010, 0.0008, 0.0006, 0.0004, 0.0002])  # NIFTYBEES fastest
    w2 = dualrot.weights_history(px, 2, "tsmom").iloc[-1]
    assert set(w2[w2 > 0].index) == {"NIFTYBEES.NS", "JUNIORBEES.NS"}   # top-2 by momentum
    assert w2[w2 > 0].tolist() == pytest.approx([0.5, 0.5])             # equal 1/K
    assert w2.sum() == pytest.approx(1.0)

    w1 = dualrot.weights_history(px, 1, "tsmom").iloc[-1]
    assert w1[w1 > 0].index.tolist() == ["NIFTYBEES.NS"] and w1.max() == pytest.approx(1.0)


@pytest.mark.parametrize("g", dualrot.GATES)
@pytest.mark.parametrize("k", dualrot.TOP_KS)
def test_weights_long_only_bounded_and_quantized(k, g):
    w = dualrot.weights_history(_rw_panel(800, seed=3), k, g)
    assert (w >= 0.0).all().all()
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()                 # sum <= 1 (cash is the rest)
    assert set(np.round(np.unique(w.values), 6)) <= {0.0, round(1.0 / k, 6)}


def test_pre_inception_column_never_held_while_nan():
    # MON100 lists 2011-03: while its price is NaN it must never carry weight.
    px = _rw_panel(800, seed=4)
    cut = 300
    px.loc[px.index[:cut], "MON100.NS"] = np.nan
    w = dualrot.weights_history(px, 2, "tsmom")
    assert (w["MON100.NS"].iloc[:cut] == 0.0).all()


def test_frozen_selection_reads_train_window_only():
    # The design freeze cannot see post-2016 data: shocking every post-TRAIN1 bar leaves
    # all four TRAIN Sharpes bit-identical (the "never touch the hold-out" guarantee).
    px = _rw_panel(3200, seed=7)
    base = dualrot.train_scores(px)
    px2 = px.copy()
    post = px2.index > pd.Timestamp(dualrot.TRAIN1)
    px2.loc[post] = px2.loc[post] * 5.0
    pd.testing.assert_frame_equal(base, dualrot.train_scores(px2))


def test_frozen_constants_are_a_registered_variant():
    assert FK in dualrot.TOP_KS and FG in dualrot.GATES


# ---- live snapshot leg: read-only, forward_track-compatible (mirrors run_trend) ----

class Spy:
    """Records every method name routed through groww_client.call; serves canned LTP."""

    PRICES = {"NSE_NIFTYBEES": 110.0, "NSE_GOLDBEES": 66.0, "NSE_JUNIORBEES": 55.0}

    def __init__(self):
        self.methods = []

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:            # the spy honors the real guard
            raise PermissionError(method)
        syms = kwargs.get("exchange_trading_symbols", ())
        return {s: self.PRICES[s] for s in syms if s in self.PRICES}


def _synthetic_dualrot():
    book = lp.Book(
        weights=pd.Series({"NIFTYBEES.NS": 0.5, "GOLDBEES.NS": 0.5}),   # K=2, both held, gross 1.0
        regime_on=True, cash_frac=0.0,
        latest_date=pd.Timestamp("2026-07-08"),
        prev_close=pd.Series({"NIFTYBEES.NS": 100.0, "GOLDBEES.NS": 60.0}),
        nsei_prev_close=float("nan"),
    )
    gates = {"NIFTYBEES.NS": True, "JUNIORBEES.NS": False, "BANKBEES.NS": False,
             "GOLDBEES.NS": True, "MON100.NS": False}
    return book, gates


def test_run_dualrot_read_only_schema_and_pnl(tmp_path, monkeypatch):
    spy = Spy()
    monkeypatch.setattr(gc, "call", spy)
    monkeypatch.setattr(lp, "current_dualrot_book", lambda **kw: _synthetic_dualrot())

    out = tmp_path / "paper_trades_dualrot.jsonl"
    row = lp.run_dualrot(path=str(out), write=True)

    # SAFETY: only read-only methods dispatched, none of them order methods.
    assert spy.methods and set(spy.methods) <= set(lp.READ_METHODS)
    assert not any(m in gc._ORDER_METHODS for m in spy.methods)

    assert sum(row["weights"].values()) == pytest.approx(1.0)   # long-only gross <= 1
    assert row["weights"]["NIFTYBEES.NS"] == pytest.approx(0.5)
    assert row["kind"] == "live_paper_dualrot_snapshot"
    assert row["hypothesis_ref"] == "RL-2026-07-26-01"
    assert row["top_k"] == FK and row["abs_gate"] == FG        # frozen variant recorded
    # book_ret = 0.5*(110/100-1) + 0.5*(66/60-1) = 0.10
    assert row["book_intraday_ret"] == pytest.approx(0.10)
    assert row["asset_intraday"]["NIFTYBEES.NS"] == pytest.approx(0.10)
    assert row["gate_states"]["BANKBEES.NS"] is False
    assert row["gate_states"]["NIFTYBEES.NS"] is True
    assert row["n_quotes_ok"] == 2 and row["n_names"] == 2

    rec = json.loads(out.read_text().strip().splitlines()[0])
    assert "panel_date" in rec and "weights" in rec            # forward_track-compatible


def test_run_dualrot_records_book_only_when_quotes_unavailable(tmp_path, monkeypatch):
    def boom(method, *a, **k):
        raise RuntimeError("no entitlement")
    monkeypatch.setattr(gc, "call", boom)
    monkeypatch.setattr(lp, "current_dualrot_book", lambda **kw: _synthetic_dualrot())

    out = tmp_path / "d.jsonl"
    row = lp.run_dualrot(path=str(out), write=True)
    assert row["book_intraday_ret"] is None        # no live P&L invented
    assert row["asset_intraday"] == {}
    assert row["n_quotes_ok"] == 0 and row["groww_ok"] is False
    assert row["weights"] and row["gate_states"]   # book + gates still recorded
    assert out.read_text().strip()                 # snapshot still written


def test_current_dualrot_book_shape_cash_and_gates(monkeypatch):
    px = _trending_panel(500, [0.0010, 0.0008, 0.0006, 0.0004, 0.0002])
    monkeypatch.setattr(lp, "etf_panel", lambda refresh=False: px)
    monkeypatch.setattr(lp, "load_yahoo_ohlcv", lambda syms, refresh=False: {})
    monkeypatch.setattr(lp, "close_prices", lambda data, field="close": px)

    book, gates = lp.current_dualrot_book()

    assert sorted(gates) == sorted(ETFS)
    assert (book.weights > 0).all() and book.weights.sum() <= 1.0 + 1e-9   # long-only, sum <= 1
    assert set(np.round(book.weights.values, 6)) <= {round(1.0 / FK, 6)}
    assert book.cash_frac == pytest.approx(1.0 - book.weights.sum())       # cash fraction correct
    assert set(book.weights.index) == {"NIFTYBEES.NS", "JUNIORBEES.NS"}    # top-2 by momentum held
    for s in ("BANKBEES.NS", "GOLDBEES.NS", "MON100.NS"):
        assert gates[s] is False and s not in book.weights.index          # ungated leg -> cash
    for s in ("NIFTYBEES.NS", "JUNIORBEES.NS"):
        assert gates[s] is True
