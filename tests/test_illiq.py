"""Tests for the RL-2026-07-26-19 Amihud illiquidity (ILLIQ) sleeve.

Two arms: (A) the TRAIN-window design study and (B) the frozen live paper book. What
matters: illiq = |ret| / rupee-turnover with zero/missing-volume days masked; a name with
< 60% valid days inside the lookback is excluded that day; the +/-3 MAD winsorization
fires; the decile L/S book LONGS the most-illiquid decile, is dollar-neutral + unit-gross
with equal-size legs; the weights read only prior-day data (the one-day lag); the study
physically refuses to read past 2016-12-31; and the frozen book's monthly turnover is well
below 100%/mo (illiquidity is persistent). The live snapshot leg mirrors run_volshock:
read-only, signed, forward_track-compatible. Fully synthetic - no network.
"""

import json

import numpy as np
import pandas as pd
import pytest

from quantlab import groww_client as gc
from quantlab import illiq as il
from quantlab import live_paper as lp


def _bidx(n, start="2013-01-02"):
    return pd.bdate_range(start=start, periods=n)


def _ipanel(n=200, m=30, start="2013-01-02", base=100.0, vol=1e5, amin=0.004, amax=0.05):
    """m names on a 100-price grid whose daily return alternates +/-a_i (so |ret| == a_i,
    a clean per-name illiquidity level rising monotonically across names) on constant
    volume. Returns (close, volume); tests pass px == close (no dividends)."""
    idx = _bidx(n, start)
    cols = [f"S{i:02d}" for i in range(m)]
    a = np.linspace(amin, amax, m)
    s = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)[:, None]
    price = base * np.cumprod(1.0 + s * a, axis=0)
    return (pd.DataFrame(price, index=idx, columns=cols),
            pd.DataFrame(vol, index=idx, columns=cols))


# ---- daily illiq: |ret| / rupee turnover, zero/missing volume masked ----

def test_zero_or_missing_volume_masked_to_nan():
    close, vol = _ipanel()
    vol = vol.copy()
    vol.iloc[-1, 0] = 0.0                                       # S00 did not trade last day
    vol.iloc[-1, 1] = np.nan                                    # S01 has no volume last day

    to = il.turnover(close, vol).iloc[-1]
    assert np.isnan(to.iloc[0]) and np.isnan(to.iloc[1])        # no rupee turnover recorded
    di = il.daily_illiq(close, close, vol).iloc[-1]
    assert np.isnan(di.iloc[0]) and np.isnan(di.iloc[1])        # that day is invalid
    assert np.isfinite(di.iloc[2:]).all()                      # every name that traded is kept
    # one bad day out of 63 is still >= 60% valid -> the name is still scored (mean skips it)
    assert np.isfinite(il.illiq(close, close, vol, lookback=63).iloc[-1, 0])


# ---- < 60% valid days inside the lookback -> excluded that day ----

def test_below_60pct_valid_days_excluded():
    close, vol = _ipanel(n=200)
    vol = vol.copy()
    vol.iloc[-30:, 0] = 0.0                                     # S00: 30 of last 63 days untraded
    sig = il.illiq(close, close, vol, lookback=63).iloc[-1]     # -> 33/63 = 52% valid < 60%
    assert np.isnan(sig.iloc[0])                               # excluded
    assert np.isfinite(sig.iloc[5])                            # a fully-traded name is scored


# ---- +/-3 MAD cross-sectional winsorization ----

def test_winsorize_clips_at_three_mad():
    row = pd.DataFrame([[0.0, 1.0, 2.0, 3.0, 4.0, 100.0]])      # median 2.5, MAD 1.5
    out = il.winsorize(row, n=3.0)
    assert out.iloc[0, -1] == pytest.approx(2.5 + 3.0 * 1.5)    # the outlier clipped in to 7.0
    assert (out.iloc[0, :5].to_numpy() == row.iloc[0, :5].to_numpy()).all()  # inliers untouched


# ---- decile L/S: dollar-neutral, unit-gross, equal legs, longs the most illiquid ----

def test_decile_dollar_neutral_unit_gross():
    close, vol = _ipanel()
    w = il.weights(close, close, vol, lookback=63, rebalance=None)
    net = w.sum(axis=1).to_numpy()
    gross = w.abs().sum(axis=1).to_numpy()
    active = gross > 0
    assert active[-1]                                          # warmed up by the end
    np.testing.assert_allclose(net[active], 0.0, atol=1e-9)    # dollar-neutral
    np.testing.assert_allclose(gross[active], 1.0, atol=1e-9)  # unit gross
    assert (gross[~active] == 0.0).all()


def test_decile_leg_sizes_equal():
    close, vol = _ipanel(m=30)                                  # floor(30 * 0.1) = 3 per leg
    w = il.weights(close, close, vol, lookback=63, rebalance=None).iloc[-1]
    assert int((w > 0).sum()) == 3 and int((w < 0).sum()) == 3


def test_longs_most_illiquid_shorts_most_liquid():
    close, vol = _ipanel()
    sig = il.winsorize(il.illiq(close, close, vol, lookback=63)).iloc[-2]  # feeds the last weight
    w = il.weights(close, close, vol, lookback=63, rebalance=None).iloc[-1]
    longs, shorts = w.index[w > 0], w.index[w < 0]
    assert sig[longs].min() > sig[shorts].max()               # every long out-illiquids every short
    assert w[longs].nunique() == 1 and w[shorts].nunique() == 1  # equal-weight within each leg


def test_monthly_held_book_is_neutral_and_deterministic():
    close, vol = _ipanel()
    w = il.weights(close, close, vol, lookback=63)             # default ME rebalance (sparse grid)
    assert w.notna().any(axis=1).sum() < len(w)                # trades only on the ME grid, not daily
    last = il.latest_weights(close, close, vol, lookback=63)   # currently-held decile (ffilled)
    assert last.abs().sum() == pytest.approx(1.0, abs=1e-9)    # gross 1
    assert last.sum() == pytest.approx(0.0, abs=1e-9)          # net 0
    pd.testing.assert_frame_equal(w, il.weights(close, close, vol, lookback=63))  # deterministic


# ---- no look-ahead: date-t weight reads only data through t-1 (fails if the lag is removed) ----

def test_weights_use_prior_day_data_only():
    close, vol = _ipanel(n=400)

    w = il.weights(close, close, vol, lookback=63, rebalance=None)
    # Perturbing the LAST bar moves NO weight: weight[t] never reads (px,close,volume)[t].
    c2, v2 = close.copy(), vol.copy()
    c2.iloc[-1] *= 1.5
    v2.iloc[-1] *= 5.0
    pd.testing.assert_frame_equal(w, il.weights(c2, c2, v2, lookback=63, rebalance=None))

    # A price jump at interior bar t spikes S00's illiquidity FROM t onward - but weight[t]
    # (which reads only data <= t-1) must not move; the change bites only from t+1.
    t = 300
    c3 = close.copy()
    c3.iloc[t, 0] *= 2.0
    pert = il.weights(c3, c3, vol, lookback=63, rebalance=None)
    pd.testing.assert_frame_equal(w.iloc[:t + 1], pert.iloc[:t + 1])   # weights <= t unchanged
    assert not w.iloc[t + 1:].equals(pert.iloc[t + 1:])               # spike bites only from t+1


def test_latest_signal_is_last_row_cross_section():
    close, vol = _ipanel()
    sig = il.latest_signal(close, close, vol, lookback=63)
    expected = il.winsorize(il.illiq(close, close, vol, lookback=63)).iloc[-1]
    pd.testing.assert_series_equal(sig, expected)
    assert sig.notna().any()                                  # a real cross-section for the corr check


# ---- (A) TRAIN study: physically refuses to read past 2016-12-31 ----

def test_train_study_clips_at_hold_out_boundary():
    close, vol = _ipanel(n=880, m=30, start="2015-01-02")      # spans well past 2016-12-31
    assert close.index[-1] > pd.Timestamp(il.TRAIN_END)        # the raw panel reaches into the hold-out
    table, rets = il.train_study(close, close, vol)
    assert len(table) == len(il.LOOKBACKS) * len(il.COST_ARMS)  # 2 variants x 3 costs = 6 rows
    for series in rets.values():
        assert series.index[-1] <= pd.Timestamp(il.TRAIN_END)  # no return computed past the boundary
        assert len(series) > 252                               # a real TRAIN window, not empty


def test_train_study_turnover_persistence():
    """Frozen L63 book on a persistent-illiquidity panel: one-sided monthly turnover well
    below 100%/mo (the registration's persistence prediction)."""
    close, vol = _ipanel(n=1500, m=40, start="2010-01-04")
    px = close.loc[il.TRAIN_START:il.TRAIN_END]
    c, v = close.loc[il.TRAIN_START:il.TRAIN_END], vol.loc[il.TRAIN_START:il.TRAIN_END]
    w = il.weights(px, c, v, lookback=il.FROZEN_LOOKBACK)      # sparse ME grid -> monthly trades
    res = il.backtest_weights(px, w, cost_bps=il.FREEZE_COST)
    turn = res.turnover
    onesided = float((turn[turn > 0] / 2.0).mean())            # avg per-rebalance (monthly) turnover
    assert onesided < 0.5                                      # persistent: far below 100%/mo
    assert il.latest_weights(px, c, v, lookback=il.FROZEN_LOOKBACK).abs().sum() == pytest.approx(1.0, abs=1e-9)


# ---- live snapshot leg: read-only, signed, forward_track-compatible (mirrors run_volshock) ----

class Spy:
    """Records every method routed through groww_client.call; serves canned LTP."""

    PRICES = {"NSE_A": 110.0, "NSE_B": 105.0, "NSE_C": 90.0, "NSE_D": 95.0}

    def __init__(self):
        self.methods = []

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:                       # the spy honors the real guard
            raise PermissionError(method)
        syms = kwargs.get("exchange_trading_symbols", ())
        return {s: self.PRICES[s] for s in syms if s in self.PRICES}


def _synthetic_illiq():
    return lp.Book(
        weights=pd.Series({"A.NS": 0.25, "B.NS": 0.25, "C.NS": -0.25, "D.NS": -0.25}),
        regime_on=True, cash_frac=0.0, latest_date=pd.Timestamp("2026-07-09"),
        prev_close=pd.Series({"A.NS": 100.0, "B.NS": 100.0, "C.NS": 100.0, "D.NS": 100.0}),
        nsei_prev_close=float("nan"),
    )


def test_run_illiq_read_only_schema_and_pnl(tmp_path, monkeypatch):
    spy = Spy()
    monkeypatch.setattr(gc, "call", spy)
    monkeypatch.setattr(lp, "current_illiq_book", lambda **kw: _synthetic_illiq())

    out = tmp_path / "paper_trades_illiq.jsonl"
    row = lp.run_illiq(path=str(out), write=True)

    # SAFETY: only read-only methods dispatched, none of them order methods.
    assert spy.methods and set(spy.methods) <= set(lp.READ_METHODS)
    assert not any(m in gc._ORDER_METHODS for m in spy.methods)

    assert row["kind"] == "live_paper_illiq_snapshot"
    assert row["hypothesis_ref"] == "RL-2026-07-26-19"
    assert row["lookback"] == il.FROZEN_LOOKBACK
    assert sum(row["weights"].values()) == pytest.approx(0.0, abs=1e-9)          # dollar-neutral
    assert sum(abs(v) for v in row["weights"].values()) == pytest.approx(1.0)    # unit gross
    assert row["n_long"] == 2 and row["n_short"] == 2
    # book_ret = .25*(.10) + .25*(.05) - .25*(-.10) - .25*(-.05) = 0.075
    assert row["book_intraday_ret"] == pytest.approx(0.075)
    assert row["n_quotes_ok"] == 4 and row["n_names"] == 4

    rec = json.loads(out.read_text().strip().splitlines()[0])
    assert "panel_date" in rec and "weights" in rec           # forward_track-compatible


def test_run_illiq_records_book_only_when_quotes_unavailable(tmp_path, monkeypatch):
    def boom(method, *a, **k):
        raise RuntimeError("no entitlement")
    monkeypatch.setattr(gc, "call", boom)
    monkeypatch.setattr(lp, "current_illiq_book", lambda **kw: _synthetic_illiq())

    out = tmp_path / "i.jsonl"
    row = lp.run_illiq(path=str(out), write=True)
    assert row["book_intraday_ret"] is None                   # no live P&L invented
    assert row["n_quotes_ok"] == 0 and row["groww_ok"] is False
    assert row["weights"]                                      # book still recorded
    assert out.read_text().strip()                            # snapshot still written


def test_current_illiq_book_dollar_neutral(monkeypatch):
    close, vol = _ipanel(n=300, m=30)
    nsei = pd.DataFrame({lp.BENCH: np.full(len(close), 100.0)}, index=close.index)
    monkeypatch.setattr(lp.illiq, "panels", lambda **kw: (close, close, vol))
    monkeypatch.setattr(lp, "load_yahoo_ohlcv", lambda syms, refresh=False: {})
    monkeypatch.setattr(lp, "close_prices", lambda data, field="adj_close": nsei)

    book = lp.current_illiq_book()
    assert book.weights.sum() == pytest.approx(0.0, abs=1e-9)     # net 0
    assert book.weights.abs().sum() == pytest.approx(1.0, abs=1e-9)  # gross 1
    assert (book.weights > 0).any() and (book.weights < 0).any()
    assert set(book.prev_close.index) == set(book.weights.index)  # priced for live P&L
