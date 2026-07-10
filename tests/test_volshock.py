"""Tests for the RL-2026-07-26-15 turnover-shock (VOL-SHOCK) sleeve.

FORWARD-ONLY construction: no performance backtest is exercised. What matters is that
the log turnover-shock is the ratio of the trailing 5d/126d mean turnover, that a
non-traded (zero/missing-volume) name drops out of that day's cross-section, that the
+/-3 MAD winsorization fires, that the decile L/S book is dollar-neutral + unit-gross
and deterministic, and that the weights use only prior-day data. The live snapshot leg
mirrors run_divcarry: read-only, signed, forward_track-compatible. Fully synthetic - no
network.
"""

import json

import numpy as np
import pandas as pd
import pytest

from quantlab import groww_client as gc
from quantlab import live_paper as lp
from quantlab import volshock as vs


def _bidx(n, start="2018-01-01"):
    return pd.bdate_range(start=start, periods=n)


def _vol_panel(n=400, m=30, seed=0, surge=5):
    """m names on a flat 100-price grid with constant baseline volume, then a
    name-specific turnover surge over the last `surge` days spanning 0.5x..3.0x the
    baseline -> a clean monotone cross-section of turnover shocks on the final days."""
    idx = _bidx(n)
    cols = [f"S{i:02d}" for i in range(m)]
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    vol = pd.DataFrame(1000.0, index=idx, columns=cols)
    mult = np.linspace(0.5, 3.0, m)
    vol.iloc[-surge:] = 1000.0 * mult                       # broadcast per-name surge
    return close, vol


# ---- shock arithmetic: log ratio of trailing 5d / 126d mean turnover ----

def test_shock_is_log_ratio_of_mean_turnover():
    n = 200
    idx = _bidx(n)
    close = pd.DataFrame({"AAA": np.full(n, 1.0)}, index=idx)   # turnover == volume
    v = np.full(n, 50.0)
    v[-5:] = 150.0                                              # a 3x surge over the last week
    vol = pd.DataFrame({"AAA": v}, index=idx)

    sh = vs.shock(close, vol)["AAA"]
    # last day: 5d mean = 150; 126d mean = (121*50 + 5*150)/126 = 6800/126
    assert sh.iloc[-1] == pytest.approx(np.log(150.0 * 126.0 / 6800.0))
    # a pre-surge day: both windows sit inside the flat region -> ratio 1 -> log 0
    assert sh.iloc[100] == pytest.approx(0.0, abs=1e-12)


# ---- zero / missing volume on the signal day is excluded ----

def test_zero_or_missing_volume_excluded():
    close, vol = _vol_panel()
    vol = vol.copy()
    vol.iloc[-1, 0] = 0.0                                       # S00 did not trade the signal day
    vol.iloc[-1, 1] = np.nan                                    # S01 has no volume that day

    sh = vs.shock(close, vol).iloc[-1]
    assert np.isnan(sh.iloc[0]) and np.isnan(sh.iloc[1])        # both excluded
    assert np.isfinite(sh.iloc[2:]).all()                      # every name that traded is kept
    # a non-traded cell is NaN turnover, never counted as a real zero-rupee day
    assert np.isnan(vs.turnover(close, vol).iloc[-1, 0])
    assert np.isnan(vs.turnover(close, vol).iloc[-1, 1])


# ---- +/-3 MAD cross-sectional winsorization ----

def test_winsorize_clips_at_three_mad():
    row = pd.DataFrame([[0.0, 1.0, 2.0, 3.0, 4.0, 100.0]])      # median 2.5, MAD 1.5
    out = vs.winsorize(row, n=3.0)
    hi = 2.5 + 3.0 * 1.5                                        # 7.0
    assert out.iloc[0, -1] == pytest.approx(hi)                # the outlier is clipped in
    assert (out.iloc[0, :5].to_numpy() == row.iloc[0, :5].to_numpy()).all()  # inliers untouched


# ---- decile L/S: dollar-neutral, unit-gross, longs the highest shock ----

def test_decile_dollar_neutral_unit_gross():
    close, vol = _vol_panel()
    w = vs.weights(close, vol, rebalance=None)                 # raw daily decile
    net = w.sum(axis=1).to_numpy()
    gross = w.abs().sum(axis=1).to_numpy()
    active = gross > 0
    assert active[-1]                                          # warmed up by the end
    np.testing.assert_allclose(net[active], 0.0, atol=1e-9)    # dollar-neutral
    np.testing.assert_allclose(gross[active], 1.0, atol=1e-9)  # unit gross
    assert (gross[~active] == 0.0).all()


def test_longs_top_shock_shorts_bottom():
    close, vol = _vol_panel()
    sh = vs.shock(close, vol).iloc[-2]                         # shock feeding the last weight (t-1)
    w = vs.weights(close, vol, rebalance=None).iloc[-1]
    longs, shorts = w[w > 0].index, w[w < 0].index
    assert sh[longs].min() > sh[shorts].max()                 # every long out-shocks every short
    assert w[longs].nunique() == 1 and w[shorts].nunique() == 1  # equal-weight within each leg


def test_monthly_held_book_is_neutral_and_deterministic():
    close, vol = _vol_panel()
    w = vs.weights(close, vol)                                 # default ME rebalance
    last = w.iloc[-1]
    assert last.abs().sum() == pytest.approx(1.0, abs=1e-9)    # gross 1
    assert last.sum() == pytest.approx(0.0, abs=1e-9)          # net 0
    pd.testing.assert_frame_equal(w, vs.weights(close, vol))   # determinism
    pd.testing.assert_series_equal(vs.latest_weights(close, vol), last)


# ---- no look-ahead: date-t weight reads only data through t-1 ----

def test_weights_use_prior_day_data_only():
    close, vol = _vol_panel(seed=3)
    w = vs.weights(close, vol, rebalance=None)

    # Perturbing the LAST bar moves NO weight: weight[t] never reads (close,volume)[t].
    c2, v2 = close.copy(), vol.copy()
    c2.iloc[-1] *= 1.5
    v2.iloc[-1] *= 5.0
    pd.testing.assert_frame_equal(w, vs.weights(c2, v2, rebalance=None))

    # A single-name volume spike at interior bar t would flip its decile membership FROM
    # t onward - but weight[t] (which reads only data <= t-1) must not move.
    t = 300
    c3, v3 = close.copy(), vol.copy()
    v3.iloc[t, 0] *= 50.0                                      # S00: huge turnover at bar t
    pert = vs.weights(c3, v3, rebalance=None)
    pd.testing.assert_frame_equal(w.iloc[:t + 1], pert.iloc[:t + 1])   # weights <= t unchanged
    assert not w.iloc[t + 1:].equals(pert.iloc[t + 1:])       # the spike bites only from t+1


# ---- live snapshot leg: read-only, signed, forward_track-compatible (mirrors run_divcarry) ----

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


def _synthetic_volshock():
    return lp.Book(
        weights=pd.Series({"A.NS": 0.25, "B.NS": 0.25, "C.NS": -0.25, "D.NS": -0.25}),
        regime_on=True, cash_frac=0.0, latest_date=pd.Timestamp("2026-07-08"),
        prev_close=pd.Series({"A.NS": 100.0, "B.NS": 100.0, "C.NS": 100.0, "D.NS": 100.0}),
        nsei_prev_close=float("nan"),
    )


def test_run_volshock_read_only_schema_and_pnl(tmp_path, monkeypatch):
    spy = Spy()
    monkeypatch.setattr(gc, "call", spy)
    monkeypatch.setattr(lp, "current_volshock_book", lambda **kw: _synthetic_volshock())

    out = tmp_path / "paper_trades_volshock.jsonl"
    row = lp.run_volshock(path=str(out), write=True)

    # SAFETY: only read-only methods dispatched, none of them order methods.
    assert spy.methods and set(spy.methods) <= set(lp.READ_METHODS)
    assert not any(m in gc._ORDER_METHODS for m in spy.methods)

    assert row["kind"] == "live_paper_volshock_snapshot"
    assert row["hypothesis_ref"] == "RL-2026-07-26-15"
    assert sum(row["weights"].values()) == pytest.approx(0.0, abs=1e-9)          # dollar-neutral
    assert sum(abs(v) for v in row["weights"].values()) == pytest.approx(1.0)    # unit gross
    assert row["n_long"] == 2 and row["n_short"] == 2
    # book_ret = .25*(.10) + .25*(.05) - .25*(-.10) - .25*(-.05) = 0.075
    assert row["book_intraday_ret"] == pytest.approx(0.075)
    assert row["n_quotes_ok"] == 4 and row["n_names"] == 4

    rec = json.loads(out.read_text().strip().splitlines()[0])
    assert "panel_date" in rec and "weights" in rec           # forward_track-compatible


def test_run_volshock_records_book_only_when_quotes_unavailable(tmp_path, monkeypatch):
    def boom(method, *a, **k):
        raise RuntimeError("no entitlement")
    monkeypatch.setattr(gc, "call", boom)
    monkeypatch.setattr(lp, "current_volshock_book", lambda **kw: _synthetic_volshock())

    out = tmp_path / "v.jsonl"
    row = lp.run_volshock(path=str(out), write=True)
    assert row["book_intraday_ret"] is None                   # no live P&L invented
    assert row["n_quotes_ok"] == 0 and row["groww_ok"] is False
    assert row["weights"]                                      # book still recorded
    assert out.read_text().strip()                            # snapshot still written


def test_current_volshock_book_dollar_neutral(monkeypatch):
    close, vol = _vol_panel()
    nsei = pd.DataFrame({lp.BENCH: np.full(len(close), 100.0)}, index=close.index)
    monkeypatch.setattr(lp.volshock, "panels", lambda **kw: (close, vol))
    monkeypatch.setattr(lp, "load_yahoo_ohlcv", lambda syms, refresh=False: {})
    monkeypatch.setattr(lp, "close_prices", lambda data, field="adj_close": nsei)

    book = lp.current_volshock_book()
    assert book.weights.sum() == pytest.approx(0.0, abs=1e-9)     # net 0
    assert book.weights.abs().sum() == pytest.approx(1.0, abs=1e-9)  # gross 1
    assert (book.weights > 0).any() and (book.weights < 0).any()
    assert set(book.prev_close.index) == set(book.weights.index)  # priced for live P&L
