"""Tests for the RL-2026-07-26-20 delivery-conviction cross-section (DELIV).

Fully synthetic - no network. Cover the locked preprocessing (no ffill beyond 3 missing
days, the <60%-valid exclusion, +/-3 MAD winsorize), the three signal variants (LEVEL /
SHOCK / SIGNED arithmetic and the sign flip), the decile L/S book's neutrality and
one-day lag causality, the hard TRAIN-boundary guarantee (no post-2016 bar enters a TRAIN
statistic), and the live snapshot leg (read-only, graceful no-quote degradation).
"""

import json

import numpy as np
import pandas as pd
import pytest

from quantlab import deliv as d
from quantlab import groww_client as gc
from quantlab import live_paper as lp


def _bidx(n, start="2015-01-01"):
    return pd.bdate_range(start=start, periods=n)


def _panels(n=1000, m=30, start="2014-06-01", seed=0):
    """A flat-price panel plus a smooth, time- and name-varying delivery-ratio panel in
    (0.1, 0.9) - deciles rotate over time, so the one-day lag is load-bearing."""
    idx = _bidx(n, start)
    cols = [f"S{i:02d}.NS" for i in range(m)]
    px = pd.DataFrame(100.0, index=idx, columns=cols)
    rng = np.random.default_rng(seed)
    ratio = pd.DataFrame(rng.uniform(0.1, 0.9, (n, m)), index=idx, columns=cols)
    ratio = ratio.rolling(10, min_periods=1).mean()          # smooth so 63d means separate
    return px, ratio


# ---- preprocessing: no ffill beyond 3 missing days ----

def test_prep_ratio_no_ffill_beyond_three_days():
    idx = _bidx(10)
    r = pd.DataFrame({"A": [0.5, np.nan, np.nan, np.nan, np.nan, 0.6, np.nan, np.nan, np.nan, 0.7]},
                     index=idx)
    out = d.prep_ratio(r, idx)["A"]
    assert out.iloc[1] == out.iloc[2] == out.iloc[3] == pytest.approx(0.5)  # bridged 3 days
    assert np.isnan(out.iloc[4])                                            # 4th gap stays NaN


# ---- preprocessing: <60% valid days in the lookback -> excluded ----

def test_masked_mean_excludes_below_sixty_percent():
    idx = _bidx(5)
    # 2 of 5 valid in the window (40% < 60%) -> NaN on the last day
    thin = pd.DataFrame({"A": [0.5, 0.5, np.nan, np.nan, np.nan]}, index=idx)
    assert np.isnan(d._masked_mean(thin, 5).iloc[-1, 0])
    # 4 of 5 valid (80% >= 60%) -> defined, equals the mean of the valid observations
    thick = pd.DataFrame({"A": [0.5, 0.5, 0.5, np.nan, 0.7]}, index=idx)
    assert d._masked_mean(thick, 5).iloc[-1, 0] == pytest.approx(0.55)


# ---- SHOCK arithmetic: log(5d mean / 126d mean) ----

def test_shock_is_log_ratio_of_mean_ratio():
    n = 200
    r = pd.DataFrame({"A": np.full(n, 0.4)}, index=_bidx(n))
    r.iloc[-5:] = 0.8                                          # a delivery-share surge
    sh = d.shock(r)["A"]
    exp = np.log(0.8 * 126.0 / (121 * 0.4 + 5 * 0.8))          # 5d mean 0.8 vs 126d mean
    assert sh.iloc[-1] == pytest.approx(exp)
    assert sh.iloc[100] == pytest.approx(0.0, abs=1e-12)       # inside the flat region


# ---- SIGNED = sign(5d price return) x SHOCK ----

def test_signed_shock_flips_with_price_direction():
    n = 200
    idx = _bidx(n)
    r = pd.DataFrame({"UP": np.full(n, 0.4), "DN": np.full(n, 0.4)}, index=idx)
    r.iloc[-5:] = 0.8                                          # both have SHOCK > 0
    up = np.full(n, 100.0); up[-1] = 110.0                     # +ve 5d return
    dn = np.full(n, 100.0); dn[-1] = 90.0                      # -ve 5d return
    px = pd.DataFrame({"UP": up, "DN": dn}, index=idx)
    ss = d.signed_shock(r, px).iloc[-1]
    assert ss["UP"] > 0 and ss["DN"] < 0                       # same shock, opposite sign


# ---- +/-3 MAD winsorization (reused verbatim from volshock) ----

def test_winsorize_clips_at_three_mad():
    row = pd.DataFrame([[0.0, 1.0, 2.0, 3.0, 4.0, 100.0]])     # median 2.5, MAD 1.5
    out = d.winsorize(row, n=3.0)
    assert out.iloc[0, -1] == pytest.approx(2.5 + 3.0 * 1.5)   # outlier clipped in
    assert (out.iloc[0, :5].to_numpy() == row.iloc[0, :5].to_numpy()).all()


# ---- decile L/S: dollar-neutral, unit-gross, longs the highest ratio ----

def test_decile_dollar_neutral_unit_gross_longs_high():
    idx = _bidx(200)
    m = 30
    cols = [f"S{i:02d}.NS" for i in range(m)]
    lvl = pd.DataFrame(np.tile(np.linspace(0.1, 0.9, m), (len(idx), 1)), index=idx, columns=cols)
    w = d.book_weights(lvl, rebalance=None)
    last = w.iloc[-1]
    assert last.sum() == pytest.approx(0.0, abs=1e-9)          # dollar-neutral
    assert last.abs().sum() == pytest.approx(1.0, abs=1e-9)    # unit gross
    longs, shorts = last[last > 0].index, last[last < 0].index
    assert lvl.iloc[-1][longs].min() > lvl.iloc[-1][shorts].max()   # longs out-rank shorts
    assert last[longs].nunique() == 1 and last[shorts].nunique() == 1  # equal-weight legs


# ---- no look-ahead: date-t weight reads only data through t-1 ----

def test_weights_use_prior_day_data_only():
    px, ratio = _panels()
    sig = d.level(ratio)
    w = d.book_weights(sig, rebalance=None)

    # Perturbing the LAST bar moves NO weight: weight[t] never reads ratio[t].
    r2 = ratio.copy(); r2.iloc[-1] = 0.5
    pd.testing.assert_frame_equal(w, d.book_weights(d.level(r2), rebalance=None))

    # The lag is load-bearing: the same book WITHOUT the shift differs (deciles rotate).
    wsig = d.winsorize(sig, d.MAD_N)
    lagged = d.decile_ls(wsig.shift(1), d.DECILE)
    nolag = d.decile_ls(wsig, d.DECILE)
    pd.testing.assert_frame_equal(w, lagged)
    assert not lagged.equals(nolag)


# ---- TRAIN boundary: nothing after 2016-12-31 enters a TRAIN statistic ----

def test_train_table_ignores_post_boundary_data():
    px, ratio = _panels(n=1000, start="2014-06-01")
    assert px.index.max() > pd.Timestamp(d.TRAIN_END)         # panel really spans past it
    base = d.train_table(px, ratio)

    px2, ratio2 = px.copy(), ratio.copy()
    post = px.index > pd.Timestamp(d.TRAIN_END)
    px2.loc[post] *= 3.0                                       # scramble the hold-out era
    ratio2.loc[post] = 0.99
    scrambled = d.train_table(px2, ratio2)
    pd.testing.assert_frame_equal(base, scrambled)            # identical -> zero leakage
    assert set(base["variant"]) == set(d.VARIANTS)


# ---- live snapshot leg: read-only, signed, graceful degradation (mirrors run_volshock) ----

class Spy:
    PRICES = {"NSE_A": 110.0, "NSE_B": 105.0, "NSE_C": 90.0, "NSE_D": 95.0}

    def __init__(self):
        self.methods = []

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:
            raise PermissionError(method)
        syms = kwargs.get("exchange_trading_symbols", ())
        return {s: self.PRICES[s] for s in syms if s in self.PRICES}


def _synthetic_book():
    return lp.Book(
        weights=pd.Series({"A.NS": 0.25, "B.NS": 0.25, "C.NS": -0.25, "D.NS": -0.25}),
        regime_on=True, cash_frac=0.0, latest_date=pd.Timestamp("2026-07-08"),
        prev_close=pd.Series({"A.NS": 100.0, "B.NS": 100.0, "C.NS": 100.0, "D.NS": 100.0}),
        nsei_prev_close=float("nan"),
    )


def test_run_deliv_read_only_schema_and_pnl(tmp_path, monkeypatch):
    spy = Spy()
    monkeypatch.setattr(gc, "call", spy)
    monkeypatch.setattr(d, "current_deliv_book", lambda **kw: _synthetic_book())

    out = tmp_path / "paper_trades_deliv.jsonl"
    row = d.run_deliv(path=str(out), write=True)

    assert spy.methods and set(spy.methods) <= set(lp.READ_METHODS)
    assert not any(mm in gc._ORDER_METHODS for mm in spy.methods)
    assert row["kind"] == "live_paper_deliv_snapshot"
    assert row["hypothesis_ref"] == "RL-2026-07-26-20"
    assert sum(row["weights"].values()) == pytest.approx(0.0, abs=1e-9)
    assert sum(abs(v) for v in row["weights"].values()) == pytest.approx(1.0)
    assert row["n_long"] == 2 and row["n_short"] == 2
    assert row["book_intraday_ret"] == pytest.approx(0.075)   # .25*.10+.25*.05+.25*.10+.25*.05
    rec = json.loads(out.read_text().strip().splitlines()[0])
    assert "panel_date" in rec and "weights" in rec


def test_run_deliv_degrades_without_quotes(tmp_path, monkeypatch):
    def boom(method, *a, **k):
        raise RuntimeError("no entitlement")
    monkeypatch.setattr(gc, "call", boom)
    monkeypatch.setattr(d, "current_deliv_book", lambda **kw: _synthetic_book())

    out = tmp_path / "v.jsonl"
    row = d.run_deliv(path=str(out), write=True)
    assert row["book_intraday_ret"] is None
    assert row["n_quotes_ok"] == 0 and row["groww_ok"] is False
    assert row["weights"] and out.read_text().strip()
