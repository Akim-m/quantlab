"""Tests for the RL-2026-07-26-08 macro-sensitivity alignment (MACRO-BETA) sleeve.

FORWARD-ONLY construction: no performance backtest is exercised. What matters is that the
closed-form rolling OLS recovers known USDINR/Brent loadings, that the macro LAG is causal
(a bar-T macro jump moves nothing on/before T - a no-lag build would), that the alignment
score flips with the macro trend sign for a pure-beta name, that a macro gap beyond the
5-day ffill limit leaves those days beta-less, that betas are +/-3 MAD winsorized, that the
decile L/S book is dollar-neutral + unit-gross and deterministic, and that the weights use
only prior-day data. The live snapshot leg mirrors run_volshock: read-only, signed,
forward_track-compatible. Fully synthetic - no network.
"""

import json

import numpy as np
import pandas as pd
import pytest

from quantlab import groww_client as gc
from quantlab import live_paper as lp
from quantlab import macrobeta as mb


def _bidx(n, start="2016-01-01"):
    return pd.bdate_range(start=start, periods=n)


def _macro_panel(n=700, m=30, seed=0, noise=1e-4, inr_drift=0.0003, oil_drift=0.0002):
    """m stocks whose daily return loads on the LAGGED macro returns with a clean
    cross-section of coefficients (beta_INR in [-2, 2], beta_oil reversed), driven by two
    independent macro return streams. Returns (px, inr_level, oil_level, a, b) where a/b are
    the true per-name INR/oil loadings. Levels are cumulative products so pct_change recovers
    the return streams exactly (perfect beta recovery when noise == 0)."""
    rng = np.random.default_rng(seed)
    idx = _bidx(n)
    inr_ret = rng.normal(inr_drift, 0.006, n); inr_ret[0] = 0.0
    oil_ret = rng.normal(oil_drift, 0.011, n); oil_ret[0] = 0.0
    inr = pd.Series(100.0 * np.cumprod(1.0 + inr_ret), index=idx)
    oil = pd.Series(80.0 * np.cumprod(1.0 + oil_ret), index=idx)

    a = np.linspace(-2.0, 2.0, m)
    b = np.linspace(2.0, -2.0, m)
    x1 = np.concatenate([[0.0], inr_ret[:-1]])            # macro at t-1 vs stock at t
    x2 = np.concatenate([[0.0], oil_ret[:-1]])
    R = np.outer(x1, a) + np.outer(x2, b)
    if noise:
        R = R + rng.normal(0.0, noise, (n, m))
    R[0, :] = 0.0
    cols = [f"S{i:02d}" for i in range(m)]
    px = pd.DataFrame(100.0 * np.cumprod(1.0 + R, axis=0), index=idx, columns=cols)
    return px, inr, oil, a, b


# ---- closed-form rolling OLS recovers the known macro loadings ----

def test_beta_recovery():
    px, inr, oil, a, b = _macro_panel(n=700, m=20, seed=1, noise=0.0)
    b_inr, b_oil = mb.betas(px.pct_change(), inr.pct_change(), oil.pct_change())
    for i in (2, 9, 17):                                  # a perfect linear fit -> exact betas
        assert b_inr[f"S{i:02d}"].iloc[-1] == pytest.approx(a[i], abs=1e-6)
        assert b_oil[f"S{i:02d}"].iloc[-1] == pytest.approx(b[i], abs=1e-6)


# ---- the macro lag is causal: a bar-T jump moves nothing on/before T ----

def test_macro_lag_is_causal():
    px, inr, oil, _, _ = _macro_panel(n=700, m=20, seed=2)
    T = 500
    base = mb.alignment_score(px, inr, oil)               # lag=1 (registered)
    inj_inr = inr.copy()
    inj_inr.iloc[T] *= 1.5                                # a large USDINR jump at bar T
    inj = mb.alignment_score(px, inj_inr, oil)

    # macro-at-t-1 clock: bar-T return enters the regressor only from T+1, and the prior-day
    # 63d trend at T reads level(T-1) - so the whole cross-section through T is byte-identical.
    thru_T = np.abs(base.iloc[:T + 1].fillna(0.0).to_numpy()
                    - inj.iloc[:T + 1].fillna(0.0).to_numpy()).max()
    assert thru_T < 1e-12                                 # nothing on/before T moves
    assert not np.allclose(base.iloc[T + 1:].fillna(0.0),
                           inj.iloc[T + 1:].fillna(0.0))   # the jump bites only from T+1

    # A NO-LAG build regresses on the SAME-day macro return, so the bar-T jump moves the
    # bar-T beta and hence the bar-T score - which is exactly what the lag prevents above.
    base0 = mb.alignment_score(px, inr, oil, lag=0)
    inj0 = mb.alignment_score(px, inj_inr, oil, lag=0)
    assert not np.allclose(base0.iloc[T].fillna(0.0), inj0.iloc[T].fillna(0.0))


# ---- alignment score flips sign with the macro trend for a pure-beta name ----

def test_alignment_flips_with_trend_sign():
    n = 500
    idx = _bidx(n)

    def build(inr_drift, seed):
        rng = np.random.default_rng(seed)
        inr_ret = rng.normal(inr_drift, 0.006, n); inr_ret[0] = 0.0
        oil_ret = rng.normal(0.0, 0.010, n); oil_ret[0] = 0.0
        inr = pd.Series(100.0 * np.cumprod(1.0 + inr_ret), index=idx)
        oil = pd.Series(80.0 * np.cumprod(1.0 + oil_ret), index=idx)
        sret = np.concatenate([[0.0], inr_ret[:-1]])      # stock ret == lagged INR -> beta_INR~1
        px = pd.DataFrame({"A": 100.0 * np.cumprod(1.0 + sret)}, index=idx)
        return px, inr, oil

    px_u, inr_u, oil_u = build(+0.002, 1)                 # USDINR trending UP
    px_d, inr_d, oil_d = build(-0.002, 2)                 # USDINR trending DOWN
    assert mb.trend_sign(inr_u).iloc[-1] == 1 and mb.trend_sign(inr_d).iloc[-1] == -1

    su = mb.alignment_score(px_u, inr_u, oil_u)["A"].iloc[-1]
    sd = mb.alignment_score(px_d, inr_d, oil_d)["A"].iloc[-1]
    assert su > 0.5 and sd < -0.5                         # beta_INR~1 dominates; sign follows trend


# ---- +/-3 MAD cross-sectional winsorization ----

def test_winsorize_clips_at_three_mad():
    row = pd.DataFrame([[0.0, 1.0, 2.0, 3.0, 4.0, 100.0]])   # median 2.5, MAD 1.5
    out = mb.winsorize(row, n=3.0)
    hi = 2.5 + 3.0 * 1.5                                     # 7.0
    assert out.iloc[0, -1] == pytest.approx(hi)              # the outlier is clipped in
    assert (out.iloc[0, :5].to_numpy() == row.iloc[0, :5].to_numpy()).all()  # inliers untouched


# ---- decile L/S: dollar-neutral, unit-gross, correct leg sizes ----

def test_decile_dollar_neutral_unit_gross():
    px, inr, oil, _, _ = _macro_panel()
    w = mb.weights(px, inr, oil, rebalance=None)             # raw daily decile
    net = w.sum(axis=1).to_numpy()
    gross = w.abs().sum(axis=1).to_numpy()
    active = gross > 0
    assert active[-1]                                        # warmed up by the end
    np.testing.assert_allclose(net[active], 0.0, atol=1e-9)  # dollar-neutral
    np.testing.assert_allclose(gross[active], 1.0, atol=1e-9)  # unit gross
    assert (gross[~active] == 0.0).all()


def test_decile_sizes():
    px, inr, oil, _, _ = _macro_panel(m=30)
    w = mb.weights(px, inr, oil, rebalance=None).iloc[-1]
    k = int(np.floor(30 * mb.DECILE))                       # 3 per leg
    assert int((w > 0).sum()) == k and int((w < 0).sum()) == k
    assert w[w > 0].nunique() == 1 and w[w < 0].nunique() == 1  # equal-weight within each leg


def test_monthly_held_book_is_neutral_and_deterministic():
    px, inr, oil, _, _ = _macro_panel()
    w = mb.weights(px, inr, oil)                            # default ME rebalance
    last = w.iloc[-1]
    assert last.abs().sum() == pytest.approx(1.0, abs=1e-9)  # gross 1
    assert last.sum() == pytest.approx(0.0, abs=1e-9)       # net 0
    pd.testing.assert_frame_equal(w, mb.weights(px, inr, oil))  # determinism
    pd.testing.assert_series_equal(mb.latest_weights(px, inr, oil), last)


# ---- calendar alignment: a macro gap longer than the ffill limit stays NaN ----

def test_calendar_gap_beyond_ffill_limit():
    panel_idx = _bidx(40, start="2020-01-01")
    macro = pd.Series(100.0, index=panel_idx).drop(panel_idx[10:19])  # a 9-business-day hole
    aligned = mb.align_level(macro, panel_idx)
    assert aligned.iloc[10:15].notna().all()               # first 5 gap days carried forward
    assert aligned.iloc[15:19].isna().all()                # beyond the 5-day limit -> NaN
    assert aligned.iloc[19:].notna().all()                 # resumes once real prints return


# ---- insufficient beta history is excluded ----

def test_insufficient_history_excluded():
    px, inr, oil, _, _ = _macro_panel(n=700, m=12, seed=4)
    px = px.copy()
    px.loc[px.index[:500], "S00"] = np.nan                 # S00 lists too late for a full window
    assert np.isnan(mb.alignment_score(px, inr, oil)["S00"].iloc[-1])
    w = mb.weights(px, inr, oil, rebalance=None).iloc[-1]
    assert w["S00"] == 0.0                                  # beta-less name never enters the book
    assert w.abs().sum() == pytest.approx(1.0, abs=1e-9)    # the rest still form a full unit-gross book


# ---- no look-ahead: date-t weight reads only data through t-1 ----

def test_weights_use_prior_day_data_only():
    px, inr, oil, _, _ = _macro_panel(seed=3)
    w = mb.weights(px, inr, oil, rebalance=None)

    # Perturbing the LAST bar moves NO weight: weight[t] never reads bar-t data (shift(1)).
    px2, inr2, oil2 = px.copy(), inr.copy(), oil.copy()
    px2.iloc[-1] *= 1.5
    inr2.iloc[-1] *= 1.2
    oil2.iloc[-1] *= 1.2
    pd.testing.assert_frame_equal(w, mb.weights(px2, inr2, oil2, rebalance=None))

    # Perturbing an interior bar t leaves every weight on/before t unchanged; it bites at t+1.
    t = 400
    px3 = px.copy()
    px3.iloc[t] *= 1.5
    pert = mb.weights(px3, inr, oil, rebalance=None)
    pd.testing.assert_frame_equal(w.iloc[:t + 1], pert.iloc[:t + 1])
    assert not w.iloc[t + 1:].equals(pert.iloc[t + 1:])


# ---- disclosure arm: cross-sectional Spearman vs an external signal ----

def test_disclosure_corr_ranks_common_names():
    score = pd.Series({"A": 3.0, "B": 1.0, "C": 2.0, "D": np.nan})
    shock = pd.Series({"A": 30.0, "B": 10.0, "C": 20.0, "E": 5.0})
    rho, n = mb.disclosure_corr(score, shock)              # A,B,C overlap and rank identically
    assert n == 3 and rho == pytest.approx(1.0)


# ---- live snapshot leg: read-only, signed, forward_track-compatible (mirrors run_volshock) ----

class Spy:
    """Records every method routed through groww_client.call; serves canned LTP."""

    PRICES = {"NSE_A": 110.0, "NSE_B": 105.0, "NSE_C": 90.0, "NSE_D": 95.0}

    def __init__(self):
        self.methods = []

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:                    # the spy honors the real guard
            raise PermissionError(method)
        syms = kwargs.get("exchange_trading_symbols", ())
        return {s: self.PRICES[s] for s in syms if s in self.PRICES}


def _synthetic_macrobeta():
    return lp.Book(
        weights=pd.Series({"A.NS": 0.25, "B.NS": 0.25, "C.NS": -0.25, "D.NS": -0.25}),
        regime_on=True, cash_frac=0.0, latest_date=pd.Timestamp("2026-07-09"),
        prev_close=pd.Series({"A.NS": 100.0, "B.NS": 100.0, "C.NS": 100.0, "D.NS": 100.0}),
        nsei_prev_close=float("nan"),
    )


def test_run_macrobeta_read_only_schema_and_pnl(tmp_path, monkeypatch):
    spy = Spy()
    monkeypatch.setattr(gc, "call", spy)
    monkeypatch.setattr(lp, "current_macrobeta_book", lambda **kw: _synthetic_macrobeta())

    out = tmp_path / "paper_trades_macrobeta.jsonl"
    row = lp.run_macrobeta(path=str(out), write=True)

    # SAFETY: only read-only methods dispatched, none of them order methods.
    assert spy.methods and set(spy.methods) <= set(lp.READ_METHODS)
    assert not any(m in gc._ORDER_METHODS for m in spy.methods)

    assert row["kind"] == "live_paper_macrobeta_snapshot"
    assert row["hypothesis_ref"] == "RL-2026-07-26-08"
    assert sum(row["weights"].values()) == pytest.approx(0.0, abs=1e-9)          # dollar-neutral
    assert sum(abs(v) for v in row["weights"].values()) == pytest.approx(1.0)    # unit gross
    assert row["n_long"] == 2 and row["n_short"] == 2
    # book_ret = .25*(.10) + .25*(.05) - .25*(-.10) - .25*(-.05) = 0.075
    assert row["book_intraday_ret"] == pytest.approx(0.075)
    assert row["n_quotes_ok"] == 4 and row["n_names"] == 4

    rec = json.loads(out.read_text().strip().splitlines()[0])
    assert "panel_date" in rec and "weights" in rec           # forward_track-compatible


def test_run_macrobeta_records_book_only_when_quotes_unavailable(tmp_path, monkeypatch):
    def boom(method, *a, **k):
        raise RuntimeError("no entitlement")
    monkeypatch.setattr(gc, "call", boom)
    monkeypatch.setattr(lp, "current_macrobeta_book", lambda **kw: _synthetic_macrobeta())

    out = tmp_path / "m.jsonl"
    row = lp.run_macrobeta(path=str(out), write=True)
    assert row["book_intraday_ret"] is None                   # no live P&L invented
    assert row["n_quotes_ok"] == 0 and row["groww_ok"] is False
    assert row["weights"]                                      # book still recorded
    assert out.read_text().strip()                            # snapshot still written


def test_current_macrobeta_book_dollar_neutral(monkeypatch):
    px, inr, oil, _, _ = _macro_panel(n=700, m=30, seed=0)
    nsei = pd.DataFrame({lp.BENCH: np.full(len(px), 100.0)}, index=px.index)
    monkeypatch.setattr(lp.macrobeta, "panels", lambda **kw: (px, inr, oil))
    monkeypatch.setattr(lp, "load_yahoo_ohlcv", lambda syms, refresh=False: {})
    monkeypatch.setattr(lp, "close_prices", lambda data, field="adj_close": nsei)

    book = lp.current_macrobeta_book()
    assert book.weights.sum() == pytest.approx(0.0, abs=1e-9)        # net 0
    assert book.weights.abs().sum() == pytest.approx(1.0, abs=1e-9)  # gross 1
    assert (book.weights > 0).any() and (book.weights < 0).any()
    assert set(book.prev_close.index) == set(book.weights.index)     # priced for live P&L
