"""Causality and construction tests for the short-term family (RL-2026-07-10).

Short-horizon signals are look-ahead-prone, so the load-bearing test is
truncation-invariance: a book built on prices[:t+1] must equal the full book at
t. A deliberately leaky variant is included to prove the harness has teeth.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.features import rolling_vol
from quantlab.short_term import (
    LO_NAMES,
    LS_NAMES,
    build_books,
    gated_rev5,
    long_short,
    st_raw_signals,
    vol_gate,
)
from quantlab.xsec import _neutral


def _panel(n=340, cols=8, seed=0):
    """Synthetic (px, mkt, ohlcv) with enough history for the 252-day beta window."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    steps = rng.normal(0.0004, 0.018, size=(n, cols))
    px = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx,
                      columns=[f"S{i}" for i in range(cols)])
    mkt = px.mean(axis=1)
    # open/close: a modest intraday wiggle around the close path (raw, unadjusted)
    close = px
    open_ = px.shift(1) * (1.0 + rng.normal(0.0, 0.004, size=(n, cols)))
    ohlcv = {"open": open_.fillna(px.iloc[0]), "close": close,
             "high": px, "low": px, "volume": pd.DataFrame(1e6, index=idx, columns=px.columns)}
    return px, mkt, ohlcv


def test_st_raw_signals_keys_and_shape():
    px, mkt, _ = _panel()
    sigs = st_raw_signals(px, mkt)
    for k in ("rev5", "rev2", "resid_rev", "mom21", "wkmom"):
        assert k in sigs and sigs[k].shape == px.shape


def test_build_books_has_full_family():
    px, mkt, ohlcv = _panel()
    books = build_books(px, mkt, ohlcv)
    assert set(books) == set(LS_NAMES) | set(LO_NAMES)
    assert len(books) == 13


@pytest.mark.parametrize("name", LS_NAMES + LO_NAMES)
def test_books_no_look_ahead(name):
    """Book weights at t use only prices <= t: prices[:t+1] reproduces full[t]."""
    px, mkt, ohlcv = _panel()
    full = build_books(px, mkt, ohlcv)[name][0]
    assert full.abs().to_numpy().sum() > 0  # non-vacuous
    for t in (300, 320, 335):
        pxt, mktt = px.iloc[: t + 1], mkt.iloc[: t + 1]
        oht = {k: v.iloc[: t + 1] for k, v in ohlcv.items()}
        trunc = build_books(pxt, mktt, oht)[name][0]
        assert (full.iloc[t] - trunc.iloc[-1]).abs().max() < 1e-9


def test_no_look_ahead_catches_future_shift():
    """A leaky reversal (tomorrow's 5-day return) must break truncation-invariance."""
    px, _, _ = _panel()
    leaky = lambda p: long_short(-p.pct_change(5).shift(-1))
    full = leaky(px)
    gap = max((full.iloc[t] - leaky(px.iloc[: t + 1]).iloc[-1]).abs().max()
              for t in (300, 320, 335))
    assert gap > 1e-6


@pytest.mark.parametrize("name", LS_NAMES)
def test_long_short_dollar_neutral_unit_gross(name):
    px, mkt, ohlcv = _panel()
    w = build_books(px, mkt, ohlcv)[name][0].dropna(how="all")
    net = w.sum(axis=1).abs()
    gross = w.abs().sum(axis=1)
    assert (net < 1e-9).all()                                  # dollar-neutral
    assert (((gross - 1.0).abs() < 1e-9) | (gross < 1e-12)).all()  # unit gross or flat


@pytest.mark.parametrize("name", LO_NAMES)
def test_long_only_sums_to_one_or_cash_and_nonneg(name):
    px, mkt, ohlcv = _panel()
    w = build_books(px, mkt, ohlcv)[name][0].dropna(how="all")
    s = w.sum(axis=1)
    assert (w >= -1e-12).to_numpy().all()                       # long-only
    assert (((s - 1.0).abs() < 1e-9) | (s.abs() < 1e-12)).all()  # invested or cash


def test_vol_gate_selects_high_vol_half():
    """vol_gate keeps the higher-vol names; gated_rev5 drops the low-vol half."""
    px, _, _ = _panel()
    gate = vol_gate(px, lb=21, top_frac=0.5)
    vol = rolling_vol(px, 21)
    row = px.index[-1]
    kept = gate.loc[row]
    # every kept name has vol >= every dropped name's vol on that date
    if kept.any() and (~kept).any():
        assert vol.loc[row][kept].min() >= vol.loc[row][~kept].max() - 1e-12
    # gated reversal leaves the dropped (low-vol) names NaN
    g = gated_rev5(px)
    assert g.loc[row][~kept].isna().all()


def test_gated_book_only_holds_gated_names():
    """LS-VOLGATE puts zero weight on names outside the high-vol half."""
    px, mkt, ohlcv = _panel()
    gate = vol_gate(px, lb=21, top_frac=0.5)
    w = build_books(px, mkt, ohlcv)["LS-VOLGATE"][0]
    row = px.index[-1]
    dropped = w.columns[~gate.loc[row].fillna(False)]
    assert (w.loc[row, dropped].abs() < 1e-12).all()
