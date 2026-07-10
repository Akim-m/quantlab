"""Synthetic tests for RL-2026-07-26-02 US-trend gate on NIFTYBEES (no network).

The load-bearing mechanic is the causality clock: a signal built on the US (`^GSPC`)
calendar must reach the NSE decision only through closes strictly before the NSE date.
These tests pin (1) no-look-ahead (a US regime flip acts on the NEXT NSE session, never
same-day; removing the one-day shift leaks it), (2) holiday-mismatch stays causal,
(3) the frozen-variant selection reads TRAIN rows only, (4) the NIFTYBEES spike guard is
applied by load_data, and (5) determinism (bit-identical headline metrics on a re-run).
"""

import numpy as np
import pandas as pd

import quantlab.us_gate_study as ug
from quantlab.us_gate_study import (
    SYMBOL,
    align_us_signal,
    evaluate,
    train_table,
    us_signal,
)


def _flip_prices(idx, flip_at):
    """^GSPC that sits at 100 then jumps to 110 from `flip_at` on: px>200d MA (and the
    21d/63d return) is False before flip_at and first True exactly at flip_at."""
    px = pd.Series(100.0, index=idx)
    px.loc[flip_at:] = 110.0
    return px


def test_no_lookahead_next_session_only():
    # Shared US/NSE calendar so the flip date lands on a common session.
    idx = pd.bdate_range("2015-01-01", periods=260)
    flip = idx[230]
    nxt = idx[231]
    gspc = _flip_prices(idx, flip)
    sig = us_signal(gspc, "ma200")
    assert not bool(sig.loc[:flip].iloc[:-1].any())  # OFF on every US date before the flip
    assert bool(sig.loc[flip])                       # first ON exactly at the flip date

    aligned = align_us_signal(sig, idx, shift=1)
    assert aligned.loc[flip] == 0.0   # NSE still OFF on the flip date (US bar dated `flip`
    assert aligned.loc[nxt] == 1.0    # is unknown at 15:30 IST); it acts only NEXT session
    assert aligned.loc[:flip].sum() == 0.0

    # Guard is load-bearing: without the shift the same-day US close leaks into the NSE
    # decision, so the position flips ON the flip date itself.
    leaked = align_us_signal(sig, idx, shift=0)
    assert leaked.loc[flip] == 1.0


def test_holiday_mismatch_uses_prior_us_close():
    # NSE trades on a day the US is closed: the aligned signal must reuse the last lagged
    # US value (<= that date), never borrow a future US bar.
    us_idx = pd.bdate_range("2015-01-01", periods=40)
    flip = us_idx[20]
    gspc = _flip_prices(us_idx, flip)
    sig = us_signal(gspc, "ma200")

    holiday = us_idx[25]                      # drop this US session
    us_open = us_idx.delete(25)
    nse_idx = us_idx                          # NSE still open on `holiday`
    sig_gap = sig.reindex(us_open)

    aligned = align_us_signal(sig_gap, nse_idx, shift=1)
    prev_us = us_open[us_open < holiday][-1]  # newest US session strictly before the gap
    assert aligned.loc[holiday] == align_us_signal(sig_gap, us_open, shift=1).loc[prev_us]


def test_frozen_selection_reads_train_only():
    idx = pd.bdate_range("2008-06-01", "2020-12-31")
    rng = np.random.default_rng(0)
    px = pd.DataFrame({SYMBOL: 100.0 * np.cumprod(1 + rng.normal(3e-4, 0.01, len(idx)))}, index=idx)
    gspc = pd.Series(100.0 * np.cumprod(1 + rng.normal(3e-4, 0.01, len(idx))), index=idx)

    base = train_table(px, gspc)

    # Corrupt every post-TRAIN row; the TRAIN table must be byte-identical.
    px2 = px.copy()
    gspc2 = gspc.copy()
    px2.loc["2017-01-01":] *= 5.0
    gspc2.loc["2017-01-01":] *= 0.2
    after = train_table(px2, gspc2)
    pd.testing.assert_frame_equal(base, after)


def test_clean_prices_applied_by_load(monkeypatch):
    idx = pd.bdate_range("2015-01-01", periods=30)
    good = pd.DataFrame({
        "open": 100.0, "high": 100.0, "low": 100.0,
        "close": 100.0, "adj_close": 100.0, "volume": 1.0,
    }, index=idx)
    spiked = good.copy()
    spiked.loc[idx[15], "adj_close"] = 5.0   # impossible 20x collapse - must be repaired

    def fake_loader(symbols, refresh=False):
        return {s.upper(): (spiked if s.upper() == SYMBOL else good).copy() for s in symbols}

    monkeypatch.setattr(ug, "load_yahoo_ohlcv", fake_loader)
    px, _ = ug.load_data()
    assert px[SYMBOL].loc[idx[15]] == 100.0   # spike guarded away (ffilled from 100)
    assert px[SYMBOL].min() == 100.0


def _fixed_data():
    idx = pd.bdate_range("2008-06-01", "2020-12-31")
    rng = np.random.default_rng(42)
    px = pd.DataFrame({SYMBOL: 100.0 * np.cumprod(1 + rng.normal(2e-4, 0.011, len(idx)))}, index=idx)
    gspc = pd.Series(100.0 * np.cumprod(1 + rng.normal(3e-4, 0.010, len(idx))), index=idx)
    return px, gspc


def test_determinism_bit_identical():
    px, gspc = _fixed_data()
    a = evaluate(px, gspc)
    b = evaluate(px, gspc)
    assert a["test"].equals(b["test"])
    assert (a["z10"], a["z20"], a["promoted"]) == (b["z10"], b["z20"], b["promoted"])
    assert a["bh"] == b["bh"] and a["disclosure"] == b["disclosure"]
