"""Synthetic tests for the turn-of-month study (no network).

Cover the three load-bearing mechanics: window selection on an irregular hand-made
calendar, the weight/return alignment (which days' returns actually land in the book),
and the round-trip cost accounting (2 sides per trip).
"""

import numpy as np
import pandas as pd

from quantlab.backtest import backtest_weights
from quantlab.tom_study import SYMBOL, tom_weights, tom_window


def test_window_selection_irregular_calendar():
    # Two months with irregular trading days (weekends/holidays skipped, unequal counts).
    month_a = ["2021-01-04", "2021-01-06", "2021-01-08", "2021-01-12",
               "2021-01-15", "2021-01-20", "2021-01-27"]  # 7 days
    month_b = ["2021-02-02", "2021-02-05", "2021-02-09", "2021-02-11",
               "2021-02-16", "2021-02-22", "2021-02-25", "2021-02-26"]  # 8 days
    idx = pd.DatetimeIndex(pd.to_datetime(month_a + month_b))

    win = tom_window(idx, n=2, m=3)
    expected = {
        "2021-01-04", "2021-01-06", "2021-01-08",  # first 3 of A
        "2021-01-20", "2021-01-27",                # last 2 of A
        "2021-02-02", "2021-02-05", "2021-02-09",  # first 3 of B
        "2021-02-25", "2021-02-26",                # last 2 of B
    }
    got = {d.strftime("%Y-%m-%d") for d in idx[win.values]}
    assert got == expected


def _price_frame(idx, factors):
    px = 100.0 * np.cumprod(factors)
    return pd.DataFrame({SYMBOL: px}, index=idx)


def test_alignment_only_window_returns_land_in_book():
    idx = pd.bdate_range("2021-01-01", "2021-03-31")
    n, m, r = 2, 3, 0.01
    win = tom_window(idx, n, m).values

    # Prices where the daily return is +1% exactly on window days, 0 elsewhere.
    factors = np.where(np.r_[False, win[1:]], 1.0 + r, 1.0)
    px = _price_frame(idx, factors)
    book = backtest_weights(px, tom_weights(px, n, m), cost_bps=0.0)

    n_win_earned = int(win[1:].sum())  # first bar has no return to earn
    assert np.isclose(book.equity.iloc[-1], (1.0 + r) ** n_win_earned)

    # Converse: returns only OFF the window -> nothing leaks into the book.
    factors_off = np.where(np.r_[False, ~win[1:]], 1.0 + r, 1.0)
    px_off = _price_frame(idx, factors_off)
    book_off = backtest_weights(px_off, tom_weights(px_off, n, m), cost_bps=0.0)
    assert np.isclose(book_off.equity.iloc[-1], 1.0)


def test_cost_two_sides_per_round_trip():
    # One interior round trip: flat prices, weights 0->1 (hold) ->0.
    idx = pd.bdate_range("2021-01-01", periods=7)
    px = pd.DataFrame({SYMBOL: 100.0}, index=idx)
    w = pd.DataFrame({SYMBOL: [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]}, index=idx)
    book = backtest_weights(px, w, cost_bps=20.0)
    assert np.isclose(book.turnover.sum(), 2.0)                       # 1 entry + 1 exit
    assert np.isclose(book.returns.sum(), -2.0 * 20.0 / 10_000)      # gross 0, cost 2 sides

    # The real TOM weights charge exactly 2 sides per contiguous held block.
    idx2 = pd.bdate_range("2021-01-01", "2021-06-30")
    px2 = pd.DataFrame({SYMBOL: 100.0}, index=idx2)
    wv = tom_weights(px2, 2, 3)[SYMBOL]
    entries = int(((wv == 1.0) & (wv.shift(1, fill_value=0.0) == 0.0)).sum())
    exits = int(((wv == 0.0) & (wv.shift(1, fill_value=0.0) == 1.0)).sum())
    book2 = backtest_weights(px2, tom_weights(px2, 2, 3), cost_bps=20.0)
    assert np.isclose(book2.turnover.sum(), entries + exits)
