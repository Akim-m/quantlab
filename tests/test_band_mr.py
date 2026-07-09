"""Synthetic tests for RL-2026-07-23 index-band mean reversion (no network).

Covers the four load-bearing mechanics: z-score causality (today's held position uses
prior-day info only), the entry/exit state machine for both exit rules on hand-made
paths, the weight/return alignment (only in-position D+1 days earn), and the Ledoit-Wolf
Sharpe-difference statistic (identical series -> 0, a known gap -> a sensible, antisymmetric z).
"""

import numpy as np
import pandas as pd

from quantlab.backtest import backtest_weights
from quantlab.band_mr import (
    SYMBOL,
    _positions,
    band_weights,
    sharpe_diff_test,
    zscore,
)


def _walk(n, seed, scale=0.02):
    rng = np.random.default_rng(seed)
    px = 100.0 * np.cumprod(1.0 + rng.normal(0.0, scale, n))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({SYMBOL: px}, index=idx)


def test_state_machine_exit_rules():
    # Enter at i=0 (z<-2), then z stays below the mean forever: mean-touch never exits,
    # the 10-day stop cuts after exactly 10 held bars.
    z = np.array([-2.5] + [-1.0] * 12)
    mean = _positions(z, k=2.0, exit_rule="mean")
    stop = _positions(z, k=2.0, exit_rule="stop10")
    assert list(mean) == [1.0] * 13
    assert list(stop) == [1.0] * 10 + [0.0] * 3

    # Mean-touch exits the bar z first reaches 0, then re-enters on the next z<-k.
    z2 = np.array([-1.0, -2.5, -1.5, -0.5, 0.3, -3.0, -2.2, -2.1])
    assert list(_positions(z2, k=2.0, exit_rule="mean")) == [0, 1, 1, 1, 0, 1, 1, 1]


def test_entry_threshold_k():
    z = np.array([-1.7, -1.7])
    assert _positions(z, k=1.5, exit_rule="mean")[0] == 1.0   # -1.7 < -1.5 enters
    assert _positions(z, k=2.0, exit_rule="mean")[0] == 0.0   # -1.7 !< -2.0 stays flat


def test_zscore_causality_prior_day_only():
    # Perturbing price on day i cannot change any position held on or before day i.
    px = _walk(220, seed=1)
    i = 130
    w0 = band_weights(px, k=2.0, exit_rule="mean")

    px2 = px.copy()
    px2.iloc[i, 0] *= 0.7  # a 30% down-print at day i
    w1 = band_weights(px2, k=2.0, exit_rule="mean")

    # weights row i-1 is the position carried INTO day i (backtest_weights lags one day),
    # so the whole [:i] prefix - including day i's held position - must be untouched.
    assert w0.iloc[:i].equals(w1.iloc[:i])
    # ...while the perturbation does propagate forward (test is not vacuous).
    assert not w0.iloc[i:].equals(w1.iloc[i:])
    # z itself uses only trailing prices: z_{i-1} unchanged, z_i changed by the print.
    z0, z1 = zscore(px[SYMBOL]), zscore(px2[SYMBOL])
    assert np.allclose(z0.to_numpy()[:i], z1.to_numpy()[:i], equal_nan=True)


def test_weight_return_alignment():
    # Only days preceded by an in-position weight earn, and they earn exactly that day's
    # market return: book.returns[d] == w[d-1] * ret[d]. Also confirms the D->D+1 timing.
    px = _walk(400, seed=7)
    w = band_weights(px, k=2.0, exit_rule="mean")
    assert w[SYMBOL].sum() > 0  # non-vacuous: the path actually trades

    book = backtest_weights(px, w, cost_bps=0.0)
    expected = w[SYMBOL].shift(1).fillna(0.0) * px[SYMBOL].pct_change().fillna(0.0)
    assert np.allclose(book.returns.to_numpy(), expected.to_numpy())
    # Flat days contribute nothing.
    held = w[SYMBOL].shift(1).fillna(0.0)
    assert np.allclose(book.returns[held == 0.0].to_numpy(), 0.0)


def test_sharpe_diff_identical_series_zero():
    r = np.random.default_rng(3).normal(0.0005, 0.01, 1500)
    sr1, sr2, z = sharpe_diff_test(r, r)
    assert sr1 == sr2
    assert z == 0.0


def test_sharpe_diff_known_gap_and_antisymmetry():
    rng = np.random.default_rng(11)
    hi = rng.normal(0.0008, 0.01, 2000)
    lo = rng.normal(0.0000, 0.01, 2000)
    sr_hi, sr_lo, z = sharpe_diff_test(hi, lo)
    assert sr_hi > sr_lo
    assert z > 1.5                                   # the better series is detectably better
    _, _, z_swap = sharpe_diff_test(lo, hi)
    assert np.isclose(z_swap, -z)                    # swapping arms flips the sign exactly
