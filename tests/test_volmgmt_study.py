"""RL-2026-07-21 vol-target overlay: causality + construction guards.

Synthetic, fast, no network. Covers the four properties the study relies on:
sigma_hat uses prior-day info only (no look-ahead), the min(1,.) cap never levers
above the base book, the scaled targets are the base weights scaled row-wise, and
a changing s_t is actually costed as turnover even when the underlying weights are
frozen.
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from quantlab.backtest import backtest_weights
from quantlab.volmgmt_study import scale_factor, sigma_hat


def _ref(n=200, seed=0):
    idx = pd.date_range("2013-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0004, 0.011, n), index=idx)


def test_sigma_hat_causality_prior_day_only():
    """s_t = f(returns through t-1). Perturbing the reference return at t (or any
    later date) cannot change s_t; it only moves scales strictly after t."""
    ref = _ref()
    s = scale_factor(ref, 0.10, 21)

    k = 120
    ref2 = ref.copy()
    ref2.iloc[k:] += 0.5  # blow up the tail from position k onward
    s2 = scale_factor(ref2, 0.10, 21)

    # s at t=k depends only on ref[..k-1], so it (and everything before) is untouched.
    pdt.assert_series_equal(s.iloc[: k + 1], s2.iloc[: k + 1])
    # The very next scale (t=k+1, whose window now includes the perturbed ref[k]) moves.
    assert s.iloc[k + 1] != s2.iloc[k + 1]


def test_cap_never_levers_above_base():
    """With realized vol far below target, sigma_target/sigma_hat > 1, so the cap
    binds: s == 1 exactly and the scaled book equals the base book (never above)."""
    idx = pd.date_range("2013-01-01", periods=120, freq="B")
    calm = pd.Series(0.0001 * np.sin(np.arange(120)), index=idx)  # ~1% ann vol << 10%
    s = scale_factor(calm, 0.10, 21)

    defined = sigma_hat(calm, 21).notna()
    assert (s <= 1.0 + 1e-12).all()
    assert np.allclose(s[defined], 1.0)

    base = pd.DataFrame({"A": 0.6, "B": 0.4}, index=idx)
    scaled = base.mul(s, axis=0)
    assert (scaled <= base + 1e-12).all().all()


def test_scaling_is_rowwise_product():
    """Scaled targets = base weight row x s_t, applied row-wise across assets."""
    idx = pd.date_range("2013-01-01", periods=3, freq="B")
    base = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=idx)
    s = pd.Series([1.0, 0.8, 0.5], index=idx)

    scaled = base.mul(s, axis=0)
    expected = pd.DataFrame({"A": [0.5, 0.4, 0.25], "B": [0.5, 0.4, 0.25]}, index=idx)
    pdt.assert_frame_equal(scaled, expected)


def test_changing_scale_costs_turnover_on_frozen_weights():
    """A changing s_t re-trades the whole book and is costed, even with the
    underlying weights frozen and prices flat (so no drift, no gross return): the
    only turnover comes from the scale changes, and returns are pure cost."""
    idx = pd.date_range("2013-01-01", periods=4, freq="B")
    prices = pd.DataFrame(1.0, index=idx, columns=["A", "B"])  # flat -> zero asset returns
    base = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    s = pd.Series([1.0, 0.8, 0.8, 0.5], index=idx)

    res = backtest_weights(prices, base.mul(s, axis=0), cost_bps=20.0)

    to = res.turnover
    assert to.iloc[1] > 0 and to.iloc[3] > 0   # s changed on days 1 and 3
    assert to.iloc[2] == 0                      # s unchanged on day 2 -> no trade
    # Flat prices: every day's return is exactly the cost of that day's turnover.
    assert np.allclose(res.returns.values, -to.values * 20.0 / 1e4)

    flat = backtest_weights(prices, base.mul(0.8), cost_bps=20.0)
    assert (flat.turnover.iloc[1:] == 0).all()  # constant scale -> only the initial trade
