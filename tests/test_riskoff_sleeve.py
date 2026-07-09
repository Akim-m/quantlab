"""RL-2026-07-16 risk-off sleeve: causality + construction guards.

Synthetic, fast, no network. Covers the three properties the study relies on:
gate/selection causality (no look-ahead), the freed-weight arithmetic (the sleeve
weight is exactly 1 - base gross, day by day), and that the sleeve's turnover is
actually costed by the combined backtest.
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from quantlab.backtest import backtest_weights
from quantlab.portfolio import rebalance_targets
from quantlab.riskoff_sleeve import GOLD, fill_freed, low_beta_book, sleeve_book


def _panel(n=320, k=5, seed=0):
    idx = pd.date_range("2013-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    px = pd.DataFrame(100 * (1 + rng.normal(0, 0.01, (n, k))).cumprod(axis=0),
                      index=idx, columns=[f"S{i}" for i in range(k)])
    mkt = pd.Series(100 * (1 + rng.normal(0.0003, 0.009, n)).cumprod(), index=idx)
    gold = pd.Series(100 * (1 + rng.normal(0.0002, 0.008, n)).cumprod(), index=idx)
    return px, mkt, gold


def test_sleeve_selection_is_causal_no_lookahead():
    """The sleeve target at date t (gold trend gate AND low-beta decile) must not
    depend on any price after t. Perturbing the entire future must leave every
    target on or before the cut bit-identical."""
    px, mkt, gold = _panel()
    cols = list(px.columns) + [GOLD]
    cut = px.index[300]

    tg = sleeve_book("gold_lowbeta", px, mkt, gold, cols)

    px2, mkt2, gold2 = px.copy(), mkt.copy(), gold.copy()
    fut = px.index[301:]
    # Overwrite the whole future with a flat extreme constant (zero future returns,
    # extreme level): a trailing rolling stat is untouched, but any centered/forward
    # window or one-day peek in the gate/beta/vol would move a pre-cut target.
    px2.loc[fut] = 1.0
    mkt2.loc[fut] = 1.0
    gold2.loc[fut] = 1.0
    tg2 = sleeve_book("gold_lowbeta", px2, mkt2, gold2, cols)

    pdt.assert_frame_equal(tg.loc[:cut], tg2.loc[:cut])


def test_low_beta_book_sums_to_one_after_warmup():
    """Premise for the freed-weight arithmetic: the low-beta sleeve is fully
    invested (rows sum to 1) once the 252d beta window is warm."""
    px, mkt, _ = _panel()
    lb = low_beta_book(px, mkt)
    warm = lb.loc[px.index[260]:]
    assert np.allclose(warm.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_freed_weight_arithmetic():
    """The sleeve occupies exactly the freed weight (1 - base gross), clipped at 0,
    day by day - never a hardcoded 0.5. Checked against a base whose gross sweeps
    0 (risk-off) -> 1 (risk-on) -> partial drift, with a fully-invested sleeve."""
    idx = pd.date_range("2015-01-01", periods=6, freq="B")
    cols = ["A", "B", GOLD]
    base_aug = pd.DataFrame(0.0, index=idx, columns=cols)
    gross = [0.0, 1.0, 0.5, 0.3, 0.0, 1.0]
    base_aug["A"] = gross                                   # all base weight in A
    sleeve = pd.DataFrame(0.0, index=idx, columns=cols)
    sleeve["B"], sleeve[GOLD] = 0.4, 0.6                    # sleeve sums to 1 each day

    combined = fill_freed(base_aug, sleeve)
    freed = combined.sub(base_aug, fill_value=0.0).sum(axis=1)   # weight routed to the sleeve
    expected = (1.0 - base_aug.sum(axis=1)).clip(lower=0.0)
    pdt.assert_series_equal(freed, expected, check_names=False)
    assert np.allclose(combined.sum(axis=1).to_numpy(), [1, 1, 1, 1, 1, 1], atol=1e-12)


def test_sleeve_turnover_is_costed():
    """A regime flip from the base book into the sleeve, and the sleeve's monthly
    refresh, must pay turnover: the flip date shows real trading and a higher cost
    strictly lowers the combined return."""
    n = 120
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    cols = ["S0", "S1", "S2", GOLD]
    prices = pd.DataFrame(100 * (1 + rng.normal(0, 0.01, (n, len(cols)))).cumprod(axis=0),
                          index=idx, columns=cols)

    base_aug = pd.DataFrame(0.0, index=idx, columns=cols)
    riskon = idx <= idx[24]                       # first ~month risk-on: base holds S2
    base_aug.loc[riskon, "S2"] = 1.0              # risk-off afterwards: base is cash
    sleeve = pd.DataFrame(0.0, index=idx, columns=cols)
    sleeve["S0"], sleeve["S1"] = 0.5, 0.5         # sleeve fills the freed weight when risk-off

    targets = rebalance_targets(fill_freed(base_aug, sleeve), "ME")
    res0 = backtest_weights(prices, targets, 0.0)
    res40 = backtest_weights(prices, targets, 40.0)

    flip = res0.turnover.index[(res0.turnover.index > idx[24]) & res0.turnover.index.isin(targets.dropna(how="all").index)][0]
    assert res0.turnover.loc[flip] > 0.5          # base S2 -> sleeve S0/S1 is a real trade
    assert res0.turnover.sum() > 0.5
    assert res40.returns.sum() < res0.returns.sum()   # sleeve turnover pays cost
