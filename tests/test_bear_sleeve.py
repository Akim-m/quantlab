"""RL-2026-07-13 bear-only reversal sleeve: causality + combination guards.

Synthetic, fast, no network. Covers the three properties the study relies on:
gate causality (no look-ahead), the returns-level combination arithmetic, and
zero sleeve P&L on interior risk-on days.
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from quantlab.backtest import backtest_weights
from quantlab.bear_sleeve import bear_gated_targets
from quantlab.blend import market_on
from quantlab.evaluation import sharpe_tstat


def _dates(n):
    return pd.date_range("2015-01-01", periods=n, freq="B")


def _trending_mkt(n=420, seed=0):
    """Market that spends a stretch above its 200d MA (bull) then below it (bear),
    so market_on(mkt, 200) toggles within the sample."""
    idx = _dates(n)
    ramp = np.concatenate([np.linspace(100, 200, 300), np.linspace(200, 120, n - 300)])
    noise = np.random.default_rng(seed).normal(0, 0.3, n)
    return pd.Series(ramp + noise, index=idx)


def _sleeve_weights(cols, idx, seed=1):
    """A daily dollar-neutral, unit-gross L/S weight matrix (hand-made, no warmup)."""
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
    demeaned = raw.sub(raw.mean(axis=1), axis=0)
    gross = demeaned.abs().sum(axis=1)
    return demeaned.div(gross, axis=0)


def test_gate_is_causal_no_lookahead():
    """The gated target at date t must not depend on any market price after t."""
    mkt = _trending_mkt()
    cols = ["A", "B", "C", "D"]
    sleeve = _sleeve_weights(cols, mkt.index)

    tg = bear_gated_targets(sleeve, "W-FRI", mkt)
    cut = mkt.index[250]
    mkt2 = mkt.copy()
    mkt2.loc[mkt2.index[251]:] = 1.0  # crash the entire future -> flips it to bear
    tg2 = bear_gated_targets(sleeve, "W-FRI", mkt2)

    # A gate that peeked ahead would see the flipped future and change targets at/before
    # cut; a causal gate leaves them bit-identical.
    pdt.assert_frame_equal(tg.loc[:cut], tg2.loc[:cut])


def test_combination_additive_and_paired_t_scale_invariant():
    """The returns-level combination is exactly additive, and the paired-t on
    (combined - base) that run() reports equals the sleeve's own t-stat regardless
    of the size s (a positive scalar cancels in the t-stat)."""
    idx = _dates(200)
    rng = np.random.default_rng(2)
    r_base = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    r_sleeve = pd.Series(rng.normal(0.0, 0.02, len(idx)), index=idx)
    t_sleeve = sharpe_tstat(r_sleeve)[1]
    for s in (0.10, 0.20):
        r_comb = r_base + s * r_sleeve
        pdt.assert_series_equal(r_comb - r_base, s * r_sleeve)
        assert abs(sharpe_tstat(r_comb - r_base)[1] - t_sleeve) < 1e-9


def test_zero_sleeve_pnl_on_interior_riskon_days():
    """On risk-on days the sleeve target is flat, so interior bull days (not the
    flip day that pays the exit cost) produce exactly zero sleeve P&L."""
    mkt = _trending_mkt()
    cols = ["A", "B", "C", "D"]
    idx = mkt.index
    rng = np.random.default_rng(3)
    prices = pd.DataFrame(100 * (1 + rng.normal(0, 0.01, (len(idx), len(cols)))).cumprod(axis=0),
                          index=idx, columns=cols)
    sleeve = _sleeve_weights(cols, idx)

    tg = bear_gated_targets(sleeve, "W-FRI", mkt)
    r = backtest_weights(prices, tg, cost_bps=20.0).returns

    on = market_on(mkt, 200).reindex(idx).fillna(False)
    interior_bull = on & on.shift(1, fill_value=False)  # risk-on today and yesterday
    warm = idx > idx[210]                                # past MA warmup
    mask = (interior_bull & warm).to_numpy()
    assert mask.sum() > 20                               # the fixture actually has such days
    assert np.allclose(r[mask].to_numpy(), 0.0, atol=1e-12)
