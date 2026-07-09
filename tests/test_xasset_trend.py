"""RL-2026-07-17 multi-asset ETF trend sleeve: causality + construction guards.

Synthetic, fast, no network. Covers the properties the study relies on: gate
causality (no look-ahead), the long-only weight arithmetic (gated-off = 0, book
sums to <=1, inverse-vol normalization), the spike-repair data cleaning, and
MON100-style pre-inception handling (leading-NaN column -> zero weight, no crash).
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quantlab.backtest import backtest_weights
from quantlab.features import rolling_vol
from quantlab.portfolio import rebalance_targets
from quantlab.xasset_trend import _gate, clean_prices, sleeve_weights


def _prices(n=400, cols=("A", "B", "C"), seed=0, drift=0.0005, vol=0.01):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    steps = np.random.default_rng(seed).normal(drift, vol, (n, len(cols)))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + steps, axis=0), index=idx, columns=list(cols))


@pytest.mark.parametrize("gate", ["tsmom", "ma"])
@pytest.mark.parametrize("weighting", ["equal", "invvol"])
def test_gate_causal_no_lookahead(gate, weighting):
    """Perturbing prices strictly after date t must not change the target at/before t."""
    px = _prices()
    cut = px.index[300]
    px2 = px.copy()
    px2.iloc[301:] *= 3.0  # arbitrary future shock (level jump + trend flip)

    w = sleeve_weights(px, gate, weighting)
    w2 = sleeve_weights(px2, gate, weighting)
    pdt.assert_frame_equal(w.loc[:cut], w2.loc[:cut])


@pytest.mark.parametrize("gate", ["tsmom", "ma"])
@pytest.mark.parametrize("weighting", ["equal", "invvol"])
def test_long_only_gated_off_zero_and_sum_le_one(gate, weighting):
    px = _prices()
    w = sleeve_weights(px, gate, weighting)
    warm = w.loc[w.index[300]:]

    assert (warm.to_numpy() >= -1e-12).all()          # long-only, no shorts
    assert (warm.sum(axis=1).to_numpy() <= 1.0 + 1e-9).all()  # never levered

    sig = _gate(px, gate)
    off = (sig <= 0) & sig.notna()                    # trend down but computable
    assert (w.where(off).fillna(0.0).abs().to_numpy() <= 1e-12).all()


def test_invvol_weights_proportional_to_inverse_vol():
    """Held (nonzero) inverse-vol weights are proportional to 1/vol, so the lowest-vol
    asset gets the largest share. A strong uptrend keeps every gate on -> the book is
    fully invested and its weights sum to 1."""
    idx = pd.date_range("2015-01-01", periods=400, freq="B")
    rng = np.random.default_rng(1)
    vols = {"A": 0.004, "B": 0.010, "C": 0.020}
    px = pd.DataFrame({c: 100.0 * np.cumprod(1.0 + rng.normal(0.004, v, 400)) for c, v in vols.items()},
                      index=idx)
    w = sleeve_weights(px, "ma", "invvol").iloc[-1]
    nz = w[w > 0]
    assert len(nz) == 3 and abs(float(w.sum()) - 1.0) < 1e-9   # strong trend -> all on

    inv = 1.0 / rolling_vol(px, 126).iloc[-1]
    ratio = nz / inv[nz.index]
    assert float(ratio.std() / ratio.mean()) < 1e-9           # w_i proportional to 1/vol_i
    assert w["A"] > w["B"] > w["C"]                            # lowest vol gets the most


def test_clean_prices_repairs_roundtrip_glitch_causally():
    """A two-day bad-print round-trip (price -> ~1/10 -> snap back) is removed with no
    residual level shift, and cleaning never uses future bars."""
    px = _prices(cols=("A",), seed=3)
    good = px["A"].iloc[210]
    px.iloc[210:212] = good / 10.0                    # 2-day collapse then snap back

    cleaned = clean_prices(px)
    assert (cleaned["A"].pct_change().abs().iloc[205:220] < 0.5).all()   # glitch gone
    assert abs(cleaned["A"].iloc[213] - px["A"].iloc[213]) < 1e-9        # no level shift after

    px2 = px.copy()
    px2.iloc[300:] *= 5.0                             # future shock
    pdt.assert_series_equal(clean_prices(px)["A"].iloc[:300], clean_prices(px2)["A"].iloc[:300])


@pytest.mark.parametrize("gate", ["tsmom", "ma"])
@pytest.mark.parametrize("weighting", ["equal", "invvol"])
def test_pre_inception_nan_zero_weight_no_crash(gate, weighting):
    """A late-listing asset (leading-NaN column, like MON100) holds zero weight while
    absent and never dilutes the assets that do trade."""
    px = _prices(cols=("A", "B", "MON"))
    px.iloc[:200, px.columns.get_loc("MON")] = np.nan

    w = sleeve_weights(px, gate, weighting)           # must not raise
    assert (w["MON"].iloc[:200].abs() <= 1e-12).all()
    assert w[["A", "B"]].iloc[300:].sum(axis=1).max() > 0.0

    res = backtest_weights(px, rebalance_targets(sleeve_weights(px, "tsmom", "invvol"), "ME"), 20.0)
    assert np.isfinite(res.returns.to_numpy()).all()
