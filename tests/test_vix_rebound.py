"""RL-2026-07-24 VIX spike-and-recede re-entry overlay: causality + construction guards.

Synthetic, fast, no network. Covers the four properties the study relies on: the
spike/recede flags are causal (no look-ahead), the override window extends exactly h
days from a trigger, the composition only ever ADDS holding on base-cash days (never
forces cash on risk-on days), and the Sharpe-difference statistic is well-behaved.
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt

from quantlab.vix_rebound import overlaid_book, override_on, sharpe_diff_z, spike_recede


def _vix(n=400, seed=0):
    idx = pd.date_range("2013-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    return pd.Series(15 + 5 * np.abs(rng.normal(0, 1, n)).cumsum() / np.sqrt(np.arange(1, n + 1)), index=idx)


def test_spike_recede_flags_are_causal():
    """spike (trailing-252d percentile) and receding (trailing 5d max) at date t must
    not depend on any VIX after t. Overwriting the whole future leaves every flag on
    or before the cut bit-identical."""
    vix = _vix()
    cut = vix.index[350]
    trig = spike_recede(vix, 90)

    vix2 = vix.copy()
    vix2.iloc[351:] = 999.0  # extreme future: any forward peek would move a past flag
    trig2 = spike_recede(vix2, 90)

    pdt.assert_series_equal(trig.loc[:cut], trig2.loc[:cut])


def test_override_window_arithmetic():
    """A trigger arms the override for exactly h days (inclusive); a fresh trigger
    inside the window extends it; it lapses h days after the last trigger. Checked on
    a hand-made trigger path with h=3."""
    idx = pd.date_range("2015-01-01", periods=12, freq="B")
    trig = pd.Series(False, index=idx)
    trig.iloc[2] = True          # isolated trigger -> on for days 2,3,4
    trig.iloc[7] = True          # trigger, then...
    trig.iloc[8] = True          # ...fresh trigger extends -> on for 7,8,9,10

    ov = override_on(trig, 3)
    expected = pd.Series(
        [False, False, True, True, True, False, False, True, True, True, True, False],
        index=idx)
    pdt.assert_series_equal(ov, expected, check_names=False)


def test_composition_adds_holding_only_off_gate():
    """overlaid = conviction x (regime_on OR override): it must HOLD (equal the
    conviction book) on days the base gate is off but the override fires, and must
    never force cash on a risk-on day (there it equals the base regardless of the
    override)."""
    idx = pd.date_range("2015-01-01", periods=4, freq="B")
    conv = pd.DataFrame({"A": [0.6, 0.6, 0.6, 0.6], "B": [0.4, 0.4, 0.4, 0.4]}, index=idx)
    on = pd.Series([True, False, False, True], index=idx)
    override = pd.Series([False, True, False, True], index=idx)

    base = overlaid_book(conv, on, pd.Series(False, index=idx))
    combined = overlaid_book(conv, on, override)

    # day 0 (on): held in both. day 1 (off+override): base cash, combined holds.
    # day 2 (off, no override): cash in both. day 3 (on+override): held, override is a no-op.
    pdt.assert_frame_equal(combined.loc[[idx[0], idx[3]]], conv.loc[[idx[0], idx[3]]])
    assert (base.loc[idx[1]] == 0.0).all() and combined.loc[idx[1]].equals(conv.loc[idx[1]])
    assert (base.loc[idx[2]] == 0.0).all() and (combined.loc[idx[2]] == 0.0).all()


def test_sharpe_diff_z_sanity():
    """Known positive gap -> clearly positive z; identical series -> exactly 0;
    swapping the arguments negates the statistic."""
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0.0010, 0.01, 3000))
    b = pd.Series(rng.normal(0.0002, 0.01, 3000))

    z = sharpe_diff_z(a, b)
    assert z > 1.5
    assert sharpe_diff_z(a, a) == 0.0
    assert abs(sharpe_diff_z(a, b) + sharpe_diff_z(b, a)) < 1e-9
