"""RL-2026-07-19 three-book blend: weight-rule arithmetic + TRAIN-only derivation.

Synthetic, fast, no network. Covers the pieces the sizing decision rests on: the four
weight rules' arithmetic (equal thirds; inverse-vol proportional to 1/std; ERC summing
to 1 and matching the closed-form uncorrelated-asset case where ERC = inverse-vol; the
regime baseline), the fixed-weight blend return arithmetic, and the guarantee that every
data-derived rule reads only the TRAIN slice (perturbing post-split returns must not move
the weights).
"""

import numpy as np
import pandas as pd

from quantlab.blend_portfolio import BOOKS, TRAIN0, TRAIN1, blend_ret, rule_weights
from quantlab.optimization import erc_weights


def _orthogonal_returns(scales, k=100):
    """Return frame whose columns are mutually orthogonal, mean-zero, so the empirical
    covariance is EXACTLY diagonal with vol_i proportional to scales[i]. Built by tiling a
    4-row Hadamard block; inverse-vol and ERC then both reduce to normalize(1/scale)."""
    block = np.array([[1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]], dtype=float).T
    pat = np.tile(block, (k, 1)) * np.asarray(list(scales.values()))
    idx = pd.date_range("2015-01-01", periods=len(pat), freq="B")
    return pd.DataFrame(pat, index=idx, columns=list(scales.keys()))


def test_equal_thirds():
    w = rule_weights("thirds", _orthogonal_returns({b: 0.01 for b in BOOKS}))
    assert list(w.index) == list(BOOKS)
    np.testing.assert_allclose(w.to_numpy(), 1.0 / 3.0)


def test_regime_baseline_is_all_regime():
    w = rule_weights("regime", _orthogonal_returns({b: 0.01 for b in BOOKS}))
    assert w["regime"] == 1.0 and w.drop("regime").abs().sum() == 0.0


def test_invvol_proportional_to_inverse_vol():
    """Weights proportional to 1/std, summing to 1, lowest-vol book gets the most."""
    scales = {"regime": 0.04, "ls": 0.01, "trend": 0.02}
    w = rule_weights("invvol", _orthogonal_returns(scales))
    expected = np.array([1 / 0.04, 1 / 0.01, 1 / 0.02])
    expected /= expected.sum()
    np.testing.assert_allclose(w.reindex(BOOKS).to_numpy(), expected, atol=1e-9)
    assert abs(float(w.sum()) - 1.0) < 1e-12
    assert w["ls"] > w["trend"] > w["regime"]


def test_erc_matches_inverse_vol_when_uncorrelated():
    """Hand-checkable ERC case: for uncorrelated assets, risk contribution_i = w_i^2 var_i,
    so equalizing gives w_i proportional to 1/vol_i. On the exactly-diagonal frame ERC must
    equal the inverse-vol weights and sum to 1."""
    scales = {"regime": 0.04, "ls": 0.01, "trend": 0.02}
    ret = _orthogonal_returns(scales)
    w = rule_weights("erc", ret)
    inv = 1.0 / ret.std()
    np.testing.assert_allclose(w.reindex(BOOKS).to_numpy(), (inv / inv.sum()).reindex(BOOKS).to_numpy(), atol=1e-4)
    assert abs(float(w.sum()) - 1.0) < 1e-6


def test_erc_primitive_two_and_three_asset_diagonal():
    """The reused primitive on a raw diagonal covariance: ERC = inverse-vol, sums to 1."""
    cov2 = pd.DataFrame(np.diag([0.04, 0.01]), index=["x", "y"], columns=["x", "y"])
    w2 = erc_weights(cov2)
    np.testing.assert_allclose(w2.to_numpy(), [1 / 3, 2 / 3], atol=1e-4)

    cov3 = pd.DataFrame(np.diag([0.01, 0.04, 0.09]), index=["a", "b", "c"], columns=["a", "b", "c"])
    w3 = erc_weights(cov3)
    inv = np.array([1 / 0.1, 1 / 0.2, 1 / 0.3])
    np.testing.assert_allclose(w3.to_numpy(), inv / inv.sum(), atol=1e-4)
    assert abs(float(w3.sum()) - 1.0) < 1e-6


def test_blend_return_arithmetic():
    """Fixed-weight blend is the exact weighted row-sum of the book returns."""
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    ret = pd.DataFrame({"regime": [0.02, -0.01, 0.00],
                        "ls": [0.00, 0.03, -0.02],
                        "trend": [0.01, 0.01, 0.04]}, index=idx)
    w = pd.Series({"regime": 0.5, "ls": 0.3, "trend": 0.2})
    got = blend_ret(ret, w)
    expected = pd.Series([0.5 * 0.02 + 0.2 * 0.01,
                          0.5 * -0.01 + 0.3 * 0.03 + 0.2 * 0.01,
                          0.3 * -0.02 + 0.2 * 0.04], index=idx)
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_weights_use_train_slice_only():
    """Perturbing post-split returns must not change any data-derived rule's weights."""
    scales = {"regime": 0.04, "ls": 0.01, "trend": 0.02}
    base = _orthogonal_returns(scales, k=200)  # 2015-01 -> ~2018-01, crosses the split

    perturbed = base.copy()
    post = perturbed.index > TRAIN1
    perturbed.loc[post] += np.random.default_rng(0).normal(0.0, 5.0, (post.sum(), len(BOOKS)))

    for rule in ("invvol", "erc"):
        w_a = rule_weights(rule, base.loc[TRAIN0:TRAIN1])
        w_b = rule_weights(rule, perturbed.loc[TRAIN0:TRAIN1])
        pd.testing.assert_series_equal(w_a, w_b)
