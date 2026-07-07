import numpy as np
import pandas as pd
import pytest

from quantlab.optimization import (
    erc_weights,
    hrp_weights,
    max_div_weights,
    min_corr_weights,
    rolling_construction,
)


def _rand_cov(n: int = 6, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3 * n, n))
    names = [f"A{i}" for i in range(n)]
    return pd.DataFrame(a.T @ a / (3 * n), index=names, columns=names)


@pytest.mark.parametrize("fn", [erc_weights, max_div_weights, hrp_weights, min_corr_weights])
def test_constructors_long_only_fully_invested(fn) -> None:
    w = fn(_rand_cov())

    assert w.min() >= -1e-9
    assert w.sum() == pytest.approx(1.0)


def test_erc_diagonal_reduces_to_inverse_vol() -> None:
    var = np.array([0.04, 0.09, 0.16, 0.25])
    cov = pd.DataFrame(np.diag(var), index=list("ABCD"), columns=list("ABCD"))
    iv = 1.0 / np.sqrt(var)

    w = erc_weights(cov)

    assert np.allclose(w.to_numpy(), iv / iv.sum(), atol=1e-6)


def test_erc_equalizes_risk_contributions() -> None:
    cov = pd.DataFrame([[0.04, 0.012], [0.012, 0.09]], index=["A", "B"], columns=["A", "B"])

    w = erc_weights(cov).to_numpy()
    rc = w * (cov.to_numpy() @ w)

    assert rc[0] == pytest.approx(rc[1], abs=1e-6)


def test_hrp_overweights_low_variance_cluster() -> None:
    # assets 0-2: tight low-vol cluster; 3-4: tight high-vol cluster; weak across
    corr = np.full((5, 5), 0.1)
    corr[:3, :3] = 0.9
    corr[3:, 3:] = 0.9
    np.fill_diagonal(corr, 1.0)
    vol = np.array([0.1, 0.1, 0.1, 0.3, 0.3])
    cov = pd.DataFrame(corr * np.outer(vol, vol), index=list("ABCDE"), columns=list("ABCDE"))

    w = hrp_weights(cov)

    assert w.min() >= -1e-9
    assert w.sum() == pytest.approx(1.0)
    assert w.iloc[:3].sum() > 0.6  # beats the 3/5 equal-weight share


@pytest.mark.parametrize("method", ["erc", "max_div", "hrp", "min_corr"])
def test_rolling_construction_sparse_month_end_targets(method) -> None:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2024-01-01", periods=100)
    rets = rng.normal(0.0004, 0.01, size=(100, 3))
    prices = pd.DataFrame(
        100 * np.exp(rets.cumsum(axis=0)), index=idx, columns=["A", "B", "C"]
    )

    w = rolling_construction(prices, method, lookback=40, rebalance="ME")

    assert w.shape == prices.shape
    assert list(w.columns) == list(prices.columns)
    month_ends = prices.groupby(pd.Grouper(freq="ME")).tail(1).index
    filled = w.dropna(how="all")
    assert len(filled) > 0
    assert filled.index.isin(month_ends).all()
    assert np.allclose(filled.sum(axis=1), 1.0)
    assert (filled.to_numpy() >= -1e-9).all()
    assert w.drop(index=filled.index).isna().all().all()
