import numpy as np
import pandas as pd
import pytest

from quantlab.optimization import (
    erc_weights_fast,
    max_sharpe_weights,
    min_variance_weights,
    rolling_mvo_weights,
)


def test_erc_fast_equalizes_risk_contributions_and_scales():
    rng = np.random.default_rng(1)
    for n in (5, 25, 150):
        idx = [f"a{i}" for i in range(n)]
        a = rng.normal(size=(n, n))
        cov = pd.DataFrame(a @ a.T / n + np.eye(n) * 0.1, columns=idx, index=idx)
        w = erc_weights_fast(cov)
        assert w.sum() == pytest.approx(1.0)
        assert (w > 0).all()                       # long-only, fully invested
        m = cov.to_numpy() @ w.to_numpy()
        rc = w.to_numpy() * m / (w.to_numpy() @ m)  # risk contributions
        assert rc.std() < 1e-6                     # equal risk contribution


def test_min_variance_weights_respect_constraints() -> None:
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.16]],
        columns=["AAA", "BBB", "CCC"],
        index=["AAA", "BBB", "CCC"],
    )

    weights = min_variance_weights(cov, max_weight=0.6)

    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= 0.0
    assert weights.max() <= 0.6 + 1e-8


def test_max_sharpe_weights_prefer_better_risk_adjusted_asset() -> None:
    mu = pd.Series({"AAA": 0.12, "BBB": 0.04})
    cov = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]],
        columns=["AAA", "BBB"],
        index=["AAA", "BBB"],
    )

    weights = max_sharpe_weights(mu, cov, max_weight=0.8)

    assert weights["AAA"] > weights["BBB"]
    assert weights.sum() == pytest.approx(1.0)


def test_rolling_mvo_waits_for_lookback() -> None:
    prices = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105],
            "BBB": [100, 100, 101, 101, 102, 102],
        },
        index=pd.date_range("2024-01-01", periods=6),
        dtype=float,
    )

    weights = rolling_mvo_weights(
        prices,
        "min_variance",
        lookback=2,
        max_weight=0.8,
        rebalance="2D",
    )

    assert weights.iloc[0].sum() == 0.0
    assert weights.iloc[-1].sum() == pytest.approx(1.0)
    assert weights.iloc[-1].max() <= 0.8 + 1e-8


def test_rolling_mvo_leaves_non_rebalance_rows_empty() -> None:
    prices = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106],
            "BBB": [100, 100, 101, 101, 102, 102, 103],
        },
        index=pd.date_range("2024-01-01", periods=7),
        dtype=float,
    )

    weights = rolling_mvo_weights(
        prices,
        "min_variance",
        lookback=2,
        max_weight=0.8,
        rebalance="3D",
    )

    assert weights.iloc[2].sum() == pytest.approx(1.0)
    assert weights.iloc[3].isna().all()
