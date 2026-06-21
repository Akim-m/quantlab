import pandas as pd
import pytest

from quantlab.backtest import backtest_weights


def test_backtest_uses_prior_day_weights() -> None:
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 121.0],
            "BBB": [100.0, 100.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )
    weights = pd.DataFrame(
        {"AAA": [1.0, 1.0, 1.0], "BBB": [0.0, 0.0, 0.0]},
        index=prices.index,
    )

    res = backtest_weights(prices, weights)

    assert res.returns.tolist() == pytest.approx([0.0, 0.1, 0.1])
    assert res.total_return == pytest.approx(0.21)


def test_backtest_charges_turnover_costs() -> None:
    prices = pd.DataFrame(
        {"AAA": [100.0, 100.0], "BBB": [100.0, 100.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    weights = pd.DataFrame(
        {"AAA": [1.0, 0.0], "BBB": [0.0, 1.0]},
        index=prices.index,
    )

    res = backtest_weights(prices, weights, cost_bps=10)

    assert res.returns.tolist() == pytest.approx([-0.001, -0.002])


def test_backtest_counts_rebalance_turnover_from_drift() -> None:
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0], "BBB": [100.0, 100.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    weights = pd.DataFrame(
        {"AAA": [0.5, 0.5], "BBB": [0.5, 0.5]},
        index=prices.index,
    )

    res = backtest_weights(prices, weights)

    assert res.returns.tolist() == pytest.approx([0.0, 0.05])
    assert res.turnover.tolist() == pytest.approx([1.0, 0.0476190476])


def test_backtest_does_not_rebalance_on_missing_target_rows() -> None:
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0], "BBB": [100.0, 100.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    weights = pd.DataFrame(
        {"AAA": [0.5, None], "BBB": [0.5, None]},
        index=prices.index,
    )

    res = backtest_weights(prices, weights)

    assert res.returns.tolist() == pytest.approx([0.0, 0.05])
    assert res.turnover.tolist() == pytest.approx([1.0, 0.0])
    assert res.weights.iloc[-1].tolist() == pytest.approx([0.5238095238, 0.4761904762])
