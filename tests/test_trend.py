import numpy as np
import pandas as pd
import pytest

from quantlab.trend import donchian, dual_momentum, tsmom


def _walk(n: int = 120, cols: int = 4, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.02, size=(n, cols))
    return pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
        columns=[f"A{i}" for i in range(cols)],
    )


def _trend_panel(n: int = 80, up: float = 0.01, down: float = -0.01) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rets = np.column_stack(
        [up + 0.003 * rng.standard_normal(n), down + 0.003 * rng.standard_normal(n)]
    )
    return pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
        columns=["UP", "DOWN"],
    )


@pytest.mark.parametrize("factor", [tsmom, donchian])
def test_no_look_ahead(factor) -> None:
    prices = _walk()
    full = factor(prices, lb=20, vol_lb=5)
    assert full.abs().sum().sum() > 0  # non-vacuous
    for t in (50, 75, 100):
        trunc = factor(prices.iloc[: t + 1], lb=20, vol_lb=5)
        assert (full.iloc[t] - trunc.iloc[-1]).abs().max() < 1e-9


def test_tsmom_signs_follow_trend() -> None:
    w = tsmom(_trend_panel(), lb=20, vol_lb=5)
    assert w.iloc[-1]["UP"] > 0
    assert w.iloc[-1]["DOWN"] < 0


def test_tsmom_unit_gross() -> None:
    gross = tsmom(_walk(), lb=20, vol_lb=5).abs().sum(axis=1)
    assert (((gross - 1.0).abs() < 1e-9) | (gross < 1e-12)).all()


def test_dual_momentum_long_only_or_cash() -> None:
    w = dual_momentum(_trend_panel(), lb=20, vol_lb=5)
    last = w.iloc[-1]
    assert last["UP"] == pytest.approx(1.0)
    assert last["DOWN"] == 0.0

    all_down = _trend_panel(up=-0.008, down=-0.012)
    cash = dual_momentum(all_down, lb=20, vol_lb=5)
    assert (cash.iloc[-1] == 0.0).all()

    sums = w.sum(axis=1)
    assert (((sums - 1.0).abs() < 1e-9) | (sums.abs() < 1e-12)).all()
