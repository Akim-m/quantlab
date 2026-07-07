import numpy as np
import pandas as pd
import pytest

from quantlab.trend import (
    _size,
    donchian,
    dual_momentum,
    ma_trend,
    tsmom,
    turn_of_month,
    volume_momentum,
)


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


def test_no_look_ahead_ma_and_volume() -> None:
    prices = _walk(n=140)
    rng = np.random.default_rng(3)
    vol = pd.DataFrame(
        rng.uniform(1e5, 1e6, size=prices.shape), index=prices.index, columns=prices.columns
    )
    runs = [
        lambda p: ma_trend(p, ma_lb=20, vol_lb=5),
        lambda p: volume_momentum(p, vol.loc[p.index], lb=20, vol_sma=5, vol_lb=5),
    ]
    for run in runs:
        full = run(prices)
        assert full.abs().sum().sum() > 0  # non-vacuous
        for t in (60, 90, 120):
            trunc = run(prices.iloc[: t + 1])
            assert (full.iloc[t] - trunc.iloc[-1]).abs().max() < 1e-9


def test_no_look_ahead_catches_future_shift() -> None:
    # a deliberately leaky variant (direction from tomorrow's close) must trip
    # the truncation check above, proving the harness has teeth
    prices = _walk(n=140)

    def leaky(p: pd.DataFrame) -> pd.DataFrame:
        return _size(np.sign(p.shift(-1) - p.rolling(20).mean()), p, 5)

    full = leaky(prices)
    gap = max(
        (full.iloc[t] - leaky(prices.iloc[: t + 1]).iloc[-1]).abs().max() for t in (60, 90, 120)
    )
    assert gap > 1e-6


def test_ma_trend_signs() -> None:
    w = ma_trend(_trend_panel(), ma_lb=20, vol_lb=5)
    assert w.iloc[-1]["UP"] > 0
    assert w.iloc[-1]["DOWN"] < 0


def test_turn_of_month_calendar() -> None:
    idx = pd.date_range("2020-01-01", "2020-03-31", freq="B")
    rng = np.random.default_rng(5)
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=(len(idx), 3)), axis=0),
        index=idx,
        columns=["A", "B", "C"],
    )
    w = turn_of_month(prices, vol_lb=5)

    # freq="B" ignores holidays, so Jan 1 counts as a trading day here
    tom = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-31",
         "2020-02-03", "2020-02-04", "2020-02-05", "2020-02-28",
         "2020-03-02", "2020-03-03", "2020-03-04", "2020-03-31"]
    )
    assert (w.drop(index=tom) == 0.0).all().all()  # cash off the ToM window

    live = w.loc[tom[3:]]  # Jan 1-3 sit inside the 5d vol warmup -> no position
    assert (live > 0.0).all().all()
    assert (live.sum(axis=1) - 1.0).abs().max() < 1e-9
    assert (w.loc[tom[:3]] == 0.0).all().all()
