from collections.abc import Callable
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

Objective = Literal["min_variance", "max_sharpe"]


def min_variance_weights(cov: pd.DataFrame, max_weight: float = 0.6) -> pd.Series:
    assets = list(cov.columns)
    c = _clean_cov(cov)

    def objective(w: np.ndarray) -> float:
        return float(w @ c @ w)

    return _solve(assets, objective, max_weight)


def max_sharpe_weights(
    mu: pd.Series,
    cov: pd.DataFrame,
    max_weight: float = 0.6,
) -> pd.Series:
    assets = list(cov.columns)
    c = _clean_cov(cov)
    m = mu.reindex(assets).fillna(0.0).to_numpy()
    if m.max() <= 0.0:
        return min_variance_weights(cov, max_weight)

    def objective(w: np.ndarray) -> float:
        vol = float((w @ c @ w) ** 0.5)
        if vol <= 0:
            return 1e9
        return -float((w @ m) / vol)

    starts = [
        min_variance_weights(cov, max_weight).to_numpy(),
        _tilted_start(len(assets), int(m.argmax()), max_weight),
    ]
    return _solve(assets, objective, max_weight, starts)


def rolling_mvo_weights(
    prices: pd.DataFrame,
    objective: Objective,
    lookback: int = 252,
    max_weight: float = 0.6,
    rebalance: str = "ME",
) -> pd.DataFrame:
    if lookback < 2:
        raise ValueError("lookback must be at least 2")

    returns = prices.pct_change().dropna()
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    for date in _rebalance_dates(prices, rebalance):
        hist = returns.loc[:date].tail(lookback)
        if len(hist) < lookback:
            continue

        mu = hist.mean() * 252
        cov = hist.cov() * 252
        if objective == "min_variance":
            weights.loc[date] = min_variance_weights(cov, max_weight)
        elif objective == "max_sharpe":
            weights.loc[date] = max_sharpe_weights(mu, cov, max_weight)
        else:
            raise ValueError(f"unknown objective: {objective}")

    return weights


def _solve(
    assets: list[str],
    objective: Callable[[np.ndarray], float],
    max_weight: float,
    starts: list[np.ndarray] | None = None,
) -> pd.Series:
    n = len(assets)
    if n == 0:
        raise ValueError("no assets to optimize")
    if max_weight * n < 1.0:
        raise ValueError("max_weight is too low for a fully invested portfolio")

    starts = starts or [np.repeat(1.0 / n, n)]
    res = None
    for start in starts:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Values in x were outside bounds")
            res = minimize(
                objective,
                start,
                bounds=tuple((0.0, max_weight) for _ in assets),
                constraints=({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},),
                method="SLSQP",
                options={"ftol": 1e-12, "maxiter": 1000},
            )
        if res.success:
            break
    if res is None or not res.success:
        msg = "no optimizer result" if res is None else res.message
        raise ValueError(f"optimization failed: {msg}")
    return pd.Series(res.x, index=assets).clip(0.0, max_weight)


def _clean_cov(cov: pd.DataFrame) -> np.ndarray:
    c = cov.fillna(0.0).to_numpy()
    c = (c + c.T) / 2.0
    return c + np.eye(c.shape[0]) * 1e-10


def _rebalance_dates(prices: pd.DataFrame, freq: str) -> pd.Index:
    return prices.groupby(pd.Grouper(freq=freq)).tail(1).index


def _tilted_start(n: int, idx: int, max_weight: float) -> np.ndarray:
    if n == 1:
        return np.array([1.0])
    weights = np.repeat((1.0 - max_weight) / (n - 1), n)
    weights[idx] = max_weight
    return weights
