import pandas as pd

from .features import rolling_momentum, rolling_vol
from .portfolio import equal_weight, inverse_vol_weight


def long_top_momentum(prices: pd.DataFrame, lookback: int, count: int) -> pd.DataFrame:
    if count < 1:
        raise ValueError("count must be positive")

    momentum = rolling_momentum(prices, lookback)
    ranks = momentum.rank(axis=1, ascending=False, method="first")
    return equal_weight(ranks.where(ranks <= count))


def inverse_vol(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return inverse_vol_weight(rolling_vol(prices, lookback))
