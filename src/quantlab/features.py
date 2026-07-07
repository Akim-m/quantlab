import numpy as np
import pandas as pd


def pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)


def rolling_momentum(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if lookback < 1:
        raise ValueError("lookback must be positive")
    return prices.pct_change(lookback)


def rolling_vol(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    return pct_returns(prices).rolling(lookback).std()


def efficiency_ratio(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Kaufman efficiency ratio: net move over total path length in [0, 1].

    1.0 is a perfectly straight trend, near 0 is choppy noise. Scale-invariant
    within an asset, so price levels across assets need not be comparable.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    net = prices.diff(window).abs()
    path = prices.diff().abs().rolling(window).sum()
    return (net / path.replace(0.0, np.nan)).fillna(0.0)
