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
