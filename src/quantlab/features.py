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


def residual_returns(prices: pd.DataFrame, market: pd.Series, lookback: int) -> pd.DataFrame:
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    r = prices.pct_change()
    m = market.pct_change()
    beta = r.rolling(lookback).cov(m).div(m.rolling(lookback).var(), axis=0)
    return r - beta.mul(m, axis=0)


def downside_beta(prices: pd.DataFrame, market: pd.Series, lookback: int) -> pd.DataFrame:
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    r = prices.pct_change()
    m = market.pct_change()
    down = m.where(m < 0)
    # up-days are NaN, so a full window of down-days never exists; require a
    # quarter-window of them instead of the default (= all) non-NaN count
    mp = max(2, lookback // 4)
    cov = r.where(m < 0, axis=0).rolling(lookback, min_periods=mp).cov(down)
    return cov.div(down.rolling(lookback, min_periods=mp).var(), axis=0)


def rolling_skew(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if lookback < 3:
        raise ValueError("lookback must be at least 3")
    return pct_returns(prices).rolling(lookback).skew()


def rolling_max_daily(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if lookback < 1:
        raise ValueError("lookback must be positive")
    return pct_returns(prices).rolling(lookback).max()


def high_ratio(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if lookback < 1:
        raise ValueError("lookback must be positive")
    return prices / prices.rolling(lookback).max()


def rolling_beta(prices: pd.DataFrame, market: pd.Series, lookback: int) -> pd.DataFrame:
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    r = prices.pct_change()
    m = market.pct_change()
    return r.rolling(lookback).cov(m).div(m.rolling(lookback).var(), axis=0)


def rolling_kurt(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if lookback < 4:
        raise ValueError("lookback must be at least 4")
    return pct_returns(prices).rolling(lookback).kurt()


def low_ratio(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if lookback < 1:
        raise ValueError("lookback must be positive")
    return prices / prices.rolling(lookback).min()


def parkinson_vol(high: pd.DataFrame, low: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Parkinson range-variance proxy: higher = more intraday range."""
    if lookback < 1:
        raise ValueError("lookback must be positive")
    return (np.log(high / low) ** 2).rolling(lookback).mean()
