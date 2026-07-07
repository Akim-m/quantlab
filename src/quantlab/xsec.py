"""Cross-sectional anomaly factors (RL-2026-07-07, ids 01-11).

Each returns daily target weights: cross-sectionally ranked, demeaned,
dollar-neutral, unit gross. Long high signal. No rebalance frequency applied.
"""

import numpy as np
import pandas as pd

from .features import (
    downside_beta,
    high_ratio,
    residual_returns,
    rolling_max_daily,
    rolling_skew,
    rolling_vol,
)


def _neutral(signal: pd.DataFrame) -> pd.DataFrame:
    s = signal.sub(signal.mean(axis=1), axis=0)         # cross-sectional demean
    gross = s.abs().sum(axis=1).replace(0.0, np.nan)
    return s.div(gross, axis=0).fillna(0.0)             # dollar-neutral, unit gross


def short_term_reversal(prices: pd.DataFrame, lb: int = 5) -> pd.DataFrame:
    return _neutral((-prices.pct_change(lb)).rank(axis=1))


def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    return _neutral((prices.shift(21) / prices.shift(252) - 1).rank(axis=1))


def long_term_reversal(prices: pd.DataFrame) -> pd.DataFrame:
    return _neutral((-(prices.shift(252) / prices.shift(1260) - 1)).rank(axis=1))


def low_volatility(prices: pd.DataFrame, lb: int = 252) -> pd.DataFrame:
    return _neutral((-rolling_vol(prices, lb)).rank(axis=1))


def idio_vol(prices: pd.DataFrame, market: pd.Series, lb: int = 252) -> pd.DataFrame:
    return _neutral((-residual_returns(prices, market, lb).rolling(lb).std()).rank(axis=1))


def max_lottery(prices: pd.DataFrame, lb: int = 21) -> pd.DataFrame:
    return _neutral((-rolling_max_daily(prices, lb)).rank(axis=1))


def high_52w(prices: pd.DataFrame, lb: int = 252) -> pd.DataFrame:
    return _neutral(high_ratio(prices, lb).rank(axis=1))


def skewness(prices: pd.DataFrame, lb: int = 252) -> pd.DataFrame:
    return _neutral((-rolling_skew(prices, lb)).rank(axis=1))


def residual_momentum(
    prices: pd.DataFrame, market: pd.Series, lb: int = 252, skip: int = 21
) -> pd.DataFrame:
    res = residual_returns(prices, market, lb)
    return _neutral((res.rolling(lb).sum() - res.rolling(skip).sum()).rank(axis=1))


def seasonality(prices: pd.DataFrame) -> pd.DataFrame:
    """Heston-Sadka: mean same-calendar-month return over prior years only.

    shift(1) inside each calendar-month group drops the current year, and the
    signal is stamped at month START (fully known then), so every day of month
    t carries the prior-years mean without ever seeing month t's own return.
    """
    mret = prices.resample("ME").last().pct_change()
    sig = mret.groupby(mret.index.month).transform(lambda s: s.expanding().mean().shift(1))
    sig.index = sig.index.to_period("M").to_timestamp()
    return _neutral(sig.reindex(prices.index, method="ffill").rank(axis=1))


def downside_beta_factor(prices: pd.DataFrame, market: pd.Series, lb: int = 252) -> pd.DataFrame:
    return _neutral((-downside_beta(prices, market, lb)).rank(axis=1))
