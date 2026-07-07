import numpy as np
import pandas as pd

from .features import efficiency_ratio, rolling_momentum, rolling_vol
from .portfolio import equal_weight, inverse_vol_weight


def long_top_momentum(prices: pd.DataFrame, lookback: int, count: int) -> pd.DataFrame:
    if count < 1:
        raise ValueError("count must be positive")

    momentum = rolling_momentum(prices, lookback)
    ranks = momentum.rank(axis=1, ascending=False, method="first")
    return equal_weight(ranks.where(ranks <= count))


def inverse_vol(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return inverse_vol_weight(rolling_vol(prices, lookback))


def efficiency_gated_trend(
    prices: pd.DataFrame,
    trend_lb: int,
    er_window: int,
    er_threshold: float,
    vol_lb: int,
) -> pd.DataFrame:
    """Long/short trend, sized inverse-vol, only when the trend is clean.

    Direction is the sign of the trailing trend; a position is taken only where
    the efficiency ratio clears `er_threshold` (smooth trend, not chop). Each
    leg is inverse-vol weighted and the book is scaled to unit gross leverage so
    turnover cost and risk stay comparable across parameter settings.
    """
    direction = np.sign(rolling_momentum(prices, trend_lb))
    gate = (efficiency_ratio(prices, er_window) >= er_threshold).astype(float)
    inv_vol = 1.0 / rolling_vol(prices, vol_lb).replace(0.0, np.nan)

    raw = direction * gate * inv_vol
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    return raw.div(gross, axis=0).fillna(0.0)


def betting_against_beta(
    prices: pd.DataFrame,
    market: pd.Series,
    beta_lb: int,
    floor: float = 0.1,
) -> pd.DataFrame:
    """Long-only book weighted inversely to each asset's beta vs the market.

    Low (or negative) beta means low market risk and earns a larger position;
    high beta is shrunk. Beta is floored at `floor` so near-zero/negative betas
    get a bounded max weight instead of blowing up. Weights sum to 1 each day.
    """
    if beta_lb < 2:
        raise ValueError("beta_lb must be at least 2")

    r = prices.pct_change()
    m = market.pct_change()
    beta = r.rolling(beta_lb).cov(m).div(m.rolling(beta_lb).var(), axis=0)

    inv = 1.0 / beta.clip(lower=floor)
    total = inv.sum(axis=1).replace(0.0, np.nan)
    return inv.div(total, axis=0).fillna(0.0)


def beta_long_short(prices: pd.DataFrame, market: pd.Series, beta_lb: int) -> pd.DataFrame:
    """Dollar-neutral long/short: long low-beta, short high-beta names.

    Each day, beta vs the market is demeaned across the cross-section, so
    below-average beta gets a long and above-average gets a short of equal
    dollar size. Gross leverage is scaled to 1 (≈0.5 long / 0.5 short).
    """
    if beta_lb < 2:
        raise ValueError("beta_lb must be at least 2")

    r = prices.pct_change()
    m = market.pct_change()
    beta = r.rolling(beta_lb).cov(m).div(m.rolling(beta_lb).var(), axis=0)

    signal = -beta.sub(beta.mean(axis=1), axis=0)
    gross = signal.abs().sum(axis=1).replace(0.0, np.nan)
    return signal.div(gross, axis=0).fillna(0.0)


def beta_timing(
    prices: pd.DataFrame,
    market: pd.Series,
    beta_lb: int,
    cap: float = 3.0,
) -> pd.DataFrame:
    """Per-asset exposure that goes WITH beta: weight = the asset's own beta.

    High beta -> larger long, negative beta -> short, capped at +/- `cap`.
    Columns are independent: each is a single-asset book, no cross-sectional
    interaction, so this can be backtested one asset at a time.
    """
    if beta_lb < 2:
        raise ValueError("beta_lb must be at least 2")

    r = prices.pct_change()
    m = market.pct_change()
    beta = r.rolling(beta_lb).cov(m).div(m.rolling(beta_lb).var(), axis=0)
    return beta.clip(-cap, cap).fillna(0.0)
