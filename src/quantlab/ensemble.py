"""Combine already-built factor outputs into ensemble strategies."""

import numpy as np
import pandas as pd

from .optimization import erc_weights


def blend_weights(frames: list[pd.DataFrame]) -> pd.DataFrame:
    w = sum(frames) / len(frames)
    gross = w.abs().sum(axis=1).replace(0.0, np.nan)
    return w.div(gross, axis=0).fillna(0.0)


def equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    return returns.mean(axis=1)


def risk_parity_returns(
    returns: pd.DataFrame, lookback: int = 126, rebalance: str = "ME"
) -> pd.Series:
    weights = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for date in returns.groupby(pd.Grouper(freq=rebalance)).tail(1).index:
        hist = returns.loc[:date].tail(lookback)
        if len(hist) < lookback:
            continue
        weights.loc[date] = erc_weights(hist.cov())

    # hold between rebalances; shift(1) so the return at t uses only weights known before t
    weights = weights.ffill().shift(1)
    return (weights * returns).sum(axis=1)
