"""The only file the autoresearch loop edits. CONFIG is the current champion."""

import numpy as np
import pandas as pd

from .features import pct_returns, rolling_momentum, rolling_vol
from .optimization import min_variance_weights
from .portfolio import equal_weight, inverse_vol_weight

CONFIG = {
    "lookback": 63,             # trend lookback in trading days
    "mode": "gate",            # gate | tilt
    "backbone": "min_variance",  # equal | inverse_vol | min_variance
    "vol_target": 0.10,         # annualized portfolio vol target
    "vol_window": 63,           # window for vol / covariance estimates
    "max_gross": 1.5,           # cap on gross exposure after vol scaling
    "rebalance": "ME",          # ME (monthly) | 2ME (bi-monthly)
}


def strategy_weights(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    trend = rolling_momentum(prices, cfg["lookback"])
    vol = rolling_vol(prices, cfg["vol_window"])
    returns = pct_returns(prices)

    out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for date in _rebalance_dates(prices, cfg["rebalance"]):
        base = _base_weights(trend.loc[date], vol.loc[date], returns.loc[:date], cfg)
        out.loc[date] = _scale_to_vol(base, returns.loc[:date], cfg)
    return out


def _base_weights(
    trend: pd.Series,
    vol: pd.Series,
    hist: pd.DataFrame,
    cfg: dict,
) -> pd.Series:
    selected = trend > 0
    if not selected.any():
        return pd.Series(0.0, index=trend.index)

    if cfg["mode"] == "tilt":
        raw = trend.where(selected).clip(lower=0.0)
        return (raw / raw.sum()).fillna(0.0)

    kind = cfg["backbone"]
    if kind == "equal":
        return equal_weight(selected.where(selected).to_frame().T).iloc[0]
    if kind == "inverse_vol":
        return inverse_vol_weight(vol.where(selected).to_frame().T).iloc[0]
    if kind == "min_variance":
        return _min_var(selected, hist, cfg)
    raise ValueError(f"unknown backbone: {kind}")


def _min_var(selected: pd.Series, hist: pd.DataFrame, cfg: dict) -> pd.Series:
    names = list(selected.index[selected])
    if len(names) < 3:
        return inverse_vol_weight(hist[names].std().to_frame().T).iloc[0].reindex(
            selected.index
        ).fillna(0.0)
    window = max(cfg["vol_window"], 60)
    cov = hist[names].tail(window).cov()
    max_weight = max(0.5, 1.0 / len(names) + 1e-9)
    w = min_variance_weights(cov, max_weight)
    return w.reindex(selected.index).fillna(0.0)


def _scale_to_vol(base: pd.Series, hist: pd.DataFrame, cfg: dict) -> pd.Series:
    port = (hist.tail(cfg["vol_window"]) * base).sum(axis=1)
    realized = float(port.std() * (252**0.5))
    if realized <= 0:
        return base
    scale = min(cfg["vol_target"] / realized, cfg["max_gross"])
    return base * scale


def _rebalance_dates(prices: pd.DataFrame, freq: str) -> pd.Index:
    step = 2 if freq == "2ME" else 1
    dates = prices.groupby(pd.Grouper(freq="ME")).tail(1).index
    return dates[::step]
