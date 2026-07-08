"""Blend / ensemble construction primitives (RL-2026-07-10).

Compose several economically-motivated signals into one book. Two output modes:
  - long-short: cross-sectionally demeaned, dollar-neutral, unit gross (factor
    evidence; not deployable in the Indian cash market where single-stock shorts
    need F&O).
  - long-only: top-quantile of the composite, inverse-vol or equal weighted,
    summing to 1 (the deployable tilt vs Nifty).

Overlays de-risk toward cash: a trend filter (market below its long MA) and a
vol-target scale. Both are causal - they use only trailing information.
"""

import numpy as np
import pandas as pd

from .features import rolling_vol
from .xsec import _neutral


def zscore_xs(signal: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per date (robust to differing scales across signals)."""
    mu = signal.mean(axis=1)
    sd = signal.std(axis=1).replace(0.0, np.nan)
    return signal.sub(mu, axis=0).div(sd, axis=0)


def composite(signals: dict[str, pd.DataFrame], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Weighted average of z-scored signals -> one composite score per name/date.

    Equal weight by default (no train-fitted weights -> less overfitting). NaNs
    (a name missing one signal) are skipped so a name is scored on what it has.
    """
    names = list(signals)
    w = {n: (weights or {}).get(n, 1.0) for n in names}
    total = sum(w.values()) or 1.0
    zs = {n: zscore_xs(signals[n]) for n in names}
    acc = None
    for n in names:
        term = zs[n] * (w[n] / total)
        acc = term if acc is None else acc.add(term, fill_value=0.0)
    return acc


def long_short(score: pd.DataFrame) -> pd.DataFrame:
    """Dollar-neutral, unit-gross book from the composite (long high score)."""
    return _neutral(score.rank(axis=1))


def long_only_topq(
    score: pd.DataFrame,
    prices: pd.DataFrame,
    top: float = 0.2,
    vol_lb: int = 63,
    weighting: str = "invvol",
) -> pd.DataFrame:
    """Long the top `top` fraction by score each date; weights sum to 1.

    invvol weighting damps the highest-vol names (a low-vol tilt on top of the
    score); "ew" is plain equal weight.
    """
    ranks = score.rank(axis=1, ascending=False)
    n = score.notna().sum(axis=1)
    keep = ranks.le((n * top).clip(lower=1.0), axis=0)
    if weighting == "invvol":
        raw = (1.0 / rolling_vol(prices, vol_lb).replace(0.0, np.nan)).where(keep)
    else:
        raw = keep.astype(float).where(keep)
    total = raw.sum(axis=1).replace(0.0, np.nan)
    return raw.div(total, axis=0).fillna(0.0)


def trend_overlay(book: pd.DataFrame, market: pd.Series, ma_lb: int = 200) -> pd.DataFrame:
    """Scale the whole book to cash on days the market closed below its `ma_lb` MA.

    A binary risk-off switch stamped from trailing prices only. Shrinking the book
    (sum < 1) parks the remainder in cash (the backtest does not lever)."""
    ma = market.rolling(ma_lb).mean()
    on = (market > ma).astype(float).reindex(book.index).fillna(0.0)
    return book.mul(on, axis=0)


def market_on(market: pd.Series, ma_lb: int = 200) -> pd.Series:
    """Causal risk-on flag: market closed above its `ma_lb` MA (known at t)."""
    return (market > market.rolling(ma_lb).mean())


def regime_switch(
    risk_on: pd.DataFrame,
    risk_off: pd.DataFrame,
    market: pd.Series,
    ma_lb: int = 200,
) -> pd.DataFrame:
    """Hold `risk_on` when the market is above its MA, else `risk_off`.

    The switch is a pre-committed, causal rule (not fitted to which book won a
    slice), so it is a deployable situation->strategy map, not hindsight."""
    on = market_on(market, ma_lb).astype(float).reindex(risk_on.index).fillna(0.0)
    return risk_on.mul(on, axis=0).add(risk_off.mul(1.0 - on, axis=0), fill_value=0.0)


def vol_target_overlay(
    book: pd.DataFrame,
    returns: pd.DataFrame,
    target: float = 0.15,
    lb: int = 21,
    cap: float = 1.5,
) -> pd.DataFrame:
    """Scale the book so its trailing realized vol tracks `target` (annualized).

    Uses the book's OWN trailing return vol from weights known at t-1, so it is
    causal. Capped at `cap` to bound leverage (cap=1.0 => never lever, only cut)."""
    book_ret = (book.shift(1) * returns).sum(axis=1)
    realized = book_ret.rolling(lb).std() * np.sqrt(252)
    scale = (target / realized.replace(0.0, np.nan)).clip(0.0, cap)
    return book.mul(scale, axis=0).fillna(0.0)
