"""RL-2026-07-10: composite-blend evaluation on the Indian single-stock universe.

Machinery only - which signals enter the blend is decided from TRAIN-window
evidence (see india_study), then frozen here before the test window is touched.
Reports train/test Sharpe, return, drawdown, turnover, and the same metrics for
a Nifty buy-and-hold benchmark so the primary bar (beat Nifty net of costs OOS)
is read directly.
"""

import numpy as np
import pandas as pd

from .backtest import BacktestResult, backtest_weights
from .blend import composite, long_only_topq, long_short, trend_overlay, vol_target_overlay
from .evaluation import sharpe_tstat
from .features import low_ratio, residual_returns, rolling_beta, rolling_vol
from .optimization import erc_weights_fast
from .portfolio import rebalance_targets


def _sector_demean(sig: pd.DataFrame, sectors: dict[str, str] | None) -> pd.DataFrame:
    """Demean each name's signal against its industry group per date. Names with no
    industry label (or when no map is given) keep their plain cross-sectional demean."""
    plain = sig.sub(sig.mean(axis=1), axis=0)
    if not sectors:
        return plain
    groups = pd.Series({c: sectors.get(c, "__none__") for c in sig.columns})
    out = plain.copy()
    for g, members in groups.groupby(groups):
        if g == "__none__":
            continue
        cols = list(members.index)
        grp = sig[cols]
        out[cols] = grp.sub(grp.mean(axis=1), axis=0)
    return out


def raw_signals(px: pd.DataFrame, mkt: pd.Series, sectors: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """Economically-motivated cross-sectional signals (higher = more attractive)."""
    mom = px.shift(21) / px.shift(252) - 1.0
    res = residual_returns(px, mkt, 252)
    return {
        "mom_12_1": mom,
        "sharpe_mom": mom / rolling_vol(px, 126),
        "resid_mom": res.rolling(252).sum() - res.rolling(21).sum(),
        "low_vol": -rolling_vol(px, 252),
        "low_beta": -rolling_beta(px, mkt, 252),
        "short_rev": -px.pct_change(5),
        "mom_6_1": px.shift(21) / px.shift(126) - 1.0,
        "off_low": low_ratio(px, 252),
        "sector_mom": _sector_demean(mom, sectors),
    }


def lo_erc(
    px: pd.DataFrame,
    score: pd.DataFrame,
    top: float = 0.2,
    lookback: int = 252,
    rebalance: str = "ME",
) -> pd.DataFrame:
    """Monthly long-only book: select the top `top` fraction by `score`, then ERC-
    weight them on the trailing `lookback`-day return covariance (else cash). Same
    monthly-selection pattern as india_study._dual_mom_erc_fast."""
    rets = px.pct_change().dropna()
    ranks = score.rank(axis=1, ascending=False)
    n = score.notna().sum(axis=1)
    keep = ranks.le((n * top).clip(lower=1.0), axis=0)
    weights = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    for date in px.groupby(pd.Grouper(freq=rebalance)).tail(1).index:
        hist = rets.loc[:date].tail(lookback)
        weights.loc[date] = 0.0
        if len(hist) < lookback or date not in keep.index:
            continue
        sel = [c for c in keep.columns[keep.loc[date]] if hist[c].notna().all()]
        if not sel:
            continue
        w = erc_weights_fast(hist[sel].cov() * 252)
        weights.loc[date, sel] = w.values
    return weights


def benchmark_returns(index: pd.Index, bench_px: pd.Series) -> pd.Series:
    """Buy-and-hold total return of the benchmark, aligned to the panel index."""
    return bench_px.reindex(index).ffill().pct_change().fillna(0.0)


def _metrics(res: BacktestResult, split: str, bench: pd.Series) -> dict:
    train = res.returns.loc[:split].iloc[:-1]
    test = res.returns.loc[split:]
    btest = bench.loc[split:]
    sr_tr, t_tr = sharpe_tstat(train)
    sr_te, t_te = sharpe_tstat(test)
    sr_b, _ = sharpe_tstat(btest)
    ann = lambda r: float((1 + r).prod() ** (252 / max(len(r), 1)) - 1)
    eq = (1 + test).cumprod()
    return {
        "train_sharpe": round(sr_tr, 3),
        "test_sharpe": round(sr_te, 3),
        "test_tstat": round(t_te, 3),
        "test_ann_return": round(ann(test), 4),
        "test_maxdd": round(float((eq / eq.cummax() - 1).min()), 4),
        "turnover": round(float(res.turnover.mean()), 4),
        "bench_test_sharpe": round(sr_b, 3),
        "bench_test_ann_return": round(ann(btest), 4),
        "beats_bench": bool(sr_te > sr_b),
    }


def evaluate_blend(
    px: pd.DataFrame,
    mkt: pd.Series,
    bench_px: pd.Series,
    signals: list[str],
    split: str,
    mode: str = "long_only",
    top: float = 0.2,
    weights: dict[str, float] | None = None,
    overlay: str | None = None,
    cost_bps: float = 20.0,
    rebalance: str = "ME",
) -> tuple[dict, BacktestResult]:
    """Build the composite book, backtest it, return (metrics, result).

    mode: 'long_only' (top-quantile tilt, deployable) or 'long_short' (dollar-
    neutral, evidence). overlay in {None,'trend','voltarget','trend+voltarget'}
    applies only to the long-only book (a neutral book has no market to time)."""
    sigs = {s: raw_signals(px, mkt)[s] for s in signals}
    score = composite(sigs, weights)

    if mode == "long_short":
        book = long_short(score)
    else:
        book = long_only_topq(score, px, top=top)
        rets = px.pct_change().fillna(0.0)
        if overlay and "trend" in overlay:
            book = trend_overlay(book, mkt)
        if overlay and "voltarget" in overlay:
            book = vol_target_overlay(book, rets, cap=1.0)

    res = backtest_weights(px, rebalance_targets(book, rebalance), cost_bps)
    return _metrics(res, split, benchmark_returns(px.index, bench_px)), res
