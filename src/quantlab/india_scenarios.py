"""RL-2026-07-10: situation -> winning-strategy sweep on the Indian universe.

Evaluates a set of named strategy books across SITUATIONS - universe breadth
(Nifty 50/200/500), cost regime, and causal market regime (bull/bear via the
200-day MA) - and reports, on the locked TEST window, net-of-cost Sharpe, annual
return, max drawdown, turnover, whether it beats the Nifty benchmark, and its
Sharpe conditioned on the bull/bear regime.

The regime split is CAUSAL (the market's 200-MA state at t-1 buckets day t's
return), so a strategy that trades on it is deployable, not hindsight. Per-regime
Sharpe is still descriptive; the deployable claim is the regime-SWITCH strategy's
own overall number (see blend.regime_switch).
"""

import numpy as np
import pandas as pd

from .backtest import backtest_weights
from .blend import market_on
from .evaluation import sharpe_tstat
from .portfolio import rebalance_targets


def _ann(r: pd.Series) -> float:
    return float((1 + r).prod() ** (252 / max(len(r), 1)) - 1)


def _maxdd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def regime_conditional_sharpe(returns: pd.Series, market: pd.Series, ma_lb: int = 200) -> tuple[float, float]:
    """(bull_sharpe, bear_sharpe): day t's return bucketed by the market's 200-MA
    state at t-1 (causal)."""
    on = market_on(market, ma_lb).shift(1).reindex(returns.index).fillna(False).astype(bool)
    bull, _ = sharpe_tstat(returns[on])
    bear, _ = sharpe_tstat(returns[~on])
    return bull, bear


def evaluate(
    strategies: dict[str, tuple[pd.DataFrame, str | None]],
    px: pd.DataFrame,
    mkt: pd.Series,
    bench_ret: pd.Series,
    split: str,
    cost_bps: float = 20.0,
) -> pd.DataFrame:
    """Backtest each named (weights, rebalance_freq) and return a metrics table."""
    bench_test = bench_ret.loc[split:]
    bench_sr, _ = sharpe_tstat(bench_test)
    rows = []
    for name, (weights, freq) in strategies.items():
        res = backtest_weights(px, rebalance_targets(weights, freq), cost_bps)
        test = res.returns.loc[split:]
        sr, t = sharpe_tstat(test)
        bull, bear = regime_conditional_sharpe(test, mkt.loc[split:])
        rows.append({
            "strategy": name,
            "test_sharpe": round(sr, 3),
            "test_tstat": round(t, 2),
            "ann_return": round(_ann(test), 4),
            "max_dd": round(_maxdd(test), 4),
            "turnover": round(float(res.turnover.mean()), 4),
            "beats_nifty": bool(sr > bench_sr),
            "bull_sharpe": round(bull, 3),
            "bear_sharpe": round(bear, 3),
        })
    table = pd.DataFrame(rows).sort_values("test_sharpe", ascending=False).reset_index(drop=True)
    table.attrs["bench_sharpe"] = round(bench_sr, 3)
    return table
