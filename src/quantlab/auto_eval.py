"""Frozen evaluator for the autoresearch loop. Do not edit to chase a result.

Objective J = consistently beat the benchmarks on TRAIN. The OOS window is touched once,
manually, via oos_report() - never inside the loop.
"""

import pandas as pd

from .auto_strategy import CONFIG, strategy_weights
from .backtest import backtest_weights
from .data import close_prices
from .experiments import _load_symbols
from .tracking import log_run

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "DBC", "GLD"]
TRAIN = ("2007-04-11", "2019-12-31")
OOS = ("2020-01-01", "2026-06-18")
COST_BPS = 10.0
MAX_DD_LIMIT = -0.25
TURNOVER_LIMIT = 12.0


def run_and_log(overrides: dict | None = None, note: str = "") -> dict:
    cfg = {**CONFIG, **(overrides or {})}
    m = _evaluate(cfg, TRAIN)
    log_run({
        "hypothesis_ref": "RL-2026-06-24-01",
        "split": "train",
        "config": cfg,
        "objective_J": m["J"],
        "metrics": m,
        "status": "keep" if (m["constraint_ok"] and m["J"] > 0 and m["beats_both"]) else "discard",
        "note": note,
    })
    return m


def oos_report(overrides: dict | None = None) -> dict:
    cfg = {**CONFIG, **(overrides or {})}
    m = _evaluate(cfg, OOS)
    log_run({
        "hypothesis_ref": "RL-2026-06-24-01",
        "split": "oos",
        "config": cfg,
        "metrics": m,
        "status": "final",
        "note": "OOS touch-once",
    })
    return m


def _evaluate(cfg: dict, window: tuple[str, str]) -> dict:
    prices = _prices().loc[window[0]:window[1]]
    targets = strategy_weights(prices, cfg)
    res = backtest_weights(prices, targets, COST_BPS)
    ew, sixty40 = _benchmarks(prices)
    return _metrics(res, ew, sixty40)


def _prices() -> pd.DataFrame:
    data = _load_symbols(UNIVERSE, False)
    return close_prices({s: data[s] for s in UNIVERSE}).dropna()


def _benchmarks(prices: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rets = prices.pct_change().fillna(0.0)
    ew = rets.mean(axis=1)
    sixty40 = 0.6 * rets["SPY"] + 0.4 * rets["IEF"]
    return ew, sixty40


def _sharpe(r: pd.Series) -> float:
    return 0.0 if r.std() == 0 else float((252**0.5) * r.mean() / r.std())


def _metrics(res, ew: pd.Series, sixty40: pd.Series) -> dict:
    r = res.returns
    n = len(r)
    ann_return = float(res.equity.iloc[-1] ** (252.0 / n) - 1.0)
    sharpe = _sharpe(r)
    downside = r[r < 0].std()
    sortino = 0.0 if downside == 0 else float((252**0.5) * r.mean() / downside)
    max_dd = res.max_drawdown
    calmar = 0.0 if max_dd == 0 else float(ann_return / abs(max_dd))
    turnover = float(res.turnover.mean() * 252)

    # objective: Information Ratio vs the tougher benchmark - the institutional measure of
    # beating a benchmark per unit of active risk. J > 0 means real alpha over the bar.
    bench = ew if _sharpe(ew) >= _sharpe(sixty40) else sixty40
    active = r - bench
    ir = 0.0 if active.std() == 0 else float((252**0.5) * active.mean() / active.std())
    j = ir

    return {
        "J": j,
        "sharpe": sharpe,
        "sharpe_ew": _sharpe(ew),
        "sharpe_6040": _sharpe(sixty40),
        "ir_vs_bench": ir,
        "beats_both": bool(sharpe > _sharpe(ew) and sharpe > _sharpe(sixty40)),
        "sortino": sortino,
        "calmar": calmar,
        "ann_return": ann_return,
        "ann_vol": float(r.std() * (252**0.5)),
        "max_drawdown": float(max_dd),
        "turnover": turnover,
        "constraint_ok": bool(max_dd >= MAX_DD_LIMIT and turnover <= TURNOVER_LIMIT),
    }
