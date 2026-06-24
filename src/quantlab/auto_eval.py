"""Frozen evaluator for the autoresearch loop. Do not edit to chase a result.

Scores a strategy CONFIG on the TRAIN window only. The OOS window is touched once,
manually, at the very end via oos_report() - never inside the loop.
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
        "status": "keep" if m["constraint_ok"] else "discard",
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
    return _metrics(res)


def _prices() -> pd.DataFrame:
    data = _load_symbols(UNIVERSE, False)
    return close_prices({s: data[s] for s in UNIVERSE}).dropna()


def _metrics(res) -> dict:
    r = res.returns
    n = len(r)
    ann_return = float(res.equity.iloc[-1] ** (252.0 / n) - 1.0)
    ann_vol = float(r.std() * (252**0.5))
    sharpe = 0.0 if r.std() == 0 else float((252**0.5) * r.mean() / r.std())
    downside = r[r < 0].std()
    sortino = 0.0 if downside == 0 else float((252**0.5) * r.mean() / downside)
    max_dd = res.max_drawdown
    calmar = 0.0 if max_dd == 0 else float(ann_return / abs(max_dd))
    turnover = float(res.turnover.mean() * 252)

    folds = r.groupby(r.index.year).apply(
        lambda x: float((252**0.5) * x.mean() / x.std()) if x.std() > 0 else 0.0
    )
    j = float(folds.mean() - 0.5 * folds.std())

    return {
        "J": j,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": float(max_dd),
        "turnover": turnover,
        "fold_sharpes": {str(y): float(v) for y, v in folds.items()},
        "constraint_ok": bool(max_dd >= MAX_DD_LIMIT and turnover <= TURNOVER_LIMIT),
    }
