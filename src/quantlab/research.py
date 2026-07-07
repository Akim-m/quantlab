"""Autoresearch loop for the efficiency-gated trend strategy.

Mirrors karpathy/autoresearch: propose a config, run it, score it on a held-back
metric, keep if it beats the running best, else discard - repeat. The twist
demanded by protocol.md: the score is computed on a DEV window only. The TEST
window is locked and evaluated exactly once, at the end, on the single best
config. Every trial is logged so the multiple-testing count is explicit.
"""

import argparse
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .backtest import backtest_weights
from .data import close_prices, load_yahoo_ohlcv
from .portfolio import rebalance_targets
from .strategies import efficiency_gated_trend
from .tracking import log_run

FX_GOLD = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X", "JPY=X", "GC=F", "SI=F"]

GRID = {
    "trend_lb": [42, 63, 126, 189, 252],
    "er_window": [21, 42, 63],
    "er_threshold": [0.0, 0.2, 0.3, 0.4],
    "vol_lb": [21, 42, 63],
    "rebalance": ["W-FRI", "ME"],
}


@dataclass(frozen=True)
class Params:
    trend_lb: int
    er_window: int
    er_threshold: float
    vol_lb: int
    rebalance: str


def evaluate(
    prices: pd.DataFrame,
    p: Params,
    cost_bps: float,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, float]:
    """Backtest on full history, then score on the [start, end] slice.

    Backtesting on full prices preserves signal warmup; slicing the resulting
    returns keeps dev and test on identical position paths with no look-ahead.
    """
    weights = rebalance_targets(
        efficiency_gated_trend(prices, p.trend_lb, p.er_window, p.er_threshold, p.vol_lb),
        p.rebalance,
    )
    res = backtest_weights(prices, weights, cost_bps)
    returns = res.returns.loc[start:end]
    turnover = res.turnover.loc[start:end]
    equity = (1.0 + returns).cumprod()
    return {
        "sharpe": _sharpe(returns),
        "ann_return": _annual_return(equity),
        "max_drawdown": _max_drawdown(equity),
        "avg_turnover": float(turnover.mean()),
    }


def search(
    prices: pd.DataFrame,
    dev_end: str,
    cost_bps: float,
    threshold: float | None,
    log_path: Path,
) -> tuple[Params, dict[str, float], pd.DataFrame]:
    configs = [Params(*combo) for combo in itertools.product(*GRID.values())]
    best: Params | None = None
    best_score = float("-inf")
    rows = []

    for i, p in enumerate(configs):
        m = evaluate(prices, p, cost_bps, end=dev_end)
        score = m["sharpe"]
        improved = score > best_score
        if improved:
            best, best_score = p, score
        rows.append({"iter": i, **asdict(p), "dev_sharpe": round(score, 4),
                     "dev_ann_return": round(m["ann_return"], 4),
                     "dev_max_dd": round(m["max_drawdown"], 4),
                     "dev_turnover": round(m["avg_turnover"], 4),
                     "status": "keep" if improved else "discard"})
        if threshold is not None and best_score >= threshold:
            break

    log = pd.DataFrame(rows)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(log_path, sep="\t", index=False)
    assert best is not None
    return best, evaluate(prices, best, cost_bps, end=dev_end), log


def _sharpe(returns: pd.Series) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float((252**0.5) * returns.mean() / returns.std())


def _annual_return(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return 0.0
    return float(equity.iloc[-1] ** (365.25 / days) - 1.0)


def _max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--dev-end", default="2018-12-31")
    parser.add_argument("--test-start", default="2019-01-01")
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=None,
                        help="stop early once dev Sharpe reaches this")
    parser.add_argument("--tag", default="fx-trend")
    parser.add_argument("--hypothesis-ref", default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    data = load_yahoo_ohlcv(FX_GOLD, refresh=args.refresh)
    prices = close_prices(data).loc[args.start:].dropna()
    print(f"universe: {len(prices.columns)} assets, "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}, {len(prices)} days")

    log_path = Path(f"experiments/autoresearch_{args.tag}.tsv")
    best, dev_m, log = search(prices, args.dev_end, args.cost_bps, args.threshold, log_path)
    test_m = evaluate(prices, best, args.cost_bps, start=args.test_start)

    print(f"\nconfigs evaluated: {len(log)} (multiple-testing count)")
    print(f"best config: {asdict(best)}")
    print(f"\n{'metric':<14}{'DEV (in-sample)':>18}{'TEST (OOS, locked)':>20}")
    for k in ["sharpe", "ann_return", "max_drawdown", "avg_turnover"]:
        print(f"{k:<14}{dev_m[k]:>18.4f}{test_m[k]:>20.4f}")

    print("\ntop 5 dev configs:")
    print(log.sort_values("dev_sharpe", ascending=False).head(5).to_string(index=False))

    log_run({
        "hypothesis_ref": args.hypothesis_ref,
        "universe": "fx_gold",
        "sample_start": str(prices.index[0].date()),
        "sample_end": str(prices.index[-1].date()),
        "n_assets": int(prices.shape[1]),
        "cost_bps": args.cost_bps,
        "strategy": "efficiency_gated_trend",
        "configs_tried": int(len(log)),
        "best_params": asdict(best),
        "metrics": {"dev": dev_m, "test": test_m},
        "status": "success",
    })


if __name__ == "__main__":
    main()
