import argparse
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult, backtest_weights
from .data import close_prices, load_yahoo_ohlcv
from .portfolio import equal_weight
from .strategies import inverse_vol, long_top_momentum

ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "GLD"]


def run_etf_baseline(
    symbols: list[str] | None = None,
    start: str = "2005-01-01",
    cost_bps: float = 5.0,
    refresh: bool = False,
    report_dir: str | Path | None = None,
) -> pd.DataFrame:
    symbols = symbols or ETF_UNIVERSE
    data = load_yahoo_ohlcv(symbols, refresh=refresh)
    prices = close_prices(data).loc[start:].dropna()

    all_assets = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    runs = {
        "equal_weight": backtest_weights(prices, equal_weight(all_assets), cost_bps),
        "inverse_vol_63d": backtest_weights(prices, inverse_vol(prices, 63), cost_bps),
        "momentum_126d_top2": backtest_weights(
            prices,
            long_top_momentum(prices, lookback=126, count=2),
            cost_bps,
        ),
    }
    if report_dir:
        benchmark = prices["SPY"].pct_change().fillna(0.0)
        _write_reports(runs, benchmark, Path(report_dir))

    return pd.DataFrame([_summary(name, res) for name, res in runs.items()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--report-dir")
    args = parser.parse_args()

    res = run_etf_baseline(
        start=args.start,
        cost_bps=args.cost_bps,
        refresh=args.refresh,
        report_dir=args.report_dir,
    )
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _summary(name: str, res: BacktestResult) -> dict[str, float | str]:
    return {
        "strategy": name,
        "total_return": res.total_return,
        "annual_return": _annual_return(res.equity),
        "sharpe": res.sharpe,
        "max_drawdown": res.max_drawdown,
        "avg_daily_turnover": float(res.turnover.mean()),
    }


def _annual_return(equity: pd.Series) -> float:
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return 0.0
    return float(equity.iloc[-1] ** (365.25 / days) - 1.0)


def _write_reports(
    runs: dict[str, BacktestResult],
    benchmark: pd.Series,
    report_dir: Path,
) -> None:
    import quantstats as qs

    report_dir.mkdir(parents=True, exist_ok=True)
    for name, res in runs.items():
        qs.reports.html(
            res.returns,
            benchmark=benchmark,
            output=str(report_dir / f"{name}.html"),
            title=f"{name} vs SPY",
        )


if __name__ == "__main__":
    main()
