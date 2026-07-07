import argparse
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult, backtest_weights
from .data import close_prices, load_yahoo_ohlcv
from .optimization import rolling_mvo_weights
from .portfolio import equal_weight, rebalance_targets
from .research import FX_GOLD
from .strategies import (
    beta_long_short,
    beta_timing,
    betting_against_beta,
    inverse_vol,
    long_top_momentum,
)
from .tracking import log_run

# Deliberate beta spread: high-beta tech/cyclical vs low-beta staples/utility.
BETA_STOCKS = ["NVDA", "TSLA", "AMD", "META", "AAPL", "KO", "PG", "JNJ", "WMT", "DUK"]

ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
REBALANCE = {"daily": None, "weekly": "W-FRI", "monthly": "ME"}

# Top 50 Nasdaq-100 constituents by market cap.
NASDAQ50 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "COST", "NFLX",
    "ADBE", "PEP", "AMD", "CSCO", "TMUS", "INTC", "INTU", "QCOM", "TXN", "AMGN",
    "ISRG", "HON", "AMAT", "BKNG", "CMCSA", "ADP", "VRTX", "GILD", "ADI", "REGN",
    "MU", "LRCX", "MDLZ", "PANW", "SBUX", "KLAC", "SNPS", "CDNS", "MAR", "CRWD",
    "ABNB", "ORLY", "CSX", "MRVL", "FTNT", "ADSK", "PYPL", "MNST", "PCAR", "CTAS",
]
NASDAQ_BENCHMARKS = {"sp500": "SPY", "nasdaq100": "QQQ"}


def run_etf_baseline(
    symbols: list[str] | None = None,
    start: str = "2005-01-01",
    cost_bps: float = 5.0,
    refresh: bool = False,
    report_dir: str | Path | None = None,
    rebalance: str = "monthly",
    split_date: str | None = "2016-01-01",
    hypothesis_ref: str | None = None,
) -> pd.DataFrame:
    if rebalance not in REBALANCE:
        raise ValueError(f"unknown rebalance: {rebalance}")

    symbols = symbols or ETF_UNIVERSE
    data = load_yahoo_ohlcv(symbols, refresh=refresh)
    prices = close_prices(data).loc[start:].dropna()
    freq = REBALANCE[rebalance]
    runs = _run_strategies(prices, cost_bps, freq)
    if report_dir:
        benchmarks = {
            "equal_weight": runs["equal_weight"].returns,
            "sp500": prices["SPY"].pct_change().fillna(0.0),
        }
        _write_reports(runs, benchmarks, Path(report_dir))

    return _summarize_and_log(
        runs, prices, "etf", rebalance, cost_bps, split_date, hypothesis_ref
    )


def run_nasdaq50(
    start: str = "2005-01-01",
    cost_bps: float = 5.0,
    refresh: bool = False,
    report_dir: str | Path | None = None,
    rebalance: str = "monthly",
    split_date: str | None = None,
    hypothesis_ref: str | None = None,
) -> pd.DataFrame:
    if rebalance not in REBALANCE:
        raise ValueError(f"unknown rebalance: {rebalance}")

    bench_symbols = list(NASDAQ_BENCHMARKS.values())
    data = _load_symbols(NASDAQ50 + bench_symbols, refresh)
    universe = [s for s in NASDAQ50 if s in data]
    missing = [s for s in NASDAQ50 if s not in data]
    if missing:
        print(f"skipped (no data): {', '.join(missing)}")

    prices = close_prices({s: data[s] for s in universe}).loc[start:].dropna()
    freq = REBALANCE[rebalance]
    runs = _run_strategies(prices, cost_bps, freq)
    if report_dir:
        bench = close_prices({s: data[s] for s in bench_symbols}).reindex(prices.index)
        benchmarks = {
            name: bench[sym].pct_change().fillna(0.0)
            for name, sym in NASDAQ_BENCHMARKS.items()
        }
        _write_reports(runs, benchmarks, Path(report_dir))

    return _summarize_and_log(
        runs, prices, "nasdaq50", rebalance, cost_bps, split_date, hypothesis_ref
    )


def run_beta_scaled(
    start: str = "2006-01-01",
    cost_bps: float = 2.0,
    beta_lb: int = 63,
    refresh: bool = False,
    report_dir: str | Path | None = "reports/beta_scaled",
    hypothesis_ref: str | None = None,
) -> pd.DataFrame:
    data = _load_symbols(FX_GOLD + ["SPY", "QQQ"], refresh)
    panel = close_prices(data).loc[start:].dropna()
    prices = panel[FX_GOLD]

    weights = rebalance_targets(
        betting_against_beta(prices, panel["SPY"], beta_lb), "ME"
    )
    runs = {"beta_scaled": backtest_weights(prices, weights, cost_bps)}

    if report_dir:
        benchmarks = {
            "sp500": panel["SPY"].pct_change().fillna(0.0),
            "nasdaq": panel["QQQ"].pct_change().fillna(0.0),
        }
        _write_reports(runs, benchmarks, Path(report_dir))

    return _summarize_and_log(
        runs, prices, "fx_gold", "monthly", cost_bps, None, hypothesis_ref
    )


def run_beta_timing(
    start: str = "2006-01-01",
    cost_bps: float = 5.0,
    beta_lb: int = 252,
    refresh: bool = False,
    hypothesis_ref: str | None = None,
) -> pd.DataFrame:
    symbols = BETA_STOCKS + FX_GOLD
    data = _load_symbols(symbols + ["SPY"], refresh)
    if "SPY" not in data:
        raise ValueError("SPY (market proxy) failed to download")
    market = close_prices({"SPY": data["SPY"]})["SPY"]

    rows = []
    for a in symbols:
        if a not in data:
            print(f"skipped (no data): {a}")
            continue
        px = close_prices({a: data[a]})[a]
        pair = pd.concat([px.rename(a), market.rename("SPY")], axis=1).loc[start:].dropna()
        if len(pair) < beta_lb + 60:
            continue

        w = beta_timing(pair[[a]], pair["SPY"], beta_lb)
        res = backtest_weights(pair[[a]], w, cost_bps)
        bh = pair[a].pct_change().fillna(0.0)
        held = w[a][w[a] != 0.0]
        rows.append({
            "asset": a,
            "years": round((pair.index[-1] - pair.index[0]).days / 365.25, 1),
            "avg_beta": round(float(held.mean()), 2),
            "strat_sharpe": round(res.sharpe, 2),
            "bh_sharpe": round(_sharpe(bh), 2),
            "ann_return": round(_annual_return(res.equity), 4),
            "max_dd": round(res.max_drawdown, 3),
            "turnover": round(float(res.turnover.mean()), 4),
        })

    summary = pd.DataFrame(rows).sort_values("avg_beta", ascending=False)
    log_run({
        "hypothesis_ref": hypothesis_ref,
        "universe": "stocks+fx_gold_individual",
        "n_assets": int(len(summary)),
        "cost_bps": cost_bps,
        "beta_lb": beta_lb,
        "strategy": "beta_timing",
        "metrics": summary.set_index("asset").to_dict(orient="index"),
        "status": "success",
    })
    return summary


def run_beta_long_short(
    start: str = "2012-06-01",
    cost_bps: float = 5.0,
    beta_lb: int = 252,
    refresh: bool = False,
    report_dir: str | Path | None = "reports/beta_long_short",
    hypothesis_ref: str | None = None,
) -> pd.DataFrame:
    data = _load_symbols(BETA_STOCKS + ["SPY", "QQQ"], refresh)
    panel = close_prices(data).loc[start:].dropna()
    prices = panel[BETA_STOCKS]

    weights = beta_long_short(prices, panel["SPY"], beta_lb)
    runs = {"beta_long_short": backtest_weights(prices, weights, cost_bps)}

    if report_dir:
        benchmarks = {
            "sp500": panel["SPY"].pct_change().fillna(0.0),
            "nasdaq": panel["QQQ"].pct_change().fillna(0.0),
        }
        _write_reports(runs, benchmarks, Path(report_dir))

    return _summarize_and_log(
        runs, prices, "us_stocks_10", "daily", cost_bps, None, hypothesis_ref
    )


def _summarize_and_log(
    runs: dict[str, BacktestResult],
    prices: pd.DataFrame,
    universe: str,
    rebalance: str,
    cost_bps: float,
    split_date: str | None,
    hypothesis_ref: str | None,
) -> pd.DataFrame:
    summary = pd.DataFrame(_summaries(runs, split_date))
    metric_cols = ["total_return", "annual_return", "sharpe", "max_drawdown", "avg_daily_turnover"]
    for name, grp in summary.groupby("strategy"):
        log_run({
            "hypothesis_ref": hypothesis_ref,
            "universe": universe,
            "sample_start": str(prices.index[0].date()),
            "sample_end": str(prices.index[-1].date()),
            "n_assets": int(prices.shape[1]),
            "rebalance": rebalance,
            "cost_bps": cost_bps,
            "strategy": name,
            "metrics": {r["split"]: {c: r[c] for c in metric_cols} for _, r in grp.iterrows()},
            "status": "success",
        })
    return summary


def _load_symbols(symbols: list[str], refresh: bool) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            data.update(load_yahoo_ohlcv([symbol], refresh=refresh))
        except Exception as err:
            print(f"download failed for {symbol}: {err}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["etf", "nasdaq50"], default="etf")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--report-dir")
    parser.add_argument("--rebalance", choices=sorted(REBALANCE), default="monthly")
    parser.add_argument("--split-date", default="2016-01-01")
    parser.add_argument("--no-split", action="store_true")
    parser.add_argument("--hypothesis-ref", help="research_log.md entry id this run tests")
    args = parser.parse_args()

    runner = run_nasdaq50 if args.universe == "nasdaq50" else run_etf_baseline
    res = runner(
        start=args.start,
        cost_bps=args.cost_bps,
        refresh=args.refresh,
        report_dir=args.report_dir,
        rebalance=args.rebalance,
        split_date=None if args.no_split else args.split_date,
        hypothesis_ref=args.hypothesis_ref,
    )
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _run_strategies(
    prices: pd.DataFrame,
    cost_bps: float,
    freq: str | None,
) -> dict[str, BacktestResult]:
    all_assets = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    return {
        "equal_weight": backtest_weights(
            prices,
            rebalance_targets(equal_weight(all_assets), freq),
            cost_bps,
        ),
        "inverse_vol_63d": backtest_weights(
            prices,
            rebalance_targets(inverse_vol(prices, 63), freq),
            cost_bps,
        ),
        "momentum_126d_top2": backtest_weights(
            prices,
            rebalance_targets(long_top_momentum(prices, lookback=126, count=2), freq),
            cost_bps,
        ),
        "mvo_min_variance": backtest_weights(
            prices,
            rolling_mvo_weights(prices, "min_variance", rebalance=freq or "D"),
            cost_bps,
        ),
        "mvo_max_sharpe": backtest_weights(
            prices,
            rolling_mvo_weights(prices, "max_sharpe", rebalance=freq or "D"),
            cost_bps,
        ),
    }


def _summaries(
    runs: dict[str, BacktestResult],
    split_date: str | None,
) -> list[dict[str, float | str]]:
    rows = []
    for name, res in runs.items():
        rows.append(_summary(name, "full", res))
        if split_date:
            split = pd.Timestamp(split_date)
            rows.append(_summary(name, "train", res, end=split - pd.Timedelta(days=1)))
            rows.append(_summary(name, "test", res, start=split))
    return rows


def _summary(
    name: str,
    split: str,
    res: BacktestResult,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict[str, float | str]:
    returns = res.returns.loc[start:end]
    equity = (1.0 + returns).cumprod()
    turnover = res.turnover.loc[start:end]
    return {
        "strategy": name,
        "split": split,
        "total_return": float(equity.iloc[-1] - 1.0),
        "annual_return": _annual_return(equity),
        "sharpe": _sharpe(returns),
        "max_drawdown": _max_drawdown(equity),
        "avg_daily_turnover": float(turnover.mean()),
    }


def _annual_return(equity: pd.Series) -> float:
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return 0.0
    return float(equity.iloc[-1] ** (365.25 / days) - 1.0)


def _sharpe(returns: pd.Series) -> float:
    if returns.std() == 0:
        return 0.0
    return float((252**0.5) * returns.mean() / returns.std())


def _max_drawdown(equity: pd.Series) -> float:
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _write_reports(
    runs: dict[str, BacktestResult],
    benchmarks: dict[str, pd.Series],
    report_dir: Path,
) -> None:
    import quantstats as qs

    for bench_name, benchmark in benchmarks.items():
        bench_dir = report_dir / f"vs_{bench_name}"
        bench_dir.mkdir(parents=True, exist_ok=True)
        for name, res in runs.items():
            qs.reports.html(
                res.returns,
                benchmark=benchmark,
                output=str(bench_dir / f"{name}.html"),
                title=f"{name} vs {bench_name}",
            )


if __name__ == "__main__":
    main()
