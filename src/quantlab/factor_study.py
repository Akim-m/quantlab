"""RL-2026-07-07 anomaly-replication study.

Backtests all 24 pre-registered factors on the locked ETF24 universe, then judges
each on the single locked TEST window (>= 2016-01-01) under a multiple-testing
correction: Benjamini-Hochberg FDR on the Sharpe t-stats plus the Deflated Sharpe
Ratio (trials=24). One fixed spec per factor - no search.
"""

import argparse

import numpy as np
import pandas as pd

from . import trend, xsec
from .backtest import backtest_weights
from .data import close_prices, load_yahoo_ohlcv
from .evaluation import benjamini_hochberg, deflated_sharpe_ratio, one_sided_p, sharpe_tstat
from .optimization import erc_weights, rolling_construction
from .portfolio import rebalance_targets
from .tracking import log_run

ETF24 = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "XLB", "XLE", "XLF", "XLI", "XLK",
         "XLP", "XLU", "XLV", "XLY", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "DBC"]
START = "2007-06-01"
SPLIT = "2016-01-01"
COST_BPS = 5.0


def _factors(px, mkt, open_px, close_px):
    """(code, name, daily-or-sparse weights, rebalance freq). None = pass through."""
    return [
        ("01", "short_term_reversal", xsec.short_term_reversal(px, 5), "W-FRI"),
        ("02", "momentum_12_1", xsec.momentum_12_1(px), "ME"),
        ("03", "long_term_reversal", xsec.long_term_reversal(px), "ME"),
        ("04", "low_volatility", xsec.low_volatility(px, 252), "ME"),
        ("05", "idio_vol", xsec.idio_vol(px, mkt, 252), "ME"),
        ("06", "max_lottery", xsec.max_lottery(px, 21), "ME"),
        ("07", "high_52w", xsec.high_52w(px, 252), "ME"),
        ("08", "skewness", xsec.skewness(px, 252), "ME"),
        ("09", "residual_momentum", xsec.residual_momentum(px, mkt, 252, 21), "ME"),
        ("10", "seasonality", xsec.seasonality(px), "ME"),
        ("11", "downside_beta", xsec.downside_beta_factor(px, mkt, 252), "ME"),
        ("12", "tsmom", trend.tsmom(px), "ME"),
        ("13", "donchian", trend.donchian(px), "ME"),
        ("14", "dual_momentum", trend.dual_momentum(px), "ME"),
        ("15", "vol_managed", trend.vol_managed(px), "ME"),
        ("16", "crash_scaled_tsmom", trend.crash_scaled_tsmom(px), "ME"),
        ("17", "overnight_intraday", trend.overnight_intraday(open_px, close_px), None),
        ("18", "bollinger", trend.bollinger(px), "W-FRI"),
        ("19", "pairs", trend.pairs(px), "ME"),
        ("20", "risk_parity_erc", rolling_construction(px, "erc"), None),
        ("21", "max_diversification", rolling_construction(px, "max_div"), None),
        ("22", "hrp", rolling_construction(px, "hrp"), None),
        ("23", "min_correlation", rolling_construction(px, "min_corr"), None),
        ("24", "dual_mom_erc", _dual_mom_erc(px), None),
    ]


def _dual_mom_erc(px, lb=252, lookback=252, rebalance="ME"):
    """Composite: each month, ERC-weight the dual-momentum qualifiers (else cash)."""
    rets = px.pct_change().dropna()
    dm = trend.dual_momentum(px, lb)
    weights = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    for date in px.groupby(pd.Grouper(freq=rebalance)).tail(1).index:
        hist = rets.loc[:date].tail(lookback)
        qual = list(dm.columns[dm.loc[date] > 0]) if date in dm.index else []
        if not qual or len(hist) < lookback:
            weights.loc[date] = 0.0
            continue
        w = erc_weights(hist[qual].cov() * 252)
        weights.loc[date] = 0.0
        weights.loc[date, qual] = w.values
    return weights


def _row(code, name, res):
    full = res.returns
    test = res.returns.loc[SPLIT:]
    sr_full, _ = sharpe_tstat(full)
    sr_test, t_test = sharpe_tstat(test)
    return {
        "code": code, "name": name,
        "full_sharpe": round(sr_full, 3),
        "test_sharpe": round(sr_test, 3),
        "test_tstat": round(t_test, 3),
        "test_p": one_sided_p(t_test, len(test)),
        "turnover": round(float(res.turnover.mean()), 4),
        "_test_returns": test,
    }


def run(refresh: bool = False) -> pd.DataFrame:
    data = load_yahoo_ohlcv(ETF24, refresh=refresh)
    px = close_prices(data).loc[START:].dropna()
    mkt = px["SPY"]
    open_px = pd.DataFrame({s: data[s]["open"] for s in px.columns}).reindex(px.index)
    close_px = pd.DataFrame({s: data[s]["close"] for s in px.columns}).reindex(px.index)
    print(f"ETF24: {px.shape[1]} assets x {len(px)} days, "
          f"{px.index[0].date()} -> {px.index[-1].date()}; test from {SPLIT}")

    rows = []
    for code, name, weights, freq in _factors(px, mkt, open_px, close_px):
        targets = rebalance_targets(weights, freq)
        res = backtest_weights(px, targets, COST_BPS)
        rows.append(_row(code, name, res))

    sr_trials = [r["_test_returns"].mean() / r["_test_returns"].std() for r in rows]
    reject = benjamini_hochberg([r["test_p"] for r in rows], q=0.10)
    for r, rej in zip(rows, reject):
        r["bh_pass"] = bool(rej)
        r["dsr"] = round(deflated_sharpe_ratio(r.pop("_test_returns"), sr_trials), 3)
        r["test_p"] = round(r["test_p"], 4)

    table = pd.DataFrame(rows).sort_values("test_sharpe", ascending=False).reset_index(drop=True)
    survivors = table[(table["bh_pass"]) & (table["dsr"] > 0.95)]
    print(table.to_string(index=False))
    print(f"\nsurvive BH-FDR(q=0.10) AND DSR>0.95 on the locked test window: "
          f"{len(survivors)} of {len(table)}")
    if len(survivors):
        print("  " + ", ".join(f"{r.code} {r.name}" for r in survivors.itertuples()))

    log_run({
        "hypothesis_ref": "RL-2026-07-07",
        "universe": "ETF24",
        "sample_start": str(px.index[0].date()),
        "sample_end": str(px.index[-1].date()),
        "n_assets": int(px.shape[1]),
        "split": SPLIT,
        "cost_bps": COST_BPS,
        "strategy": "anomaly_replication_24",
        "n_trials": len(table),
        "metrics": table.to_dict(orient="records"),
        "survivors": survivors["name"].tolist(),
        "status": "success",
    })
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="24-factor anomaly replication (RL-2026-07-07)")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    run(refresh=args.refresh)


if __name__ == "__main__":
    main()
