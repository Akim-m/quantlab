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
# NSE sector indices (survivorship-bias-free), market = Nifty 50 (^NSEI)
NSE_SECTORS = ["^NSEBANK", "^CNXIT", "^CNXAUTO", "^CNXPHARMA", "^CNXFMCG", "^CNXMETAL",
               "^CNXENERGY", "^CNXREALTY", "^CNXINFRA", "^CNXPSUBANK", "^CNXMEDIA", "^CNXPSE"]
START = "2007-06-01"
SPLIT = "2016-01-01"
COST_BPS = 5.0


def _factors(px, mkt, open_px, close_px, high_px, low_px, volume_px):
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
        ("25", "bab_beta", xsec.bab_beta(px, mkt, 252), "ME"),
        ("26", "sharpe_momentum", xsec.sharpe_momentum(px), "ME"),
        ("27", "kurtosis", xsec.kurtosis(px, 252), "ME"),
        ("28", "low_52w", xsec.low_52w(px, 252), "ME"),
        ("29", "parkinson_lowrange", xsec.parkinson_lowrange(high_px, low_px, 21), "ME"),
        ("30", "turn_of_month", trend.turn_of_month(px), None),
        ("31", "ma_trend", trend.ma_trend(px), "ME"),
        ("32", "volume_momentum", trend.volume_momentum(px, volume_px), "ME"),
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


def _row(code, name, res, split):
    full = res.returns
    test = res.returns.loc[split:]
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


def run(universe=None, market_sym="SPY", cost_bps=COST_BPS, split=SPLIT, start=START,
        label="ETF24", hypothesis_ref="RL-2026-07-07", refresh=False) -> pd.DataFrame:
    universe = universe or ETF24
    data = load_yahoo_ohlcv(list(dict.fromkeys(universe + [market_sym])), refresh=refresh)
    panel = close_prices(data).loc[start:].dropna()
    px = panel[[s for s in universe if s in panel.columns]]
    mkt = panel[market_sym]
    ohlcv = {c: pd.DataFrame({s: data[s][c] for s in px.columns}).reindex(px.index)
             for c in ("open", "close", "high", "low", "volume")}
    print(f"{label}: {px.shape[1]} assets x {len(px)} days, "
          f"{px.index[0].date()} -> {px.index[-1].date()}; test from {split}, cost {cost_bps}bps")

    rows = []
    factors = _factors(px, mkt, ohlcv["open"], ohlcv["close"],
                       ohlcv["high"], ohlcv["low"], ohlcv["volume"])
    for code, name, weights, freq in factors:
        targets = rebalance_targets(weights, freq)
        res = backtest_weights(px, targets, cost_bps)
        rows.append(_row(code, name, res, split))

    # per-period test Sharpes for the DSR benchmark; drop dead (zero-variance)
    # factors like an inapplicable pairs book, whose NaN would poison every DSR
    sr_trials = [s for r in rows
                 if np.isfinite(s := r["_test_returns"].mean() / r["_test_returns"].std())]
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
        "hypothesis_ref": hypothesis_ref,
        "universe": label,
        "sample_start": str(px.index[0].date()),
        "sample_end": str(px.index[-1].date()),
        "n_assets": int(px.shape[1]),
        "split": split,
        "cost_bps": cost_bps,
        "strategy": f"anomaly_replication_{len(table)}",
        "n_trials": len(table),
        "metrics": table.to_dict(orient="records"),
        "survivors": survivors["name"].tolist(),
        "status": "success",
    })
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Anomaly replication study (RL-2026-07-07/08/09)")
    parser.add_argument("--market", choices=["us", "india"], default="us")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.market == "india":
        run(universe=NSE_SECTORS, market_sym="^NSEI", cost_bps=20.0, split="2018-01-01",
            start="2011-08-01", label="NSE12", hypothesis_ref="RL-2026-07-09",
            refresh=args.refresh)
    else:
        run(refresh=args.refresh)


if __name__ == "__main__":
    main()
