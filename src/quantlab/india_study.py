"""RL-2026-07-10: 32-factor family on Nifty 200 single stocks.

The prior India study (RL-2026-07-09) ran the same factor family on 12 NSE sector
PRICE indices and found nothing survived correction. This re-runs on a real
single-stock cross-section with TOTAL-return data (adj_close via .NS) and a longer
history - the two limitations that study's own conclusion flagged. Same factor
specs, same evaluation (BH-FDR + Deflated Sharpe), so the comparison is clean.

`pairs` uses hard-coded US symbols and is inapplicable here (flat book); it is
excluded from the DSR trial set, as in RL-2026-07-09.
"""

import argparse

import numpy as np
import pandas as pd

from . import trend, xsec
from .backtest import backtest_weights
from .evaluation import benjamini_hochberg, deflated_sharpe_ratio, one_sided_p, sharpe_tstat
from .india import BENCHMARK, TRADEABLE_BENCH, india_panel
from .optimization import erc_weights_fast, rolling_construction
from .portfolio import rebalance_targets
from .tracking import log_run

COST_BPS = 20.0  # STT + spread + impact; matches RL-2026-07-09


def _dual_mom_erc_fast(px, lb=252, lookback=252, rebalance="ME"):
    """Factor 24 with scalable ERC: ERC-weight the dual-momentum qualifiers each
    month (else cash). Same rule as factor_study._dual_mom_erc but erc_weights_fast
    so it survives a few-hundred-name universe."""
    rets = px.pct_change().dropna()
    dm = trend.dual_momentum(px, lb)
    weights = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    for date in px.groupby(pd.Grouper(freq=rebalance)).tail(1).index:
        hist = rets.loc[:date].tail(lookback)
        qual = list(dm.columns[dm.loc[date] > 0]) if date in dm.index else []
        if not qual or len(hist) < lookback:
            weights.loc[date] = 0.0
            continue
        w = erc_weights_fast(hist[qual].cov() * 252)
        weights.loc[date] = 0.0
        weights.loc[date, qual] = w.values
    return weights


def _scalable_factors(px, mkt, ohlcv):
    """The RL-2026-07-07/08 factor family, rebuilt for a few-hundred-name universe.

    Specs are copied verbatim from factor_study._factors (kept identical for a clean
    cross-market comparison) EXCEPT construction: ERC (20) and dual-mom-ERC (24) use
    the fast CCD ERC, and max_diversification (21) - SLSQP-only, intractable at this
    size - is DROPPED and disclosed (HRP + min_corr remain the construction reps). We
    build the list directly instead of via _factors, whose eager SLSQP ERC would
    crash before any filtering. So the applicable family is 31 factors, not 32."""
    o, c, h, low, v = (ohlcv["open"], ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"])
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
        ("17", "overnight_intraday", trend.overnight_intraday(o, c), None),
        ("18", "bollinger", trend.bollinger(px), "W-FRI"),
        ("19", "pairs", trend.pairs(px), "ME"),   # US symbols -> flat here, excluded from DSR
        ("20", "risk_parity_erc", rolling_construction(px, "erc_fast"), None),
        ("22", "hrp", rolling_construction(px, "hrp"), None),
        ("23", "min_correlation", rolling_construction(px, "min_corr"), None),
        ("24", "dual_mom_erc", _dual_mom_erc_fast(px), None),
        ("25", "bab_beta", xsec.bab_beta(px, mkt, 252), "ME"),
        ("26", "sharpe_momentum", xsec.sharpe_momentum(px), "ME"),
        ("27", "kurtosis", xsec.kurtosis(px, 252), "ME"),
        ("28", "low_52w", xsec.low_52w(px, 252), "ME"),
        ("29", "parkinson_lowrange", xsec.parkinson_lowrange(h, low, 21), "ME"),
        ("30", "turn_of_month", trend.turn_of_month(px), None),
        ("31", "ma_trend", trend.ma_trend(px), "ME"),
        ("32", "volume_momentum", trend.volume_momentum(px, v), "ME"),
    ]


def _row(code, name, res, split):
    """Per-factor metrics with a TRAIN-window view (pre-split) alongside TEST.

    Train metrics exist so the blend can be designed on train evidence only; the
    test window stays untouched until the final frozen evaluation.
    """
    train = res.returns.loc[:split].iloc[:-1]   # strictly before the split date
    test = res.returns.loc[split:]
    sr_full, _ = sharpe_tstat(res.returns)
    sr_train, t_train = sharpe_tstat(train)
    sr_test, t_test = sharpe_tstat(test)
    return {
        "code": code, "name": name,
        "full_sharpe": round(sr_full, 3),
        "train_sharpe": round(sr_train, 3),
        "train_tstat": round(t_train, 3),
        "test_sharpe": round(sr_test, 3),
        "test_tstat": round(t_test, 3),
        "test_p": one_sided_p(t_test, len(test)),
        "turnover": round(float(res.turnover.mean()), 4),
        "_test_returns": test,
    }


def run(
    start: str = "2010-01-01",
    split: str = "2017-01-01",
    cost_bps: float = COST_BPS,
    index: str = "nifty500",
    label: str | None = None,
    hypothesis_ref: str = "RL-2026-07-10",
    refresh: bool = False,
) -> pd.DataFrame:
    label = label or index.upper()
    px, mkt, ohlcv, kept = india_panel(start=start, index=index, refresh=refresh)
    print(f"{label}: {px.shape[1]} stocks x {len(px)} days, "
          f"{px.index[0].date()} -> {px.index[-1].date()}; test from {split}, cost {cost_bps}bps")

    factors = _scalable_factors(px, mkt, ohlcv)
    rows = []
    for code, name, weights, freq in factors:
        targets = rebalance_targets(weights, freq)
        res = backtest_weights(px, targets, cost_bps)
        rows.append(_row(code, name, res, split))

    # DSR benchmark spread: drop dead/inapplicable books (e.g. US-only `pairs`)
    # whose zero-variance test returns would poison the trial variance
    sr_trials = [r["_test_returns"].mean() / sd
                 for r in rows if (sd := r["_test_returns"].std()) > 0]
    reject = benjamini_hochberg([r["test_p"] for r in rows], q=0.10)
    for r, rej in zip(rows, reject):
        r["bh_pass"] = bool(rej)
        r["dsr"] = round(deflated_sharpe_ratio(r.pop("_test_returns"), sr_trials), 3)
        r["test_p"] = round(r["test_p"], 4)

    table = pd.DataFrame(rows).sort_values("test_sharpe", ascending=False).reset_index(drop=True)
    survivors = table[(table["bh_pass"]) & (table["dsr"] > 0.95)]
    print(table.to_string(index=False))
    print(f"\nsurvive BH-FDR(q=0.10) AND DSR>0.95 on test window: {len(survivors)} of {len(table)}")
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
        "strategy": f"india_anomaly_replication_{len(table)}",
        "n_trials": len(table),
        "metrics": table.to_dict(orient="records"),
        "survivors": survivors["name"].tolist(),
        "status": "success",
    })
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Indian single-stock factor study (RL-2026-07-10)")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--split", default="2017-01-01")
    parser.add_argument("--cost", type=float, default=COST_BPS)
    parser.add_argument("--index", default="nifty500")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    run(start=args.start, split=args.split, cost_bps=args.cost, index=args.index, refresh=args.refresh)


if __name__ == "__main__":
    main()
