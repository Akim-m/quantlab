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

from .backtest import backtest_weights
from .evaluation import benjamini_hochberg, deflated_sharpe_ratio, one_sided_p, sharpe_tstat
from .factor_study import _factors
from .india import BENCHMARK, TRADEABLE_BENCH, india_panel
from .portfolio import rebalance_targets
from .tracking import log_run

COST_BPS = 20.0  # STT + spread + impact; matches RL-2026-07-09


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

    factors = _factors(px, mkt, ohlcv["open"], ohlcv["close"],
                       ohlcv["high"], ohlcv["low"], ohlcv["volume"])
    rows = []
    for code, name, weights, freq in factors:
        targets = rebalance_targets(weights, freq)
        res = backtest_weights(px, targets, cost_bps)
        rows.append(_row(code, name, res, split))

    # DSR benchmark spread: drop dead/inapplicable books (e.g. US-only `pairs`)
    # whose zero-variance test returns would poison the trial variance
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
