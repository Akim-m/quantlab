"""RL-2026-07-19: sizing the three promoted books (REGIME + F&O L/S + multi-asset trend).

Returns-level blend: r = sum_i w_i * r_i with FIXED weights and NO rebalancing dynamics.
Each book already charges its own turnover cost internally (its construction backtests
at the headline cost), so a fixed-weight combination just re-weights three daily P&L
streams - there is no cross-book rebalancing to cost. Four locked a-priori sizing rules;
ONE is chosen on the TRAIN (2010->2016-12-31) combined Sharpe, frozen, then read ONCE on
test (2017+) vs REGIME-alone.

The three books are rebuilt via their frozen constructions in `xasset_trend`:
  REGIME long-only  (bear_sleeve.base_book)   -> test SR 1.865  [reproduction gate],
  F&O-shortable L/S (india_ls LS-FNO-THIN)     -> test SR ~0.846,
  5-ETF trend sleeve (incl. its 2019-12 data repair) -> test SR ~1.057.
No Groww live call: the L/S short universe reads the cached instrument master from disk.

Weights are derived once from the 20 bps headline TRAIN returns and applied UNCHANGED
across the 10/20/40 bps test cost checks (they are a-priori sizing, not a per-cost fit).
Deployment bar (pre-registered): paired-t of the daily (blend - REGIME) difference > 1
AND blend maxDD better-or-equal to REGIME, robust at all three costs. A fail is a valid
deliverable (~50% prior - REGIME's own Sharpe is a high hurdle).
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from .evaluation import sharpe_tstat
from .optimization import erc_weights
from .tracking import log_run
from .xasset_trend import FROZEN_GATE, FROZEN_WEIGHTING, base_returns, etf_panel, sleeve_ret

BOOKS = ("regime", "ls", "trend")
RULES = ("regime", "thirds", "invvol", "erc")
TRAIN0, TRAIN1 = "2010-01-01", "2016-12-31"
# Frozen on TRAIN (2010->2016-12-31) combined Sharpe over the four rules (invvol 1.564 >
# erc 1.561 > thirds 1.512 > regime 1.290), before the single test read.
FROZEN_RULE = "invvol"
CUMULATIVE_INDIA_TRIALS = 72  # 68 prior India trials + these 4 sizing rules


def book_returns(cost_bps: float, refresh: bool = False) -> pd.DataFrame:
    """The three promoted books' daily returns on the REGIME (deployed book) calendar.

    REGIME and the F&O L/S sleeve share that calendar exactly; the 5-ETF trend sleeve
    (its own ETF calendar) is reindexed onto it and flat (0) on any non-overlapping day.
    All three are rebuilt via their frozen constructions in xasset_trend."""
    regime, ls = base_returns(cost_bps, refresh=refresh)
    trend = sleeve_ret(etf_panel(refresh=refresh), FROZEN_GATE, FROZEN_WEIGHTING, cost_bps)
    df = pd.DataFrame(index=regime.index)
    df["regime"], df["ls"], df["trend"] = regime, ls, trend.reindex(regime.index)
    return df.fillna(0.0)


def rule_weights(rule: str, train_ret: pd.DataFrame) -> pd.Series:
    """A-priori sizing weights (sum to 1) from TRAIN daily returns only.

    regime = 100% the deployed book [baseline]; thirds = equal; invvol proportional to
    1/std_i; erc = equal-risk-contribution on the TRAIN covariance. All four use only the
    passed TRAIN slice, so post-split returns cannot move them."""
    cols = list(train_ret.columns)
    if rule == "regime":
        w = pd.Series([1.0] + [0.0] * (len(cols) - 1), index=cols)
    elif rule == "thirds":
        w = pd.Series(1.0 / len(cols), index=cols)
    elif rule == "invvol":
        iv = 1.0 / train_ret.std()
        w = iv / iv.sum()
    elif rule == "erc":
        w = erc_weights(train_ret.cov())
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return w.reindex(cols)


def blend_ret(ret: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Fixed-weight returns-level blend r = sum_i w_i * r_i."""
    return ret[list(w.index)] @ w


def _metrics(r: pd.Series) -> tuple[float, float, float]:
    """(Sharpe, annualized return, max drawdown) of a daily return series."""
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    ann = float((1.0 + r).prod() ** (252 / max(len(r), 1)) - 1.0)
    return round(sharpe_tstat(r)[0], 3), round(ann, 4), round(dd, 4)


def run(split: str = "2017-01-01", refresh: bool = False) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 30)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        r20 = book_returns(20.0, refresh=refresh)
        regime_sr, _, _ = _metrics(r20["regime"].loc[split:])
        ls_sr = _metrics(r20["ls"].loc[split:])[0]
        trend_sr = _metrics(r20["trend"].loc[split:])[0]

        print(f"3-book blend  {r20.index[0].date()}->{r20.index[-1].date()}  test>={split}")
        print(f"[BASE reproduction]  REGIME test SR={regime_sr} (gate 1.865)  "
              f"L/S={ls_sr} (~0.846)  TREND={trend_sr} (~1.057)")
        if abs(regime_sr - 1.865) > 0.001:
            print("STOP: REGIME reproduction gate FAILED (test SR != 1.865); aborting the read.")
            return

        train = r20.loc[TRAIN0:TRAIN1]
        weights = {rule: rule_weights(rule, train) for rule in RULES}

        train_rows = []
        for rule in RULES:
            sr, ann, dd = _metrics(blend_ret(train, weights[rule]))
            train_rows.append({"rule": rule, "w_regime": round(weights[rule]["regime"], 4),
                               "w_ls": round(weights[rule]["ls"], 4), "w_trend": round(weights[rule]["trend"], 4),
                               "train_sharpe": sr, "train_ann": ann, "train_maxdd": dd})
        train_tbl = pd.DataFrame(train_rows).sort_values("train_sharpe", ascending=False).reset_index(drop=True)
        argmax = str(train_tbl.iloc[0]["rule"])
        print(f"\n[TRAIN {TRAIN0}->{TRAIN1} @20bps | 4 rules -> FREEZE on combined Sharpe]")
        print(train_tbl.to_string(index=False))
        print(f"FROZEN: {FROZEN_RULE}  (TRAIN argmax={argmax}; "
              f"{'matches' if argmax == FROZEN_RULE else 'MISMATCH - freeze held per protocol'})")

        print(f"\n[TEST {split}+ ONE read | all rules @10/20/40 bps | verdict binds to {FROZEN_RULE} vs regime]")
        print(f"{'cost':>4} {'rule':>8} | {'SR':>6} {'ann':>8} {'maxDD':>8} {'pairT_vs_regime':>16}")
        test_log = {}
        for cost in (10.0, 20.0, 40.0):
            rc = book_returns(cost, refresh=refresh).loc[split:]
            reg = rc["regime"]
            reg_dd = _metrics(reg)[2]
            for rule in RULES:
                b = blend_ret(rc, weights[rule])
                sr, ann, dd = _metrics(b)
                pair_t = 0.0 if rule == "regime" else round(sharpe_tstat(b - reg)[1], 2)
                print(f"{int(cost):>4} {rule:>8} | {sr:6.3f} {ann:8.4f} {dd:8.4f} {pair_t:16.2f}")
                test_log.setdefault(cost, {})[rule] = {"sharpe": sr, "ann": ann, "maxdd": dd,
                                                        "paired_t_vs_regime": pair_t,
                                                        "dmaxdd_vs_regime": round(dd - reg_dd, 4)}
            print()

    def passes(cost: float) -> bool:
        m = test_log[cost][FROZEN_RULE]
        return m["paired_t_vs_regime"] > 1.0 and m["dmaxdd_vs_regime"] >= 0.0
    bar = {int(c): passes(c) for c in (10.0, 20.0, 40.0)}
    deploy = all(bar.values())
    print(f"[DEPLOYMENT BAR] paired-t>1 AND maxDD better-or-equal vs REGIME, per cost {bar}  -> deploy={deploy}")

    fw = weights[FROZEN_RULE]
    for cost, mrow in test_log.items():
        log_run({"hypothesis_ref": "RL-2026-07-19", "universe": "REGIME+FNO_LS+ETF_TREND",
                 "cost_bps": cost, "split": split, "strategy": f"blend_{FROZEN_RULE}",
                 "frozen_rule": FROZEN_RULE, "frozen_weights": {k: round(float(fw[k]), 4) for k in BOOKS},
                 "metrics": mrow[FROZEN_RULE], "all_rules": mrow, "deploy": bool(passes(cost)),
                 "n_trials_family": 4, "n_trials_cumulative": CUMULATIVE_INDIA_TRIALS, "status": "success"})


def main() -> None:
    p = argparse.ArgumentParser(description="RL-2026-07-19 three-book portfolio blend")
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(split=a.split, refresh=a.refresh)


if __name__ == "__main__":
    main()
