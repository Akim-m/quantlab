"""RL-2026-07-26-01: 5-ETF dual-momentum defensive rotation (DUAL-ROT). FORWARD-ONLY.

Antonacci (2014) dual momentum on the same five NSE ETFs as the RL-17 trend sleeve -
NIFTYBEES (large-cap), JUNIORBEES (next-50), BANKBEES (banks), GOLDBEES (gold-INR),
MON100 (Nasdaq-INR). Relative momentum (12-1 total return) ranks the five; the top-K
are held EQUAL-weighted at 1/K each, but a selected sleeve is kept only while its own
absolute momentum is up (Moskowitz-Ooi-Pedersen 2012 crash gate) - a gated-out
selection holds cash, so the book sums to <=1. Concentrating in the strongest 1-2
streams (K in {1,2}) is the bet against the RL-17 sleeve's hold-all-uptrends
inverse-vol diversification.

DESIGN FREEZE on TRAIN 2010-01-01..2016-12-31 ONLY: the four variants (top-K in {1,2}
x absolute gate in {12-1 tsmom sign, px>200d MA}) are scored on combined net Sharpe
(20 bps) and the single best is frozen in FROZEN_TOP_K / FROZEN_GATE below, before any
forward read. This module is FORWARD-ONLY - it computes NO post-2016 backtest number;
the deliverable is the frozen construction and its live paper-track leg
(live_paper.run_dualrot -> paper_trades_dualrot.jsonl). MON100 lists 2011-03 and is
ineligible until it has 12 months of history (disclosed). Reuses xasset_trend's cleaned
panel and _gate primitive (single source of truth for the gate math); the daily weights
are lagged one trading day so a rebalance-date target uses only data through t-1
(prior-day signals, locked spec).
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from .backtest import backtest_weights
from .evaluation import sharpe_tstat
from .portfolio import rebalance_targets
from .xasset_trend import _gate, etf_panel

TOP_KS = (1, 2)
GATES = ("tsmom", "ma")
TRAIN0, TRAIN1 = "2010-01-01", "2016-12-31"
COST_BPS = 20.0
# Frozen on TRAIN (2010->2016-12-31) combined net Sharpe, before any forward read
# (TRAIN argmax over the four variants: K=2 / tsmom @ Sharpe 0.693; see run()):
FROZEN_TOP_K, FROZEN_GATE = 2, "tsmom"


def weights_history(px: pd.DataFrame, top_k: int, gate: str) -> pd.DataFrame:
    """Daily long-only dual-momentum weights: rank the ETFs by 12-1 relative momentum,
    hold the top `top_k` EQUAL-weighted at 1/top_k each, but zero (cash) any selected
    sleeve whose own absolute gate is down. Weights are lagged one trading day, so the
    target on date t is a function of prices through t-1 only (prior-day signals)."""
    rel = _gate(px, "tsmom")                    # 12-1 total return, the relative-strength score
    up = _gate(px, gate) > 0                     # absolute crash gate (tsmom sign, or px>200d MA)
    eligible = rel.notna() & px.notna()
    rank = rel.where(eligible).rank(axis=1, ascending=False, method="first")
    held = (rank <= top_k) & up & eligible
    w = held.astype(float) / float(top_k)        # 1/K per selected+gated slot; the rest is cash
    return w.shift(1).fillna(0.0)


def latest_weights(px: pd.DataFrame, top_k: int = FROZEN_TOP_K,
                   gate: str = FROZEN_GATE) -> pd.Series:
    """Frozen-variant target weights on the last panel date (for the live paper book)."""
    return weights_history(px, top_k, gate).iloc[-1]


def sleeve_ret(px: pd.DataFrame, top_k: int, gate: str, cost_bps: float) -> pd.Series:
    """Monthly-rebalanced (ME) net return series of one dual-momentum variant."""
    w = rebalance_targets(weights_history(px, top_k, gate), "ME")
    return backtest_weights(px, w, cost_bps).returns


def train_scores(px: pd.DataFrame) -> pd.DataFrame:
    """The four variants' TRAIN-window combined net Sharpe (the design-freeze evidence).

    Reads px.loc[TRAIN0:TRAIN1] ONLY - the frozen choice cannot see post-2016 data
    (weights at date t use prices through t-1, so TRAIN-sliced returns never depend on a
    later bar)."""
    rows = []
    for k in TOP_KS:
        for g in GATES:
            sr = sharpe_tstat(sleeve_ret(px, k, g, COST_BPS).loc[TRAIN0:TRAIN1])[0]
            rows.append({"top_k": k, "gate": g, "train_sharpe": round(sr, 3)})
    return pd.DataFrame(rows).sort_values("train_sharpe", ascending=False).reset_index(drop=True)


def run(refresh: bool = False) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 30)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        px = etf_panel(refresh=refresh)
        scores = train_scores(px)
    argmax = (int(scores.iloc[0]["top_k"]), scores.iloc[0]["gate"])
    frozen = (FROZEN_TOP_K, FROZEN_GATE)

    print(f"DUAL-ROT 5-ETF dual momentum  {px.index[0].date()}->{px.index[-1].date()}  FORWARD-ONLY")
    print(f"MON100 first valid {px['MON100.NS'].first_valid_index().date()} (pre-inception = ineligible)")
    print(f"\n[DESIGN FREEZE  TRAIN {TRAIN0}->{TRAIN1} @{COST_BPS:.0f}bps | 4 variants -> combined Sharpe]")
    print(scores.to_string(index=False))
    print(f"FROZEN: top_k={FROZEN_TOP_K} gate={FROZEN_GATE}  "
          f"(TRAIN argmax = K{argmax[0]}/{argmax[1]}; "
          f"{'matches' if argmax == frozen else 'MISMATCH - freeze held per protocol'})")
    print("FORWARD-ONLY: no test-window backtest is computed or reported (protocol).")


def main() -> None:
    p = argparse.ArgumentParser(
        description="RL-2026-07-26-01 ETF dual-momentum rotation (DUAL-ROT) design-freeze print")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(refresh=a.refresh)


if __name__ == "__main__":
    main()
