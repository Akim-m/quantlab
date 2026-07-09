"""RL-2026-07-16: risk-off sleeve - should the defensive half own more than cash?

The deployed book (top-decile conviction momentum scaled to cash by the
(200d-MA OR India-VIX) overlay; RL-2026-07-11, test SR 1.865) parks the freed
weight in cash whenever the overlay fires (~1/3 of test days, including today).
This tests four locked sleeves for that freed weight and picks ONE on TRAIN
(2010->2016) evidence, freezing it before a single TEST read (2017+):
  (a) cash   - the deployed baseline,
  (b) gold   - GOLDBEES.NS held only while above its own 200d MA, else cash,
  (c) lowbeta- the panel's lowest-beta decile, inverse-vol weighted,
  (d) gold_lowbeta - 50/50 of the two.

Construction is at the WEIGHTS level on a panel augmented with GOLDBEES.NS: each
day the sleeve fills the freed weight (1 - base gross), so one backtest_weights
call charges every turnover - base rebalancing, sleeve entry/exit on regime flips,
and the sleeve's own monthly refresh. Causal: every target at date t uses only
prices through t (the backtest applies it to earn from t+1), exactly as the
deployed overlay does. A wash/negative is the pre-registered ~40%-likely outcome
and a valid deliverable.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from .backtest import backtest_weights
from .blend import composite, conviction_topq, long_only_topq, regime_on
from .data import close_prices, load_yahoo_ohlcv
from .evaluation import sharpe_tstat
from .features import rolling_beta
from .india import india_panel, sector_map
from .india_blend_study import raw_signals
from .portfolio import rebalance_targets
from .tracking import log_run

CORE = ("mom_12_1", "sharpe_mom", "resid_mom")
GOLD = "GOLDBEES.NS"
VIX_SYM = "^INDIAVIX"
VARIANTS = ("cash", "gold", "lowbeta", "gold_lowbeta")
# Frozen on TRAIN (2010->2016) combined-Sharpe evidence (lowbeta 1.60 > gold_lowbeta
# 1.46 > cash 1.28 > gold 1.14), before the test read:
FROZEN_VARIANT = "lowbeta"
# 60 prior India trials (RL-07-10/11/13) + these 4 sleeve variants.
CUMULATIVE_INDIA_TRIALS = 64


def base_book(px, mkt, vix, sectors):
    """The deployed book: top-decile conviction momentum scaled to cash by the
    (200d-MA OR India-VIX) regime overlay. Weights sum to 1 when risk-on, 0 when
    risk-off (the freed weight the sleeve fills)."""
    sigs = raw_signals(px, mkt, sectors)
    book = conviction_topq(composite({k: sigs[k] for k in CORE}), top=0.1)
    on = regime_on(mkt, vix, 200, 252, 0.80).reindex(book.index).fillna(False).astype(float)
    return book.mul(on, axis=0)


def gold_gate(gold, lb=200):
    """Causal gold-trend flag: GOLDBEES above its own `lb`-day MA (known at t)."""
    return gold > gold.rolling(lb).mean()


def low_beta_book(px, mkt, top=0.1):
    """Lowest-beta decile of the panel, inverse-vol weighted, sums to 1. Reuses the
    deployed invvol top-quantile primitive on the -beta score (raw_signals.low_beta)."""
    return long_only_topq(-rolling_beta(px, mkt, 252), px, top=top, weighting="invvol")


def sleeve_book(variant, px, mkt, gold, cols):
    """Daily sleeve weights on the augmented column set `cols` (panel + GOLD).

    Rows sum to 1 where fully invested, less when the gold gate is off (that share
    stays cash). 'cash' is all zeros - the deployed baseline."""
    z = pd.DataFrame(0.0, index=px.index, columns=cols)
    if variant == "cash":
        return z
    gold_leg = z.copy()
    gold_leg[GOLD] = gold_gate(gold).reindex(px.index).fillna(False).astype(float)
    lb = low_beta_book(px, mkt).reindex(columns=cols).fillna(0.0)
    if variant == "gold":
        return gold_leg
    if variant == "lowbeta":
        return lb
    if variant == "gold_lowbeta":
        return 0.5 * gold_leg + 0.5 * lb
    raise ValueError(f"unknown variant {variant!r}")


def fill_freed(base_aug, sleeve):
    """Combined daily book: route the freed weight (1 - base gross, clipped at 0)
    into the sleeve. `base_aug` and `sleeve` share the augmented columns."""
    freed = (1.0 - base_aug.sum(axis=1)).clip(lower=0.0)
    return base_aug.add(sleeve.mul(freed, axis=0), fill_value=0.0)


def combined_book(variant, px, pxa, mkt, gold, base):
    """Daily combined weights: base on risk-on days, base + sleeve-filled freed
    weight on risk-off days, on the augmented columns."""
    sleeve = sleeve_book(variant, px, mkt, gold, pxa.columns)
    base_aug = base.reindex(columns=pxa.columns).fillna(0.0)
    return fill_freed(base_aug, sleeve)


def variant_returns(variant, px, pxa, mkt, gold, base, cost):
    book = combined_book(variant, px, pxa, mkt, gold, base)
    return backtest_weights(pxa, rebalance_targets(book, "ME"), cost).returns


def _metrics(r):
    """(sharpe, ann_return, max_drawdown) of a daily return series."""
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    ann = float((1.0 + r).prod() ** (252 / max(len(r), 1)) - 1.0)
    return round(sharpe_tstat(r)[0], 3), round(ann, 4), round(dd, 4)


def run(start="2010-01-01", split="2017-01-01", refresh=False):
    pd.set_option("display.width", 220, "display.max_columns", 40)

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        px, mkt, _, _ = india_panel(start=start, index="nifty500", ret_clip=0.40, refresh=refresh)
        vix = close_prices(load_yahoo_ohlcv([VIX_SYM], refresh=refresh))[VIX_SYM].reindex(px.index).ffill()
        gold = close_prices(load_yahoo_ohlcv([GOLD], refresh=refresh))[GOLD].reindex(px.index).ffill()
        base = base_book(px, mkt, vix, sector_map("nifty500"))
    pxa = px.copy()
    pxa[GOLD] = gold

    print(f"N500 {px.shape[1]} stocks x {len(px)} days  {px.index[0].date()}->{px.index[-1].date()}  "
          f"test>={split}  +{GOLD}")

    # ---------------- TRAIN (->2016): pick the variant by combined Sharpe, then FREEZE ----------------
    rets20 = {v: variant_returns(v, px, pxa, mkt, gold, base, 20.0) for v in VARIANTS}
    rows = []
    for v in VARIANTS:
        tr = rets20[v].loc[:split].iloc[:-1]
        sr, ann, dd = _metrics(tr)
        rows.append({"variant": v, "train_SR": sr, "train_ann": ann, "train_maxDD": dd})
    train = pd.DataFrame(rows).sort_values("train_SR", ascending=False).reset_index(drop=True)
    chosen = str(train.iloc[0]["variant"])
    print("\n[TRAIN 2010->2016 @20bps: combined book per variant]")
    print(train.to_string(index=False))
    print(f"FROZEN (max TRAIN combined SR): {chosen}   (module FROZEN_VARIANT={FROZEN_VARIANT})")

    # ---------------- ONE TEST READ (2017+): all variants + paired-t vs the cash baseline ----------------
    print(f"\n[TEST {split}+ ONE read]  frozen sleeve = {FROZEN_VARIANT}")
    hdr = (f"{'cost':>4} {'variant':>13} | {'SR':>6} {'ann':>8} {'maxDD':>8} | "
           f"{'dSR_vs_cash':>11} {'dMaxDD':>7} {'pairT_vs_cash':>13}")
    print(hdr)
    test_log = {}
    for cost in (10.0, 20.0, 40.0):
        rc = {v: variant_returns(v, px, pxa, mkt, gold, base, cost).loc[split:] for v in VARIANTS}
        b_sr, b_ann, b_dd = _metrics(rc["cash"])
        for v in VARIANTS:
            sr, ann, dd = _metrics(rc[v])
            diff = (rc[v] - rc["cash"]).dropna()
            pair_t = sharpe_tstat(diff)[1] if v != "cash" else 0.0
            print(f"{int(cost):>4} {v:>13} | {sr:6.3f} {ann:8.4f} {dd:8.4f} | "
                  f"{sr - b_sr:+11.3f} {dd - b_dd:+7.3f} {pair_t:13.2f}")
            test_log.setdefault(cost, {})[v] = {
                "sharpe": sr, "ann": ann, "maxdd": dd,
                "d_sharpe_vs_cash": round(sr - b_sr, 3), "d_maxdd_vs_cash": round(dd - b_dd, 4),
                "paired_t_vs_cash": round(pair_t, 2)}

    # ---------------- deployment bar (pre-registered): paired-t > 1 AND maxDD not worse by >2pts, robust 10/40 ----------------
    def passes(cost):
        m = test_log[cost][FROZEN_VARIANT]
        return m["paired_t_vs_cash"] > 1.0 and m["d_maxdd_vs_cash"] >= -0.02
    bar = {int(c): passes(c) for c in (20.0, 10.0, 40.0)}
    deploy = bar[20] and bar[10] and bar[40]
    print(f"\n[DEPLOYMENT BAR] paired-t>1 AND maxDD not worse by >2pts, per cost {bar}  "
          f"-> deploy={deploy}")

    for cost, mrow in test_log.items():
        log_run({"hypothesis_ref": "RL-2026-07-16", "universe": "NIFTY500+GOLDBEES",
                 "cost_bps": cost, "split": split, "strategy": f"REGIME+{FROZEN_VARIANT}_sleeve",
                 "frozen_variant": FROZEN_VARIANT, "metrics": mrow[FROZEN_VARIANT],
                 "all_variants": mrow, "deploy": bool(passes(cost)),
                 "n_trials_cumulative": CUMULATIVE_INDIA_TRIALS, "status": "success"})


def main():
    p = argparse.ArgumentParser(description="RL-2026-07-16 risk-off sleeve study")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(start=a.start, split=a.split, refresh=a.refresh)


if __name__ == "__main__":
    main()
