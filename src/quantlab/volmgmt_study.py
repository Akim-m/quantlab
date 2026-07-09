"""RL-2026-07-21: continuous vol-target overlay on the deployed REGIME book.

Moreira-Muir (2017): scaling exposure by inverse realized variance can raise a
momentum book's Sharpe, because volatility is persistent while expected return is
not proportional to it. The deployed book's regime overlay (RL-2026-07-11, test
SR 1.865) is BINARY - fully invested or cash. This tests a CONTINUOUS vol-target
that instead de-risks smoothly into vol clusters, applied on top of the deployed
book by scaling its daily weight rows by s_t = min(1, sigma_target / sigma_hat_t).

Circularity guard (the whole study turns on this): sigma_hat_t is the trailing
realized vol of a FIXED reference series - the UNSCALED deployed book's own
returns, computed once (gross, so the one overlay is re-costed at each bps level
rather than redefined per cost) and causal (sigma_hat at t uses returns through
t-1 only). It is NOT the scaled book's returns, which depend on s_t and would
make the estimate circular.

The scaled book's daily targets = the unscaled book's actual daily (drifted) held
weights x s_t, fed to backtest_weights EVERY day so the daily scale trading is
costed for real: a changing s_t re-trades the whole book even when the underlying
weights are frozen. At s_t == 1 this reproduces the deployed baseline exactly, so
the incremental cost drag is attributable purely to the scaling.

Four locked variants: sigma_target in {10%, 15%} x window in {21d, 63d}. ONE
chosen on TRAIN (2010->2016) Sharpe, frozen, one TEST read (2017+) vs the unscaled
baseline. Deployment bar: paired-t of daily (scaled - baseline) > 1 AND maxDD
better-or-equal, robust at 10/20/40 bps. A wash/negative is the pre-registered
~40%-likely outcome and a valid deliverable.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from .backtest import backtest_weights
from .data import close_prices, load_yahoo_ohlcv
from .evaluation import sharpe_tstat
from .india import india_panel, sector_map
from .portfolio import rebalance_targets
from .riskoff_sleeve import GOLD, VIX_SYM, base_book
from .tracking import log_run

VARIANTS = ((0.10, 21), (0.10, 63), (0.15, 21), (0.15, 63))
# Frozen on TRAIN (2010->2016) Sharpe: 10%/63d (1.529) > 15%/63d (1.382) >
# 10%/21d (1.333) > 15%/21d (1.305), before the test read.
FROZEN = (0.10, 63)
BASE_TEST_SR = 1.865  # deployed-book reproduction gate @20bps


def sigma_hat(ref, window):
    """Annualized trailing realized vol of `ref`, using returns through t-1 only."""
    return ref.rolling(window).std().shift(1) * np.sqrt(252)


def scale_factor(ref, target_vol, window):
    """s_t = min(1, target_vol / sigma_hat_t), and 1.0 where sigma_hat is undefined.

    Capped at 1 so the overlay only ever de-risks - it never levers above the base
    book (long-only, implementable)."""
    s = (target_vol / sigma_hat(ref, window)).clip(upper=1.0)
    return s.fillna(1.0)


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
    monthly = rebalance_targets(base.reindex(columns=pxa.columns).fillna(0.0), "ME")

    # Unscaled book, computed once: the daily held-weight path (cost-independent) and
    # the gross-return reference that feeds sigma_hat.
    unscaled = backtest_weights(pxa, monthly, 0.0)
    W_base, ref = unscaled.weights, unscaled.returns

    print(f"N500 {px.shape[1]} stocks x {len(px)} days  {px.index[0].date()}->{px.index[-1].date()}  "
          f"test>={split}  +{GOLD}")

    # ---------------- reproduction gate: unscaled book must give test SR 1.865 @20bps ----------------
    base20 = backtest_weights(pxa, monthly, 20.0)
    base_sr = _metrics(base20.returns.loc[split:])[0]
    print(f"\n[GATE] unscaled deployed-book test SR@20bps = {base_sr:.3f} (expect {BASE_TEST_SR})")
    assert abs(base_sr - BASE_TEST_SR) < 0.01, f"base reproduction drifted to {base_sr}"

    # ---------------- TRAIN (->2016): pick the variant by scaled-book Sharpe, then FREEZE ----------------
    rows = []
    for tv, w in VARIANTS:
        s = scale_factor(ref, tv, w)
        r = backtest_weights(pxa, W_base.mul(s, axis=0), 20.0).returns.loc[:split].iloc[:-1]
        sr, ann, dd = _metrics(r)
        rows.append({"sigma_target": tv, "window": w, "train_SR": sr, "train_ann": ann,
                     "train_maxDD": dd, "mean_s": round(float(s.loc[:split].mean()), 3)})
    train = pd.DataFrame(rows).sort_values("train_SR", ascending=False).reset_index(drop=True)
    chosen = (float(train.iloc[0]["sigma_target"]), int(train.iloc[0]["window"]))
    print("\n[TRAIN 2010->2016 @20bps: scaled book per variant]")
    print(train.to_string(index=False))
    print(f"FROZEN (max TRAIN SR): {chosen}   (module FROZEN={FROZEN})")

    # ---------------- ONE TEST READ (2017+): frozen scaled book vs unscaled baseline ----------------
    tv, w = FROZEN
    s = scale_factor(ref, tv, w)
    scaled_tgt = W_base.mul(s, axis=0)
    n_test = len(base20.returns.loc[split:])
    yrs = n_test / 252

    print(f"\n[TEST {split}+ ONE read]  frozen = sigma_target {tv:.0%}, window {w}d  "
          f"(mean_s={float(s.loc[split:].mean()):.3f})")
    hdr = (f"{'cost':>4} {'book':>9} | {'SR':>6} {'ann':>8} {'maxDD':>8} | "
           f"{'dSR':>7} {'dMaxDD':>7} {'pairT':>7} {'cost_drag':>10} {'inc_drag':>9}")
    print(hdr)
    test_log = {}
    for cost in (10.0, 20.0, 40.0):
        b = backtest_weights(pxa, monthly, cost)
        sc = backtest_weights(pxa, scaled_tgt, cost)
        b_r, sc_r = b.returns.loc[split:], sc.returns.loc[split:]
        b_sr, b_ann, b_dd = _metrics(b_r)
        sr, ann, dd = _metrics(sc_r)
        pair_t = sharpe_tstat((sc_r - b_r).dropna())[1]
        b_drag = float(b.turnover.loc[split:].sum()) * cost / 1e4 / yrs
        sc_drag = float(sc.turnover.loc[split:].sum()) * cost / 1e4 / yrs
        inc = sc_drag - b_drag
        print(f"{int(cost):>4} {'baseline':>9} | {b_sr:6.3f} {b_ann:8.4f} {b_dd:8.4f} | "
              f"{'':>7} {'':>7} {'':>7} {b_drag:10.4%} {'':>9}")
        print(f"{int(cost):>4} {'scaled':>9} | {sr:6.3f} {ann:8.4f} {dd:8.4f} | "
              f"{sr - b_sr:+7.3f} {dd - b_dd:+7.3f} {pair_t:7.2f} {sc_drag:10.4%} {inc:+9.4%}")
        test_log[cost] = {"sharpe": sr, "ann": ann, "maxdd": dd,
                          "baseline_sharpe": b_sr, "baseline_ann": b_ann, "baseline_maxdd": b_dd,
                          "d_sharpe": round(sr - b_sr, 3), "d_maxdd": round(dd - b_dd, 4),
                          "paired_t": round(pair_t, 2), "inc_cost_drag": round(inc, 6)}

    # ---------------- deployment bar (pre-registered): paired-t>1 AND maxDD better-or-equal ----------------
    def passes(cost):
        m = test_log[cost]
        return m["paired_t"] > 1.0 and m["d_maxdd"] >= 0.0
    bar = {int(c): passes(c) for c in (20.0, 10.0, 40.0)}
    deploy = all(bar.values())
    print(f"\n[DEPLOYMENT BAR] paired-t>1 AND maxDD better-or-equal, per cost {bar}  -> deploy={deploy}")

    for cost, m in test_log.items():
        log_run({"hypothesis_ref": "RL-2026-07-21", "universe": "NIFTY500+GOLDBEES",
                 "cost_bps": cost, "split": split, "strategy": "REGIME+voltarget_overlay",
                 "frozen_sigma_target": tv, "frozen_window": w, "metrics": m,
                 "deploy": bool(passes(cost)), "n_trials_family": len(VARIANTS), "status": "success"})


def main():
    p = argparse.ArgumentParser(description="RL-2026-07-21 vol-target overlay study")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(start=a.start, split=a.split, refresh=a.refresh)


if __name__ == "__main__":
    main()
