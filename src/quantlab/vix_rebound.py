"""RL-2026-07-24: VIX spike-and-recede re-entry overlay on the deployed book.

The deployed book (top-decile conviction momentum gated to cash by the (200d-MA OR
India-VIX) regime overlay; RL-2026-07-11, test SR 1.865) waits out high-VIX risk-off
in cash. India-VIX spikes overshoot and mean-revert, and equity returns in the weeks
after a spike RECEDES are abnormally high (the risk premium realizing). This overlay
overrides the regime gate to risk-ON while a spike is receding, re-holding the
momentum book instead of cash for h days, to harvest that recovery earlier.

Flags (all prior-day info, trailing windows only - causal):
  spike     = VIX above its trailing-252d p-th percentile,
  receding  = VIX below its trailing 5-day maximum.
While both hold the override is armed for the next h days (a fresh trigger inside the
window extends it). Composition: conviction weights x (regime_on OR override) - the
override can only ADD holding on days the base gate says cash, never force cash on a
risk-on day. Costs on every extra turnover flow through backtest_weights on the same
monthly rebalance as the base book, so the overlay is sampled exactly where the base
regime gate is. POWER: the verdict rests on ~10-13 distinct re-entry episodes; a pass
is promoted only to forward confirmation, a null is weak evidence (locked up front).
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from .backtest import backtest_weights
from .blend import composite, conviction_topq, regime_on
from .data import close_prices, load_yahoo_ohlcv
from .evaluation import sharpe_tstat
from .india import india_panel, sector_map
from .india_blend_study import raw_signals
from .portfolio import rebalance_targets
from .riskoff_sleeve import base_book
from .tracking import log_run

CORE = ("mom_12_1", "sharpe_mom", "resid_mom")
VIX_SYM = "^INDIAVIX"
VARIANTS = ((90, 10), (90, 21), (95, 10), (95, 21))
# Frozen on TRAIN (2010->2016) combined-Sharpe evidence (p90/h10 1.743 > p90/h21
# 1.604 > p95/h21 1.526 > p95/h10 1.447), before the test read:
FROZEN = (90, 10)
BASE_TEST_SR = 1.865  # deployed book @20bps test SR - the reproduction gate
CUMULATIVE_INDIA_TRIALS = 84  # ~80 through RL-07-21 + these 4 (approximate family tally)


def spike_recede(vix, p):
    """Both trigger flags AND-ed: spike (VIX above its trailing-252d p-th percentile)
    and receding (VIX below its trailing 5-day maximum). Causal - both use trailing
    windows only, so perturbing any future VIX cannot move a past flag."""
    spike = vix > vix.rolling(252).quantile(p / 100.0)
    receding = vix < vix.rolling(5).max()
    return spike & receding


def override_on(trigger, h):
    """Risk-ON override: True for h days from each trigger day (inclusive); a fresh
    trigger inside the window extends it. Depends only on triggers on/before t."""
    return trigger.astype(float).rolling(h, min_periods=1).max().fillna(0.0) > 0


def overlaid_book(conviction, on, override):
    """Composed daily book: conviction weights held whenever the base gate is on OR
    the override fires (holds on base-cash days, never forces cash on risk-on days)."""
    hold = (on | override.reindex(conviction.index).fillna(False)).astype(float)
    return conviction.mul(hold, axis=0)


def sharpe_diff_z(r1, r2):
    """Ledoit-Wolf (2008) HAC Sharpe-difference z: (SR1 - SR2) / se, with se from a
    Parzen-kernel HAC covariance of the (mean, uncentered 2nd moment) estimators and
    an Andrews (1991) AR(1)-plug-in bandwidth. Robust to the serial dependence and
    non-normality of daily book returns that Jobson-Korkie's IID form assumes away;
    on IID data it reproduces the closed-form JK-Memmel z. z > 0 favors r1."""
    df = pd.concat([r1, r2], axis=1).dropna()
    x, y = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    T = len(x)
    if T < 8:
        return 0.0
    mx, my = x.mean(), y.mean()
    qx, qy = (x * x).mean(), (y * y).mean()
    Dx, Dy = qx - mx * mx, qy - my * my
    if Dx <= 0 or Dy <= 0:
        return 0.0
    delta = mx / np.sqrt(Dx) - my / np.sqrt(Dy)
    grad = np.array([qx / Dx**1.5, -qy / Dy**1.5, -mx / (2 * Dx**1.5), my / (2 * Dy**1.5)])

    U = np.column_stack([x, y, x * x, y * y])
    U = U - U.mean(axis=0)
    S = _andrews_bandwidth(U, T)
    psi = (U.T @ U) / T
    for lag in range(1, min(T - 1, int(np.ceil(S))) + 1):
        k = _parzen(lag / S)
        if k > 0.0:
            g = (U[lag:].T @ U[:-lag]) / T
            psi += k * (g + g.T)

    var = grad @ psi @ grad / T
    return float(delta / np.sqrt(var)) if var > 0 else 0.0


def _andrews_bandwidth(U, T):
    """Andrews (1991) Parzen bandwidth from AR(1) fits to each moment series."""
    num = den = 0.0
    for j in range(U.shape[1]):
        u0, u1 = U[:-1, j], U[1:, j]
        s00 = (u0 * u0).sum()
        rho = min(max((u0 * u1).sum() / s00, -0.999), 0.999) if s00 > 0 else 0.0
        sig2 = np.mean((u1 - rho * u0) ** 2)
        num += 4 * rho**2 * sig2**2 / (1 - rho) ** 8
        den += sig2**2 / (1 - rho) ** 4
    alpha2 = num / den if den > 0 else 0.0
    return 2.6614 * (alpha2 * T) ** 0.2 if alpha2 > 0 else 1.0


def _parzen(a):
    a = abs(a)
    if a <= 0.5:
        return 1 - 6 * a**2 + 6 * a**3
    if a <= 1.0:
        return 2 * (1 - a) ** 3
    return 0.0


def _metrics(r):
    """(sharpe, ann_return, max_drawdown) of a daily return series."""
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    ann = float((1.0 + r).prod() ** (252 / max(len(r), 1)) - 1.0)
    return round(sharpe_tstat(r)[0], 3), round(ann, 4), round(dd, 4)


def _returns(weights, px, cost):
    return backtest_weights(px, rebalance_targets(weights, "ME"), cost).returns


def run(start="2010-01-01", split="2017-01-01", refresh=False):
    pd.set_option("display.width", 220, "display.max_columns", 40)

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        px, mkt, _, _ = india_panel(start=start, index="nifty500", ret_clip=0.40, refresh=refresh)
        vix = close_prices(load_yahoo_ohlcv([VIX_SYM], refresh=refresh))[VIX_SYM].reindex(px.index).ffill()
        sectors = sector_map("nifty500")
        base = base_book(px, mkt, vix, sectors)
        sigs = raw_signals(px, mkt, sectors)
        conviction = conviction_topq(composite({k: sigs[k] for k in CORE}), top=0.1)
        on = regime_on(mkt, vix, 200, 252, 0.80).reindex(conviction.index).fillna(False)

    # Weights-level anchor: the reconstructed baseline IS the proven deployed book.
    assert np.allclose(overlaid_book(conviction, on, pd.Series(False, index=on.index)).values, base.values), \
        "reconstructed baseline diverged from riskoff_sleeve.base_book"

    print(f"N500 {px.shape[1]} stocks x {len(px)} days  {px.index[0].date()}->{px.index[-1].date()}  test>={split}")

    # Returns-level reproduction GATE: reproduce the deployed test SR 1.865 @20bps first.
    base_ret = {c: _returns(base, px, c) for c in (10.0, 20.0, 40.0)}
    b20 = _metrics(base_ret[20.0].loc[split:])
    if round(b20[0], 3) != BASE_TEST_SR:
        raise SystemExit(f"REPRODUCTION FAILED: base test SR {b20[0]} != {BASE_TEST_SR}; aborting before test read")
    print(f"[REPRODUCTION] deployed base @20bps test SR {b20[0]} == {BASE_TEST_SR}  (ann {b20[1]}, maxDD {b20[2]})")

    over_ret = {}
    for v in VARIANTS:
        book = overlaid_book(conviction, on, override_on(spike_recede(vix, v[0]), v[1]))
        over_ret[v] = {c: _returns(book, px, c) for c in (10.0, 20.0, 40.0)}

    # ---------------- TRAIN (->2016): pick by combined Sharpe @20bps, then FREEZE ----------------
    rows = []
    for v in VARIANTS:
        sr, ann, dd = _metrics(over_ret[v][20.0].loc[:split].iloc[:-1])
        rows.append({"variant": f"p{v[0]}_h{v[1]}", "train_SR": sr, "train_ann": ann, "train_maxDD": dd})
    train = pd.DataFrame(rows).sort_values("train_SR", ascending=False).reset_index(drop=True)
    chosen = tuple(int(x) for x in train.iloc[0]["variant"][1:].split("_h"))
    print(f"\n[TRAIN {start[:4]}->2016 @20bps: overlaid combined Sharpe]  base_train_SR "
          f"{_metrics(base_ret[20.0].loc[:split].iloc[:-1])[0]}")
    print(train.to_string(index=False))
    print(f"FROZEN (max TRAIN combined SR): p{chosen[0]}_h{chosen[1]}   (module FROZEN=p{FROZEN[0]}_h{FROZEN[1]})")

    # ---------------- ONE TEST READ (2017+): frozen vs baseline + all variants disclosed ----------------
    fp, fh = FROZEN
    ov_test = override_on(spike_recede(vix, fp), fh).reindex(conviction.index).fillna(False).loc[split:]
    episodes = int((ov_test.astype(int).diff() == 1).sum() + (1 if bool(ov_test.iloc[0]) else 0))
    print(f"\n[TEST {split}+ ONE read]  frozen = p{fp}_h{fh}   "
          f"distinct re-entry episodes = {episodes}  (override-active days {int(ov_test.sum())})")
    print(f"{'cost':>4} {'variant':>9} | {'SR':>6} {'ann':>8} {'maxDD':>8} | "
          f"{'dSR':>7} {'dMaxDD_pts':>10} {'LW_z':>7}")
    test_log = {}
    for cost in (10.0, 20.0, 40.0):
        b_sr, b_ann, b_dd = _metrics(base_ret[cost].loc[split:])
        print(f"{int(cost):>4} {'base':>9} | {b_sr:6.3f} {b_ann:8.4f} {b_dd:8.4f} |")
        for v in VARIANTS:
            r = over_ret[v][cost].loc[split:]
            sr, ann, dd = _metrics(r)
            z = sharpe_diff_z(r, base_ret[cost].loc[split:])
            flag = "  <-frozen" if v == FROZEN else ""
            print(f"{int(cost):>4} {'p%d_h%d' % v:>9} | {sr:6.3f} {ann:8.4f} {dd:8.4f} | "
                  f"{sr - b_sr:+7.3f} {(dd - b_dd) * 100:+10.2f} {z:7.3f}{flag}")
            if v == FROZEN:
                test_log[cost] = {"sharpe": sr, "ann": ann, "maxdd": dd,
                                  "baseline_sharpe": b_sr, "baseline_ann": b_ann, "baseline_maxdd": b_dd,
                                  "d_sharpe": round(sr - b_sr, 3), "d_maxdd_pts": round((dd - b_dd) * 100, 2),
                                  "lw_sharpe_diff_z": round(z, 3)}

    # ---------------- bar (pre-registered, idiom-correct): LW z>1 AND maxDD not worse by >2pts, robust 10/40 ----------------
    def passes(cost):
        m = test_log[cost]
        return m["lw_sharpe_diff_z"] > 1.0 and m["d_maxdd_pts"] >= -2.0
    bar = {int(c): passes(c) for c in (20.0, 10.0, 40.0)}
    deploy = bar[20] and bar[10] and bar[40]
    print(f"\n[DEPLOYMENT BAR] LW Sharpe-diff z>1 AND maxDD not worse by >2pts, per cost {bar}  -> deploy={deploy}")
    print(f"  (verdict rests on {episodes} episodes - a pass would go to FORWARD confirmation, not deployment)")

    for cost, m in test_log.items():
        log_run({"hypothesis_ref": "RL-2026-07-24", "universe": "NIFTY500",
                 "cost_bps": cost, "split": split, "strategy": f"REGIME+vix_rebound_p{fp}_h{fh}",
                 "frozen_variant": f"p{fp}_h{fh}", "episodes": episodes, "metrics": m,
                 "deploy": bool(passes(cost)), "n_trials_cumulative": CUMULATIVE_INDIA_TRIALS,
                 "status": "success"})


def main():
    p = argparse.ArgumentParser(description="RL-2026-07-24 VIX spike-and-recede re-entry overlay")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(start=a.start, split=a.split, refresh=a.refresh)


if __name__ == "__main__":
    main()
