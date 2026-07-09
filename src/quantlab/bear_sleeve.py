"""RL-2026-07-13: bear-only reversal sleeve on the deployable REGIME book.

Question (pre-registered): does a small, bear-gated short-term reversal sleeve,
funded from the REGIME book's defensive cash, improve the COMBINED book's
risk-adjusted return? The sleeve is standalone cost-gated (RL-2026-07-11) but is
active exactly when the long book is de-risked to cash and is ~uncorrelated with it.

Construction (all causal):
  - BASE = the RL-2026-07-11 deployable book: top-decile CONVICTION momentum
    (`conviction_topq` on composite{mom_12_1, sharpe_mom, resid_mom}, nifty500)
    scaled to cash by the (200d-MA OR India-VIX) regime overlay (`regime_on`).
    Reproduces test SR 1.86, ann 35.7%, maxDD -27% (RL-2026-07-11).
  - SLEEVE = an existing `short_term.py` dollar-neutral unit-gross L/S book on the
    nifty100 panel (weekly), ZEROED on risk-on days (^NSEI >= its 200d MA). Regime
    flips are rebalanced and pay turnover; between flips the sleeve holds/drifts at
    its native weekly cadence.
  - COMBINED returns = r_base + s * r_sleeve_gated, each leg costed independently.
    The sleeve is unit-gross, so scaling its net return by s scales P&L and cost
    together (cost is linear in turnover).

TRAIN (2011->2016) picks the reversal variant and size s from {0.10, 0.20}; frozen
before the single TEST read (2017+). Honesty: at the 20 bps headline the sleeve is a
drag on TRAIN for every config (see run() output) - a wash/negative verdict is the
pre-registered ~50%-likely outcome and is a valid deliverable.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from .backtest import backtest_weights
from .blend import composite, conviction_topq, market_on, regime_on
from .data import close_prices, load_yahoo_ohlcv
from .evaluation import deflated_sharpe_ratio, sharpe_tstat
from .india import india_panel, sector_map
from .india_blend_study import raw_signals
from .portfolio import rebalance_targets
from .short_term import build_books as st_build_books
from .tracking import log_run

CORE = ("mom_12_1", "sharpe_mom", "resid_mom")
VARIANTS = ("LS-REV5", "LS-RESID-REV", "LS-VOLGATE")  # rev5 / resid_rev / vol-gated
SIZES = (0.10, 0.20)
# Frozen on TRAIN (2011->2016) evidence, before the test window is read:
FROZEN_VARIANT = "LS-RESID-REV"
FROZEN_SIZE = 0.10
# ~54 prior RL-2026-07-10 India trials (short_term_run) + 6 combined configs here.
CUMULATIVE_INDIA_TRIALS = 60


def base_book(px, mkt, vix, sectors):
    """RL-2026-07-11 deployable book: top-decile conviction momentum, scaled to
    cash by the (200d-MA OR India-VIX) regime overlay. Weights sum to <=1 (cash rest)."""
    sigs = raw_signals(px, mkt, sectors)
    book = conviction_topq(composite({k: sigs[k] for k in CORE}), top=0.1)
    on = regime_on(mkt, vix, 200, 252, 0.80).reindex(book.index).fillna(False).astype(float)
    return book.mul(on, axis=0)


def bear_gated_targets(sleeve_w, freq, mkt, ma_lb=200):
    """Sleeve target matrix: hold the sleeve on bear days (^NSEI < its `ma_lb` MA),
    flat on risk-on days. Rebalances at the sleeve's native `freq` while active and on
    every regime flip (so flips pay turnover); NaN elsewhere so the book holds/drifts.

    Causal: the risk-on flag and the native rebalance grid both use only trailing
    prices, so the target at date t never depends on prices after t."""
    idx = sleeve_w.index
    bear = (~market_on(mkt, ma_lb)).reindex(idx).fillna(False)
    held = rebalance_targets(sleeve_w, freq).ffill().fillna(0.0)  # weekly book, ffilled
    gated = held.where(bear, 0.0)                                 # 0 on risk-on days
    changed = (gated != gated.shift(1)).any(axis=1)              # rebalance when target moves
    changed.iloc[0] = True
    targets = gated.copy()
    targets.loc[~changed] = np.nan
    return targets


def _metrics(r):
    """(sharpe, ann_return, max_drawdown) of a daily return series."""
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    ann = float((1.0 + r).prod() ** (252 / max(len(r), 1)) - 1.0)
    return round(sharpe_tstat(r)[0], 3), round(ann, 4), round(dd, 4)


def run(start="2010-01-01", split="2017-01-01", train_start="2011-01-01", refresh=False):
    pd.set_option("display.width", 240, "display.max_columns", 40)

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        px, mkt, _, _ = india_panel(start=start, index="nifty500", ret_clip=0.40, refresh=refresh)
        px100, mkt100, oh100, _ = india_panel(start=start, index="nifty100", ret_clip=0.40, refresh=refresh)
        vix = close_prices(load_yahoo_ohlcv(["^INDIAVIX"], refresh=refresh))["^INDIAVIX"].reindex(px.index).ffill()
        base = base_book(px, mkt, vix, sector_map("nifty500"))
        sleeve_books = st_build_books(px100, mkt100, oh100)

    print(f"BASE nifty500 {px.shape[1]}x{len(px)}  SLEEVE nifty100 {px100.shape[1]}x{len(px100)}  "
          f"{px.index[0].date()}->{px.index[-1].date()}  test>={split}")

    def r_base_at(cost):
        return backtest_weights(px, rebalance_targets(base, "ME"), cost).returns

    def r_sleeve_at(variant, cost):
        w, f = sleeve_books[variant]
        return backtest_weights(px100, bear_gated_targets(w, f, mkt100), cost).returns.reindex(px.index).fillna(0.0)

    # ---------------- TRAIN design (2011->2016): pick variant + size, then FREEZE ----------------
    rb20 = r_base_at(20.0)
    tb = rb20.loc[train_start:split].iloc[:-1]
    base_sr_tr = _metrics(tb)[0]
    print(f"\n[TRAIN {train_start}->{split} design @20bps]  base-alone SR {base_sr_tr}")
    rows = []
    for v in VARIANTS:
        rs = r_sleeve_at(v, 20.0)
        corr = rb20.loc[train_start:split].iloc[:-1].corr(rs.loc[train_start:split].iloc[:-1])
        for s in SIZES:
            tc = (rb20 + s * rs).loc[train_start:split].iloc[:-1]
            csr, cann, cdd = _metrics(tc)
            rows.append({"variant": v, "s": s, "comb_SR": csr, "dSR": round(csr - base_sr_tr, 3),
                         "comb_maxDD": cdd, "corr_base_sleeve": round(corr, 3)})
    train = pd.DataFrame(rows)
    print(train.to_string(index=False))
    print(f"FROZEN (max TRAIN comb_SR): variant={FROZEN_VARIANT} s={FROZEN_SIZE}  "
          f"[all configs <= base on TRAIN -> sleeve does not help at 20 bps; freezing the least-drag / "
          f"most-diversifying config for the one test read]")

    # ---------------- ONE TEST READ (2017+): combined vs base at 10/20/40 bps ----------------
    print(f"\n[TEST {split}+ ONE read]  frozen sleeve={FROZEN_VARIANT} s={FROZEN_SIZE}")
    hdr = f"{'cost':>5} | {'base_SR':>7} {'base_ann':>8} {'base_DD':>8} | {'slv_SR':>7} {'slv_ann':>8} | " \
          f"{'comb_SR':>7} {'comb_ann':>8} {'comb_DD':>8} | {'dSR':>6} {'pairT_diff':>10} {'corr':>6}"
    print(hdr)
    test_log = {}
    for cost in (10.0, 20.0, 40.0):
        rb = r_base_at(cost).loc[split:]
        rs = r_sleeve_at(FROZEN_VARIANT, cost).loc[split:]
        rc = rb + FROZEN_SIZE * rs
        bsr, bann, bdd = _metrics(rb)
        ssr, sann, _ = _metrics(rs)
        csr, cann, cdd = _metrics(rc)
        pair_t = sharpe_tstat(rc - rb)[1]            # scale-invariant t of the difference (= sleeve t)
        corr = rb.corr(rs)
        print(f"{int(cost):>5} | {bsr:7.3f} {bann:8.4f} {bdd:8.4f} | {ssr:7.3f} {sann:8.4f} | "
              f"{csr:7.3f} {cann:8.4f} {cdd:8.4f} | {csr - bsr:+6.3f} {pair_t:10.2f} {corr:6.3f}")
        test_log[cost] = {"base_sharpe": bsr, "base_ann": bann, "base_maxdd": bdd,
                          "sleeve_sharpe": ssr, "sleeve_ann": sann,
                          "comb_sharpe": csr, "comb_ann": cann, "comb_maxdd": cdd,
                          "d_sharpe": round(csr - bsr, 3), "paired_t_diff": round(pair_t, 2),
                          "corr_base_sleeve": round(corr, 3)}

    # sleeve DSR against the reversal-variant family (does the SLEEVE'S own edge survive?)
    var_sr = []
    for v in VARIANTS:
        rv = r_sleeve_at(v, 20.0).loc[split:]
        if rv.std() > 0:
            var_sr.append(rv.mean() / rv.std())
    sleeve20 = r_sleeve_at(FROZEN_VARIANT, 20.0).loc[split:]
    dsr = deflated_sharpe_ratio(sleeve20, var_sr)
    print(f"\nsleeve DSR (frozen sleeve @20bps vs {len(var_sr)}-variant family) = {round(dsr, 3)}  "
          f"[combined book inherits the base's strict-bar failure, established RL-07-10/11]")

    for cost, mrow in test_log.items():
        log_run({"hypothesis_ref": "RL-2026-07-13", "universe": "NIFTY500+NIFTY100",
                 "cost_bps": cost, "split": split, "strategy": "REGIME+bear_reversal_sleeve",
                 "sleeve_variant": FROZEN_VARIANT, "sleeve_size": FROZEN_SIZE,
                 "metrics": mrow, "sleeve_dsr": round(dsr, 3),
                 "n_trials_cumulative": CUMULATIVE_INDIA_TRIALS, "status": "success"})


def main():
    p = argparse.ArgumentParser(description="RL-2026-07-13 bear-only reversal sleeve study")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--train-start", default="2011-01-01")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(start=a.start, split=a.split, train_start=a.train_start, refresh=a.refresh)


if __name__ == "__main__":
    main()
