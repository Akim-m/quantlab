"""RL-2026-07-10: frozen 9-trial deployable-strategy sweep on the Indian universe.

Runs the pre-registered strategy family (see the study spec) ONCE on the locked
TEST window (>=2017). Nothing here is tuned to test-window results: the only fitted
number is LO-VT's target vol, which is the LO-CORE-NOOVERLAY book's realized vol on
TRAIN, frozen by construction.

Benchmarks:
  - raw ^NSEI (price index, no dividends)              -> `beats_nifty`
  - TR-proxy = ^NSEI daily + 0.014/252 (dividend carry) -> active / alpha yardstick
  - EW-N (equal weight across the panel, monthly, net)  -> KEY alpha yardstick

Long books exclude low_vol/low_beta/short_rev (those are the regime risk-off book).
Long-short books are factor evidence (net Sharpe/t, FDR/DSR), NOT judged vs Nifty.
"""

import argparse

import numpy as np
import pandas as pd

from . import trend
from .backtest import backtest_weights
from .blend import (
    composite, long_only_topq, long_only_topq_banded, long_short,
    regime_switch, trend_overlay, vol_target_overlay,
)
from .evaluation import benjamini_hochberg, deflated_sharpe_ratio, one_sided_p, sharpe_tstat
from .india import india_panel, sector_map
from .india_blend_study import lo_erc, raw_signals
from .india_scenarios import evaluate2
from .india_study import _scalable_factors
from .optimization import rolling_construction
from .portfolio import rebalance_targets
from .tracking import log_run

CORE = ("mom_12_1", "sharpe_mom", "resid_mom")
EXT = ("mom_12_1", "sharpe_mom", "resid_mom", "mom_6_1", "off_low", "sector_mom")
NINE = ["LO-CORE", "LO-CORE-NOOVERLAY", "LO-EXT", "LO-ERC", "LO-VT",
        "REGIME", "LO-BAND", "LS-CORE", "LS-RESID2"]
SWEEP_STRATS = ["LO-CORE", "LS-CORE", "REGIME", "LO-BAND"]
CUMULATIVE_INDIA_TRIALS = 41  # 32 (RL-07-08/09 factor family) + 9 (this study)


def benchmarks(px: pd.DataFrame, mkt: pd.Series, cost_bps: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(raw_nsei_ret, tr_proxy_ret, ew_panel_net_ret) aligned to the panel index."""
    nsei = mkt.reindex(px.index).pct_change().fillna(0.0)
    tr = nsei + 0.014 / 252.0
    ew = pd.DataFrame(1.0 / px.shape[1], index=px.index, columns=px.columns)
    ew_ret = backtest_weights(px, rebalance_targets(ew, "ME"), cost_bps).returns
    return nsei, tr, ew_ret


def _core_regime_band_ls(px, mkt, sigs):
    """The four cost-independent books shared by the primary cell and the sweep."""
    core_score = composite({k: sigs[k] for k in CORE})
    noov = long_only_topq(core_score, px, top=0.2)
    lo_core = trend_overlay(noov, mkt, 200)
    risk_off = 0.5 * long_only_topq(composite({k: sigs[k] for k in ("low_vol", "low_beta")}), px, top=0.2)
    regime = regime_switch(noov, risk_off, mkt, 200)
    lo_band = trend_overlay(long_only_topq_banded(core_score, px, 0.15, 0.35), mkt, 200)
    ls_core = long_short(composite({k: sigs[k] for k in ("resid_mom", "sharpe_mom", "mom_12_1")}))
    return core_score, noov, {
        "LO-CORE": (lo_core, "ME"),
        "LS-CORE": (ls_core, "ME"),
        "REGIME": (regime, "ME"),
        "LO-BAND": (lo_band, "ME"),
    }


def build_books(px, mkt, sectors, split, cost_bps) -> tuple[dict[str, tuple[pd.DataFrame, str | None]], float]:
    """All 9 frozen trials as (weights, rebalance_freq) books, plus LO-VT's target vol."""
    sigs = raw_signals(px, mkt, sectors)
    core_score, noov, shared = _core_regime_band_ls(px, mkt, sigs)
    rets = px.pct_change().fillna(0.0)

    # LO-VT target = TRAIN realized annualized vol of the NO-overlay book (frozen).
    noov_ret = backtest_weights(px, rebalance_targets(noov, "ME"), cost_bps).returns
    target_vol = float(noov_ret.loc[:split].iloc[:-1].std() * np.sqrt(252))

    ext_score = composite({k: sigs[k] for k in EXT})
    books = {
        "LO-CORE": shared["LO-CORE"],
        "LO-CORE-NOOVERLAY": (noov, "ME"),
        "LO-EXT": (trend_overlay(long_only_topq(ext_score, px, top=0.2), mkt, 200), "ME"),
        "LO-ERC": (trend_overlay(lo_erc(px, core_score, top=0.2), mkt, 200), "ME"),
        "LO-VT": (vol_target_overlay(noov, rets, target=target_vol, cap=1.0), "ME"),
        "REGIME": shared["REGIME"],
        "LO-BAND": shared["LO-BAND"],
        "LS-CORE": shared["LS-CORE"],
        "LS-RESID2": (long_short(composite({k: sigs[k] for k in CORE},
                                           weights={"resid_mom": 2.0, "sharpe_mom": 1.0, "mom_12_1": 1.0})), "ME"),
    }
    return books, target_vol


def fdr_dsr_verdict(books, px, mkt, ohlcv, split, cost_bps) -> pd.DataFrame:
    """BH-FDR (q=0.10) over the 9-trial family + Deflated Sharpe against the FULL
    searched family (31 prior factors + these 9 blends).

    The DSR trial set MUST be the whole search, not the 9 self-similar winners: a
    narrow winners-only set has tiny variance -> tiny benchmark -> inflated DSR
    (verified: LO-CORE DSR 0.999 on the 9-set vs 0.006 on the full spread). Using
    the full family keeps the strict verdict honest and consistent with RL-07-08/09."""
    rets = {n: backtest_weights(px, rebalance_targets(w, f), cost_bps).returns.loc[split:]
            for n, (w, f) in books.items()}
    # full-family DSR trial set: the 31 prior factors + these blends, per-period Sharpes
    fac_sr = []
    for _code, _name, fw, ff in _scalable_factors(px, mkt, ohlcv):
        fr = backtest_weights(px, rebalance_targets(fw, ff), cost_bps).returns.loc[split:]
        if fr.std() > 0:
            fac_sr.append(fr.mean() / fr.std())
    sr_trials = fac_sr + [r.mean() / r.std() for r in rets.values() if r.std() > 0]
    names = list(rets)
    pvals = [one_sided_p(sharpe_tstat(rets[n])[1], len(rets[n])) for n in names]
    reject = benjamini_hochberg(pvals, q=0.10)
    rows = []
    for n, p, rej in zip(names, pvals, reject):
        dsr = deflated_sharpe_ratio(rets[n], sr_trials)
        rows.append({"strategy": n, "test_sharpe": round(sharpe_tstat(rets[n])[0], 3),
                     "test_p": round(p, 4), "bh_pass": bool(rej), "dsr": round(dsr, 3),
                     "survives": bool(rej and dsr > 0.95)})
    return pd.DataFrame(rows).sort_values("test_sharpe", ascending=False).reset_index(drop=True)


def run(start="2010-01-01", split="2017-01-01", primary_cost=20.0, refresh=False) -> None:
    pd.set_option("display.width", 260, "display.max_columns", 60)

    # ---------------- PRIMARY CELL: Nifty 500, 20 bps ----------------
    px, mkt, ohlcv, _ = india_panel(start=start, index="nifty500", ret_clip=0.40, refresh=refresh)
    sectors = sector_map("nifty500")
    print(f"PRIMARY CELL  N500 {px.shape[1]} stocks x {len(px)} days "
          f"{px.index[0].date()}->{px.index[-1].date()}  test>={split}  cost {primary_cost}bps")

    nsei, tr, ew = benchmarks(px, mkt, primary_cost)
    books, target_vol = build_books(px, mkt, sectors, split, primary_cost)
    print(f"LO-VT target vol (TRAIN realized, LO-CORE-NOOVERLAY) = {target_vol:.4f}")

    strategies = {n: books[n] for n in NINE}
    strategies["dual_momentum"] = (trend.dual_momentum(px), "ME")
    strategies["hrp"] = (rolling_construction(px, "hrp"), None)
    table = evaluate2(strategies, px, mkt, nsei, split, tr, ew, cost_bps=primary_cost,
                      extra_returns={"TR-proxy": tr, "EW-N-net": ew})
    print(f"\n[PRIMARY METRICS]  raw ^NSEI test Sharpe = {table.attrs['bench_sharpe']}")
    print(table.to_string(index=False))

    # ---------------- BH-FDR / DSR verdict over the 9-trial family ----------------
    verdict = fdr_dsr_verdict({n: books[n] for n in NINE}, px, mkt, ohlcv, split, primary_cost)
    print("\n[BH-FDR (q=0.10, 9-trial family) + Deflated-Sharpe (full 40-trial search)]")
    print(verdict.to_string(index=False))
    n_surv = int(verdict["survives"].sum())
    print(f"survive BH-FDR AND DSR>0.95: {n_surv} of 9")
    print(f"NOTE: DSR trial set = the FULL searched family (31 prior factors + these 9). "
          f"Computing DSR on the 9 self-similar winners alone inflates it to ~1.0; the full "
          f"spread (factor Sharpes -2.3..+1.6) is the honest bar - and gives ~0 survivors, "
          f"consistent with RL-07-08/09. Cumulative India trial count ~{CUMULATIVE_INDIA_TRIALS}.")

    # ---------------- log one row per trial ----------------
    trow = {r["strategy"]: r for r in table.to_dict("records")}
    for n in NINE:
        v = verdict[verdict["strategy"] == n].iloc[0]
        log_run({"hypothesis_ref": "RL-2026-07-10", "universe": "NIFTY500",
                 "cost_bps": primary_cost, "split": split, "strategy": n,
                 "n_assets": int(px.shape[1]), "metrics": trow[n],
                 "bh_pass": bool(v["bh_pass"]), "dsr": float(v["dsr"]),
                 "n_trials_family": len(NINE), "n_trials_cumulative": CUMULATIVE_INDIA_TRIALS,
                 "status": "success"})

    # ---------------- SWEEP: universe x cost ----------------
    print("\n[SWEEP]  LO-CORE / LS-CORE / REGIME / LO-BAND  x  {n50,n200,n500} x {10,20,40}bps")
    sweep_rows = []
    for univ in ("nifty50", "nifty200", "nifty500"):
        if univ == "nifty500":
            pxu, mktu = px, mkt
        else:
            pxu, mktu, _, _ = india_panel(start=start, index=univ, ret_clip=0.40, refresh=refresh)
        sigsu = raw_signals(pxu, mktu)
        _, _, sweep_books = _core_regime_band_ls(pxu, mktu, sigsu)
        nseu = mktu.reindex(pxu.index).pct_change().fillna(0.0)
        for cost in (10.0, 20.0, 40.0):
            _, tru, ewu = benchmarks(pxu, mktu, cost)
            tbl = evaluate2(sweep_books, pxu, mktu, nseu, split, tru, ewu, cost_bps=cost)
            for r in tbl.to_dict("records"):
                sweep_rows.append({"strategy": r["strategy"], "universe": univ, "cost": int(cost),
                                   "test_sharpe": r["test_sharpe"], "ann_return": r["ann_return"],
                                   "maxDD": r["max_dd"], "active_t_vs_EW": r["act_t_ew"],
                                   "beats_EW": r["beats_ew"]})
    sweep = pd.DataFrame(sweep_rows).sort_values(["strategy", "universe", "cost"]).reset_index(drop=True)
    print(sweep.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="RL-2026-07-10 frozen strategy sweep")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--cost", type=float, default=20.0, help="primary-cell cost (bps)")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(start=a.start, split=a.split, primary_cost=a.cost, refresh=a.refresh)


if __name__ == "__main__":
    main()
