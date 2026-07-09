"""RL-2026-07-10: frozen short-term (daily/weekly) strategy sweep on Indian equity.

Runs the pre-registered 13-book short-term family ONCE on the locked TEST window
(>=2017). The central question: does any short-horizon signal have a GROSS edge
large enough to survive realistic Indian costs (5/10/20 bps/side) at weekly/daily
turnover? Every book is reported GROSS and NET side by side.

Design is frozen in short_term.build_books before the test window is touched. The
OOS window is multi-use across the RL-2026-07-10 program (disclosed). Nothing here
is tuned to test-window results.

Benchmarks (reused from india_run.benchmarks):
  - raw ^NSEI (price index, no dividends)                -> `beats_nifty`
  - TR-proxy = ^NSEI + 0.014/252 dividend carry          -> active yardstick
  - EW-N = equal-weight panel, monthly, net              -> KEY alpha yardstick
    (a short-term long-only tilt must beat simply owning the universe, net).

Long-short books are dollar-neutral factor evidence (judge on standalone net
Sharpe + FDR/DSR, NOT vs EW). Long-only books are judged by paired active-t vs EW.
"""

import argparse

import pandas as pd

from .backtest import backtest_weights
from .evaluation import benjamini_hochberg, deflated_sharpe_ratio, one_sided_p, sharpe_tstat
from .india import india_panel
from .india_run import benchmarks
from .india_scenarios import evaluate2
from .india_study import _scalable_factors
from .portfolio import rebalance_targets
from .short_term import LO_NAMES, LS_NAMES, build_books

# Prior cumulative RL-2026-07-10 India trials (32 factor family + 9 blends = 41),
# per india_run.CUMULATIVE_INDIA_TRIALS; + this 13-book short-term family.
CUMULATIVE_INDIA_TRIALS = 41 + 13  # ~54
FDR_COST = 10.0  # representative liquid-large-cap cost for the significance verdict


def sharpe_at(books, px, split, cost):
    """{strategy: test-window Sharpe} at one cost."""
    out = {}
    for n, (w, f) in books.items():
        r = backtest_weights(px, rebalance_targets(w, f), cost).returns.loc[split:]
        out[n] = round(sharpe_tstat(r)[0], 3)
    return out


def primary_table(books, ev10, px, split):
    """GROSS + NET@5/10/20 Sharpe per book, joined to evaluate2's net@10 metrics.

    Flag `survives`: net@10 Sharpe > 0.5 AND positive net paired-t vs EW (long-only);
    long-short books have no EW frame, so their flag is net@10 Sharpe > 0.5 alone.
    """
    g = sharpe_at(books, px, split, 0.0)
    n5 = sharpe_at(books, px, split, 5.0)
    n20 = sharpe_at(books, px, split, 20.0)
    ev = ev10.set_index("strategy")
    rows = []
    for n in books:
        kind = "LO" if n in LO_NAMES else "LS"
        net10 = float(ev.loc[n, "test_sharpe"])
        act_t = float(ev.loc[n, "act_t_ew"])
        survives = net10 > 0.5 and (act_t > 0 if kind == "LO" else True)
        rows.append({
            "strategy": n, "type": kind,
            "grossSR": g[n], "net5": n5[n], "net10": round(net10, 3), "net20": n20[n],
            "ann@10": float(ev.loc[n, "ann_return"]), "maxDD@10": float(ev.loc[n, "max_dd"]),
            "turnover": float(ev.loc[n, "turnover"]),
            "act_t_ew": round(act_t, 2) if kind == "LO" else None,
            "survives": survives,
        })
    return pd.DataFrame(rows).sort_values(["type", "net10"], ascending=[True, False]).reset_index(drop=True)


def regime_table(ev0, ev10):
    """Gross vs net@10 regime slices: 200-MA bull/bear and hi/lo trailing-vol."""
    g = ev0.set_index("strategy")
    n = ev10.set_index("strategy")
    rows = []
    for s in ev0["strategy"]:
        rows.append({
            "strategy": s,
            "g_bull": g.loc[s, "bull_sharpe"], "g_bear": g.loc[s, "bear_sharpe"],
            "g_hivol": g.loc[s, "hivol_sharpe"], "g_lovol": g.loc[s, "lovol_sharpe"],
            "n10_bull": n.loc[s, "bull_sharpe"], "n10_bear": n.loc[s, "bear_sharpe"],
            "n10_hivol": n.loc[s, "hivol_sharpe"],
        })
    return pd.DataFrame(rows)


def fdr_dsr_verdict(books, px, mkt, ohlcv, split, cost):
    """BH-FDR (q=0.10) over the 13-book family + Deflated Sharpe against the FULL
    searched spread (31 prior scalable factors on THIS panel + these 13 books),
    all net returns on the test window. Trials on the full spread, not the self-
    similar winners, keeps the DSR benchmark honest (see india_run.fdr_dsr_verdict).
    """
    rets = {n: backtest_weights(px, rebalance_targets(w, f), cost).returns.loc[split:]
            for n, (w, f) in books.items()}
    fac_sr = []
    for _code, _name, fw, ff in _scalable_factors(px, mkt, ohlcv):
        fr = backtest_weights(px, rebalance_targets(fw, ff), cost).returns.loc[split:]
        if fr.std() > 0:
            fac_sr.append(fr.mean() / fr.std())
    sr_trials = fac_sr + [r.mean() / r.std() for r in rets.values() if r.std() > 0]
    names = list(rets)
    pvals = [one_sided_p(sharpe_tstat(rets[n])[1], len(rets[n])) for n in names]
    reject = benjamini_hochberg(pvals, q=0.10)
    rows = []
    for n, p, rej in zip(names, pvals, reject):
        dsr = deflated_sharpe_ratio(rets[n], sr_trials)
        rows.append({"strategy": n, "net_sharpe": round(sharpe_tstat(rets[n])[0], 3),
                     "p": round(p, 4), "bh_pass": bool(rej), "dsr": round(dsr, 3),
                     "survives": bool(rej and dsr > 0.95)})
    return pd.DataFrame(rows).sort_values("net_sharpe", ascending=False).reset_index(drop=True), len(sr_trials)


def run(start="2010-01-01", split="2017-01-01", refresh=False):
    pd.set_option("display.width", 260, "display.max_columns", 60)

    # ---------------- PRIMARY CELL: Nifty 100, weekly ----------------
    px, mkt, ohlcv, _ = india_panel(start=start, index="nifty100", ret_clip=0.40, refresh=refresh)
    print(f"PRIMARY CELL  nifty100 {px.shape[1]} stocks x {len(px)} days "
          f"{px.index[0].date()}->{px.index[-1].date()}  test>={split}")
    books = build_books(px, mkt, ohlcv)
    nsei, tr, ew = benchmarks(px, mkt, FDR_COST)

    ev10 = evaluate2(books, px, mkt, nsei, split, tr, ew, cost_bps=FDR_COST,
                     extra_returns={"EW-N-net": ew, "TR-proxy": tr})
    ev0 = evaluate2(books, px, mkt, nsei, split, tr, ew, cost_bps=0.0)
    ew_sr, _ = sharpe_tstat(ew.loc[split:])

    print(f"\n[1. PRIMARY: gross vs net Sharpe @ 5/10/20 bps]  raw ^NSEI test SR={ev10.attrs['bench_sharpe']}  "
          f"EW-N-net test SR={round(ew_sr, 3)}")
    pt = primary_table(books, ev10, px, split)
    print(pt.to_string(index=False))
    surv = pt[pt["survives"]]
    print(f"net@10 SR>0.5 AND (LO: positive net paired-t vs EW): {len(surv)} of {len(pt)}"
          + (f" -> {', '.join(surv['strategy'])}" if len(surv) else " -> NONE"))

    print("\n[4. REGIME slices]  gross vs net@10: 200-MA bull/bear, hi/lo trailing-vol")
    print(regime_table(ev0, ev10).to_string(index=False))

    # ---------------- BH-FDR / DSR over the short-term family ----------------
    verdict, n_trials = fdr_dsr_verdict(books, px, mkt, ohlcv, split, FDR_COST)
    print(f"\n[3. BH-FDR (q=0.10, 13-book family) + Deflated Sharpe (full {n_trials}-trial spread), net@{int(FDR_COST)}bps]")
    print(verdict.to_string(index=False))
    print(f"survive BH-FDR AND DSR>0.95: {int(verdict['survives'].sum())} of {len(verdict)}.  "
          f"Cumulative RL-2026-07-10 India trials ~{CUMULATIVE_INDIA_TRIALS}; DSR bar rises with the "
          f"trial count, so a family that fails at {n_trials} fails harder at ~{CUMULATIVE_INDIA_TRIALS}.")

    # ---------------- NIFTY 50 (lowest-cost) cell: does cheap liquidity rescue the
    # short-term signals that DIED ON COST? Test the top-3 long-short reversal books
    # (the ones with a real gross edge cost killed) at 5 and 10 bps on the most-liquid
    # universe, and read their bear-regime net Sharpe.
    top3 = pt[pt["type"] == "LS"].sort_values("grossSR", ascending=False)["strategy"].head(3).tolist()
    print(f"\n[2. NIFTY 50 (most-liquid / lowest-cost) rescue test for the cost-killed LS signals: {', '.join(top3)}]")
    px5, mkt5, oh5, _ = india_panel(start=start, index="nifty50", ret_clip=0.40, refresh=refresh)
    books5 = build_books(px5, mkt5, oh5)
    nsei5, tr5, ew5 = benchmarks(px5, mkt5, 5.0)
    ev5 = evaluate2({n: books5[n] for n in top3}, px5, mkt5, nsei5, split, tr5, ew5, cost_bps=5.0)
    ev5b = ev5.set_index("strategy")  # net@5 metrics (incl. bear_sharpe)
    print(f"nifty50 {px5.shape[1]} stocks")
    rows = []
    for n in top3:
        g = sharpe_tstat(backtest_weights(px5, rebalance_targets(*books5[n]), 0.0).returns.loc[split:])[0]
        n10 = sharpe_tstat(backtest_weights(px5, rebalance_targets(*books5[n]), 10.0).returns.loc[split:])[0]
        rows.append({"strategy": n, "grossSR": round(g, 3), "net5SR": ev5b.loc[n, "test_sharpe"],
                     "net10SR": round(n10, 3), "turnover": ev5b.loc[n, "turnover"],
                     "net5_bearSR": ev5b.loc[n, "bear_sharpe"]})
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="RL-2026-07-10 short-term strategy sweep")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(start=a.start, split=a.split, refresh=a.refresh)


if __name__ == "__main__":
    main()
