"""RL-2026-07-12: implementable market-neutral sleeve - resid-mom L/S, F&O-only shorts.

Measures how much restricting the short leg of the RL-2026-07-10 residual-momentum
long-short (the LS-RESID2 book, test SR 0.86) to F&O-shortable names degrades the
edge. ONE test read (>=2017), design frozen on TRAIN (2010->2016-12-31) first.

Frozen short-leg choice = THIN: mask the unrestricted book's short side to F&O
single-stock-futures underlyings, then rescale short gross to long gross. Decided on
TRAIN evidence (see RL-2026-07-12): the two candidates were statistically
indistinguishable on TRAIN (Sharpe 0.95 thin vs 0.98 fill, well inside the ~0.3
Sharpe SE), and THIN changes ONLY the short leg (long leg bit-identical to the
unrestricted book) while the re-rank "fill" alternative contaminated the long leg via
long/short cancellation. Costs: 20 bps both legs headline; carry credit, futures roll
drag, and 10/40 bps are disclosed sensitivities on the FROZEN weights (not trials).
"""

import argparse

import numpy as np
import pandas as pd

from .backtest import backtest_weights
from .blend import composite, fno_long_short, long_short
from .evaluation import deflated_sharpe_ratio, sharpe_tstat
from .india import BENCHMARK, fno_shortable, india_panel, sector_map
from .india_blend_study import raw_signals
from .india_run import _core_regime_band_ls
from .india_scenarios import _capm
from .india_study import _scalable_factors
from .portfolio import rebalance_targets
from .tracking import log_run

CORE = ("resid_mom", "sharpe_mom", "mom_12_1")
WEIGHTS = {"resid_mom": 2.0, "sharpe_mom": 1.0, "mom_12_1": 1.0}  # LS-RESID2 (RL-07-10)
CUMULATIVE_INDIA_TRIALS = 51  # ~50 prior India trials (RL-07-08/09/10/11) + this headline


def _ann(r: pd.Series) -> float:
    return float((1 + r).prod() ** (252 / max(len(r), 1)) - 1)


def _maxdd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def _metrics(ret: pd.Series, split: str, nsei_ret: pd.Series) -> dict:
    """Test-window metrics in the repo idiom: Sharpe/t, ann, maxDD, CAPM beta vs Nifty."""
    test = ret.loc[split:]
    sr, t = sharpe_tstat(test)
    beta, _, alpha_t = _capm(test, nsei_ret.loc[split:])
    return {"sharpe": round(sr, 3), "t": round(t, 2), "ann_return": round(_ann(test), 4),
            "maxdd": round(_maxdd(test), 4), "capm_beta_nifty": round(beta, 3),
            "capm_alpha_t": round(alpha_t, 2)}


def _sensitivities(base_test: pd.Series, short_gross: pd.Series) -> pd.DataFrame:
    """Disclosed sensitivities on the FROZEN headline weights (not new trials): short-leg
    carry credit (+0/+3/+5% ann on short gross), futures roll drag (12 bps/yr), and the
    10/40 bps cost check (passed in as base returns already backtested at those costs)."""
    sg = short_gross.reindex(base_test.index).fillna(0.0)
    rows = []
    for label, carry, roll in [("headline 20bps", 0.0, 0.0), ("carry +3%", 0.03, 0.0),
                               ("carry +5%", 0.05, 0.0), ("roll drag 12bps/yr", 0.0, 0.0012),
                               ("carry +3% - roll 12bps", 0.03, 0.0012)]:
        adj = base_test + (carry - roll) / 252.0 * sg
        sr, t = sharpe_tstat(adj)
        rows.append({"scenario": label, "sharpe": round(sr, 3), "t": round(t, 2),
                     "ann_return": round(_ann(adj), 4)})
    return pd.DataFrame(rows)


def run(start: str = "2010-01-01", split: str = "2017-01-01", cost_bps: float = 20.0,
        refresh: bool = False) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 40)
    px, mkt, ohlcv, _ = india_panel(start=start, index="nifty500", ret_clip=0.40, refresh=refresh)
    sectors = sector_map("nifty500")
    shortable = {s.upper() for s in fno_shortable(refresh=refresh)}
    overlap = set(px.columns) & shortable
    nsei_ret = mkt.pct_change().reindex(px.index).fillna(0.0)
    print(f"N500 {px.shape[1]} stocks x {len(px)} days  {px.index[0].date()}->{px.index[-1].date()}  "
          f"test>={split}  cost {cost_bps}bps")
    print(f"F&O-shortable single stocks {len(shortable)}; overlap with universe {len(overlap)}/{px.shape[1]}")

    score = composite({k: raw_signals(px, mkt, sectors)[k] for k in CORE}, weights=WEIGHTS)
    head = fno_long_short(score, shortable)
    books = {"LS-UNRESTRICTED": long_short(score), "LS-FNO-THIN": head}

    # ---- one test read: headline (F&O-only) + apples-to-apples unrestricted baseline ----
    rets, metrics, short_gross = {}, {}, None
    for name, w in books.items():
        res = backtest_weights(px, rebalance_targets(w, "ME"), cost_bps)
        rets[name] = res.returns
        metrics[name] = _metrics(res.returns, split, nsei_ret)
        if name == "LS-FNO-THIN":
            short_gross = res.weights.clip(upper=0.0).abs().sum(axis=1)

    # correlation of the sleeve with the deployable REGIME long book (test window)
    _, _, shared = _core_regime_band_ls(px, mkt, raw_signals(px, mkt, sectors))
    regime_ret = backtest_weights(px, rebalance_targets(shared["REGIME"][0], "ME"), cost_bps).returns
    corr = float(rets["LS-FNO-THIN"].loc[split:].corr(regime_ret.loc[split:]))

    # DSR against the full searched family (31 factors + the two L/S books), as india_run
    fac_sr = []
    for _c, _n, fw, ff in _scalable_factors(px, mkt, ohlcv):
        fr = backtest_weights(px, rebalance_targets(fw, ff), cost_bps).returns.loc[split:]
        if fr.std() > 0:
            fac_sr.append(fr.mean() / fr.std())
    sr_trials = fac_sr + [rets[n].loc[split:].mean() / rets[n].loc[split:].std() for n in books]
    dsr = deflated_sharpe_ratio(rets["LS-FNO-THIN"].loc[split:], sr_trials)

    print("\n[HEADLINE + BASELINE, 20 bps both legs, test window]")
    tbl = pd.DataFrame({n: metrics[n] for n in books}).T
    print(tbl.to_string())
    degr = metrics["LS-UNRESTRICTED"]["sharpe"] - metrics["LS-FNO-THIN"]["sharpe"]
    print(f"degradation (unrestricted - F&O-only) Sharpe: {degr:+.3f}")
    print(f"corr(F&O L/S, REGIME long book) = {corr:+.3f}")
    print(f"Deflated Sharpe vs full family (n_trials={len(sr_trials)}): {dsr:.3f}  "
          f"(cumulative India trials ~{CUMULATIVE_INDIA_TRIALS})")

    sens = _sensitivities(rets["LS-FNO-THIN"].loc[split:], short_gross)
    cost_rows = []
    for c in (10.0, 40.0):
        r = backtest_weights(px, rebalance_targets(head, "ME"), c).returns.loc[split:]
        sr, t = sharpe_tstat(r)
        cost_rows.append({"scenario": f"cost {int(c)}bps", "sharpe": round(sr, 3),
                          "t": round(t, 2), "ann_return": round(_ann(r), 4)})
    print("\n[DISCLOSED SENSITIVITIES on frozen F&O-only weights]")
    print(sens.to_string(index=False))
    print(pd.DataFrame(cost_rows).to_string(index=False))

    # ---- log one row per book (headline carries sensitivities + DSR) ----
    for name in books:
        row = {"hypothesis_ref": "RL-2026-07-12", "universe": "NIFTY500", "cost_bps": cost_bps,
               "split": split, "strategy": name, "n_assets": int(px.shape[1]),
               "n_shortable": len(shortable), "n_overlap": len(overlap),
               "short_mode": "thin", "metrics": metrics[name], "status": "success"}
        if name == "LS-FNO-THIN":
            row.update({"corr_regime": round(corr, 3), "dsr": round(dsr, 3),
                        "n_trials_family": len(sr_trials),
                        "n_trials_cumulative": CUMULATIVE_INDIA_TRIALS,
                        "sensitivities": sens.to_dict("records"), "cost_check": cost_rows})
        log_run(row)


def main() -> None:
    p = argparse.ArgumentParser(description="RL-2026-07-12 F&O-shortable L/S sleeve")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--cost", type=float, default=20.0)
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(start=a.start, split=a.split, cost_bps=a.cost, refresh=a.refresh)


if __name__ == "__main__":
    main()
