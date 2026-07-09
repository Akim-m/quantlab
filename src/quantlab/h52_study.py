"""RL-2026-07-14: 52-week-strength long book - does anchoring diversify momentum?

Same deployed construction as the RL-2026-07-11 book (top-decile conviction +
(200d-MA OR India-VIX) regime overlay); only the SIGNAL differs. The question is
NOT whether 52w-strength beats equal-weight (momentum is stronger) but whether it
DIVERSIFIES the deployed momentum book enough to lift a 50/50 signal blend.

The 52w variant (off_low vs George-Hwang proximity to the high) is chosen on
TRAIN-window evidence, frozen, then the TEST window is read once.
"""

import argparse

import pandas as pd

from .backtest import backtest_weights
from .blend import composite, conviction_topq, regime_on
from .data import close_prices, load_yahoo_ohlcv
from .evaluation import sharpe_tstat
from .features import high_ratio
from .india import india_panel
from .india_blend_study import raw_signals
from .india_run import CORE, benchmarks
from .india_scenarios import evaluate2
from .portfolio import rebalance_targets
from .tracking import log_run

VIX_SYM = "^INDIAVIX"


def h52_variants(px: pd.DataFrame, sigs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The two pre-registered 52-week-strength signals (higher = stronger; both
    point-in-time causal, no shift - same NaN discipline as raw_signals):
      off_low = px / 252d-min (distance above the 52w low; existing signal),
      gh_high = px / 252d-max (George-Hwang 2004 proximity to the 52w high)."""
    return {"off_low": sigs["off_low"], "gh_high": high_ratio(px, 252)}


def deployed_book(score: pd.DataFrame, mkt: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """Frozen deployed construction: top-decile conviction weighting scaled to cash
    by the (200d-MA OR India-VIX) regime overlay. Only `score` differs across the
    momentum / 52w / blend books."""
    on = regime_on(mkt, vix).reindex(score.index).fillna(False).astype(float)
    return conviction_topq(score, top=0.1).mul(on, axis=0)


def _book_ret(book: pd.DataFrame, px: pd.DataFrame, cost_bps: float) -> pd.Series:
    return backtest_weights(px, rebalance_targets(book, "ME"), cost_bps).returns


def _ann(r: pd.Series) -> float:
    return float((1 + r).prod() ** (252 / max(len(r), 1)) - 1)


def freeze_signal(px, mkt, vix, variants, ew, split, cost_bps) -> tuple[str, pd.DataFrame]:
    """Pick the 52w variant on TRAIN evidence ONLY, then freeze. Selector = TRAIN
    deployed-book Sharpe; the top-decile conviction active return vs EW-277 is
    reported alongside as the decile gradient. Returns (chosen, evidence_table)."""
    ew_tr = ew.loc[:split].iloc[:-1]
    rows = []
    for name, sc in variants.items():
        book_tr = _book_ret(deployed_book(sc, mkt, vix), px, cost_bps).loc[:split].iloc[:-1]
        decile_tr = _book_ret(conviction_topq(sc, top=0.1), px, cost_bps).loc[:split].iloc[:-1]
        rows.append({"variant": name,
                     "train_book_sharpe": round(sharpe_tstat(book_tr)[0], 3),
                     "decile_active_vs_ew": round(_ann(decile_tr) - _ann(ew_tr), 4),
                     "train_ann": round(_ann(book_tr), 4)})
    ev = pd.DataFrame(rows).sort_values("train_book_sharpe", ascending=False).reset_index(drop=True)
    return str(ev.iloc[0]["variant"]), ev


def build_books(px, mkt, vix, sigs, chosen) -> dict[str, tuple[pd.DataFrame, str]]:
    """The three test books: momentum (deployed), 52w (frozen signal), and the ONE
    pre-registered 50/50 signal blend - all under the identical construction."""
    mom = composite({k: sigs[k] for k in CORE})
    h52 = h52_variants(px, sigs)[chosen]
    blend = composite({"mom": mom, "h52": h52})
    return {"H52": (deployed_book(h52, mkt, vix), "ME"),
            "MOM": (deployed_book(mom, mkt, vix), "ME"),
            "BLEND": (deployed_book(blend, mkt, vix), "ME")}


def cost_table(books, px, mkt, split, costs=(10.0, 20.0, 40.0)) -> pd.DataFrame:
    """Test-window Sharpe / ann / maxDD / paired-active-t-vs-EW for each book and
    EW-277, across the three cost regimes."""
    rows = []
    for cost in costs:
        nsei, tr, ew = benchmarks(px, mkt, cost)
        tbl = evaluate2(books, px, mkt, nsei, split, tr, ew, cost_bps=cost,
                        extra_returns={"EW-277": ew})
        for r in tbl.to_dict("records"):
            rows.append({"cost_bps": int(cost), "book": r["strategy"],
                         "test_sharpe": r["test_sharpe"], "ann_return": r["ann_return"],
                         "max_dd": r["max_dd"], "act_t_ew": r["act_t_ew"]})
    return pd.DataFrame(rows)


def diversification(books, px, mkt, ew, split, cost_bps=20.0) -> dict:
    """Does the 52w book diversify momentum? Active-return correlation vs the
    momentum book, and the ONE pre-registered blend-vs-momentum test: blend Sharpe
    and the paired t of the daily (blend - momentum) difference. Promotion needs
    blend > momentum on Sharpe AND that paired t > 1."""
    h = _book_ret(books["H52"][0], px, cost_bps).loc[split:]
    m = _book_ret(books["MOM"][0], px, cost_bps).loc[split:]
    bl = _book_ret(books["BLEND"][0], px, cost_bps).loc[split:]
    ewt = ew.loc[split:]
    diff = (bl - m).dropna()
    t_impr = float(diff.mean() / (diff.std(ddof=1) / len(diff) ** 0.5))
    mom_sr, blend_sr = sharpe_tstat(m)[0], sharpe_tstat(bl)[0]
    return {"active_corr_vs_mom": round((h - ewt).corr(m - ewt), 3),
            "raw_corr_vs_mom": round(h.corr(m), 3),
            "mom_sharpe": round(mom_sr, 3), "blend_sharpe": round(blend_sr, 3),
            "blend_minus_mom_t": round(t_impr, 3),
            "promoted": bool(blend_sr > mom_sr and t_impr > 1.0)}


def run(start="2010-01-01", split="2017-01-01", refresh=False) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 30)
    px, mkt, _, _ = india_panel(start=start, index="nifty500", ret_clip=0.40, refresh=refresh)
    vix = close_prices(load_yahoo_ohlcv([VIX_SYM], refresh=refresh))[VIX_SYM].dropna()
    sigs = raw_signals(px, mkt)
    _, _, ew = benchmarks(px, mkt, 20.0)

    chosen, evidence = freeze_signal(px, mkt, vix, h52_variants(px, sigs), ew, split, 20.0)
    print(f"N500 {px.shape[1]} stocks x {len(px)} days  test>={split}")
    print("[TRAIN evidence -> FROZEN 52w signal]")
    print(evidence.to_string(index=False))
    print(f"FROZEN signal: {chosen}\n")

    books = build_books(px, mkt, vix, sigs, chosen)
    table = cost_table(books, px, mkt, split)
    print("[TEST read | H52 / MOM / BLEND / EW-277  x  10/20/40 bps]")
    print(table.to_string(index=False))

    div = diversification(books, px, mkt, ew, split)
    print("\n[DIVERSIFICATION @ 20bps]  (promotion = blend beats MOM Sharpe AND blend-MOM paired t > 1)")
    for k, v in div.items():
        print(f"  {k}: {v}")

    for r in table[(table["cost_bps"] == 20) & table["book"].isin(("H52", "MOM", "BLEND"))].to_dict("records"):
        log_run({"hypothesis_ref": "RL-2026-07-14", "universe": "NIFTY500",
                 "cost_bps": 20.0, "split": split, "strategy": f"H52-{r['book']}",
                 "frozen_signal": chosen, "n_assets": int(px.shape[1]),
                 "metrics": r, "diversification": div, "status": "success"})


def main() -> None:
    p = argparse.ArgumentParser(description="RL-2026-07-14 52-week-strength diversification study")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(start=a.start, split=a.split, refresh=a.refresh)


if __name__ == "__main__":
    main()
