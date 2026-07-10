"""RL-2026-07-26-02: US-close trend spillover gate on NIFTYBEES.NS (US-GATE).

US equity returns lead non-US markets at weekly-monthly horizons (Rapach-Strauss-Zhou
2013, JF - gradual information diffusion from the world's price-setting market). India's
deployed gate is purely local (200MA/VIX); a `^GSPC` trend gate may carry information
local prices have not fully impounded, especially at global risk transitions. The book
holds NIFTYBEES when the US trend is ON, else cash.

Causality clock (the load-bearing guard): at the NSE close on date t (15:30 IST) the
latest COMPLETE US session is t-1 (its close falls ~02:00 IST on t, before 15:30). The
`^GSPC` bar dated t is NOT yet known then (it closes ~02:00 IST on t+1). So the signal
is built on the US calendar, shifted ONE US trading day, then as-of forward-filled onto
the NSE calendar: NSE date t receives a signal that depends only on `^GSPC` closes
strictly before t. backtest_weights earns a target set at date D on D+1, so the weight
placed at NSE date t (= that signal) earns close_t->close_{t+1} - exactly the locked
close-to-close book. Removing the one-day shift leaks a same-day US close into the NSE
decision; the no-look-ahead test proves the shift is what prevents it.

Four locked variants, ONE frozen on TRAIN (2010->2016) net Sharpe @10bps, one TEST read
(2017+). NIFTYBEES goes through the RL-2026-07-17 spike repair (2019-12 fabricated
prints land inside the test window). The LW Sharpe-difference test is RL-23's helper.
"""

import argparse
import warnings

import pandas as pd

from .backtest import backtest_weights
from .band_mr import sharpe_diff_test
from .data import close_prices, load_yahoo_ohlcv
from .evaluation import sharpe_tstat
from .tracking import log_run
from .xasset_trend import clean_prices

SYMBOL = "NIFTYBEES.NS"
US = "^GSPC"
VARIANTS = ["ret21", "ret63", "ma200", "ret21_ma200"]
LABEL = {
    "ret21": "GSPC 21d ret>0",
    "ret63": "GSPC 63d ret>0",
    "ma200": "GSPC px>200d MA",
    "ret21_ma200": "21d>0 AND px>200MA",
}
TRAIN0, TRAIN1 = "2010-01-01", "2016-12-31"
HEADLINE_BPS = 10.0
# Frozen on TRAIN (2010->2016-12-31) net Sharpe @10 bps, before the single test read:
# ma200 is the TRAIN argmax (SR 0.390, vs ret63 0.199 / ret21 0.189 / ret21_ma200 0.156).
FROZEN = "ma200"


def us_signal(gspc: pd.Series, variant: str) -> pd.Series:
    """Boolean US-trend signal on the `^GSPC` (US) calendar, from trailing closes only.
    A bar dated d uses closes through d; NaN warm-up comparisons resolve to False (cash)."""
    up21 = gspc.pct_change(21) > 0.0
    up63 = gspc.pct_change(63) > 0.0
    above = gspc > gspc.rolling(200).mean()
    return {
        "ret21": up21,
        "ret63": up63,
        "ma200": above,
        "ret21_ma200": up21 & above,
    }[variant]


def align_us_signal(sig_us: pd.Series, nse_index: pd.DatetimeIndex, shift: int = 1) -> pd.Series:
    """Lag the US-dated signal one US trading day, then as-of forward-fill onto the NSE
    calendar. NSE date t receives the signal from `^GSPC` closes strictly before t (US
    day t-1 closes ~02:00 IST on t, ahead of the 15:30 IST NSE close; the US bar dated t
    is not yet known). shift=0 removes the buffer and leaks a same-day close - it exists
    only so the guard can be proven load-bearing."""
    lagged = sig_us.astype(float).shift(shift)
    on_union = lagged.reindex(sig_us.index.union(nse_index)).ffill()
    return on_union.reindex(nse_index).fillna(0.0)


def strat_book(px: pd.DataFrame, gspc: pd.Series, variant: str, cost_bps: float, shift: int = 1):
    w = align_us_signal(us_signal(gspc, variant), px.index, shift).to_frame(SYMBOL)
    return backtest_weights(px, w, cost_bps)


def india_gate_book(px: pd.DataFrame, cost_bps: float):
    """Disclosure arm: NIFTYBEES gated by its OWN 200d MA (the local gate). The decision
    at NSE close t uses the NIFTYBEES close at t; backtest_weights lags it into t+1."""
    sig = (px[SYMBOL] > px[SYMBOL].rolling(200).mean()).astype(float)
    return backtest_weights(px, sig.to_frame(SYMBOL), cost_bps)


def _metrics(r: pd.Series) -> tuple[float, float, float]:
    """(Sharpe, annualized return, max drawdown) of a daily return series."""
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    ann = float((1.0 + r).prod() ** (252 / max(len(r), 1)) - 1.0)
    return round(sharpe_tstat(r)[0], 3), round(ann, 4), round(dd, 4)


def _aligned(a: pd.Series, b: pd.Series):
    df = pd.concat([a, b], axis=1).dropna()
    return df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()


def load_data(refresh: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    px = clean_prices(close_prices(load_yahoo_ohlcv([SYMBOL], refresh=refresh))[[SYMBOL]])
    gspc = close_prices(load_yahoo_ohlcv([US], refresh=refresh))[US]
    return px, gspc


def train_table(px: pd.DataFrame, gspc: pd.Series) -> pd.DataFrame:
    """TRAIN-only: each variant's net Sharpe @10 bps over 2010->2016. No test rows enter."""
    rows = []
    for v in VARIANTS:
        r = strat_book(px, gspc, v, HEADLINE_BPS).returns.loc[TRAIN0:TRAIN1]
        sr, ann, dd = _metrics(r)
        rows.append({"variant": v, "label": LABEL[v], "train_sharpe": sr,
                     "train_ann": ann, "train_maxdd": dd})
    return pd.DataFrame(rows).sort_values("train_sharpe", ascending=False).reset_index(drop=True)


def evaluate(px: pd.DataFrame, gspc: pd.Series, split: str = "2017-01-01") -> dict:
    """Pure (no printing/logging): TRAIN selection, the frozen variant's TEST read across
    costs, the pre-registered bar, and the disclosure arm. Deterministic in its inputs."""
    train = train_table(px, gspc)
    argmax = train.iloc[0]["variant"]

    bh = px[SYMBOL].pct_change().loc[split:].dropna()
    bh_sr, bh_ann, bh_dd = _metrics(bh)

    test_rows = []
    for cost in (5.0, 10.0, 20.0):
        book = strat_book(px, gspc, FROZEN, cost)
        r = book.returns.loc[split:]
        turn = float(book.turnover.loc[split:].sum())
        sr, ann, dd = _metrics(r)
        _, _, z = sharpe_diff_test(*_aligned(r, bh))
        test_rows.append({"cost_bps": int(cost), "test_sharpe": sr, "ann_return": ann,
                          "max_dd": dd, "turnover": round(turn, 1), "lw_z_vs_bh": round(z, 3)})
    test = pd.DataFrame(test_rows)
    head = test[test["cost_bps"] == 10].iloc[0]
    z10 = float(test[test["cost_bps"] == 10]["lw_z_vs_bh"].iloc[0])
    z20 = float(test[test["cost_bps"] == 20]["lw_z_vs_bh"].iloc[0])
    maxdd_better = bool(head["max_dd"] > bh_dd)
    promoted = bool(z10 > 1.0 and z20 > 1.0 and maxdd_better)

    dr = india_gate_book(px, HEADLINE_BPS).returns.loc[split:]
    di_sr, di_ann, di_dd = _metrics(dr)
    _, _, di_z = sharpe_diff_test(*_aligned(dr, bh))

    return {
        "train": train, "argmax": argmax, "frozen": FROZEN,
        "bh": {"sharpe": bh_sr, "ann": bh_ann, "maxdd": bh_dd, "days": int(len(bh))},
        "test": test, "z10": round(z10, 3), "z20": round(z20, 3),
        "maxdd_better": maxdd_better, "promoted": promoted,
        "disclosure": {"sharpe": di_sr, "ann": di_ann, "maxdd": di_dd, "lw_z_vs_bh": round(di_z, 3)},
    }


def run(split: str = "2017-01-01", refresh: bool = False) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        px, gspc = load_data(refresh=refresh)
        res = evaluate(px, gspc, split)

    train, test, bh, dis = res["train"], res["test"], res["bh"], res["disclosure"]
    argmax, frozen = res["argmax"], res["frozen"]
    print(f"US-GATE NIFTYBEES  {px.index[0].date()}->{px.index[-1].date()}  "
          f"^GSPC {gspc.index[0].date()}->{gspc.index[-1].date()}  test>={split}")
    print(f"\n[TRAIN {TRAIN0}->{TRAIN1} @{int(HEADLINE_BPS)}bps | 4 variants -> FREEZE on net Sharpe]")
    print(train.to_string(index=False))
    print(f"FROZEN: {frozen} ({LABEL[frozen]})  (TRAIN argmax = {argmax}; "
          f"{'matches' if argmax == frozen else 'MISMATCH - freeze held per protocol'})")
    print(f"\n[TEST {split}+ ONE read | frozen {frozen} | 5/10/20 bps | LW z vs B&H NIFTYBEES]")
    print(test.to_string(index=False))
    print(f"B&H NIFTYBEES test Sharpe = {bh['sharpe']}  ann = {bh['ann']}  maxDD = {bh['maxdd']}  "
          f"({bh['days']} days)")
    print(f"\n[BAR]  LW z(strat - B&H) > 1 @10bps AND surviving 20bps AND maxDD better than B&H")
    print(f"  z@10 = {res['z10']}  z@20 = {res['z20']}  maxDD_better = {res['maxdd_better']}  "
          f"-> {'PASS' if res['promoted'] else 'FAIL'}")
    print(f"\n[DISCLOSURE ARM - no bearing on the verdict]  NIFTYBEES gated by its own INDIA 200d MA")
    print(f"  Sharpe = {dis['sharpe']}  ann = {dis['ann']}  maxDD = {dis['maxdd']}  "
          f"LW z vs B&H = {dis['lw_z_vs_bh']}")

    head = test[test["cost_bps"] == 10].iloc[0]
    log_run({
        "hypothesis_ref": "RL-2026-07-26-02", "universe": SYMBOL, "us_signal": US,
        "cost_bps": HEADLINE_BPS, "split": split,
        "strategy": f"us_gate_{frozen}", "frozen_variant": frozen, "train_argmax": argmax,
        "metrics": {"test_sharpe": float(head["test_sharpe"]), "ann_return": float(head["ann_return"]),
                    "max_dd": float(head["max_dd"]), "turnover": float(head["turnover"]),
                    "lw_z_vs_bh": float(head["lw_z_vs_bh"])},
        "z10": res["z10"], "z20": res["z20"], "maxdd_better": res["maxdd_better"],
        "bh_sharpe": bh["sharpe"], "bh_maxdd": bh["maxdd"],
        "disclosure_india_gate": dis, "promoted": res["promoted"],
        "n_trials_family": len(VARIANTS), "status": "success",
    })


def main() -> None:
    p = argparse.ArgumentParser(description="RL-2026-07-26-02 US-trend gate on NIFTYBEES")
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(split=a.split, refresh=a.refresh)


if __name__ == "__main__":
    main()
