"""RL-2026-07-20: turn-of-month effect on NIFTYBEES.NS (short-term, cost-light).

Equity returns concentrate around month boundaries (Ariel 1987; in India plausibly
amplified by SIP auto-debits clustering in the first trading days plus month-end
window-dressing). The book holds the index ETF only over the turn-of-month window
[last N trading days of month m, first M of month m+1] and sits in cash otherwise -
~24 one-way legs a year on a liquid ETF, so costs cannot kill it by construction; the
only question is whether the concentration survives out of sample.

Trading-day months come from the price index itself (not the calendar): the last N and
first M rows of each month-group. TRAIN (2010->2016-12-31) picks the (N, M) variant on
Sharpe from exactly four locked candidates and FREEZES it before the single TEST read
(2017+). Promotion bar (pre-registered): TOM Sharpe > buy-and-hold Sharpe AND
inside-vs-outside two-sample t >= 2 at 10 bps, surviving 20 bps. A miss is a valid
negative - global evidence says the effect has decayed post-2000.

Alignment: backtest_weights earns a target set at date D on D+1, so to earn the WINDOW
days' returns the 1.0 weights are shifted back one trading day (enter at the close before
the window, exit at its last close). Prices go through the RL-2026-07-17 spike repair -
the 2019-12 fabricated prints hit NIFTYBEES and land inside the test window.
"""

import argparse
import warnings

import pandas as pd
from scipy.stats import ttest_ind

from .backtest import backtest_weights
from .data import close_prices, load_yahoo_ohlcv
from .evaluation import sharpe_tstat
from .tracking import log_run
from .xasset_trend import clean_prices

SYMBOL = "NIFTYBEES.NS"
VARIANTS = [(3, 2), (2, 3), (1, 3), (3, 1)]
TRAIN0, TRAIN1 = "2010-01-01", "2016-12-31"
HEADLINE_BPS = 10.0
# Frozen on TRAIN (2010->2016-12-31) Sharpe @10 bps, before the single test read:
FROZEN_N, FROZEN_M = 2, 3


def tom_window(index: pd.DatetimeIndex, n: int, m: int) -> pd.Series:
    """Boolean mask, True on turn-of-month days: the first M and last N trading days of
    each month-group of the price index (calendar-foreseeable, so no look-ahead)."""
    g = pd.Series(0, index=index).groupby(index.to_period("M"))
    start = g.cumcount()
    end = g.transform("size") - 1 - start
    return (start < m) | (end < n)


def tom_weights(prices: pd.DataFrame, n: int, m: int) -> pd.DataFrame:
    """1.0 on the day BEFORE each window day, else 0 - so backtest_weights (target set at
    D earns on D+1) lands exactly the window days' returns in the book."""
    win = tom_window(prices.index, n, m)
    w = win.shift(-1, fill_value=False).astype(float)
    return w.to_frame(prices.columns[0])


def _metrics(r: pd.Series) -> tuple[float, float, float]:
    """(Sharpe, annualized return, max drawdown) of a daily return series."""
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    ann = float((1.0 + r).prod() ** (252 / max(len(r), 1)) - 1.0)
    return round(sharpe_tstat(r)[0], 3), round(ann, 4), round(dd, 4)


def load_prices(refresh: bool = False) -> pd.DataFrame:
    return clean_prices(close_prices(load_yahoo_ohlcv([SYMBOL], refresh=refresh))[[SYMBOL]])


def tom_returns(prices: pd.DataFrame, n: int, m: int, cost_bps: float) -> pd.Series:
    return backtest_weights(prices, tom_weights(prices, n, m), cost_bps).returns


def run(split: str = "2017-01-01", refresh: bool = False) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        px = load_prices(refresh=refresh)

        train_rows = []
        for n, m in VARIANTS:
            sr, ann, dd = _metrics(tom_returns(px, n, m, HEADLINE_BPS).loc[TRAIN0:TRAIN1])
            train_rows.append({"N": n, "M": m, "train_sharpe": sr, "train_ann": ann,
                               "train_maxdd": dd})
        train = pd.DataFrame(train_rows).sort_values("train_sharpe", ascending=False).reset_index(drop=True)
        argmax = (int(train.iloc[0]["N"]), int(train.iloc[0]["M"]))

        ret = px[SYMBOL].pct_change()
        win = tom_window(px.index, FROZEN_N, FROZEN_M)
        inside = ret.loc[split:][win.loc[split:]].dropna()
        outside = ret.loc[split:][~win.loc[split:]].dropna()
        t_stat, p_val = ttest_ind(inside, outside, equal_var=False)
        bh_sr = round(sharpe_tstat(ret.loc[split:])[0], 3)

        test_rows = []
        for cost in (5.0, 10.0, 20.0):
            sr, ann, dd = _metrics(tom_returns(px, FROZEN_N, FROZEN_M, cost).loc[split:])
            test_rows.append({"cost_bps": int(cost), "test_sharpe": sr, "ann_return": ann,
                              "max_dd": dd})
        test = pd.DataFrame(test_rows)
        sr10 = test[test["cost_bps"] == 10].iloc[0]["test_sharpe"]
        sr20 = test[test["cost_bps"] == 20].iloc[0]["test_sharpe"]
        promoted = bool(sr10 > bh_sr and t_stat >= 2.0 and sr20 > bh_sr)

    print(f"TOM NIFTYBEES  {px.index[0].date()}->{px.index[-1].date()}  test>={split}")
    print(f"\n[TRAIN {TRAIN0}->{TRAIN1} @{int(HEADLINE_BPS)}bps | 4 variants -> FREEZE on Sharpe]")
    print(train.to_string(index=False))
    print(f"FROZEN: (N={FROZEN_N},M={FROZEN_M})  (TRAIN argmax = {argmax}; "
          f"{'matches' if argmax == (FROZEN_N, FROZEN_M) else 'MISMATCH - freeze held per protocol'})")
    print(f"\n[TEST {split}+ ONE read | frozen (N={FROZEN_N},M={FROZEN_M}) | 5/10/20 bps]")
    print(test.to_string(index=False))
    print(f"B&H test Sharpe = {bh_sr}  ({len(px[SYMBOL].loc[split:].dropna())} days)")
    print(f"inside-vs-outside daily return  t={t_stat:.3f} p={p_val:.4f}  "
          f"(inside {inside.mean()*1e4:.1f}bps/d x{len(inside)} vs outside {outside.mean()*1e4:.1f}bps/d x{len(outside)})")
    print(f"\n[PROMOTION]  bar: TOM SR>B&H SR AND t>=2 @10bps, surviving 20bps  |  "
          f"SR@10={sr10} SR@20={sr20} B&H={bh_sr} t={t_stat:.2f}  -> {'PROMOTED' if promoted else 'NOT PROMOTED'}")

    for row in test_rows:
        log_run({"hypothesis_ref": "RL-2026-07-20", "universe": SYMBOL,
                 "cost_bps": float(row["cost_bps"]), "split": split,
                 "strategy": f"tom_N{FROZEN_N}_M{FROZEN_M}",
                 "frozen_n": FROZEN_N, "frozen_m": FROZEN_M, "metrics": row,
                 "bh_sharpe": bh_sr, "inside_outside_t": round(float(t_stat), 3),
                 "inside_outside_p": round(float(p_val), 4), "promoted": promoted,
                 "n_trials_family": len(VARIANTS), "status": "success"})


def main() -> None:
    p = argparse.ArgumentParser(description="RL-2026-07-20 turn-of-month effect on NIFTYBEES")
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(split=a.split, refresh=a.refresh)


if __name__ == "__main__":
    main()
