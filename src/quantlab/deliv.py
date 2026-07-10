"""RL-2026-07-26-20: delivery-percentage conviction cross-section (DELIV).

TRAIN-design + FORWARD-ONLY. Economic hypothesis (Llorente-Michaely-Saar-Wang 2002):
NSE's daily DELIVERABLE quantity is India's direct observable of informed/conviction
flow - the slice of a day's volume that actually settled to demat rather than being
round-tripped intraday. A high delivery share is accumulation, not churn, and returns
continue after informed-trading volume. Delivery is a genuinely NON-price input.

The raw signal per name/day is the delivery ratio = deliverable_qty / traded_qty in
(0,1], sourced from the MTO archive (`nse_mto`). Missing MTO days are left missing and
bridged only up to 3 trading days; a name with <60% valid days in a lookback is excluded
from that day's cross-section. Three pre-registered variants (all LONG the high end):

  LEVEL  = 63d mean delivery ratio                         (steady conviction level)
  SHOCK  = log(5d mean ratio / 126d mean ratio)            (a delivery-share surge)
  SIGNED = sign(5d price return) x SHOCK                    (conviction-CONFIRMED moves)

Each becomes a decile long/short book (long the top decile, short the bottom),
equal-weight, dollar-neutral, +/-3 MAD cross-sectional winsorization, weights lagged one
trading day, held on a month-end grid - the same construction the lab's other decile
sleeves use (`volshock.winsorize` / `volshock.decile_ls` are reused verbatim).

The TRAIN design reads 2011-07-01 (the measured MTO archive floor) -> 2016-12-31 ONLY;
nothing after 2016-12-31 enters a TRAIN performance statistic (enforced by slicing the
panel before any backtest, and test-asserted). The 2017 -> 2025-10 hold-out is NEVER
read. `freeze` = argmax TRAIN net Sharpe at 20 bps; 40 bps is the disclosed sensitivity.
The frozen variant then runs forward as a live paper book (`run_deliv` ->
paper_trades_deliv.jsonl), and `collect_mto_today` (in `nse_mto`) rescues each new day's
delivery file.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from . import live_paper as lp
from . import nse_mto
from .nse_events import IST
from .backtest import backtest_weights
from .data import close_prices, load_yahoo_ohlcv
from .evaluation import sharpe_tstat
from .india import india_panel
from .portfolio import rebalance_targets
from .tracking import log_run
from .volshock import decile_ls, winsorize

LEVEL_WIN = 63          # LEVEL: trailing mean-ratio window
SHORT = 5               # SHOCK: recent-surge window
LONG = 126              # SHOCK: half-year baseline window
MIN_FRAC = 0.60         # a name needs >= this share of valid days in a lookback
FFILL_LIMIT = 3         # bridge at most this many missing MTO days
DECILE = 0.10
MAD_N = 3.0
REBALANCE = "ME"

TRAIN_START = "2011-07-01"
TRAIN_END = "2016-12-31"

# Frozen at the TRAIN freeze = argmax TRAIN net Sharpe @20bps. Measured on the archive
# 2011-07-01->2016-12-31: LEVEL 0.914 (t 2.38) vs SHOCK -0.124 vs SIGNED -1.013 -> LEVEL.
FROZEN_VARIANT = "LEVEL"
VARIANTS = ("LEVEL", "SHOCK", "SIGNED")
DELIV_SNAPSHOT_PATH = "experiments/paper_trades_deliv.jsonl"
BENCH = lp.BENCH


# ------------------------------------------------------------------- ratio -> signal

def prep_ratio(r: pd.DataFrame, calendar: pd.DatetimeIndex | None = None,
               limit: int = FFILL_LIMIT) -> pd.DataFrame:
    """Align the delivery-ratio panel to the trading calendar and bridge SHORT gaps only:
    forward-fill at most `limit` consecutive missing MTO days. Longer gaps stay NaN, so a
    name with stale delivery data is not silently carried."""
    if calendar is not None:
        r = r.reindex(calendar)
    return r.ffill(limit=limit)


def _masked_mean(r: pd.DataFrame, window: int, min_frac: float = MIN_FRAC) -> pd.DataFrame:
    """Trailing mean of the ratio over `window`, defined only where at least
    `min_frac`*window of the days are valid (non-NaN). This IS the <60%-valid exclusion:
    a name without enough delivery observations in the lookback drops to NaN and leaves
    the cross-section."""
    cnt = r.notna().astype(float).rolling(window, min_periods=1).sum()
    mean = r.rolling(window, min_periods=1).mean()      # pandas skips NaN in the mean
    return mean.where(cnt >= min_frac * window)


def level(r: pd.DataFrame) -> pd.DataFrame:
    """LEVEL = 63d mean delivery ratio (higher = steadier conviction)."""
    return _masked_mean(r, LEVEL_WIN)


def shock(r: pd.DataFrame) -> pd.DataFrame:
    """SHOCK = log(5d mean ratio / 126d mean ratio). Ratios are strictly positive where
    defined, so the log is finite; both means carry the <60%-valid exclusion."""
    with np.errstate(all="ignore"):
        return np.log(_masked_mean(r, SHORT) / _masked_mean(r, LONG))


def signed_shock(r: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """SIGNED = sign(5d price return) x SHOCK: a delivery surge counts as a positive
    conviction signal only when price also rose over the same 5 days (and vice-versa)."""
    with np.errstate(all="ignore"):
        return np.sign(px.pct_change(SHORT)) * shock(r)


def signals(r: pd.DataFrame, px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The three pre-registered raw signal panels, keyed by variant."""
    return {"LEVEL": level(r), "SHOCK": shock(r), "SIGNED": signed_shock(r, px)}


def book_weights(signal: pd.DataFrame, q: float = DECILE, n_mad: float = MAD_N,
                 rebalance: str | None = REBALANCE) -> pd.DataFrame:
    """Signed decile L/S target weights from a raw signal panel: +/-`n_mad` MAD
    cross-sectional winsorize, lag one trading day, long the top-`q` decile / short the
    bottom (equal-weight, dollar-neutral, unit gross), then hold on a monthly grid."""
    w = decile_ls(winsorize(signal, n_mad).shift(1), q)
    if rebalance:
        w = rebalance_targets(w, rebalance).ffill().fillna(0.0)
    return w


# --------------------------------------------------------------------------- panels

def _join_ratios(kept: list[str], calendar: pd.DatetimeIndex,
                 mto_dir: str = nse_mto.DATA_DIR) -> pd.DataFrame:
    """Delivery-ratio panel joined to the universe: bare MTO SYMBOL -> SYMBOL.NS, kept to
    the panel's current names and aligned+gap-bridged to its calendar."""
    raw = nse_mto.load_ratios(mto_dir)
    if raw.empty:
        return pd.DataFrame(index=calendar, columns=kept, dtype=float)
    raw = raw.copy()
    raw.columns = [str(c).upper() + ".NS" for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.duplicated()].reindex(columns=kept)
    return prep_ratio(raw, calendar)


def panels(start: str = "2010-01-01", index: str = "nifty500", refresh: bool = False,
           mto_dir: str = nse_mto.DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """(px_adj, delivery_ratio, raw_close, market). `px_adj` is the winsorized total-return
    panel (returns + 5d sign); `delivery_ratio` is the gap-bridged ratio panel over the
    same universe/calendar; `raw_close` is the split-adjusted close for the live baseline."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        px, mkt, ohlcv, kept = india_panel(start=start, index=index, ret_clip=0.40, refresh=refresh)
    ratio = _join_ratios(kept, px.index, mto_dir)
    return px, ratio, ohlcv["close"], mkt


def latest_signals(r: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """The winsorized cross-sectional score of each variant on the last date (columns =
    variants, index = symbol). What the orchestrator correlates across signals/sleeves."""
    out = {name: winsorize(sig, MAD_N).iloc[-1] for name, sig in signals(r, px).items()}
    return pd.DataFrame(out).dropna(how="all")


def latest_signal(variant: str = FROZEN_VARIANT, start: str = "2010-01-01",
                  index: str = "nifty500", refresh: bool = False) -> pd.Series:
    """Latest winsorized cross-sectional score for one variant, as a Series."""
    px, ratio, _, _ = panels(start=start, index=index, refresh=refresh)
    return winsorize(signals(ratio, px)[variant], MAD_N).iloc[-1].dropna()


# ----------------------------------------------------------------------- TRAIN study

def _ann_return(r: pd.Series) -> float:
    r = r.dropna()
    return float((1.0 + r).prod() ** (252 / max(len(r), 1)) - 1.0)


def _ann_turnover(res) -> float:
    n = max(len(res.turnover), 1)
    return float(res.turnover.sum() * 252 / n)


def train_table(px: pd.DataFrame, ratio: pd.DataFrame, costs=(20.0, 40.0),
                end: str = TRAIN_END) -> pd.DataFrame:
    """Per-variant TRAIN net metrics at each cost. The panel is sliced to `end` BEFORE any
    signal or backtest, so no post-TRAIN bar can enter a statistic."""
    px_tr, ratio_tr = px.loc[:end], ratio.loc[:end]
    assert px_tr.index.max() <= pd.Timestamp(end), "TRAIN slice leaked past the boundary"
    rows = []
    for name, sig in signals(ratio_tr, px_tr).items():
        w = book_weights(sig)
        for cost in costs:
            res = backtest_weights(px_tr, w, cost_bps=cost)
            sr, t = sharpe_tstat(res.returns)
            assert res.returns.index.max() <= pd.Timestamp(end), "TRAIN return leaked past boundary"
            rows.append({"variant": name, "cost_bps": int(cost),
                         "net_sharpe": round(sr, 3), "t_stat": round(t, 2),
                         "ann_return": round(_ann_return(res.returns), 4),
                         "max_dd": round(res.max_drawdown, 4),
                         "turnover": round(_ann_turnover(res), 1)})
    return pd.DataFrame(rows)


def freeze(table: pd.DataFrame, cost_bps: float = 20.0) -> tuple[str, float]:
    """Frozen variant = argmax TRAIN net Sharpe at `cost_bps`; margin = its Sharpe minus
    the runner-up's."""
    t = table[table["cost_bps"] == int(cost_bps)].sort_values("net_sharpe", ascending=False)
    chosen = str(t.iloc[0]["variant"])
    margin = float(t.iloc[0]["net_sharpe"] - t.iloc[1]["net_sharpe"]) if len(t) > 1 else float("nan")
    return chosen, round(margin, 3)


def coverage(px: pd.DataFrame, mto_dir: str = nse_mto.DATA_DIR,
             kept: list[str] | None = None, end: str = TRAIN_END) -> dict:
    """TRAIN MTO coverage of the universe (measured on the RAW join, no gap-bridging):
    matched names, the archive floor, and the median per-name share of valid TRAIN days."""
    raw = nse_mto.load_ratios(mto_dir, end=end)
    kept = kept if kept is not None else list(px.columns)
    if raw.empty:
        return {"floor": None, "matched": 0, "median_valid_pct": 0.0, "train_days": 0}
    raw = raw.copy()
    raw.columns = [str(c).upper() + ".NS" for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.duplicated()].reindex(columns=kept)
    cal = px.loc[:end].index
    aligned = raw.reindex(cal)
    valid = aligned.notna()
    matched = [c for c in kept if valid[c].any()]
    floor = raw.dropna(how="all").index.min()
    med = float(valid[matched].mean().median() * 100) if matched else 0.0
    return {"floor": None if floor is pd.NaT else str(floor.date()),
            "matched": len(matched), "median_valid_pct": round(med, 1),
            "train_days": int(len(cal))}


def run(start: str = "2010-01-01", refresh: bool = False, mto_dir: str = nse_mto.DATA_DIR) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 30)
    px, ratio, _, _ = panels(start=start, refresh=refresh, mto_dir=mto_dir)

    cov = coverage(px, mto_dir, kept=list(px.columns))
    print(f"DELIV  N500-{px.shape[1]}  panel {px.index[0].date()}->{px.index[-1].date()}")
    print(f"[coverage TRAIN<= {TRAIN_END}]  archive floor {cov['floor']}  matched {cov['matched']}/"
          f"{px.shape[1]}  median valid-day% {cov['median_valid_pct']}  ({cov['train_days']} train days)")

    table = train_table(px, ratio)
    print("\n[TRAIN table  variant x {20,40}bps]  (net, all read <= 2016-12-31)")
    print(table.to_string(index=False))

    chosen, margin = freeze(table)
    print(f"\nFROZEN (argmax TRAIN net Sharpe @20bps): {chosen}  (margin over runner-up {margin} SR)")
    if chosen != FROZEN_VARIANT:
        print(f"  NOTE: module FROZEN_VARIANT={FROZEN_VARIANT} differs from measured argmax {chosen}")

    latest = latest_signals(ratio, px)
    print(f"\n[latest cross-signal corr  {px.index[-1].date()}]")
    print(latest.corr().round(3).to_string())

    for r in table.to_dict("records"):
        log_run({"hypothesis_ref": "RL-2026-07-26-20", "kind": "deliv_train",
                 "universe": "NIFTY500", "window": f"{TRAIN_START}->{TRAIN_END}",
                 "strategy": f"DELIV-{r['variant']}", "cost_bps": float(r["cost_bps"]),
                 "frozen_variant": chosen, "freeze_margin_sr": margin,
                 "archive_floor": cov["floor"], "coverage": cov,
                 "metrics": r, "status": "success"})


# ----------------------------------------------------------------------- live book

def current_deliv_book(start: str = "2010-01-01", index: str = "nifty500",
                       refresh: bool = False, variant: str | None = None) -> lp.Book:
    """Reconstruct the frozen DELIV decile L/S sleeve on the latest panel date; SIGNED
    dollar-neutral weights (top-ratio decile +, bottom -). The intraday baseline is the
    raw (split-adjusted) close, matching the raw Groww LTP - `live_paper` conventions."""
    variant = variant or FROZEN_VARIANT
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        px, ratio, raw_close, _ = panels(start=start, index=index, refresh=refresh)
        w_full = book_weights(signals(ratio, px)[variant]).iloc[-1]
        nsei = close_prices(load_yahoo_ohlcv([BENCH]))[BENCH].reindex(px.index).ffill()

    last = px.index[-1]
    w = w_full[w_full.abs() > 1e-12].sort_values(ascending=False)

    today = datetime.now(IST).date()
    completed = raw_close[raw_close.index.map(lambda d: d.date() < today)]
    prev_row = completed.iloc[-1] if len(completed) else raw_close.iloc[-1]
    nsei_prev = nsei[nsei.index.map(lambda d: d.date() < today)]
    nsei_prev_close = float(nsei_prev.iloc[-1] if len(nsei_prev) else nsei.iloc[-1])

    return lp.Book(weights=w, regime_on=bool(len(w) > 0),
                   cash_frac=float(1.0 - w.abs().sum()), latest_date=last,
                   prev_close=prev_row.reindex(w.index), nsei_prev_close=nsei_prev_close)


def run_deliv(start: str = "2010-01-01", index: str = "nifty500", refresh: bool = False,
              path: str = DELIV_SNAPSHOT_PATH, write: bool = True,
              variant: str | None = None) -> dict:
    """Snapshot the frozen delivery-conviction decile L/S sleeve to its own ledger.
    Dollar-neutral (net ~0, gross ~1); SIGNED weights; graceful no-quote degradation."""
    variant = variant or FROZEN_VARIANT
    book = current_deliv_book(start=start, index=index, refresh=refresh, variant=variant)
    gross, net = float(book.weights.abs().sum()), float(book.weights.sum())
    n_long, n_short = int((book.weights > 0).sum()), int((book.weights < 0).sum())
    book_ret, _, n_ok, n_req, err = lp.live_book_pnl(book)
    nifty_ret, proxy = lp.nifty_intraday(book.nsei_prev_close)

    row = {
        "hypothesis_ref": "RL-2026-07-26-20", "kind": "live_paper_deliv_snapshot",
        "asof_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "panel_date": str(book.latest_date.date()), "universe": index,
        "variant": variant, "gross": round(gross, 4), "net": round(net, 6),
        "n_long": n_long, "n_short": n_short,
        "book_intraday_ret": None if book_ret is None else round(book_ret, 6),
        "nifty_intraday_ret": None if nifty_ret is None else round(nifty_ret, 6),
        "nifty_proxy": proxy, "n_names": int(n_req), "n_quotes_ok": n_ok,
        "groww_ok": err is None and n_ok > 0, "note": err or "ok",
        "weights": {s: round(float(w), 6) for s, w in book.weights.items()},
    }
    if write:
        log_run(row, path=path)

    print(f"[DELIV live paper] panel {row['panel_date']}  variant={variant}  "
          f"gross={gross:.2f}  net={net:+.4f}  long={n_long} short={n_short}")
    print("TOP 8 longs / TOP 8 shorts:")
    for s, wt in book.weights.head(8).items():
        print(f"  + {s:16s} {wt*100:6.2f}%")
    for s, wt in book.weights.tail(8).items():
        print(f"  - {s:16s} {wt*100:6.2f}%")
    if book_ret is None:
        print(f"live sleeve P&L: UNAVAILABLE (quotes ok {n_ok}/{n_req}; {err})")
    else:
        print(f"live sleeve intraday {book_ret*100:+.2f}% (dollar-neutral target ~0); "
              f"quotes ok {n_ok}/{n_req}")
    print(f"snapshot {'appended to '+path if write else 'NOT written (dry run)'}")
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="RL-2026-07-26-20 DELIV: TRAIN study / live snapshot")
    p.add_argument("--mode", choices=("train", "live"), default="train")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.mode == "live":
        run_deliv(start=a.start, refresh=a.refresh, write=not a.dry_run)
    else:
        run(start=a.start, refresh=a.refresh)


if __name__ == "__main__":
    main()
