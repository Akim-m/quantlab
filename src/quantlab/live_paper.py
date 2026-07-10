"""Live paper-observation harness for the deployable Indian books.

READ-ONLY. Four sleeves, separate ledgers: the RL-2026-07-10 long-only REGIME book
(`run` -> `paper_trades.jsonl`), the RL-2026-07-12 F&O-shortable residual-momentum
long-short sleeve (`run_ls` -> `paper_trades_ls.jsonl`, signed weights), the
RL-2026-07-17 multi-asset trend sleeve (`run_trend` -> `paper_trades_trend.jsonl`,
five NSE ETFs, per-asset gate states), and the RL-2026-07-16-flagged gold_lowbeta
risk-off variant (`run_gl` -> `paper_trades_gl.jsonl`, REGIME base + the 50/50
trend-gated-GOLDBEES / low-beta sleeve filling the freed weight on risk-off days).
Each builds the current target book from (optionally refreshed) Yahoo history via its
study's own frozen construction, fetches LIVE Groww LTP for the held names, computes
the book's live intraday P&L, and APPENDS one snapshot row so daily re-runs accumulate
a forward track record.

Nothing here mutates account state: only read-only market-data methods are called,
through the rate-limited `quantlab.groww_client.call` wrapper that itself refuses
order methods. When Groww live data is unavailable - the API key lacks live-data
entitlement, or the cash market is shut - the run degrades gracefully: it still
records the target book and regime state, with the live P&L fields left null.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import divcarry
from . import dualrot
from . import groww_client as gc
from . import illiq
from . import macrobeta
from . import pairs_rv
from . import volshock
from .backtest import backtest_weights
from .blend import composite, fno_long_short, market_on, regime_on
from .data import close_prices, load_yahoo_ohlcv
from .india import fno_shortable, india_panel, sector_map
from .india_blend_study import raw_signals
from .india_ls import CORE as LS_CORE, WEIGHTS as LS_WEIGHTS
from .india_run import _core_regime_band_ls
from .riskoff_sleeve import GOLD, VIX_SYM, base_book, combined_book
from .tracking import log_run
from .xasset_trend import ETFS, FROZEN_GATE, FROZEN_WEIGHTING, etf_panel, sleeve_weights

SEGMENT = "CASH"                       # NSE cash equity
# read-only Groww methods this harness may route through gc.call (order methods are
# refused by gc.call itself; the method-spy test proves only these are ever reached).
READ_METHODS = ("get_ltp", "get_all_instruments")
NIFTY_PROXIES = ("NSE_NIFTY", "NSE_NIFTYBEES")  # index first, then the tradeable ETF
BENCH = "^NSEI"                        # Nifty 50 price index, the forward-track benchmark
IST = timezone(timedelta(hours=5, minutes=30))
SNAPSHOT_PATH = "experiments/paper_trades.jsonl"
LS_SNAPSHOT_PATH = "experiments/paper_trades_ls.jsonl"
TREND_SNAPSHOT_PATH = "experiments/paper_trades_trend.jsonl"
GL_SNAPSHOT_PATH = "experiments/paper_trades_gl.jsonl"
DUALROT_SNAPSHOT_PATH = "experiments/paper_trades_dualrot.jsonl"
DIVCARRY_SNAPSHOT_PATH = "experiments/paper_trades_divcarry.jsonl"
VOLSHOCK_SNAPSHOT_PATH = "experiments/paper_trades_volshock.jsonl"
MACROBETA_SNAPSHOT_PATH = "experiments/paper_trades_macrobeta.jsonl"
ILLIQ_SNAPSHOT_PATH = "experiments/paper_trades_illiq.jsonl"
PAIRS_SNAPSHOT_PATH = "experiments/paper_trades_pairs.jsonl"


def to_groww(yahoo_sym: str) -> str:
    """'RELIANCE.NS' -> 'NSE_RELIANCE' (Groww exchange_symbol for the NSE cash leg)."""
    return "NSE_" + yahoo_sym.upper().removesuffix(".NS")


def _price(v) -> float | None:
    """Pull a last price out of a Groww LTP/OHLC payload value, tolerant of shape."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in ("ltp", "last_price", "value", "close", "tp"):
            if isinstance(v.get(k), (int, float)):
                return float(v[k])
    return None


@dataclass
class Book:
    weights: pd.Series          # nonzero target weights, indexed by Yahoo symbol
    regime_on: bool             # True = risk-on (^NSEI above its 200d MA)
    cash_frac: float            # 1 - sum(weights)
    latest_date: pd.Timestamp   # last panel date the weights were computed on
    prev_close: pd.Series       # last completed-session raw close per held name
    nsei_prev_close: float      # ^NSEI last completed-session close


def current_book(start: str = "2010-01-01", index: str = "nifty500",
                 refresh: bool = False) -> Book:
    """Reconstruct the REGIME book and return its target positions on the latest date."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            px, mkt, ohlcv, _ = india_panel(start=start, index=index,
                                             ret_clip=0.40, refresh=refresh)
            sigs = raw_signals(px, mkt, sector_map(index))
            _, _, shared = _core_regime_band_ls(px, mkt, sigs)

    regime = shared["REGIME"][0]
    last = regime.index[-1]
    w = regime.loc[last]
    w = w[w > 0].sort_values(ascending=False)

    # prev_close = last close strictly before today's IST date, so an in-progress
    # (partial) same-day Yahoo bar never becomes its own baseline.
    today = datetime.now(IST).date()
    raw = ohlcv["close"]
    completed = raw[raw.index.map(lambda d: d.date() < today)]
    prev_row = completed.iloc[-1] if len(completed) else raw.iloc[-1]
    nsei_prev = mkt[mkt.index.map(lambda d: d.date() < today)]
    nsei_prev_close = float(nsei_prev.iloc[-1] if len(nsei_prev) else mkt.iloc[-1])

    on = bool(market_on(mkt, 200).reindex(regime.index).loc[last])
    return Book(weights=w, regime_on=on, cash_frac=float(1.0 - w.sum()),
                latest_date=last, prev_close=prev_row.reindex(w.index),
                nsei_prev_close=nsei_prev_close)


LTP_BATCH = 50  # Groww caps get_ltp at 50 symbols per request


def fetch_ltp(exchange_symbols: list[str]) -> tuple[dict[str, float], str | None]:
    """Batched read-only Groww LTP (<=50 symbols/call). Returns (prices, error_or_None).

    Never raises: a Groww/network failure yields whatever was fetched so far plus the
    reason, so the caller records a (partial) snapshot instead of crashing the run.
    """
    out: dict[str, float] = {}
    for i in range(0, len(exchange_symbols), LTP_BATCH):
        chunk = exchange_symbols[i:i + LTP_BATCH]
        try:
            payload = gc.call("get_ltp", exchange_trading_symbols=tuple(chunk), segment=SEGMENT)
        except Exception as e:  # auth/entitlement/network/rate — all non-fatal here
            return out, f"{type(e).__name__}: {e}"
        for s in chunk:
            if (p := _price(payload.get(s))) is not None:
                out[s] = p
    return out, None


def nifty_intraday(nsei_prev_close: float) -> tuple[float | None, str | None]:
    """Live Nifty intraday move from the first Groww proxy that quotes."""
    prices, _ = fetch_ltp(list(NIFTY_PROXIES))
    for proxy in NIFTY_PROXIES:
        if proxy in prices and nsei_prev_close:
            return prices[proxy] / nsei_prev_close - 1.0, proxy
    return None, None


def live_book_pnl(book: Book) -> tuple[float | None, dict[str, float], int, int, str | None]:
    """Fetch live LTP for the held names and compute the book's signed intraday
    return vs each name's prev close. Works for long-only and signed L/S books
    alike (shorts have negative weight -> negative contribution on an up move).
    Returns (book_ret_or_None, per_name_live_ret, n_quotes_ok, n_names_requested,
    error_or_None)."""
    groww_syms = [to_groww(s) for s in book.weights.index]
    prices, err = fetch_ltp(groww_syms)
    live_ret = {s: prices[g] / book.prev_close[s] - 1.0
                for s, g in zip(book.weights.index, groww_syms)
                if g in prices and book.prev_close.get(s, 0)}
    n_ok = len(live_ret)
    book_ret = float(sum(book.weights[s] * r for s, r in live_ret.items())) if n_ok else None
    return book_ret, live_ret, n_ok, len(groww_syms), err


def run(start: str = "2010-01-01", index: str = "nifty500", refresh: bool = False,
        path: str = SNAPSHOT_PATH, write: bool = True) -> dict:
    book = current_book(start=start, index=index, refresh=refresh)
    state = "risk_on" if book.regime_on else "risk_off"
    book_ret, _, n_ok, n_req, err = live_book_pnl(book)
    nifty_ret, proxy = nifty_intraday(book.nsei_prev_close)

    row = {
        "hypothesis_ref": "RL-2026-07-10", "kind": "live_paper_snapshot",
        "asof_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "panel_date": str(book.latest_date.date()), "universe": index,
        "regime_state": state, "cash_frac": round(book.cash_frac, 4),
        "book_intraday_ret": None if book_ret is None else round(book_ret, 6),
        "nifty_intraday_ret": None if nifty_ret is None else round(nifty_ret, 6),
        "nifty_proxy": proxy, "n_names": int(len(book.weights)),
        "n_quotes_ok": n_ok, "groww_ok": err is None and n_ok > 0,
        "note": err or "ok",
        "weights": {s: round(float(w), 6) for s, w in book.weights.items()},
    }
    if write:
        log_run(row, path=path)

    print(f"[REGIME live paper] panel {row['panel_date']}  regime={state}  "
          f"cash={book.cash_frac:.0%}  names={len(book.weights)}")
    print("TOP 15 target holdings:")
    for s, wt in book.weights.head(15).items():
        print(f"  {s:16s} {wt*100:6.2f}%")
    if book_ret is None:
        print(f"live book P&L: UNAVAILABLE (quotes ok {n_ok}/{n_req}; {err})")
    else:
        nb = "n/a" if nifty_ret is None else f"{nifty_ret*100:+.2f}%"
        print(f"live book intraday {book_ret*100:+.2f}% vs Nifty {nb} "
              f"({proxy}); quotes ok {n_ok}/{n_req}")
    print(f"snapshot {'appended to '+path if write else 'NOT written (dry run)'}")
    return row


def current_ls_book(start: str = "2010-01-01", index: str = "nifty500",
                    refresh: bool = False) -> Book:
    """Reconstruct the RL-2026-07-12 F&O-shortable resid-mom L/S sleeve on the latest
    panel date and return its SIGNED target weights (longs +, shorts -)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            px, mkt, ohlcv, _ = india_panel(start=start, index=index,
                                             ret_clip=0.40, refresh=refresh)
            score = composite({k: raw_signals(px, mkt, sector_map(index))[k] for k in LS_CORE},
                              weights=LS_WEIGHTS)
            shortable = {s.upper() for s in fno_shortable(refresh=refresh)}
            book = fno_long_short(score, shortable)

    last = book.index[-1]
    w = book.loc[last]
    w = w[w.abs() > 1e-9].sort_values(ascending=False)

    today = datetime.now(IST).date()
    raw = ohlcv["close"]
    completed = raw[raw.index.map(lambda d: d.date() < today)]
    prev_row = completed.iloc[-1] if len(completed) else raw.iloc[-1]
    nsei_prev = mkt[mkt.index.map(lambda d: d.date() < today)]
    nsei_prev_close = float(nsei_prev.iloc[-1] if len(nsei_prev) else mkt.iloc[-1])

    on = bool(market_on(mkt, 200).reindex(book.index).loc[last])
    return Book(weights=w, regime_on=on, cash_frac=float(1.0 - w.abs().sum()),
                latest_date=last, prev_close=prev_row.reindex(w.index),
                nsei_prev_close=nsei_prev_close)


def run_ls(start: str = "2010-01-01", index: str = "nifty500", refresh: bool = False,
           path: str = LS_SNAPSHOT_PATH, write: bool = True) -> dict:
    """Snapshot the implementable market-neutral sleeve. Separate ledger from the
    long-only REGIME track; rows carry SIGNED weights (shorts negative)."""
    book = current_ls_book(start=start, index=index, refresh=refresh)
    gross, net = float(book.weights.abs().sum()), float(book.weights.sum())
    n_long, n_short = int((book.weights > 0).sum()), int((book.weights < 0).sum())
    book_ret, _, n_ok, n_req, err = live_book_pnl(book)
    nifty_ret, proxy = nifty_intraday(book.nsei_prev_close)

    row = {
        "hypothesis_ref": "RL-2026-07-12", "kind": "live_paper_ls_snapshot",
        "asof_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "panel_date": str(book.latest_date.date()), "universe": index,
        "gross": round(gross, 4), "net": round(net, 6),
        "n_long": n_long, "n_short": n_short,
        "book_intraday_ret": None if book_ret is None else round(book_ret, 6),
        "nifty_intraday_ret": None if nifty_ret is None else round(nifty_ret, 6),
        "nifty_proxy": proxy, "n_names": int(n_req), "n_quotes_ok": n_ok,
        "groww_ok": err is None and n_ok > 0, "note": err or "ok",
        "weights": {s: round(float(w), 6) for s, w in book.weights.items()},
    }
    if write:
        log_run(row, path=path)

    print(f"[F&O L/S live paper] panel {row['panel_date']}  gross={gross:.2f}  net={net:+.4f}  "
          f"long={n_long} short={n_short}")
    print("TOP 8 longs / TOP 8 shorts:")
    for s, wt in book.weights.head(8).items():
        print(f"  + {s:16s} {wt*100:6.2f}%")
    for s, wt in book.weights.tail(8).items():
        print(f"  - {s:16s} {wt*100:6.2f}%")
    if book_ret is None:
        print(f"live sleeve P&L: UNAVAILABLE (quotes ok {n_ok}/{n_req}; {err})")
    else:
        print(f"live sleeve intraday {book_ret*100:+.2f}% (market-neutral target ~0); "
              f"quotes ok {n_ok}/{n_req}")
    print(f"snapshot {'appended to '+path if write else 'NOT written (dry run)'}")
    return row


def current_trend_book(refresh: bool = False) -> tuple[Book, dict[str, bool]]:
    """Reconstruct the RL-2026-07-17 5-ETF trend sleeve (frozen tsmom gate + inverse-vol
    weights) on the latest cleaned panel date. Returns its long-only target weights and
    each ETF's held/cash state (weight > 0 == its own trend is up and it is held)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            px = etf_panel(refresh=refresh)
            w_full = sleeve_weights(px, FROZEN_GATE, FROZEN_WEIGHTING).iloc[-1]

    last = px.index[-1]
    gates = {s: bool(w_full.get(s, 0.0) > 0) for s in ETFS}
    w = w_full[w_full > 0].sort_values(ascending=False)

    # Raw (unadjusted) close is the intraday baseline the raw Groww LTP compares to;
    # the warm cache from etf_panel is reused (refresh=False here avoids a double pull).
    raw = close_prices(load_yahoo_ohlcv(ETFS), field="close")[ETFS].reindex(px.index)
    today = datetime.now(IST).date()
    completed = raw[raw.index.map(lambda d: d.date() < today)]
    prev_row = completed.iloc[-1] if len(completed) else raw.iloc[-1]

    book = Book(weights=w, regime_on=bool(len(w) > 0), cash_frac=float(1.0 - w.sum()),
                latest_date=last, prev_close=prev_row.reindex(w.index),
                nsei_prev_close=float("nan"))
    return book, gates


def run_trend(refresh: bool = False, path: str = TREND_SNAPSHOT_PATH, write: bool = True) -> dict:
    """Snapshot the RL-2026-07-17 multi-asset trend sleeve to its own ledger. Long-only,
    weights sum to <=1 (cash for off-trend legs); forward_track-compatible."""
    book, gates = current_trend_book(refresh=refresh)
    book_ret, live_ret, n_ok, n_req, err = live_book_pnl(book)

    row = {
        "hypothesis_ref": "RL-2026-07-17", "kind": "live_paper_trend_snapshot",
        "asof_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "panel_date": str(book.latest_date.date()), "universe": "NSE-ETF5",
        "cash_frac": round(book.cash_frac, 4),
        "book_intraday_ret": None if book_ret is None else round(book_ret, 6),
        "n_names": int(len(book.weights)), "n_quotes_ok": n_ok,
        "groww_ok": err is None and n_ok > 0, "note": err or "ok",
        "gate_states": gates,
        "asset_intraday": {s: round(r, 6) for s, r in live_ret.items()},
        "weights": {s: round(float(w), 6) for s, w in book.weights.items()},
    }
    if write:
        log_run(row, path=path)

    print(f"[TREND live paper] panel {row['panel_date']}  held={len(book.weights)}/5  "
          f"cash={book.cash_frac:.0%}")
    for s, wt in book.weights.items():
        mv = live_ret.get(s)
        mvs = "n/a" if mv is None else f"{mv*100:+.2f}%"
        print(f"  {s:16s} {wt*100:6.2f}%  intraday {mvs}")
    off = [s for s, on in gates.items() if not on]
    print(f"off-trend (cash): {', '.join(off) if off else 'none'}")
    if book_ret is None:
        print(f"live sleeve P&L: UNAVAILABLE (quotes ok {n_ok}/{n_req}; {err})")
    else:
        print(f"live sleeve intraday {book_ret*100:+.2f}%; quotes ok {n_ok}/{n_req}")
    print(f"snapshot {'appended to '+path if write else 'NOT written (dry run)'}")
    return row


def current_dualrot_book(refresh: bool = False) -> tuple[Book, dict[str, bool]]:
    """Reconstruct the RL-2026-07-26-01 5-ETF dual-momentum rotation (frozen top-K / gate)
    on the latest cleaned panel date. Returns its long-only target weights (top-K held at
    1/K each, cash for selected-but-gated-out or unselected sleeves) and each ETF's
    held/cash state (weight > 0 == it is in the top-K and its absolute gate is up)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            px = etf_panel(refresh=refresh)
            w_full = dualrot.latest_weights(px)

    last = px.index[-1]
    gates = {s: bool(w_full.get(s, 0.0) > 0) for s in ETFS}
    w = w_full[w_full > 0].sort_values(ascending=False)

    # Raw (unadjusted) close is the intraday baseline the raw Groww LTP compares to;
    # the warm cache from etf_panel is reused (refresh=False here avoids a double pull).
    raw = close_prices(load_yahoo_ohlcv(ETFS), field="close")[ETFS].reindex(px.index)
    today = datetime.now(IST).date()
    completed = raw[raw.index.map(lambda d: d.date() < today)]
    prev_row = completed.iloc[-1] if len(completed) else raw.iloc[-1]

    book = Book(weights=w, regime_on=bool(len(w) > 0), cash_frac=float(1.0 - w.sum()),
                latest_date=last, prev_close=prev_row.reindex(w.index),
                nsei_prev_close=float("nan"))
    return book, gates


def run_dualrot(refresh: bool = False, path: str = DUALROT_SNAPSHOT_PATH, write: bool = True) -> dict:
    """Snapshot the RL-2026-07-26-01 dual-momentum rotation to its own ledger. Long-only,
    weights sum to <=1 (cash for ungated/unselected legs); forward_track-compatible."""
    book, gates = current_dualrot_book(refresh=refresh)
    book_ret, live_ret, n_ok, n_req, err = live_book_pnl(book)

    row = {
        "hypothesis_ref": "RL-2026-07-26-01", "kind": "live_paper_dualrot_snapshot",
        "asof_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "panel_date": str(book.latest_date.date()), "universe": "NSE-ETF5",
        "cash_frac": round(book.cash_frac, 4),
        "book_intraday_ret": None if book_ret is None else round(book_ret, 6),
        "n_names": int(len(book.weights)), "n_quotes_ok": n_ok,
        "groww_ok": err is None and n_ok > 0, "note": err or "ok",
        "gate_states": gates,
        "asset_intraday": {s: round(r, 6) for s, r in live_ret.items()},
        "top_k": dualrot.FROZEN_TOP_K, "abs_gate": dualrot.FROZEN_GATE,
        "weights": {s: round(float(w), 6) for s, w in book.weights.items()},
    }
    if write:
        log_run(row, path=path)

    print(f"[DUAL-ROT live paper] panel {row['panel_date']}  held={len(book.weights)}/5  "
          f"cash={book.cash_frac:.0%}  (K={dualrot.FROZEN_TOP_K} gate={dualrot.FROZEN_GATE})")
    for s, wt in book.weights.items():
        mv = live_ret.get(s)
        mvs = "n/a" if mv is None else f"{mv*100:+.2f}%"
        print(f"  {s:16s} {wt*100:6.2f}%  intraday {mvs}")
    off = [s for s, on in gates.items() if not on]
    print(f"cash (unselected or gated-out): {', '.join(off) if off else 'none'}")
    if book_ret is None:
        print(f"live sleeve P&L: UNAVAILABLE (quotes ok {n_ok}/{n_req}; {err})")
    else:
        print(f"live sleeve intraday {book_ret*100:+.2f}%; quotes ok {n_ok}/{n_req}")
    print(f"snapshot {'appended to '+path if write else 'NOT written (dry run)'}")
    return row


def current_divcarry_book(start: str = "2010-01-01", index: str = "nifty500",
                          refresh: bool = False) -> Book:
    """Reconstruct the RL-2026-07-26-13 dividend-yield decile L/S sleeve on the latest
    panel date; SIGNED dollar-neutral weights (top-yield decile +, bottom -). The
    intraday baseline is the raw (split-adjusted) close, matching the raw Groww LTP."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            close, adj = divcarry.panels(start=start, index=index, refresh=refresh)
            w_full = divcarry.latest_weights(close, adj)
            nsei = close_prices(load_yahoo_ohlcv([BENCH]))[BENCH].reindex(close.index).ffill()

    last = close.index[-1]
    w = w_full[w_full.abs() > 1e-12].sort_values(ascending=False)

    today = datetime.now(IST).date()
    completed = close[close.index.map(lambda d: d.date() < today)]
    prev_row = completed.iloc[-1] if len(completed) else close.iloc[-1]
    nsei_prev = nsei[nsei.index.map(lambda d: d.date() < today)]
    nsei_prev_close = float(nsei_prev.iloc[-1] if len(nsei_prev) else nsei.iloc[-1])

    return Book(weights=w, regime_on=bool(len(w) > 0),
                cash_frac=float(1.0 - w.abs().sum()), latest_date=last,
                prev_close=prev_row.reindex(w.index), nsei_prev_close=nsei_prev_close)


def run_divcarry(start: str = "2010-01-01", index: str = "nifty500", refresh: bool = False,
                 path: str = DIVCARRY_SNAPSHOT_PATH, write: bool = True) -> dict:
    """Snapshot the dividend-carry decile L/S sleeve to its own ledger. Dollar-neutral
    (net ~0, gross ~1); rows carry SIGNED weights (shorts negative). forward_track-compatible."""
    book = current_divcarry_book(start=start, index=index, refresh=refresh)
    gross, net = float(book.weights.abs().sum()), float(book.weights.sum())
    n_long, n_short = int((book.weights > 0).sum()), int((book.weights < 0).sum())
    book_ret, _, n_ok, n_req, err = live_book_pnl(book)
    nifty_ret, proxy = nifty_intraday(book.nsei_prev_close)

    row = {
        "hypothesis_ref": "RL-2026-07-26-13", "kind": "live_paper_divcarry_snapshot",
        "asof_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "panel_date": str(book.latest_date.date()), "universe": index,
        "gross": round(gross, 4), "net": round(net, 6),
        "n_long": n_long, "n_short": n_short,
        "book_intraday_ret": None if book_ret is None else round(book_ret, 6),
        "nifty_intraday_ret": None if nifty_ret is None else round(nifty_ret, 6),
        "nifty_proxy": proxy, "n_names": int(n_req), "n_quotes_ok": n_ok,
        "groww_ok": err is None and n_ok > 0, "note": err or "ok",
        "weights": {s: round(float(w), 6) for s, w in book.weights.items()},
    }
    if write:
        log_run(row, path=path)

    print(f"[DIV-CARRY live paper] panel {row['panel_date']}  gross={gross:.2f}  net={net:+.4f}  "
          f"long={n_long} short={n_short}")
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


def current_volshock_book(start: str = "2010-01-01", index: str = "nifty500",
                          refresh: bool = False) -> Book:
    """Reconstruct the RL-2026-07-26-15 turnover-shock decile L/S sleeve on the latest
    panel date; SIGNED dollar-neutral weights (top-shock decile +, bottom -). The
    intraday baseline is the raw (split-adjusted) close, matching the raw Groww LTP."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            close, volume = volshock.panels(start=start, index=index, refresh=refresh)
            w_full = volshock.latest_weights(close, volume)
            nsei = close_prices(load_yahoo_ohlcv([BENCH]))[BENCH].reindex(close.index).ffill()

    last = close.index[-1]
    w = w_full[w_full.abs() > 1e-12].sort_values(ascending=False)

    today = datetime.now(IST).date()
    completed = close[close.index.map(lambda d: d.date() < today)]
    prev_row = completed.iloc[-1] if len(completed) else close.iloc[-1]
    nsei_prev = nsei[nsei.index.map(lambda d: d.date() < today)]
    nsei_prev_close = float(nsei_prev.iloc[-1] if len(nsei_prev) else nsei.iloc[-1])

    return Book(weights=w, regime_on=bool(len(w) > 0),
                cash_frac=float(1.0 - w.abs().sum()), latest_date=last,
                prev_close=prev_row.reindex(w.index), nsei_prev_close=nsei_prev_close)


def run_volshock(start: str = "2010-01-01", index: str = "nifty500", refresh: bool = False,
                 path: str = VOLSHOCK_SNAPSHOT_PATH, write: bool = True) -> dict:
    """Snapshot the turnover-shock decile L/S sleeve to its own ledger. Dollar-neutral
    (net ~0, gross ~1); rows carry SIGNED weights (shorts negative). forward_track-compatible."""
    book = current_volshock_book(start=start, index=index, refresh=refresh)
    gross, net = float(book.weights.abs().sum()), float(book.weights.sum())
    n_long, n_short = int((book.weights > 0).sum()), int((book.weights < 0).sum())
    book_ret, _, n_ok, n_req, err = live_book_pnl(book)
    nifty_ret, proxy = nifty_intraday(book.nsei_prev_close)

    row = {
        "hypothesis_ref": "RL-2026-07-26-15", "kind": "live_paper_volshock_snapshot",
        "asof_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "panel_date": str(book.latest_date.date()), "universe": index,
        "gross": round(gross, 4), "net": round(net, 6),
        "n_long": n_long, "n_short": n_short,
        "book_intraday_ret": None if book_ret is None else round(book_ret, 6),
        "nifty_intraday_ret": None if nifty_ret is None else round(nifty_ret, 6),
        "nifty_proxy": proxy, "n_names": int(n_req), "n_quotes_ok": n_ok,
        "groww_ok": err is None and n_ok > 0, "note": err or "ok",
        "weights": {s: round(float(w), 6) for s, w in book.weights.items()},
    }
    if write:
        log_run(row, path=path)

    print(f"[VOL-SHOCK live paper] panel {row['panel_date']}  gross={gross:.2f}  net={net:+.4f}  "
          f"long={n_long} short={n_short}")
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


def current_macrobeta_book(start: str = "2010-01-01", index: str = "nifty500",
                           refresh: bool = False) -> Book:
    """Reconstruct the RL-2026-07-26-08 macro-sensitivity alignment decile L/S sleeve on
    the latest panel date; SIGNED dollar-neutral weights (most-aligned decile +, most-
    misaligned -). The intraday baseline is the raw (split-adjusted) close, matching the
    raw Groww LTP."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            px, inr, oil = macrobeta.panels(start=start, index=index, refresh=refresh)
            w_full = macrobeta.latest_weights(px, inr, oil)
            w = w_full[w_full.abs() > 1e-12].sort_values(ascending=False)
            # raw (split-adjusted, dividend-UNadjusted) close is the live intraday baseline
            # the raw Groww LTP compares to - not the dividend-adjusted panel px the betas
            # are built on. Fetched for the held names only (cache is already warm).
            raw = close_prices(load_yahoo_ohlcv(list(w.index)), field="close").reindex(px.index).ffill()
            nsei = close_prices(load_yahoo_ohlcv([BENCH]))[BENCH].reindex(px.index).ffill()

    last = px.index[-1]
    today = datetime.now(IST).date()
    completed = raw[raw.index.map(lambda d: d.date() < today)]
    prev_row = completed.iloc[-1] if len(completed) else raw.iloc[-1]
    nsei_prev = nsei[nsei.index.map(lambda d: d.date() < today)]
    nsei_prev_close = float(nsei_prev.iloc[-1] if len(nsei_prev) else nsei.iloc[-1])

    return Book(weights=w, regime_on=bool(len(w) > 0),
                cash_frac=float(1.0 - w.abs().sum()), latest_date=last,
                prev_close=prev_row.reindex(w.index), nsei_prev_close=nsei_prev_close)


def run_macrobeta(start: str = "2010-01-01", index: str = "nifty500", refresh: bool = False,
                  path: str = MACROBETA_SNAPSHOT_PATH, write: bool = True) -> dict:
    """Snapshot the macro-alignment decile L/S sleeve to its own ledger. Dollar-neutral
    (net ~0, gross ~1); rows carry SIGNED weights (shorts negative). forward_track-compatible."""
    book = current_macrobeta_book(start=start, index=index, refresh=refresh)
    gross, net = float(book.weights.abs().sum()), float(book.weights.sum())
    n_long, n_short = int((book.weights > 0).sum()), int((book.weights < 0).sum())
    book_ret, _, n_ok, n_req, err = live_book_pnl(book)
    nifty_ret, proxy = nifty_intraday(book.nsei_prev_close)

    row = {
        "hypothesis_ref": "RL-2026-07-26-08", "kind": "live_paper_macrobeta_snapshot",
        "asof_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "panel_date": str(book.latest_date.date()), "universe": index,
        "gross": round(gross, 4), "net": round(net, 6),
        "n_long": n_long, "n_short": n_short,
        "book_intraday_ret": None if book_ret is None else round(book_ret, 6),
        "nifty_intraday_ret": None if nifty_ret is None else round(nifty_ret, 6),
        "nifty_proxy": proxy, "n_names": int(n_req), "n_quotes_ok": n_ok,
        "groww_ok": err is None and n_ok > 0, "note": err or "ok",
        "weights": {s: round(float(w), 6) for s, w in book.weights.items()},
    }
    if write:
        log_run(row, path=path)

    print(f"[MACRO-BETA live paper] panel {row['panel_date']}  gross={gross:.2f}  net={net:+.4f}  "
          f"long={n_long} short={n_short}")
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


def current_illiq_book(start: str = "2010-01-01", index: str = "nifty500",
                       refresh: bool = False) -> Book:
    """Reconstruct the RL-2026-07-26-19 Amihud-illiquidity decile L/S sleeve (frozen L63)
    on the latest panel date; SIGNED dollar-neutral weights (most-illiquid decile +, most-
    liquid -). Signal computed on the FULL panel (warm-up, not a hold-out read). The intraday
    baseline is the raw (split-adjusted) close, matching the raw Groww LTP."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            px, close, volume = illiq.panels(start=start, index=index, refresh=refresh)
            w_full = illiq.latest_weights(px, close, volume)
            nsei = close_prices(load_yahoo_ohlcv([BENCH]))[BENCH].reindex(close.index).ffill()

    last = close.index[-1]
    w = w_full[w_full.abs() > 1e-12].sort_values(ascending=False)

    today = datetime.now(IST).date()
    completed = close[close.index.map(lambda d: d.date() < today)]
    prev_row = completed.iloc[-1] if len(completed) else close.iloc[-1]
    nsei_prev = nsei[nsei.index.map(lambda d: d.date() < today)]
    nsei_prev_close = float(nsei_prev.iloc[-1] if len(nsei_prev) else nsei.iloc[-1])

    return Book(weights=w, regime_on=bool(len(w) > 0),
                cash_frac=float(1.0 - w.abs().sum()), latest_date=last,
                prev_close=prev_row.reindex(w.index), nsei_prev_close=nsei_prev_close)


def run_illiq(start: str = "2010-01-01", index: str = "nifty500", refresh: bool = False,
              path: str = ILLIQ_SNAPSHOT_PATH, write: bool = True) -> dict:
    """Snapshot the Amihud-illiquidity decile L/S sleeve to its own ledger. Dollar-neutral
    (net ~0, gross ~1); rows carry SIGNED weights (shorts negative). forward_track-compatible."""
    book = current_illiq_book(start=start, index=index, refresh=refresh)
    gross, net = float(book.weights.abs().sum()), float(book.weights.sum())
    n_long, n_short = int((book.weights > 0).sum()), int((book.weights < 0).sum())
    book_ret, _, n_ok, n_req, err = live_book_pnl(book)
    nifty_ret, proxy = nifty_intraday(book.nsei_prev_close)

    row = {
        "hypothesis_ref": "RL-2026-07-26-19", "kind": "live_paper_illiq_snapshot",
        "asof_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "panel_date": str(book.latest_date.date()), "universe": index,
        "lookback": illiq.FROZEN_LOOKBACK,
        "gross": round(gross, 4), "net": round(net, 6),
        "n_long": n_long, "n_short": n_short,
        "book_intraday_ret": None if book_ret is None else round(book_ret, 6),
        "nifty_intraday_ret": None if nifty_ret is None else round(nifty_ret, 6),
        "nifty_proxy": proxy, "n_names": int(n_req), "n_quotes_ok": n_ok,
        "groww_ok": err is None and n_ok > 0, "note": err or "ok",
        "weights": {s: round(float(w), 6) for s, w in book.weights.items()},
    }
    if write:
        log_run(row, path=path)

    print(f"[ILLIQ live paper] panel {row['panel_date']}  L{illiq.FROZEN_LOOKBACK}  "
          f"gross={gross:.2f}  net={net:+.4f}  long={n_long} short={n_short}")
    print("TOP 8 longs (most illiquid) / TOP 8 shorts (most liquid):")
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


def _last_row(path: str) -> dict | None:
    """The last JSON row of a ledger (the pairs harness's state), or None if absent/empty."""
    p = Path(path)
    if not p.exists():
        return None
    last = None
    for line in p.read_text().splitlines():
        if line.strip():
            last = line
    return json.loads(last) if last else None


def current_pairs_book(prev_open: list[dict], refresh: bool = False
                       ) -> tuple[Book, list[dict], dict[tuple[str, str], float]]:
    """Reconstruct the RL-2026-07-26-09 cointegration-pairs book on the latest panel date.

    Path-dependent: today's open pairs come from `prev_open` (the last ledger row's state)
    plus today's rolling-63d z per FROZEN pair. Returns (book, open_pairs, z_now); the book
    carries SIGNED equal-dollar weights (each open pair dollar-neutral, gross 0.1)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            log_px, close_raw, mkt, _ = pairs_rv.build_panel(refresh=refresh)

    dates = log_px.index
    today = dates[-1]
    z_now = pairs_rv.current_z(log_px)
    open_pairs = pairs_rv.update_open_pairs(prev_open, z_now, dates, today)
    w = pairs_rv.target_weights(open_pairs).sort_values(ascending=False)

    today_date = datetime.now(IST).date()
    completed = close_raw[close_raw.index.map(lambda d: d.date() < today_date)]
    prev_row = completed.iloc[-1] if len(completed) else close_raw.iloc[-1]
    nsei_prev = mkt[mkt.index.map(lambda d: d.date() < today_date)]
    nsei_prev_close = float(nsei_prev.iloc[-1] if len(nsei_prev) else mkt.iloc[-1])

    book = Book(weights=w, regime_on=bool(len(w) > 0), cash_frac=float(1.0 - w.abs().sum()),
                latest_date=today, prev_close=prev_row.reindex(w.index),
                nsei_prev_close=nsei_prev_close)
    return book, open_pairs, z_now


def run_pairs(refresh: bool = False, path: str = PAIRS_SNAPSHOT_PATH, write: bool = True) -> dict:
    """Snapshot the cointegration-pairs relative-value book to its own ledger. Signed
    equal-dollar legs, per-pair dollar-neutral (book net ~0, gross 0.1*n_open); the ledger
    IS the state (last row's open_pairs drives today's transitions). forward_track-compatible."""
    prev = _last_row(path)
    prev_open = list((prev or {}).get("open_pairs", []))
    book, open_pairs, z_now = current_pairs_book(prev_open, refresh=refresh)
    gross, net = float(book.weights.abs().sum()), float(book.weights.sum())
    book_ret, _, n_ok, n_req, err = live_book_pnl(book)
    nifty_ret, proxy = nifty_intraday(book.nsei_prev_close)

    row = {
        "hypothesis_ref": "RL-2026-07-26-09", "kind": "live_paper_pairs_snapshot",
        "asof_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "panel_date": str(book.latest_date.date()), "universe": "F&O-cap-N500",
        "open_pairs": open_pairs, "n_open": len(open_pairs),
        "gross": round(gross, 4), "net": round(net, 6),
        "book_intraday_ret": None if book_ret is None else round(book_ret, 6),
        "nifty_intraday_ret": None if nifty_ret is None else round(nifty_ret, 6),
        "nifty_proxy": proxy, "n_names": int(n_req), "n_quotes_ok": n_ok,
        "quotes_ok": err is None and n_ok > 0, "groww_ok": err is None and n_ok > 0,
        "note": err or "ok",
        "weights": {s: round(float(w), 6) for s, w in book.weights.items()},
    }
    if write:
        log_run(row, path=path)

    print(f"[PAIRS-RV live paper] panel {row['panel_date']}  open={len(open_pairs)}/{pairs_rv.MAX_PAIRS}  "
          f"gross={gross:.2f}  net={net:+.4f}")
    for p in open_pairs:
        z = z_now.get((p["a"], p["b"]))
        zs = "n/a" if z is None else f"{z:+.2f}"
        print(f"  {p['direction']:8s} {p['a']:14s}/{p['b']:14s}  z_entry={p['z_entry']:+.2f}  "
              f"z_now={zs}  since {p['entry_date']}")
    if not open_pairs:
        print("  no open pairs (all spreads inside the +/-2 band) - book is all cash")
    elif book_ret is None:
        print(f"live book P&L: UNAVAILABLE (quotes ok {n_ok}/{n_req}; {err})")
    else:
        print(f"live book intraday {book_ret*100:+.2f}% (per-pair dollar-neutral); "
              f"quotes ok {n_ok}/{n_req}")
    print(f"snapshot {'appended to '+path if write else 'NOT written (dry run)'}")
    return row


def current_gl_book(refresh: bool = False) -> Book:
    """Reconstruct the RL-2026-07-16 gold_lowbeta risk-off VARIANT of the deployed book:
    the RL-16 base (top-decile conviction momentum scaled by the 200MA-or-VIX overlay)
    plus the 50/50 trend-gated-GOLDBEES / low-beta sleeve filling the freed weight on
    risk-off days. Reuses the study's frozen `base_book` and `combined_book` code path;
    on a risk-off day (base fully in cash) the combined book is the pure sleeve."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            px, mkt, ohlcv, _ = india_panel(start="2010-01-01", index="nifty500",
                                             ret_clip=0.40, refresh=refresh)
            gold_data = load_yahoo_ohlcv([GOLD], refresh=refresh)
            gold_adj = close_prices(gold_data)[GOLD].reindex(px.index).ffill()
            vix = close_prices(load_yahoo_ohlcv([VIX_SYM], refresh=refresh))[VIX_SYM].reindex(px.index).ffill()
            base = base_book(px, mkt, vix, sector_map("nifty500"))
            pxa = px.copy()
            pxa[GOLD] = gold_adj
            book_w = combined_book("gold_lowbeta", px, pxa, mkt, gold_adj, base)
            on = bool(regime_on(mkt, vix, 200, 252, 0.80).reindex(px.index).fillna(False).iloc[-1])

    last = book_w.index[-1]
    w = book_w.loc[last]
    w = w[w > 1e-9].sort_values(ascending=False)

    today = datetime.now(IST).date()
    def _last_completed(obj):
        c = obj[obj.index.map(lambda d: d.date() < today)]
        return c.iloc[-1] if len(c) else obj.iloc[-1]
    prev = _last_completed(ohlcv["close"]).copy()                 # raw close per panel name
    prev[GOLD] = float(_last_completed(close_prices(gold_data, field="close")[GOLD]))
    nsei_prev = mkt[mkt.index.map(lambda d: d.date() < today)]
    nsei_prev_close = float(nsei_prev.iloc[-1] if len(nsei_prev) else mkt.iloc[-1])

    return Book(weights=w, regime_on=on, cash_frac=float(1.0 - w.sum()), latest_date=last,
                prev_close=prev.reindex(w.index), nsei_prev_close=nsei_prev_close)


def run_gl(refresh: bool = False, path: str = GL_SNAPSHOT_PATH, write: bool = True) -> dict:
    """Snapshot the gold_lowbeta risk-off variant to its own ledger. Long-only combined
    book (gross ~1 on risk-off days, the GOLDBEES leg included); forward_track-compatible."""
    book = current_gl_book(refresh=refresh)
    state = "risk_on" if book.regime_on else "risk_off"
    book_ret, _, n_ok, n_req, err = live_book_pnl(book)
    nifty_ret, proxy = nifty_intraday(book.nsei_prev_close)
    gross = float(book.weights.sum())

    row = {
        "hypothesis_ref": "RL-2026-07-16", "kind": "live_paper_gl_snapshot",
        "asof_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "panel_date": str(book.latest_date.date()), "universe": "NIFTY500+GOLDBEES",
        "regime_state": state, "gross": round(gross, 4), "cash_frac": round(book.cash_frac, 4),
        "gold_weight": round(float(book.weights.get(GOLD, 0.0)), 6),
        "book_intraday_ret": None if book_ret is None else round(book_ret, 6),
        "nifty_intraday_ret": None if nifty_ret is None else round(nifty_ret, 6),
        "nifty_proxy": proxy, "n_names": int(len(book.weights)), "n_quotes_ok": n_ok,
        "groww_ok": err is None and n_ok > 0, "note": err or "ok",
        "weights": {s: round(float(w), 6) for s, w in book.weights.items()},
    }
    if write:
        log_run(row, path=path)

    print(f"[gold_lowbeta live paper] panel {row['panel_date']}  regime={state}  "
          f"gross={gross:.2f}  names={len(book.weights)}  gold_leg={row['gold_weight']*100:.1f}%")
    print("TOP 12 target holdings:")
    for s, wt in book.weights.head(12).items():
        print(f"  {s:16s} {wt*100:6.2f}%")
    if book_ret is None:
        print(f"live book P&L: UNAVAILABLE (quotes ok {n_ok}/{n_req}; {err})")
    else:
        nb = "n/a" if nifty_ret is None else f"{nifty_ret*100:+.2f}%"
        print(f"live book intraday {book_ret*100:+.2f}% vs Nifty {nb} "
              f"({proxy}); quotes ok {n_ok}/{n_req}")
    print(f"snapshot {'appended to '+path if write else 'NOT written (dry run)'}")
    return row


def _read_books(path: str) -> dict[pd.Timestamp, dict[str, float]]:
    """Snapshot books keyed by panel_date, keeping the LAST row per date.

    Legacy rows without a `weights` KEY predate the forward track and are skipped;
    an explicit empty book ({}) is a real all-cash day and rebalances to cash.
    A missing ledger reads as zero books (the forward report then reports that cleanly).
    """
    p = Path(path)
    if not p.exists():
        return {}
    books: dict[pd.Timestamp, dict[str, float]] = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "weights" in rec:
            books[pd.Timestamp(rec["panel_date"])] = rec["weights"]  # last write wins
    return books


def forward_track(path: str = SNAPSHOT_PATH, cost_bps: float = 20.0,
                  refresh: bool = False) -> pd.DataFrame | None:
    """Rigorous forward return of the recorded REGIME books vs the Nifty.

    Each snapshot's held weights are applied at that panel date's adjusted close and
    earn the realized close-to-close return until the next snapshot (backtest_weights
    semantics). Pure report: prints a per-day and cumulative book-vs-benchmark table
    and never touches Groww.
    """
    books = _read_books(path)
    if len(books) < 2:
        print(f"[forward] need >= 2 snapshot days to compute a forward return; "
              f"have {len(books)} (run a daily snapshot first)")
        return None

    dates = sorted(books)
    symbols = sorted({s for w in books.values() for s in w})
    W = pd.DataFrame(0.0, index=pd.DatetimeIndex(dates), columns=symbols)
    for d in dates:
        for s, wt in books[d].items():
            W.at[d, s] = float(wt)

    prices = close_prices(load_yahoo_ohlcv(symbols + [BENCH], refresh=refresh)).loc[dates[0]:]
    priced = [s for s in symbols if s in prices.columns and prices[s].notna().any()]
    missing = [s for s in symbols if s not in priced]

    held = W.abs().sum(axis=1)          # GROSS held weight: correct for L/S (net ~0) and
    priced_w = W[priced].abs().sum(axis=1)  # identical to net for a long-only book
    cov = (priced_w / held).where(held > 0, 1.0)   # all-cash day: nothing to price
    print(f"[forward] priced share of gross weight: min {cov.min():.1%}, "
          f"mean {cov.mean():.1%} across {len(dates)} snapshot day(s)")
    if missing:
        print(f"[forward] WARNING: dropped {len(missing)} unpriced symbol(s), weights "
              f"NOT renormalized: {', '.join(missing)}")
        for d in dates:
            print(f"  {d.date()}: priced {priced_w[d]:.1%} of {held[d]:.1%} held")

    res = backtest_weights(prices[priced], W, cost_bps=cost_bps)
    book = res.returns
    nsei = prices[BENCH].pct_change().reindex(book.index).fillna(0.0)
    daily = pd.DataFrame({"book": book, "nsei": nsei, "active": book - nsei})

    print(f"{'date':<12}{'book':>10}{'nifty':>10}{'active':>10}")
    for d, r in daily.iterrows():
        print(f"{str(d.date()):<12}{r.book*100:>9.2f}%{r.nsei*100:>9.2f}%{r.active*100:>9.2f}%")
    cum_b = float((1.0 + book).prod() - 1.0)
    cum_n = float((1.0 + nsei).prod() - 1.0)
    drag = float((res.turnover * cost_bps / 10_000).sum())
    print(f"cumulative  book {cum_b*100:+.2f}%  nifty {cum_n*100:+.2f}%  "
          f"active {(cum_b - cum_n)*100:+.2f}%")
    print(f"turnover cost drag {drag*100:.2f}% over {len(daily)} tracked day(s)")
    return daily


def main() -> None:
    p = argparse.ArgumentParser(description="Live paper snapshot for a deployable/forward book")
    p.add_argument("--sleeve",
                   choices=("regime", "ls", "trend", "gl", "dualrot", "divcarry", "volshock",
                            "macrobeta", "illiq"),
                   default="regime",
                   help="regime = RL-07-10 long-only book; ls = RL-07-12 F&O-shortable L/S; "
                        "trend = RL-07-17 5-ETF trend sleeve; gl = RL-07-16 gold_lowbeta variant; "
                        "dualrot = RL-07-26-01 5-ETF dual-momentum rotation; "
                        "divcarry = RL-07-26-13 dividend-yield decile L/S; "
                        "volshock = RL-07-26-15 turnover-shock decile L/S; "
                        "macrobeta = RL-07-26-08 macro-alignment decile L/S; "
                        "illiq = RL-07-26-19 Amihud-illiquidity decile L/S")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--index", default="nifty500")
    p.add_argument("--refresh", action="store_true",
                   help="refresh Yahoo history to the latest close before building the book")
    p.add_argument("--path", default=None, help="ledger path (defaults per sleeve)")
    p.add_argument("--dry-run", action="store_true", help="print but do not append a row")
    p.add_argument("--forward", action="store_true",
                   help="print the forward-return report from recorded books, no snapshot")
    p.add_argument("--cost-bps", type=float, default=20.0,
                   help="turnover cost for the forward report (default 20)")
    a = p.parse_args()
    paths = {"regime": SNAPSHOT_PATH, "ls": LS_SNAPSHOT_PATH,
             "trend": TREND_SNAPSHOT_PATH, "gl": GL_SNAPSHOT_PATH,
             "dualrot": DUALROT_SNAPSHOT_PATH, "divcarry": DIVCARRY_SNAPSHOT_PATH,
             "volshock": VOLSHOCK_SNAPSHOT_PATH, "macrobeta": MACROBETA_SNAPSHOT_PATH,
             "illiq": ILLIQ_SNAPSHOT_PATH}
    path = a.path or paths[a.sleeve]
    if a.forward:
        forward_track(path=path, cost_bps=a.cost_bps, refresh=a.refresh)
        return
    write = not a.dry_run
    if a.sleeve == "trend":
        run_trend(refresh=a.refresh, path=path, write=write)
    elif a.sleeve == "dualrot":
        run_dualrot(refresh=a.refresh, path=path, write=write)
    elif a.sleeve == "gl":
        run_gl(refresh=a.refresh, path=path, write=write)
    elif a.sleeve == "divcarry":
        run_divcarry(start=a.start, index=a.index, refresh=a.refresh, path=path, write=write)
    elif a.sleeve == "volshock":
        run_volshock(start=a.start, index=a.index, refresh=a.refresh, path=path, write=write)
    elif a.sleeve == "macrobeta":
        run_macrobeta(start=a.start, index=a.index, refresh=a.refresh, path=path, write=write)
    elif a.sleeve == "illiq":
        run_illiq(start=a.start, index=a.index, refresh=a.refresh, path=path, write=write)
    else:
        runner = run_ls if a.sleeve == "ls" else run
        runner(start=a.start, index=a.index, refresh=a.refresh, path=path, write=write)


if __name__ == "__main__":
    main()
