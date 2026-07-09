"""Live paper-observation harness for the deployable Indian books.

READ-ONLY. Two sleeves, separate ledgers: the RL-2026-07-10 long-only REGIME book
(`run` -> `paper_trades.jsonl`) and the RL-2026-07-12 F&O-shortable residual-momentum
long-short sleeve (`run_ls` -> `paper_trades_ls.jsonl`, signed weights). Each builds
the current target book from (optionally refreshed) Yahoo history, fetches LIVE Groww
LTP for the held names, computes the book's live intraday P&L, and APPENDS one
snapshot row so daily re-runs accumulate a forward track record.

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

from . import groww_client as gc
from .backtest import backtest_weights
from .blend import composite, fno_long_short, market_on
from .data import close_prices, load_yahoo_ohlcv
from .india import fno_shortable, india_panel, sector_map
from .india_blend_study import raw_signals
from .india_ls import CORE as LS_CORE, WEIGHTS as LS_WEIGHTS
from .india_run import _core_regime_band_ls
from .tracking import log_run

SEGMENT = "CASH"                       # NSE cash equity
# read-only Groww methods this harness may route through gc.call (order methods are
# refused by gc.call itself; the method-spy test proves only these are ever reached).
READ_METHODS = ("get_ltp", "get_all_instruments")
NIFTY_PROXIES = ("NSE_NIFTY", "NSE_NIFTYBEES")  # index first, then the tradeable ETF
BENCH = "^NSEI"                        # Nifty 50 price index, the forward-track benchmark
IST = timezone(timedelta(hours=5, minutes=30))
SNAPSHOT_PATH = "experiments/paper_trades.jsonl"
LS_SNAPSHOT_PATH = "experiments/paper_trades_ls.jsonl"


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


def live_book_pnl(book: Book) -> tuple[float | None, int, int, str | None]:
    """Fetch live LTP for the held names and compute the book's signed intraday
    return vs each name's prev close. Works for long-only and signed L/S books
    alike (shorts have negative weight -> negative contribution on an up move).
    Returns (book_ret_or_None, n_quotes_ok, n_names_requested, error_or_None)."""
    groww_syms = [to_groww(s) for s in book.weights.index]
    prices, err = fetch_ltp(groww_syms)
    live_ret = {s: prices[g] / book.prev_close[s] - 1.0
                for s, g in zip(book.weights.index, groww_syms)
                if g in prices and book.prev_close.get(s, 0)}
    n_ok = len(live_ret)
    book_ret = float(sum(book.weights[s] * r for s, r in live_ret.items())) if n_ok else None
    return book_ret, n_ok, len(groww_syms), err


def run(start: str = "2010-01-01", index: str = "nifty500", refresh: bool = False,
        path: str = SNAPSHOT_PATH, write: bool = True) -> dict:
    book = current_book(start=start, index=index, refresh=refresh)
    state = "risk_on" if book.regime_on else "risk_off"
    book_ret, n_ok, n_req, err = live_book_pnl(book)
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
    book_ret, n_ok, n_req, err = live_book_pnl(book)
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
    p = argparse.ArgumentParser(description="Live paper snapshot: REGIME long book or F&O L/S sleeve")
    p.add_argument("--sleeve", choices=("regime", "ls"), default="regime",
                   help="regime = RL-07-10 long-only book; ls = RL-07-12 F&O-shortable L/S")
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
    path = a.path or (LS_SNAPSHOT_PATH if a.sleeve == "ls" else SNAPSHOT_PATH)
    if a.forward:
        forward_track(path=path, cost_bps=a.cost_bps, refresh=a.refresh)
        return
    runner = run_ls if a.sleeve == "ls" else run
    runner(start=a.start, index=a.index, refresh=a.refresh, path=path, write=not a.dry_run)


if __name__ == "__main__":
    main()
