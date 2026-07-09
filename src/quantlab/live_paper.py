"""RL-2026-07-10 live paper-observation harness for the REGIME strategy.

READ-ONLY. Builds the current REGIME target book from (optionally refreshed) Yahoo
history, fetches LIVE Groww LTP for the held names, computes the book's live
intraday P&L against a Nifty proxy, and APPENDS one snapshot row to
`experiments/paper_trades.jsonl` so daily re-runs accumulate a forward track record.

Nothing here mutates account state: only read-only market-data methods are called,
through the rate-limited `quantlab.groww_client.call` wrapper that itself refuses
order methods. When Groww live data is unavailable - the API key lacks live-data
entitlement, or the cash market is shut - the run degrades gracefully: it still
records the target book and regime state, with the live P&L fields left null.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import groww_client as gc
from .blend import market_on
from .india import india_panel, sector_map
from .india_blend_study import raw_signals
from .india_run import _core_regime_band_ls
from .tracking import log_run

SEGMENT = "CASH"                       # NSE cash equity
READ_METHODS = ("get_ltp",)            # the only Groww methods this harness ever calls
NIFTY_PROXIES = ("NSE_NIFTY", "NSE_NIFTYBEES")  # index first, then the tradeable ETF
IST = timezone(timedelta(hours=5, minutes=30))
SNAPSHOT_PATH = "experiments/paper_trades.jsonl"


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


def run(start: str = "2010-01-01", index: str = "nifty500", refresh: bool = False,
        path: str = SNAPSHOT_PATH, write: bool = True) -> dict:
    book = current_book(start=start, index=index, refresh=refresh)
    state = "risk_on" if book.regime_on else "risk_off"
    groww_syms = [to_groww(s) for s in book.weights.index]

    prices, err = fetch_ltp(groww_syms)
    live_ret = {s: prices[g] / book.prev_close[s] - 1.0
                for s, g in zip(book.weights.index, groww_syms)
                if g in prices and book.prev_close.get(s, 0)}
    n_ok = len(live_ret)
    book_ret = (float(sum(book.weights[s] * r for s, r in live_ret.items()))
                if n_ok else None)
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
    }
    if write:
        log_run(row, path=path)

    print(f"[REGIME live paper] panel {row['panel_date']}  regime={state}  "
          f"cash={book.cash_frac:.0%}  names={len(book.weights)}")
    print("TOP 15 target holdings:")
    for s, wt in book.weights.head(15).items():
        print(f"  {s:16s} {wt*100:6.2f}%")
    if book_ret is None:
        print(f"live book P&L: UNAVAILABLE (quotes ok {n_ok}/{len(groww_syms)}; {err})")
    else:
        nb = "n/a" if nifty_ret is None else f"{nifty_ret*100:+.2f}%"
        print(f"live book intraday {book_ret*100:+.2f}% vs Nifty {nb} "
              f"({proxy}); quotes ok {n_ok}/{len(groww_syms)}")
    print(f"snapshot {'appended to '+path if write else 'NOT written (dry run)'}")
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="RL-2026-07-10 REGIME live paper snapshot")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--index", default="nifty500")
    p.add_argument("--refresh", action="store_true",
                   help="refresh Yahoo history to the latest close before building the book")
    p.add_argument("--path", default=SNAPSHOT_PATH)
    p.add_argument("--dry-run", action="store_true", help="print but do not append a row")
    a = p.parse_args()
    run(start=a.start, index=a.index, refresh=a.refresh, path=a.path, write=not a.dry_run)


if __name__ == "__main__":
    main()
