"""Indian single-stock universe (Nifty 200) for RL-2026-07-10.

Fetches the current Nifty 200 constituent list from NSE, downloads per-stock
total-return daily data from Yahoo (.NS suffix -> adj_close includes dividends,
unlike the ^ price indices used in RL-2026-07-09), and builds a clean price panel
under a documented inclusion rule.

SURVIVORSHIP: the constituent list is *current* membership. No free point-in-time
source exists, so results carry survivorship bias (today's members are past
survivors) and the history filter adds a mild older-listing tilt. Both are
disclosed; cross-sectional/relative signals are less sensitive to it than
long-only absolute bets, and a broad-set robustness check reports the direction.
"""

from pathlib import Path
import csv
import io
from urllib.request import Request, urlopen

import pandas as pd

from .data import close_prices, load_yahoo_ohlcv

INDEX_URL = "https://nsearchives.nseindia.com/content/indices/ind_{index}list.csv"
BENCHMARK = "^NSEI"           # Nifty 50 price index (benchmark, not traded)
TRADEABLE_BENCH = "NIFTYBEES.NS"  # Nifty ETF with dividends (deployable benchmark)


def nse_index_symbols(index: str = "nifty200", cache_dir: str | Path = "data/raw",
                      refresh: bool = False) -> list[str]:
    """Current constituents of an NSE index as Yahoo tickers (SYMBOL.NS), cached.

    `index` is the NSE slug: nifty50, nifty200, nifty500, niftytotalmarket, ...
    """
    path = Path(cache_dir) / f"ind_{index}list.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not path.exists():
        req = Request(INDEX_URL.format(index=index),
                      headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"})
        with urlopen(req, timeout=30) as res:
            path.write_bytes(res.read())
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8", errors="replace"))))
    return [f"{r['Symbol'].strip()}.NS" for r in rows if r.get("Symbol")]


def nifty200_symbols(**kw) -> list[str]:
    return nse_index_symbols("nifty200", **kw)


def _winsorize_prices(px: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Rebuild the price panel from daily returns clipped to +/-cap.

    NSE stocks mostly have +/-20% daily circuit limits, so a daily move beyond ~50%
    is almost always a bad print / unadjusted action, not a real return. Clipping
    bounds any single glitch's impact; disclosed as a pre-registered cleaning step."""
    r = px.pct_change().clip(-cap, cap).fillna(0.0)
    return px.iloc[0] * (1.0 + r).cumprod()


def india_panel(
    start: str,
    end: str | None = None,
    index: str = "nifty200",
    symbols: list[str] | None = None,
    min_coverage: float = 0.95,
    ffill_limit: int = 3,
    ret_clip: float | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict[str, pd.DataFrame], list[str]]:
    """Build a total-return price panel for an NSE universe.

    Inclusion rule (pre-registered, applied before any factor is computed):
      a name is kept iff it was listed on/before `start` (first price in its FULL
      history <= start) AND covers >= `min_coverage` of the Nifty (^NSEI) trading
      days in [start, end]. Small gaps are forward-filled up to `ffill_limit` days.
      Listing/coverage are judged on full history and the index calendar - NOT the
      windowed panel, whose first in-window date is always > start.

    Returns (px, market, ohlcv_fields, kept_symbols).
    """
    syms = symbols if symbols is not None else nse_index_symbols(index, refresh=refresh)
    all_syms = list(dict.fromkeys(syms + [BENCHMARK, TRADEABLE_BENCH]))
    data = load_yahoo_ohlcv(all_syms, refresh=refresh)

    full = close_prices(data)                          # full history, all names
    cal = data[BENCHMARK.upper()].loc[start:end].index  # Nifty trading calendar
    start_ts = pd.Timestamp(start)

    kept = []
    for s in syms:
        col = s.upper()
        if col not in full.columns:
            continue
        first = full[col].first_valid_index()
        if first is None or first > start_ts:          # listed after window start
            continue
        if full[col].reindex(cal).notna().mean() < min_coverage:
            continue
        kept.append(col)

    px = full[kept].reindex(cal).ffill(limit=ffill_limit).dropna()
    if ret_clip is not None:
        px = _winsorize_prices(px, ret_clip)
    mkt = full[BENCHMARK.upper()].reindex(px.index).ffill(limit=ffill_limit)
    ohlcv = {c: pd.DataFrame({s: data[s][c] for s in kept}).reindex(px.index).ffill(limit=ffill_limit)
             for c in ("open", "close", "high", "low", "volume")}
    return px, mkt, ohlcv, kept
