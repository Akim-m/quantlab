"""Intraday 5-minute bar archive (RL-2026-07-25).

ARCHIVE-BEFORE-EXPIRY. Groww retains only ~90 trailing days of intraday candles
(measured 2026-07-09) and no free source has Indian intraday history, so this
collector fetches 5-minute OHLCV bars since each symbol's last archived session
and appends them to `data/raw/intraday/<symbol>.csv`. History accrues at 1:1 real
time; a gap up to ~90 days self-heals on the next run. No strategy claim is made
or implied - this locks the data program only (ORB / VWAP studies get their own
pre-registration after >=12 months of bars).

READ-ONLY: candles come through `gc.call('get_historical_candle_data', ...)`, which
rate-limits to <=7 req/s and refuses order methods. A symbol that fails (rename /
suspension / transient) is recorded and skipped; the run never crashes on one bad
symbol. One coverage row per run -> `experiments/intraday_archive.jsonl` (committed)
so gaps are auditable in git even though the bars are git-ignored bulk.

Candle wire format (probed 2026-07-09): dict with `candles` = list of
[epoch_seconds, open, high, low, close, volume]; the epoch converts to IST
wall-clock (09:15 = market open). Index bars carry volume=None. The NIFTY index
serves on trading_symbol 'NIFTY', segment CASH - the NSE_ prefix is rejected.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import groww_client as gc
from .tracking import log_run

IST = timezone(timedelta(hours=5, minutes=30))
STORE_DIR = "data/raw/intraday"
AUDIT_PATH = "experiments/intraday_archive.jsonl"
EXCHANGE, SEGMENT, INTERVAL = "NSE", "CASH", 5
WINDOW_DAYS = 14          # 5-min candles cap at 15d/request; page in <=14d windows
LOOKBACK_DAYS = 92        # first-run reach (Groww retains ~90 trailing days)
DT_FMT = "%Y-%m-%d %H:%M:%S"
COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
READ_METHODS = ("get_historical_candle_data",)

# Frozen universe (RL-2026-07-25): NIFTY index + the nifty100 constituents resolved
# 2026-07-09, written literally so the archived set is reproducible regardless of
# later index reconstitution. Membership drift is a disclosed limitation.
NIFTY = "NIFTY"
NIFTY100 = (
    "ABB", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER",
    "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT", "DMART", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BAJAJHLDNG", "BANKBARODA", "BEL", "BPCL",
    "BHARTIARTL", "BOSCHLTD", "BRITANNIA", "CGPOWER", "CANBK", "CHOLAFIN", "CIPLA",
    "COALINDIA", "CUMMINSIND", "DLF", "DIVISLAB", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GAIL", "GODREJCP", "GRASIM", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE",
    "HINDALCO", "HAL", "HINDUNILVR", "HINDZINC", "HYUNDAI", "ICICIBANK", "ITC",
    "INDHOTEL", "IOC", "IRFC", "INFY", "INDIGO", "JSWSTEEL", "JINDALSTEL", "JIOFIN",
    "KOTAKBANK", "LTM", "LT", "LODHA", "M&M", "MARUTI", "MAXHEALTH", "MAZDOCK",
    "MUTHOOTFIN", "NTPC", "NESTLEIND", "ONGC", "PIDILITIND", "PFC", "POWERGRID",
    "PNB", "RECLTD", "RELIANCE", "SBILIFE", "MOTHERSON", "SHREECEM", "SHRIRAMFIN",
    "ENRIN", "SIEMENS", "SOLARINDS", "SBIN", "SUNPHARMA", "TVSMOTOR", "TATACAP",
    "TCS", "TATACONSUM", "TMCV", "TMPV", "TATAPOWER", "TATASTEEL", "TECHM", "TITAN",
    "TORNTPHARM", "TRENT", "ULTRACEMCO", "UNIONBANK", "UNITDSPR", "VBL", "VEDL",
    "WIPRO", "ZYDUSLIFE",
)
UNIVERSE = (NIFTY,) + NIFTY100


def paged_windows(start: datetime, end: datetime,
                  max_days: int = WINDOW_DAYS) -> list[tuple[datetime, datetime]]:
    """Split [start, end] into consecutive sub-windows each spanning <= max_days,
    keeping every 5-minute request under Groww's 15-day cap. Empty when start>=end
    (symbol already current). Consecutive windows share a boundary instant - the
    duplicated boundary bar is removed by the timestamp dedupe on merge."""
    out: list[tuple[datetime, datetime]] = []
    step = timedelta(days=max_days)
    s = start
    while s < end:
        e = min(s + step, end)
        out.append((s, e))
        s = e
    return out


def last_archived(path: str | Path) -> datetime | None:
    """Last archived bar timestamp (IST-aware) for a per-symbol CSV, or None if the
    file is absent/empty. Bars are stored sorted, so the final row is the max."""
    p = Path(path)
    if not p.exists():
        return None
    last = None
    with p.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            last = row.get("timestamp")
    if not last:
        return None
    return datetime.strptime(last, DT_FMT).replace(tzinfo=IST)


def _to_rows(candles) -> list[dict]:
    """Convert Groww candle lists [epoch, o, h, l, c, v] to storage rows keyed by
    IST timestamp string. Malformed candles are skipped; index volume (None) -> ''."""
    rows = []
    for c in candles or []:
        if not isinstance(c, (list, tuple)) or len(c) < 6:
            continue
        try:
            ts = datetime.fromtimestamp(float(c[0]), tz=IST).strftime(DT_FMT)
        except (TypeError, ValueError, OSError):
            continue
        rows.append({"timestamp": ts, "open": c[1], "high": c[2], "low": c[3],
                     "close": c[4], "volume": "" if c[5] is None else c[5]})
    return rows


def fetch_symbol(symbol: str, start: datetime,
                 end: datetime) -> tuple[list[dict], str | None]:
    """Fetch 5-min bars for [start, end] paged in <=14d windows. Returns
    (rows, first_error_or_None); never raises. A failing window is skipped, not
    fatal, so a transient blip never wipes the rest of the symbol's coverage."""
    rows: list[dict] = []
    err: str | None = None
    for s, e in paged_windows(start, end):
        try:
            payload = gc.call("get_historical_candle_data", trading_symbol=symbol,
                              exchange=EXCHANGE, segment=SEGMENT,
                              start_time=s.strftime(DT_FMT), end_time=e.strftime(DT_FMT),
                              interval_in_minutes=INTERVAL)
        except Exception as ex:
            err = err or f"{type(ex).__name__}: {ex}"
            continue
        rows.extend(_to_rows(payload.get("candles") if isinstance(payload, dict) else None))
    return rows, err


def merge_bars(path: str | Path, rows: list[dict], write: bool = True) -> int:
    """Merge new bars into a per-symbol CSV: dedupe on timestamp, keep sorted
    (lexicographic == chronological for fixed-width timestamps). Returns the count
    of net-new timestamps. write=False computes the count without touching disk."""
    p = Path(path)
    bars: dict[str, dict] = {}
    if p.exists():
        with p.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                bars[row["timestamp"]] = row
    before = len(bars)
    for row in rows:
        bars[row["timestamp"]] = row
    if write:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            for ts in sorted(bars):
                w.writerow(bars[ts])
    return len(bars) - before


def collect(now: datetime | None = None, universe: tuple[str, ...] = UNIVERSE,
            store_dir: str | Path = STORE_DIR, write: bool = True,
            path: str = AUDIT_PATH, verbose: bool = True) -> dict:
    now = now or datetime.now(IST)
    n_new = n_partial = n_zero = 0
    failed: list[str] = []
    first_session = last_session = None

    for sym in universe:
        fp = Path(store_dir) / f"{sym}.csv"
        start = last_archived(fp) or (now - timedelta(days=LOOKBACK_DAYS))
        rows, err = fetch_symbol(sym, start, now)
        if err and not rows:
            failed.append(sym)
            continue
        if err:
            n_partial += 1
        if not rows:                    # already current / no data: nothing to persist
            n_zero += 1
            continue
        n_new += merge_bars(fp, rows, write=write)
        first, last = rows[0]["timestamp"][:10], rows[-1]["timestamp"][:10]
        first_session = min(first_session or first, first)
        last_session = max(last_session or last, last)

    n_failed = len(failed)
    note = "ok"
    if failed:
        note = f"failed={','.join(failed[:20])}" + (" ..." if n_failed > 20 else "")
    if n_partial:
        note = f"{note}; partial={n_partial}"

    row = {
        "hypothesis_ref": "RL-2026-07-25", "kind": "intraday_archive",
        "asof_ist": datetime.now(IST).strftime(DT_FMT),
        "run_date": now.date().isoformat(),
        "n_symbols": len(universe), "n_ok": len(universe) - n_failed,
        "n_failed": n_failed, "n_partial": n_partial, "n_zero_new": n_zero,
        "n_new_bars": n_new,
        "first_session": first_session, "last_session": last_session,
        "failed": failed[:20], "note": note,
    }
    if write:
        log_run(row, path=path)
    if verbose:
        _print_summary(row, path, write)
    return row


def _print_summary(row: dict, path: str, write: bool) -> None:
    print(f"[intraday archive] {row['run_date']}  symbols={row['n_symbols']}  "
          f"ok={row['n_ok']}  failed={row['n_failed']}  partial={row['n_partial']}")
    print(f"new_bars={row['n_new_bars']}  sessions {row['first_session']} -> "
          f"{row['last_session']}  zero_new={row['n_zero_new']}")
    print(f"note: {row['note']}")
    print(f"audit row {'appended to ' + path if write else 'NOT written (dry run)'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Archive 5-minute NSE bars before they expire")
    p.add_argument("--dry-run", action="store_true",
                   help="fetch and report but write neither bars nor an audit row")
    p.add_argument("--store-dir", default=STORE_DIR)
    p.add_argument("--path", default=AUDIT_PATH)
    a = p.parse_args()
    collect(write=not a.dry_run, store_dir=a.store_dir, path=a.path)


if __name__ == "__main__":
    main()
