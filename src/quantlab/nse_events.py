"""NSE event-data forward archivers: F&O ban list + index reconstitution.

ARCHIVE-BEFORE-EXPIRY. Two independent daily rescues, each losing every uncollected
day forever, feeding two pre-registered event studies:

  - `collect_ban_list`  -> RL-2026-07-26-11 (F&O ban-list / MWPL crowding events).
    NSE publishes the daily F&O security ban list. We try, in order: (a) the JSON API
    `https://www.nseindia.com/api/foSecBan` (browser-like headers + a cookie warm-up GET
    to the NSE root, which its anti-bot layer requires); (b) the CSV archive
    `https://nsearchives.nseindia.com/content/fo/fo_secban.csv`, which is more
    scraper-tolerant. One row/day -> `experiments/nse_ban_list.jsonl`.
    Live-probed 2026-07-10: the JSON API was blocked from the lab (root 403, api 404
    behind an anti-bot script); the CSV archive returned 200 with shape
    `Securities in Ban For Trade Date DD-MON-YYYY:` then `<serial>,<SYMBOL>` rows. The
    CSV is therefore the authoritative source here; the JSON leg is kept first per the
    spec and works where the anti-bot posture allows it.

  - `collect_index_changes` -> RL-2026-07-26-17 (index-reconstitution flow events).
    Fetches the CURRENT official Nifty 50 / Nifty Next 50 constituent CSVs
    (`.../content/indices/ind_{index}list.csv` - the same endpoint family this repo
    already uses for nifty500) and DIFFS today's membership against the last stored
    membership; the additions/deletions ARE the reconstitution events, caught at the
    effective date. Full membership is stored in every ledger row, so the committed
    `experiments/nse_index_changes.jsonl` IS the point-in-time archive and the diff
    baseline needs no git-ignored side file. First run per index = baseline
    (note="baseline"). FUTURE UPGRADE: scraping the NSE Indices press releases would
    capture the earlier announcement date (anticipation window) instead of only the
    effective date - not needed for the set-diff design, noted for later.

No look-ahead is possible (pure forward archiving). No Groww dependency. No strategy
logic. Network uses stdlib `urllib.request` (matching `india.py`'s existing NSE-CSV
fetch) so no new / undeclared dependency surface is added. Every run records exactly
one audit row per leg via `tracking.log_run`; a failed fetch degrades to an error row
(note=<reason>) and never crashes the caller.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from http.cookiejar import CookieJar
from pathlib import Path

from .tracking import log_run

IST = timezone(timedelta(hours=5, minutes=30))
DT_FMT = "%Y-%m-%d %H:%M:%S"

BAN_JSON_URL = "https://www.nseindia.com/api/foSecBan"
BAN_CSV_URL = "https://nsearchives.nseindia.com/content/fo/fo_secban.csv"
BAN_PATH = "experiments/nse_ban_list.jsonl"

INDEX_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_{index}list.csv"
CHANGES_PATH = "experiments/nse_index_changes.jsonl"
INDICES = ("nifty50", "niftynext50")

NSE_ROOT = "https://www.nseindia.com"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/csv, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# NSE tickers: uppercase alnum plus & . - (e.g. M&M, BAJAJ-AUTO). Sentinels are header
# remnants / empty-day markers that must not be mistaken for symbols.
_SYM_RE = re.compile(r"^[A-Z0-9][A-Z0-9&.\-]{0,19}$")
_SENTINELS = {"NIL", "NA", "NONE", "NULL", "SYMBOL", "SR", "SRNO", "SRNO."}
_DATE_RE = re.compile(r"(\d{1,2}-[A-Za-z]{3}-\d{4})")


def _looks_like_symbol(s: str) -> bool:
    s = s.strip().upper()
    return bool(_SYM_RE.match(s)) and s not in _SENTINELS


# ---------------------------------------------------------------- network (stdlib)

def http_get(url: str, warmup: bool = False, timeout: int = 25) -> tuple[str | None, int | None, str | None]:
    """One polite GET with browser-like headers. `warmup=True` first does a cookie
    warm-up GET to the NSE root (the JSON API's anti-bot layer needs it). Single
    request per call - no retry loop to hammer with. Returns
    (text_or_None, status_or_None, err_or_None); never raises."""
    try:
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar()))
        if warmup:
            try:
                opener.open(urllib.request.Request(NSE_ROOT, headers=_HEADERS), timeout=timeout)
                time.sleep(0.5)                      # be polite between warmup and API hit
            except Exception:
                pass                                 # warmup failure is non-fatal
        with opener.open(urllib.request.Request(url, headers=_HEADERS), timeout=timeout) as res:
            return res.read().decode("utf-8", errors="replace"), res.status, None
    except urllib.error.HTTPError as e:              # 401/403/404/... recorded honestly
        return None, e.code, f"HTTP {e.code}"
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


# ------------------------------------------------------------------- ban-list parse

def _parse_ban_header_date(text: str) -> str | None:
    """Extract the ban-file's declared trade date (DD-MON-YYYY -> ISO), or None."""
    m = _DATE_RE.search(text)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%d-%b-%Y").date().isoformat()
    except ValueError:
        return None


def parse_ban_csv(text: str) -> tuple[list[str], str | None]:
    """Parse the fo_secban.csv archive. Real shape (probed 2026-07-10):
        line 1: 'Securities in Ban For Trade Date DD-MON-YYYY:'
        rows:   '<serial>,<SYMBOL>'
    Returns (symbols, trade_date_iso_or_None). The serial-prefixed rows are the
    canonical layout; a bare-symbol layout (no serials) is parsed as a fallback so a
    format change still yields data. An empty/Nil day yields []."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    trade_date = next((_parse_ban_header_date(ln) for ln in lines
                       if "trade date" in ln.lower()), None)

    symbols: list[str] = []
    seen: set[str] = set()

    def add(sym: str) -> None:
        s = sym.strip().strip('"').upper()
        if _looks_like_symbol(s) and s not in seen:
            seen.add(s)
            symbols.append(s)

    # primary: '<serial>,<SYMBOL>' rows
    for ln in lines:
        fields = [f.strip().strip('"') for f in ln.split(",")]
        if len(fields) >= 2 and fields[0].isdigit():
            add(fields[1])
    if symbols:
        return symbols, trade_date

    # fallback: bare-symbol layout (skip header / sentence lines)
    for ln in lines:
        low = ln.lower()
        if "trade date" in low or low.startswith("securities"):
            continue
        for field in ln.split(","):
            if not field.strip().isdigit():
                add(field)
    return symbols, trade_date


_JSON_SYM_KEYS = {"symbol", "symbolname"}


def _symbols_from_json(obj) -> list[str]:
    """Best-effort symbol extraction from a foSecBan JSON payload. The exact shape is
    not live-verifiable (the API was anti-bot blocked from the lab 2026-07-10), so this
    is tolerant: a bare list of strings is taken as symbols, otherwise values under
    'symbol'/'symbolName' keys are collected recursively. Non-symbol strings are
    filtered out."""
    raw: list[str] = []
    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        raw = list(obj)
    else:
        def walk(x) -> None:
            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(v, str) and str(k).lower() in _JSON_SYM_KEYS:
                        raw.append(v)
                    else:
                        walk(v)
            elif isinstance(x, list):
                for it in x:
                    walk(it)
        walk(obj)

    out, seen = [], set()
    for s in (t.strip().upper() for t in raw):
        if _looks_like_symbol(s) and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def collect_ban_list(today: date | None = None, write: bool = True, path: str = BAN_PATH,
                     fetch=http_get, verbose: bool = True) -> dict:
    """Archive today's F&O ban list: JSON API first, CSV archive fallback. Writes one
    row to `path`. On total failure, records an error row (symbols=[], source=None)
    rather than raising."""
    today = today or datetime.now(IST).date()
    symbols: list[str] | None = None
    source: str | None = None
    trade_date: str | None = None
    notes: list[str] = []

    # (a) JSON API with cookie warm-up
    text, status, err = fetch(BAN_JSON_URL, warmup=True)
    if err or status != 200 or not text:
        notes.append(f"json:{err or f'HTTP {status}'}")
    else:
        try:
            syms = _symbols_from_json(json.loads(text))
        except (ValueError, TypeError):
            syms, notes = [], notes + ["json:parse_error"]
        if syms:
            symbols, source = syms, "json_api"
        elif "json:parse_error" not in notes:
            notes.append("json:empty")

    # (b) CSV archive fallback
    if symbols is None:
        text, status, err = fetch(BAN_CSV_URL)
        if err or status != 200 or not text:
            notes.append(f"csv:{err or f'HTTP {status}'}")
        else:
            symbols, trade_date = parse_ban_csv(text)
            source = "csv_archive"

    ok = symbols is not None
    if not ok:
        symbols = []
    row = {
        "hypothesis_ref": "RL-2026-07-26-11", "kind": "nse_ban_list",
        "asof_ist": datetime.now(IST).strftime(DT_FMT),
        "date": today.isoformat(), "trade_date": trade_date,
        "symbols": symbols, "n_banned": len(symbols), "source": source,
        "note": "; ".join(notes) or "ok",
    }
    if write:
        log_run(row, path=path)
    if verbose:
        print(f"[ban list] {row['date']}  n_banned={row['n_banned']}  "
              f"source={source}  trade_date={trade_date}")
        print(f"symbols: {', '.join(symbols) if symbols else '(none)'}")
        print(f"note: {row['note']}  {'written to ' + path if write else 'NOT written (dry run)'}")
    return row


# ------------------------------------------------------------- index reconstitution

def parse_constituents(text: str) -> list[str]:
    """Raw NSE symbols (no .NS) from a constituent CSV's 'Symbol' column, deduped,
    order preserved."""
    out, seen = [], set()
    for r in csv.DictReader(io.StringIO(text or "")):
        s = (r.get("Symbol") or "").strip()
        if s and s.upper() not in seen:
            seen.add(s.upper())
            out.append(s)
    return out


def diff_membership(prev: list[str] | None, current: list[str]) -> tuple[list[str], list[str]]:
    """(added, removed) as sorted lists. prev=None -> baseline: both empty."""
    if prev is None:
        return [], []
    p, c = set(prev), set(current)
    return sorted(c - p), sorted(p - c)


def _last_members(path: str, index: str) -> list[str] | None:
    """Most-recent successfully-fetched membership for `index` from the committed
    ledger, or None if there is no prior successful row. Error rows carry members=null
    and are skipped, so a failed fetch never corrupts the next run's diff baseline."""
    p = Path(path)
    if not p.exists():
        return None
    last = None
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if (rec.get("kind") == "nse_index_changes" and rec.get("index") == index
                    and isinstance(rec.get("members"), list) and rec.get("members")):
                last = rec["members"]
    return last


def collect_index_changes(today: date | None = None, indices=INDICES, write: bool = True,
                          path: str = CHANGES_PATH, fetch=http_get,
                          verbose: bool = True) -> list[dict]:
    """Diff current Nifty 50 / Next 50 membership against the last stored membership;
    write one row per index. First run per index = baseline. Full membership is stored
    per row (the point-in-time archive). A failed fetch yields an error row with
    members=null so it is skipped as a future diff baseline."""
    today = today or datetime.now(IST).date()
    rows: list[dict] = []
    for index in indices:
        prev = _last_members(path, index)
        text, status, err = fetch(INDEX_CSV_URL.format(index=index))
        base = {"hypothesis_ref": "RL-2026-07-26-17", "kind": "nse_index_changes",
                "asof_ist": datetime.now(IST).strftime(DT_FMT),
                "date": today.isoformat(), "index": index}

        if err or status != 200 or not text:
            row = {**base, "added": [], "removed": [], "members": None,
                   "n_members": 0, "note": f"error: {err or f'HTTP {status}'}"}
        else:
            current = parse_constituents(text)
            if not current:
                row = {**base, "added": [], "removed": [], "members": None,
                       "n_members": 0, "note": "error: empty_constituents"}
            else:
                added, removed = diff_membership(prev, current)
                note = ("baseline" if prev is None else
                        "no_change" if not added and not removed else
                        f"added={len(added)} removed={len(removed)}")
                row = {**base, "added": added, "removed": removed,
                       "members": sorted(current), "n_members": len(current), "note": note}
        rows.append(row)
        if write:
            log_run(row, path=path)
        if verbose:
            print(f"[index changes] {index}  n={row['n_members']}  "
                  f"added={row['added']}  removed={row['removed']}  note={row['note']}")
    if verbose:
        print(f"{'rows written to ' + path if write else 'NOT written (dry run)'}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Archive NSE F&O ban list + index-reconstitution membership")
    p.add_argument("--dry-run", action="store_true", help="fetch and report but write no rows")
    p.add_argument("--ban-only", action="store_true", help="run only the ban-list leg")
    p.add_argument("--index-only", action="store_true", help="run only the index-changes leg")
    p.add_argument("--ban-path", default=BAN_PATH)
    p.add_argument("--changes-path", default=CHANGES_PATH)
    a = p.parse_args()
    write = not a.dry_run
    if not a.index_only:
        collect_ban_list(write=write, path=a.ban_path)
    if not a.ban_only:
        collect_index_changes(write=write, path=a.changes_path)


if __name__ == "__main__":
    main()
