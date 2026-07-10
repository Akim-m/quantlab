"""NSE MTO (Market-wide To-be-delivered) delivery-data archive + parser.

Feeds RL-2026-07-26-20 (DELIV): the delivery-percentage conviction cross-section.
NSE publishes, per trading day, each security's DELIVERABLE quantity - the slice of
traded volume that actually settled (was taken to demat) rather than being round-tripped
intraday. The daily archive lives at

    https://nsearchives.nseindia.com/archives/equities/mto/MTO_DDMMYYYY.DAT

verified serving from 2011 to today (probe 2026-07-10: 2011-07-01, 2013-07-01 and
2026-07-09 all returned 200). Files land under `data/raw/nse_mto/` (git-ignored via the
`data/raw/` rule) and are NEVER committed.

Record layout (record-type-20 lines, comma-separated):
    20,<SrNo>,<Name of Security>,<Series>,<Quantity Traded>,<Deliverable Quantity>,<%>
so the delivery ratio = deliverable_qty / traded_qty is a direct read of fields [5]/[4];
the trailing %-field is the same number x100 and is used only as an independent check.
Series is filtered to EQ (BE / GS / GC / debt series dropped and counted). Older files
carry two settlement-type sections (D + N); if a symbol legitimately appears more than
once in EQ on a date its traded and deliverable quantities are SUMMED before the ratio,
which is the correct day-level delivery share (deliverable <= traded per row -> ratio in
(0,1]). Malformed lines are skipped and counted; nothing is fabricated.

Only forward archiving + local parsing: no look-ahead, no Groww dependency, no strategy
logic. The polite nsearchives HTTP path is reused from `nse_events.http_get` so no new
network-dependency surface is added.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .nse_events import IST, http_get

DATA_DIR = "data/raw/nse_mto"
MANIFEST = "data/raw/nse_mto/manifest.json"
MTO_URL = "https://nsearchives.nseindia.com/archives/equities/mto/MTO_{d}.DAT"
BHAV_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"

# The two pre-registered backfill spans. The 2017-01-01 -> 2025-10-31 hold-out is
# deliberately NOT downloaded, so it stays physically absent from the lab.
TRAIN_SPAN = ("2011-07-01", "2016-12-31")
LIVE_START = "2025-11-01"

SERIES_KEEP = "EQ"


# ------------------------------------------------------------------- date helpers

def _ddmmyyyy(d: date) -> str:
    return d.strftime("%d%m%Y")


def _date_from_name(name: str) -> date | None:
    """'MTO_09072026.DAT' -> date(2026, 7, 9), or None if it doesn't match."""
    stem = name.upper()
    if not (stem.startswith("MTO_") and stem.endswith(".DAT")):
        return None
    try:
        return datetime.strptime(stem[4:-4], "%d%m%Y").date()
    except ValueError:
        return None


def _weekdays(start: str, end: str):
    d = pd.Timestamp(start).date()
    last = pd.Timestamp(end).date()
    while d <= last:
        if d.weekday() < 5:                       # Mon-Fri; NSE never trades weekends
            yield d
        d += timedelta(days=1)


# ------------------------------------------------------------------------- parsing

def parse_delivery(text: str, series_keep: str = SERIES_KEEP) -> tuple[dict[str, float], dict]:
    """Parse one MTO .DAT into {SYMBOL: delivery_ratio} for the kept series (EQ).

    ratio = deliverable_qty / traded_qty, computed from the raw quantities (fields
    [5]/[4]) - NOT the file's rounded %-field. Non-kept series are dropped and counted;
    malformed record-type-20 lines (too few fields, non-numeric quantities) are skipped
    and counted; a zero traded quantity yields no ratio (counted). Duplicate EQ rows for
    one symbol (multi-settlement-type files) are aggregated by summing before the ratio.
    Returns (ratios, stats)."""
    traded_sum: dict[str, int] = {}
    deliv_sum: dict[str, int] = {}
    stats = {"eq_rows": 0, "dropped_series": 0, "malformed": 0, "zero_traded": 0}

    for ln in (text or "").splitlines():
        f = ln.split(",")
        if f[0].strip() != "20":                  # header / title / other record types
            continue
        if len(f) < 7:
            stats["malformed"] += 1
            continue
        if f[3].strip().upper() != series_keep:
            stats["dropped_series"] += 1
            continue
        try:
            traded = int(f[4]); deliv = int(f[5])
        except ValueError:
            stats["malformed"] += 1
            continue
        if traded <= 0:
            stats["zero_traded"] += 1
            continue
        sym = f[2].strip().upper()
        stats["eq_rows"] += 1
        traded_sum[sym] = traded_sum.get(sym, 0) + traded
        deliv_sum[sym] = deliv_sum.get(sym, 0) + deliv

    ratios = {s: deliv_sum[s] / traded_sum[s] for s in traded_sum}
    stats["symbols"] = len(ratios)
    stats["dup_eq"] = stats["eq_rows"] - len(ratios)  # EQ rows folded by summing
    return ratios, stats


def parse_bhavcopy_delivery(text: str) -> dict[str, float]:
    """{SYMBOL: DELIV_QTY/TTL_TRD_QNTY} for EQ rows of the modern full bhavcopy
    (sec_bhavdata_full). The independent QC reference for the MTO ratios."""
    out: dict[str, float] = {}
    for r in csv.DictReader(io.StringIO(text or ""), skipinitialspace=True):
        if (r.get("SERIES") or "").strip() != "EQ":
            continue
        try:
            traded = float(r["TTL_TRD_QNTY"]); deliv = float(r["DELIV_QTY"])
        except (KeyError, ValueError, TypeError):
            continue
        if traded > 0:
            out[(r.get("SYMBOL") or "").strip().upper()] = deliv / traded
    return out


def load_ratios(data_dir: str = DATA_DIR, start: str | None = None,
                end: str | None = None) -> pd.DataFrame:
    """Assemble a date x bare-SYMBOL delivery-ratio panel from the downloaded .DAT files
    (optionally restricted to [start, end]). A day with no file is simply absent; a
    symbol absent on a day is NaN."""
    lo = pd.Timestamp(start).date() if start else None
    hi = pd.Timestamp(end).date() if end else None
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for fp in sorted(Path(data_dir).glob("MTO_*.DAT")):
        dt = _date_from_name(fp.name)
        if dt is None or (lo and dt < lo) or (hi and dt > hi):
            continue
        ratios, _ = parse_delivery(fp.read_text(encoding="utf-8", errors="replace"))
        if ratios:
            rows[pd.Timestamp(dt)] = ratios
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


# ------------------------------------------------------------------- backfill / fetch

def _looks_like_mto(text: str) -> bool:
    head = (text or "")[:200]
    return "MTO" in head or "Security Wise Delivery" in head


def _load_manifest(path: str) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return {}
    return {}


def _save_manifest(path: str, manifest: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, sort_keys=True, indent=0), encoding="utf-8")


def _fetch_one(day: date, data_dir: str, fetch, sleep: float) -> str:
    """Download a single day's MTO file if absent. Returns the manifest status:
    'ok' | 'cached' | '404' | 'error:<reason>'. One retry on a transient failure
    (network error or a non-404 HTTP status); a clean 404 (holiday) is definitive."""
    fp = Path(data_dir) / f"MTO_{_ddmmyyyy(day)}.DAT"
    if fp.exists():
        return "cached"
    url = MTO_URL.format(d=_ddmmyyyy(day))
    text, status, err = fetch(url)
    if (err is not None or status not in (200, 404)):   # transient -> one polite retry
        time.sleep(sleep)
        text, status, err = fetch(url)
    if status == 200 and text and _looks_like_mto(text):
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(text, encoding="utf-8")
        return "ok"
    if status == 404:
        return "404"
    return f"error:{err or f'HTTP {status}'}"


def backfill(spans=(TRAIN_SPAN,), data_dir: str = DATA_DIR, manifest_path: str = MANIFEST,
             sleep: float = 0.25, fetch=http_get, verbose: bool = True) -> dict:
    """Resumable backfill of the MTO archive over the given (start, end) spans. Skips
    weekends and files already on disk; records each attempted date's status to the
    manifest; polite ~`sleep`s between network requests. Returns a summary dict."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    counts = {"ok": 0, "cached": 0, "404": 0, "error": 0}
    for start, end in spans:
        for d in _weekdays(start, end):
            iso = d.isoformat()
            st = _fetch_one(d, data_dir, fetch, sleep)
            manifest[iso] = "ok" if st == "cached" else st
            counts["ok" if st in ("ok", "cached") else
                   "404" if st == "404" else "error"] += 1
            if st != "cached":
                time.sleep(sleep)               # only rate-limit actual requests
                if counts["ok"] % 100 == 0 and verbose:
                    _save_manifest(manifest_path, manifest)
                    print(f"  ... {iso} {st}  (ok={counts['ok']} 404={counts['404']} "
                          f"err={counts['error']})")
    _save_manifest(manifest_path, manifest)
    total_bytes = sum(p.stat().st_size for p in Path(data_dir).glob("MTO_*.DAT"))
    summary = {"counts": counts, "files_on_disk": len(list(Path(data_dir).glob("MTO_*.DAT"))),
               "total_bytes": total_bytes, "manifest_entries": len(manifest)}
    if verbose:
        print(f"[backfill] ok/cached={counts['ok']} 404={counts['404']} "
              f"error={counts['error']}  bytes={total_bytes:,}")
    return summary


def collect_mto_today(today: date | None = None, data_dir: str = DATA_DIR,
                      manifest_path: str = MANIFEST, fetch=http_get,
                      verbose: bool = True) -> dict:
    """Daily forward collector: fetch today's MTO file after EOD. Idempotent (skips a
    file already on disk) and never raises - a fetch failure returns a status row and
    leaves the archive untouched. NOT wired into scripts/snapshot.py by design."""
    today = today or datetime.now(IST).date()
    st = _fetch_one(today, data_dir, fetch, sleep=0.0)
    manifest = _load_manifest(manifest_path)
    manifest[today.isoformat()] = "ok" if st == "cached" else st
    _save_manifest(manifest_path, manifest)
    row = {"date": today.isoformat(), "status": st}
    if verbose:
        print(f"[mto collect] {row['date']}  {st}")
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill / collect the NSE MTO delivery archive")
    p.add_argument("--span", choices=("train", "live", "both"), default="both")
    p.add_argument("--live-end", default=None, help="live span end (default: today IST)")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--sleep", type=float, default=0.25)
    p.add_argument("--today", action="store_true", help="collect only today's file")
    a = p.parse_args()
    if a.today:
        collect_mto_today(data_dir=a.data_dir)
        return
    live_end = a.live_end or datetime.now(IST).date().isoformat()
    spans = {"train": [TRAIN_SPAN], "live": [(LIVE_START, live_end)],
             "both": [TRAIN_SPAN, (LIVE_START, live_end)]}[a.span]
    backfill(spans=spans, data_dir=a.data_dir, sleep=a.sleep)


if __name__ == "__main__":
    main()
