"""Forward-only observational event collectors (RL-2026-07-26-14 / -16).

FORWARD-ONLY. Events are detected from registration (2026-07-10) forward; nothing
scans history (a historical read would spend the exhausted test window). No strategy
book, no Sharpe, no backtest - each run detects/archives events and, separately,
measures the 21-trading-day forward abnormal return of events that have matured.

Two independent legs feeding two pre-registered studies:

  - JUMP-MOM (RL-2026-07-26-14): a price JUMP with a volume spike is a free news-event
    proxy (Chan 2003 post-news drift). `collect_jumps` scans the LATEST completed
    session in the N500-277 Yahoo panel (prior-day cache) for names whose daily |return|
    >= 3x their trailing-63d daily sigma (sigma from data strictly before the event day,
    so no look-ahead). Variant (b) flags volume > 95th pct of the trailing 63d. A 21-
    trading-day per-name blackout dedupes overlapping events to the first. One row per
    event + one audit row -> experiments/event_jumps.jsonl.

  - SSF-LIST (RL-2026-07-26-16): when NSE adds/removes a stock from F&O, the short
    constraint is released/re-imposed (Miller 1977). The F&O underlying set IS the
    per-name basis-dict keys of the daily F&O collector (fno_daily.jsonl), so
    additions/deletions are a set-diff of the last two snapshots. One row per change +
    one audit row -> experiments/event_ssf_list.jsonl. First run (one snapshot only) is
    a baseline. A same-day batch shares one event_date (clustered-inference caveat lives
    in the registration).

Maturity (`measure_matured`, both ledgers): for an archived event whose event_date + 21
trading days has arrived, the 21d abnormal return = the stock's forward cumulative
return minus its NSE-industry equal-weight peer basket (industry from the nifty500 CSV;
the basket EXCLUDES the event stock). Measured once per (event_date, symbol) - idempotent.
No significance testing here; that happens only at the locked 252-day read.

Never-crash discipline: every leg catches everything and records a note row instead of
raising. All Yahoo access goes through the existing cache (refresh=False); no Groww
dependency.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .india import india_panel, sector_map
from .tracking import log_run

IST = timezone(timedelta(hours=5, minutes=30))
DT_FMT = "%Y-%m-%d %H:%M:%S"

PANEL_START = "2010-01-01"          # the N500-277 panel window (yields 277 kept names)
JUMP_PATH = "experiments/event_jumps.jsonl"
SSF_PATH = "experiments/event_ssf_list.jsonl"
FNO_PATH = "experiments/fno_daily.jsonl"

JUMP_REF = "RL-2026-07-26-14"
SSF_REF = "RL-2026-07-26-16"

K_SIGMA = 3.0                        # jump threshold: |ret| >= K_SIGMA * sigma63
SIGMA_WIN = 63                       # trailing daily-return window for sigma / vol pct
VOL_PCT = 0.95                       # variant (b): volume > this pct of trailing window
BLACKOUT = 21                        # per-name dedupe: trading days after an event
HORIZON = 21                         # forward abnormal-return window (trading days)

EVENT_KINDS = ("jump", "ssf_change")  # ledger rows measure_matured treats as events


# --------------------------------------------------------------------- utilities

def _r(x, n: int) -> float | None:
    """Round to n places; None-safe and NaN-safe."""
    if x is None:
        return None
    x = float(x)
    return None if pd.isna(x) else round(x, n)


def _read_ledger(path: str | Path) -> list[dict]:
    """All JSON rows from a jsonl ledger, skipping blanks/malformed; [] if absent."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _panel_col(sym: str, px: pd.DataFrame) -> str | None:
    """Resolve a ledger symbol (bare 'ABB' or 'ABB.NS') to its panel column, or None.

    JUMP symbols are already panel columns (SYMBOL.NS); SSF symbols are the bare F&O
    underlying names, resolved by appending .NS."""
    s = str(sym).upper()
    if s in px.columns:
        return s
    ns = s if s.endswith(".NS") else s + ".NS"
    return ns if ns in px.columns else None


def load_panel(refresh: bool = False):
    """(px, volume, industries) for the N500-277 panel. Yahoo via the cache only."""
    px, _, ohlcv, _ = india_panel(start=PANEL_START, index="nifty500",
                                  ret_clip=0.40, refresh=refresh)
    industries = {k.upper(): v for k, v in sector_map("nifty500").items()}
    return px, ohlcv["volume"], industries


# ------------------------------------------------------------------- JUMP-MOM leg

def _in_blackout(px: pd.DataFrame, event_date, last_dt, blackout: int = BLACKOUT) -> bool:
    """True if `event_date` falls within `blackout` trading days of a name's prior
    event (same-day or up to blackout-1 sessions later). last_dt=None -> not blocked."""
    if last_dt is None:
        return False
    try:
        gap = px.index.get_loc(pd.Timestamp(event_date)) - px.index.get_loc(pd.Timestamp(last_dt))
    except KeyError:
        return False
    return 0 <= gap < blackout


def _last_jump_dates(path: str | Path) -> dict[str, pd.Timestamp]:
    """Most-recent archived jump event_date per symbol, for the blackout dedupe."""
    last: dict[str, pd.Timestamp] = {}
    for r in _read_ledger(path):
        if r.get("kind") == "jump" and r.get("symbol") and r.get("event_date"):
            d = pd.Timestamp(r["event_date"])
            s = str(r["symbol"])
            if s not in last or d > last[s]:
                last[s] = d
    return last


def detect_jumps(px: pd.DataFrame, volume: pd.DataFrame, industries: dict,
                 event_date=None, last_event: dict | None = None, k: float = K_SIGMA,
                 sigma_win: int = SIGMA_WIN, vol_win: int = SIGMA_WIN,
                 vol_pct: float = VOL_PCT, blackout: int = BLACKOUT) -> list[dict]:
    """Jump events on `event_date` (default: the latest panel session).

    Event iff |ret_event| >= k * sigma63, where sigma63 is the sample std (ddof=1) of the
    `sigma_win` daily returns STRICTLY BEFORE the event day - the jump day itself never
    enters its own sigma. Names inside their per-name blackout are dropped. Returns event
    dicts (no ledger metadata); the caller stamps hypothesis_ref/kind and writes."""
    last_event = last_event or {}
    ed = pd.Timestamp(event_date) if event_date is not None else px.index[-1]
    i = px.index.get_loc(ed)
    if i < sigma_win + 1:                       # need a full pre-event return window
        return []

    rets = px.pct_change()
    ret = rets.iloc[i]                          # event-day return per name
    sigma = rets.iloc[i - sigma_win:i].std(ddof=1)   # strictly pre-event window
    vwin = volume.iloc[i - vol_win:i]
    p95 = vwin.quantile(vol_pct)
    vol_ev = volume.iloc[i]

    events = []
    for sym in px.columns:
        s, sig = ret[sym], sigma[sym]
        if pd.isna(s) or pd.isna(sig) or sig <= 0:
            continue
        z = s / sig
        if abs(z) < k:
            continue
        if _in_blackout(px, ed, last_event.get(sym), blackout):
            continue
        v, thr = vol_ev.get(sym), p95.get(sym)
        confirmed = bool(v is not None and thr is not None
                         and not pd.isna(v) and not pd.isna(thr) and v > thr)
        events.append({
            "event_date": ed.date().isoformat(), "symbol": sym,
            "ret": _r(s, 6), "sigma63": _r(sig, 6), "z": _r(z, 3),
            "direction": "up" if s > 0 else "down",
            "vol_confirmed": confirmed, "industry": industries.get(sym), "note": "ok",
        })
    return events


def collect_jumps(px: pd.DataFrame | None = None, volume: pd.DataFrame | None = None,
                  industries: dict | None = None, today: date | None = None,
                  refresh: bool = False, write: bool = True, path: str = JUMP_PATH,
                  verbose: bool = True) -> dict:
    """Detect jumps on the latest completed panel session and archive them plus an audit
    row. Never raises: on any failure an audit row with note=<reason> is written."""
    today = today or datetime.now(IST).date()
    try:
        if px is None:
            px, volume, industries = load_panel(refresh=refresh)
        events = detect_jumps(px, volume, industries, last_event=_last_jump_dates(path))
        session = px.index[-1].date().isoformat()
        n_scanned = px.shape[1]
        note = f"session={session}"
    except Exception as e:                      # never crash the orchestrator
        events, session, n_scanned = [], None, 0
        note = f"error: {type(e).__name__}: {e}"

    if write:
        for ev in events:
            log_run({"hypothesis_ref": JUMP_REF, "kind": "jump",
                     "asof_ist": datetime.now(IST).strftime(DT_FMT), **ev}, path=path)
    audit = {"hypothesis_ref": JUMP_REF, "kind": "audit",
             "asof_ist": datetime.now(IST).strftime(DT_FMT),
             "run_date": today.isoformat(), "n_events": len(events),
             "n_names_scanned": n_scanned, "note": note}
    if write:
        log_run(audit, path=path)
    if verbose:
        print(f"[JUMP-MOM] {note}  events={len(events)}  scanned={n_scanned}")
        for ev in events[:10]:
            print(f"  {ev['symbol']:16s} ret={ev['ret']} z={ev['z']} "
                  f"{ev['direction']} vol_confirmed={ev['vol_confirmed']}")
        print(f"audit row {'appended to ' + path if write else 'NOT written (dry run)'}")
    return audit


# ------------------------------------------------------------------- SSF-LIST leg

def _fno_snapshots(fno_path: str | Path) -> list[dict]:
    """F&O daily-snapshot rows (those carrying a basis dict), in file order."""
    return [r for r in _read_ledger(fno_path)
            if r.get("kind") == "fno_daily_snapshot" and isinstance(r.get("basis"), dict)]


def diff_universe(prev: set | None, curr: set) -> tuple[list[str], list[str]]:
    """(added, removed) sorted. prev=None -> baseline: both empty."""
    if prev is None:
        return [], []
    return sorted(curr - prev), sorted(prev - curr)


def collect_ssf_changes(fno_path: str = FNO_PATH, today: date | None = None,
                        write: bool = True, path: str = SSF_PATH,
                        verbose: bool = True) -> dict:
    """Set-diff the F&O underlying set between the last two F&O snapshots; archive one
    row per add/remove plus an audit row. One snapshot only -> baseline (no events).
    Never raises."""
    today = today or datetime.now(IST).date()
    changes: list[dict] = []
    event_date = None
    n_universe = 0
    try:
        snaps = _fno_snapshots(fno_path)
        if len(snaps) < 2:
            note = "baseline" if len(snaps) == 1 else "no_fno_data"
            if snaps:
                n_universe = len(snaps[-1]["basis"])
        else:
            prev = set(snaps[-2]["basis"].keys())
            curr = set(snaps[-1]["basis"].keys())
            n_universe = len(curr)
            event_date = str(snaps[-1].get("collect_date"))
            added, removed = diff_universe(prev, curr)
            for sym in added:
                changes.append({"event_date": event_date, "symbol": sym,
                                "change": "added", "n_universe": n_universe, "note": "ok"})
            for sym in removed:
                changes.append({"event_date": event_date, "symbol": sym,
                                "change": "removed", "n_universe": n_universe, "note": "ok"})
            note = ("no_change" if not changes
                    else f"added={len(added)} removed={len(removed)}")
    except Exception as e:
        note = f"error: {type(e).__name__}: {e}"

    if write:
        for ch in changes:
            log_run({"hypothesis_ref": SSF_REF, "kind": "ssf_change",
                     "asof_ist": datetime.now(IST).strftime(DT_FMT), **ch}, path=path)
    n_added = sum(c["change"] == "added" for c in changes)
    audit = {"hypothesis_ref": SSF_REF, "kind": "audit",
             "asof_ist": datetime.now(IST).strftime(DT_FMT),
             "run_date": today.isoformat(), "event_date": event_date,
             "n_added": n_added, "n_removed": len(changes) - n_added,
             "n_universe": n_universe, "note": note}
    if write:
        log_run(audit, path=path)
    if verbose:
        print(f"[SSF-LIST] {note}  universe={n_universe}")
        for ch in changes:
            print(f"  {ch['change']:8s} {ch['symbol']}")
        print(f"audit row {'appended to ' + path if write else 'NOT written (dry run)'}")
    return audit


# ------------------------------------------------------------ maturity measurement

def abnormal_return(px: pd.DataFrame, industries: dict, col: str, i: int,
                    horizon: int = HORIZON) -> tuple[float | None, int]:
    """21d abnormal return = stock forward cumulative return minus the equal-weight
    forward return of its NSE-industry peers (the event stock EXCLUDED). Returns
    (abn_or_None, n_peers); abn is None when the name has no industry peers."""
    j = i + horizon
    stock = px[col].iloc[j] / px[col].iloc[i] - 1.0
    ind = industries.get(col)
    peers = [c for c in px.columns
             if c != col and ind is not None and industries.get(c) == ind]
    prets = [px[c].iloc[j] / px[c].iloc[i] - 1.0 for c in peers]
    prets = [p for p in prets if not pd.isna(p)]
    if not prets:
        return None, 0
    return float(stock - sum(prets) / len(prets)), len(prets)


def _measured_keys(path: str | Path) -> set:
    """(event_date, symbol) pairs already carrying a matured row (idempotency guard)."""
    return {(r.get("event_date"), r.get("symbol")) for r in _read_ledger(path)
            if r.get("kind") == "matured"}


def measure_matured(path: str, px: pd.DataFrame | None = None,
                    industries: dict | None = None, ref: str | None = None,
                    horizon: int = HORIZON, refresh: bool = False,
                    write: bool = True, verbose: bool = True) -> list[dict]:
    """Measure every archived event in `path` that has aged `horizon` trading days and is
    not yet measured; append one matured row each. Measured from the first panel session
    on/after the event_date, so an F&O event_date that isn't itself a Yahoo session still
    resolves. Idempotent per (event_date, symbol). Never raises on a single event."""
    today = datetime.now(IST).date().isoformat()
    try:
        if px is None:
            px, _, industries = load_panel(refresh=refresh)
    except Exception as e:
        if verbose:
            print(f"[maturity {path}] panel load failed: {type(e).__name__}: {e}")
        return []

    measured = _measured_keys(path)
    last_pos = len(px) - 1
    rows: list[dict] = []
    for ev in _read_ledger(path):
        if ev.get("kind") not in EVENT_KINDS:
            continue
        key = (ev.get("event_date"), ev.get("symbol"))
        if key in measured or None in key:
            continue
        try:
            col = _panel_col(ev["symbol"], px)
            if col is None:
                continue                        # not in panel -> unmeasurable, stays pending
            i = int(px.index.searchsorted(pd.Timestamp(ev["event_date"])))
            if i >= len(px) or i + horizon > last_pos:
                continue                        # event in the future / not yet matured
            abn, n_peers = abnormal_return(px, industries, col, i, horizon)
            row = {"hypothesis_ref": ref or ev.get("hypothesis_ref"), "kind": "matured",
                   "asof_ist": datetime.now(IST).strftime(DT_FMT),
                   "event_date": ev["event_date"], "symbol": ev["symbol"],
                   "abn_21d": _r(abn, 6), "n_peers": n_peers,
                   "matured_date": px.index[i + horizon].date().isoformat(),
                   "note": "ok" if abn is not None else "no_peers"}
        except Exception as e:                  # one bad event never sinks the batch
            row = None
            if verbose:
                print(f"  measure error {key}: {type(e).__name__}: {e}")
        if row is None:
            continue
        rows.append(row)
        measured.add(key)
        if write:
            log_run(row, path=path)
    if verbose:
        print(f"[maturity {path}] matured {len(rows)} event(s) as of {today}")
    return rows


# --------------------------------------------------------------------------- run

def run_all(refresh: bool = False, write: bool = True, verbose: bool = True) -> None:
    """Build the panel once and run all three legs (jumps, SSF changes, maturity for
    both ledgers). The orchestrator may call this or the individual legs."""
    px, volume, industries = load_panel(refresh=refresh)
    collect_jumps(px=px, volume=volume, industries=industries, write=write, verbose=verbose)
    collect_ssf_changes(write=write, verbose=verbose)
    measure_matured(JUMP_PATH, px=px, industries=industries, ref=JUMP_REF,
                    write=write, verbose=verbose)
    measure_matured(SSF_PATH, px=px, industries=industries, ref=SSF_REF,
                    write=write, verbose=verbose)


def main() -> None:
    p = argparse.ArgumentParser(description="Forward event collectors: jumps + F&O list changes")
    p.add_argument("--dry-run", action="store_true", help="detect and report but write no rows")
    p.add_argument("--refresh", action="store_true", help="refresh the Yahoo panel cache first")
    a = p.parse_args()
    run_all(refresh=a.refresh, write=not a.dry_run)


if __name__ == "__main__":
    main()
