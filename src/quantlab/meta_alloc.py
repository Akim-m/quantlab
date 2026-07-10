"""RL-2026-07-26-21 META-ALLOC: regime-conditional capital map across the live
cash-equity paper sleeves (a book-of-books), recorded daily next to a frozen
STATIC 50/50 baseline.

FORWARD-ONLY by construction: 7 of the 10 sleeves were born 2026-07-10, so no
joint history exists and this module has no backtest path. The daily snapshot
records the state and BOTH capital maps; the forward record combines each
sleeve's rigorous close-to-close forward return (`live_paper.forward_track`)
under the latest map recorded STRICTLY BEFORE the return day - that join is
the causal lag. First locked read >=252 forward days: Sharpe(META) -
Sharpe(STATIC) Ledoit-Wolf z > 1 AND maxDD(META) <= maxDD(STATIC) + 2 pts.

State = the deployed gate verbatim (`blend.regime_on`: ^NSEI >= 200d MA AND
^INDIAVIX < causal rolling-252d 80th pct). No new estimation in this layer.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import deliv, live_paper
from .blend import regime_on
from .data import close_prices, load_yahoo_ohlcv
from .tracking import log_run

HYP = "RL-2026-07-26-21"
META_SNAPSHOT_PATH = "experiments/meta_alloc.jsonl"

SLEEVES = {
    "regime": live_paper.SNAPSHOT_PATH,
    "trend": live_paper.TREND_SNAPSHOT_PATH,
    "dualrot": live_paper.DUALROT_SNAPSHOT_PATH,
    "ls": live_paper.LS_SNAPSHOT_PATH,
    "pairs": live_paper.PAIRS_SNAPSHOT_PATH,
    "divcarry": live_paper.DIVCARRY_SNAPSHOT_PATH,
    "volshock": live_paper.VOLSHOCK_SNAPSHOT_PATH,
    "illiq": live_paper.ILLIQ_SNAPSHOT_PATH,
    "macrobeta": live_paper.MACROBETA_SNAPSHOT_PATH,
    "deliv": deliv.DELIV_SNAPSHOT_PATH,
}
DIRECTIONAL = ("regime", "trend", "dualrot")
NEUTRAL = ("ls", "pairs", "divcarry", "volshock", "illiq", "macrobeta", "deliv")

# Frozen at registration (research_log.md RL-2026-07-26-21). Do not tune.
ON_DIRECTIONAL = {"regime": 0.50, "trend": 0.125, "dualrot": 0.075}
ON_NEUTRAL_POOL = 0.30
OFF_NEUTRAL_POOL = 0.60  # remainder is cash
STATIC_DIR_POOL = 0.50
STATIC_NEU_POOL = 0.50


def alloc(on: bool) -> dict[str, float]:
    """The frozen META capital map for a regime state. Fractions sum to 1 with CASH."""
    if on:
        a = dict(ON_DIRECTIONAL)
        a.update({s: ON_NEUTRAL_POOL / len(NEUTRAL) for s in NEUTRAL})
    else:
        a = {s: OFF_NEUTRAL_POOL / len(NEUTRAL) for s in NEUTRAL}
    a["CASH"] = round(1.0 - sum(a.values()), 12)
    return a


def static_alloc() -> dict[str, float]:
    """The frozen STATIC 50/50 baseline map (never re-mapped)."""
    a = {s: STATIC_DIR_POOL / len(DIRECTIONAL) for s in DIRECTIONAL}
    a.update({s: STATIC_NEU_POOL / len(NEUTRAL) for s in NEUTRAL})
    a["CASH"] = 0.0
    return a


def regime_now(refresh: bool = False) -> tuple[bool, pd.Timestamp]:
    """Current state from the deployed gate on cached ^NSEI/^INDIAVIX closes."""
    px = close_prices(load_yahoo_ohlcv(["^NSEI", "^INDIAVIX"], refresh=refresh))
    mkt = px["^NSEI"].dropna()
    vix = px["^INDIAVIX"].reindex(mkt.index).ffill()
    on = regime_on(mkt, vix)
    return bool(on.iloc[-1]), mkt.index[-1]


def _dedupe(rows: list[dict]) -> list[dict]:
    """Last row per panel_date, in date order (missed/repeated days: last wins)."""
    by_date: dict[str, dict] = {}
    for r in rows:
        if "panel_date" in r:
            by_date[r["panel_date"]] = r
    return [by_date[d] for d in sorted(by_date)]


def combine(rows: list[dict], sleeve_rets: dict[str, pd.Series]) -> pd.DataFrame | None:
    """Daily META and STATIC returns from recorded maps + sleeve forward returns.

    The return realized on day D uses the map from the latest row with
    panel_date STRICTLY BEFORE D (the causal lag). A sleeve with no return on D
    contributes 0 (cash). `meta_coverage` = the invested META weight actually
    priced that day.
    """
    rows = _dedupe(rows)
    if not rows or not sleeve_rets:
        return None
    change = pd.DatetimeIndex([pd.Timestamp(r["panel_date"]) for r in rows])
    rets = pd.DataFrame(sleeve_rets).sort_index()
    rets = rets.loc[rets.index > change[0]]
    if rets.empty:
        return None
    out = []
    for d, day in rets.iterrows():
        i = int(change.searchsorted(d, side="left")) - 1
        if i < 0:
            continue
        meta_map = rows[i]["meta_alloc"]
        stat_map = rows[i]["static_alloc"]
        avail = day.dropna()
        meta = float(sum(meta_map.get(s, 0.0) * r for s, r in avail.items()))
        stat = float(sum(stat_map.get(s, 0.0) * r for s, r in avail.items()))
        cov = float(sum(meta_map.get(s, 0.0) for s in avail.index))
        out.append((d, meta, stat, cov))
    if not out:
        return None
    return pd.DataFrame(out, columns=["date", "meta", "static", "meta_coverage"]).set_index("date")


def snapshot(write: bool = True, refresh: bool = False,
             path: str = META_SNAPSHOT_PATH) -> dict:
    """Record today's state and both frozen maps (one ledger row)."""
    on, panel_date = regime_now(refresh=refresh)
    row = {
        "hypothesis_ref": HYP, "kind": "meta_alloc_snapshot",
        "asof_ist": datetime.now(live_paper.IST).strftime("%Y-%m-%d %H:%M:%S"),
        "panel_date": str(panel_date.date()),
        "regime_state": "risk_on" if on else "risk_off",
        "meta_alloc": alloc(on), "static_alloc": static_alloc(),
    }
    if write:
        log_run(row, path=path)
    invested = 1.0 - row["meta_alloc"]["CASH"]
    print(f"[META-ALLOC] panel {row['panel_date']}  state={row['regime_state']}  "
          f"invested={invested:.0%} (cash {row['meta_alloc']['CASH']:.0%})")
    return row


def forward_record(cost_bps: float = 20.0, path: str = META_SNAPSHOT_PATH) -> pd.DataFrame | None:
    """META vs STATIC forward record from the sleeve ledgers (quiet per-sleeve calls)."""
    p = Path(path)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    if not rows:
        print("[META-ALLOC forward] no meta rows yet")
        return None
    sleeve_rets: dict[str, pd.Series] = {}
    for name, ledger in SLEEVES.items():
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                daily = live_paper.forward_track(path=ledger, cost_bps=cost_bps)
        except Exception:
            daily = None
        if daily is not None and not daily.empty:
            sleeve_rets[name] = daily["book"]
    if not sleeve_rets:
        print(f"[META-ALLOC forward] 0/{len(SLEEVES)} sleeves have >=2 forward days yet - "
              f"record starts accruing tomorrow")
        return None
    res = combine(rows, sleeve_rets)
    if res is None:
        print("[META-ALLOC forward] maps recorded but no return day follows the first row yet")
        return None
    cum_m = float((1.0 + res["meta"]).prod() - 1.0)
    cum_s = float((1.0 + res["static"]).prod() - 1.0)
    print(f"[META-ALLOC forward] {len(res)} day(s), {len(sleeve_rets)}/{len(SLEEVES)} sleeves "
          f"priced: META {cum_m:+.4%} vs STATIC {cum_s:+.4%} "
          f"(coverage last day {res['meta_coverage'].iloc[-1]:.0%} of invested weight)")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="RL-2026-07-26-21 META-ALLOC snapshot / forward record")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write the ledger")
    ap.add_argument("--forward", action="store_true", help="print the forward record only")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if args.forward:
        forward_record()
    else:
        snapshot(write=not args.dry_run, refresh=args.refresh)


if __name__ == "__main__":
    main()
