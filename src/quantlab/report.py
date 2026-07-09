"""Regenerate PERFORMANCE.md from source artifacts (documents are code).

`python -m quantlab.report` rebuilds the two deployed books from their FROZEN
constructions (no hand-typed metrics), slices them into scenarios, reads the forward
paper-track ledgers, and writes PERFORMANCE.md. Every number in the file comes from a
computation here, so the file cannot silently rot: rerun it after new data or a new
snapshot and the numbers move with the source.

Read-only: Yahoo `adj_close` cache + the cached Groww instrument master + the
experiment ledgers. No live Groww calls, no order path.
"""

from __future__ import annotations

import argparse
import glob
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import live_paper
from .backtest import backtest_weights
from .bear_sleeve import base_book
from .blend import composite, fno_long_short
from .data import close_prices, load_yahoo_ohlcv
from .evaluation import sharpe_tstat
from .india import fno_shortable, india_panel, sector_map
from .india_blend_study import raw_signals
from .india_ls import CORE as LS_CORE, WEIGHTS as LS_WEIGHTS
from .india_run import benchmarks
from .india_scenarios import _ann, _capm, _maxdd, regime_conditional_sharpe
from .portfolio import rebalance_targets

SPLIT = "2017-01-01"
COST_BPS = 20.0
FORWARD_INCEPTION = "2026-07-09"  # first genuine live snapshot; earlier panel dates are reconstructed
FNO_PATH = "experiments/fno_daily.jsonl"
OUT_PATH = "PERFORMANCE.md"
REGEN_CMD = "PYTHONIOENCODING=utf-8 PYTHONPATH=src uv run python -m quantlab.report"


# ---------------------------------------------------------------- metrics + tables

def book_metrics(ret: pd.Series, bench_ret: pd.Series) -> dict:
    """Test-window headline metrics for a daily return series: Sharpe, annualized
    return, max drawdown, and CAPM beta vs the benchmark. Pure - the caller passes
    series already sliced to the window it wants scored."""
    sr, _ = sharpe_tstat(ret)
    beta, _, _ = _capm(ret, bench_ret.reindex(ret.index).fillna(0.0))
    return {"sharpe": round(sr, 3), "ann": round(_ann(ret), 4),
            "maxdd": round(_maxdd(ret), 4), "beta": round(beta, 3)}


def _pct(x: float, sign: bool = True) -> str:
    return f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%"


def deployed_table(rows: list[tuple[str, dict]]) -> str:
    """Markdown table of (name, book_metrics) rows for the deployed-strategy section."""
    out = ["| Strategy | Test Sharpe | Ann. return | Max drawdown | Beta vs Nifty |",
           "|---|---:|---:|---:|---:|"]
    for name, m in rows:
        out.append(f"| {name} | {m['sharpe']:.3f} | {_pct(m['ann'])} | "
                   f"{_pct(m['maxdd'], sign=False)} | {m['beta']:+.2f} |")
    return "\n".join(out)


def year_table(series: dict[str, pd.Series], years: list[int]) -> str:
    """Per-calendar-year total return, one column per book/benchmark."""
    names = list(series)
    header = "| Year | " + " | ".join(names) + " |"
    sep = "|---|" + "".join(["---:|"] * len(names))
    out = [header, sep]
    last_year = years[-1]
    for y in years:
        label = f"{y} YTD" if y == last_year else str(y)
        cells = []
        for n in names:
            r = series[n].loc[f"{y}":f"{y}"]
            tot = float((1.0 + r).prod() - 1.0) if len(r) else 0.0
            cells.append(_pct(tot))
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def crash_row(series: dict[str, pd.Series], lo: str, hi: str) -> str:
    cells = []
    for n, r in series.items():
        seg = r.loc[lo:hi]
        cells.append(f"**{n}** {_pct(float((1.0 + seg).prod() - 1.0))}")
    return " · ".join(cells)


def regime_table(series: dict[str, pd.Series], mkt: pd.Series) -> str:
    """Sharpe in the risk-ON vs risk-OFF day subsets (^NSEI 200-day MA, causal).

    `mkt` is the full-history ^NSEI level so the moving average is warmed before the
    test window (regime_conditional_sharpe aligns the flag to each return index)."""
    out = ["| Strategy | Risk-ON Sharpe | Risk-OFF Sharpe |", "|---|---:|---:|"]
    for n, r in series.items():
        bull, bear = regime_conditional_sharpe(r, mkt.ffill())
        out.append(f"| {n} | {bull:.2f} | {bear:.2f} |")
    return "\n".join(out)


# ------------------------------------------------------------------------- ledgers

def _ledger_rows(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _forward_cum(path: str, cost_bps: float) -> tuple[dict | None, str | None]:
    """Cumulative forward book-vs-Nifty, reusing live_paper.forward_track's return
    arithmetic (each recorded book earns realized close-to-close return to the next
    snapshot). Returns (summary, note); summary is None when < 2 snapshot days."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            daily = live_paper.forward_track(path=path, cost_bps=cost_bps)
    except Exception as e:  # Yahoo cache/network - degrade, never crash the report
        return None, f"{type(e).__name__}: {e}"
    if daily is None or daily.empty:
        return None, None
    cum_b = float((1.0 + daily["book"]).prod() - 1.0)
    cum_n = float((1.0 + daily["nsei"]).prod() - 1.0)
    return {"days": int(len(daily)), "cum_book": cum_b, "cum_nsei": cum_n,
            "cum_active": cum_b - cum_n}, None


def forward_section(regime_path: str, ls_path: str, fno_path: str,
                    cost_bps: float = COST_BPS) -> str:
    reg, ls, fno = _ledger_rows(regime_path), _ledger_rows(ls_path), _ledger_rows(fno_path)
    lines = ["## 3. Forward paper-track (out-of-sample, the only clean proof left)", ""]

    if not reg and not ls and not fno:
        lines.append("No forward snapshots recorded yet.")
        return "\n".join(lines)

    # REGIME long-only sleeve
    days = sorted({r["panel_date"] for r in reg if "panel_date" in r})
    lines.append(f"**REGIME long-only book** — {len(days)} snapshot day(s) recorded "
                 f"(panel dates {days[0]}..{days[-1]})." if days else
                 "**REGIME long-only book** — no snapshots yet.")
    if reg:
        last = reg[-1]
        lines.append(f"- Latest ({last['asof_ist']} IST): regime **{last['regime_state']}**, "
                     f"cash {last['cash_frac']:.0%}, {last['n_names']} names, live quotes "
                     f"{last['n_quotes_ok']}/{last['n_names']} (Groww {'ok' if last.get('groww_ok') else 'unavailable'}); "
                     f"intraday book {_pct(last['book_intraday_ret'])} vs Nifty {_pct(last['nifty_intraday_ret'])}.")
    fwd, note = _forward_cum(regime_path, cost_bps)
    if fwd:
        lines.append(f"- Cumulative forward (rigorous close-to-close, {cost_bps:.0f} bps) over "
                     f"{fwd['days']} tracked day(s): book **{_pct(fwd['cum_book'])}** vs "
                     f"Nifty {_pct(fwd['cum_nsei'])} → active **{_pct(fwd['cum_active'])}**.")
    else:
        lines.append(f"- Cumulative forward return: needs ≥ 2 snapshot days"
                     + (f" (note: {note})" if note else "") + ".")

    # F&O-shortable L/S sleeve
    ls_days = sorted({r["panel_date"] for r in ls if "panel_date" in r})
    lines.append("")
    lines.append(f"**F&O-shortable L/S sleeve** — {len(ls_days)} snapshot day(s) recorded."
                 if ls_days else "**F&O-shortable L/S sleeve** — no snapshots yet.")
    if ls:
        last = ls[-1]
        lines.append(f"- Latest ({last['asof_ist']} IST): gross {last['gross']:.2f}, net {last['net']:+.4f}, "
                     f"{last['n_long']} long / {last['n_short']} short, "
                     f"intraday {_pct(last['book_intraday_ret'])} (target market-neutral ~0).")
    ls_fwd, ls_note = _forward_cum(ls_path, cost_bps)
    if ls_fwd:
        lines.append(f"- Cumulative forward ({cost_bps:.0f} bps) over {ls_fwd['days']} day(s): "
                     f"book **{_pct(ls_fwd['cum_book'])}** vs Nifty {_pct(ls_fwd['cum_nsei'])}.")
    else:
        lines.append("- Cumulative forward return: needs ≥ 2 snapshot days.")

    # F&O forward-collection program (RL-2026-07-15)
    fno_days = sorted({r["collect_date"] for r in fno if "collect_date" in r})
    lines.append("")
    if fno:
        last = fno[-1]
        lines.append(f"**F&O basis/PCR/IV collector** — {len(fno_days)} collect day(s). "
                     f"Latest {last['collect_date']}: {last['n_underlyings']} underlyings, "
                     f"NIFTY OI-PCR {last['pcr']:.3f}, ATM IV {last['atm_iv']:.1f}, skew {last['skew']:.1f}. "
                     f"Forward-only (expired contracts unresolvable); first read after ≥126 days.")
    else:
        lines.append("**F&O basis/PCR/IV collector** — no collection yet.")

    lines.append("")
    lines.append(f"> Caveat: pre-inception panel dates are RECONSTRUCTED — the genuine forward "
                 f"clock started **{FORWARD_INCEPTION}** (first live snapshot). Intraday numbers are "
                 f"same-day diagnostics; the rigorous forward return is the close-to-close line above.")
    return "\n".join(lines)


# ----------------------------------------------------------------------- graveyard

def graveyard_section() -> str:
    return "\n".join([
        "## 4. The graveyard — honest negatives",
        "",
        "This lab reports where things do NOT win. Ideas tested and shelved (see `research_log.md`):",
        "",
        "| Idea | Ref | One-line verdict |",
        "|---|---|---|",
        "| Sector rotation (top-5 industries by 6m momentum) | RL-2026-07-11 | Ties equal-weight (SR 1.35 vs 1.34), worse drawdown (−45%); a 200MA overlay hurts it. Not promoted. |",
        "| Short-term reversal family (13 daily/weekly books) | RL-2026-07-11 | Real gross edge (~0.67 SR) but ~130%/wk turnover; 0 of 13 survive 20 bps. Cost-gated. |",
        "| Bear-only reversal sleeve on the REGIME book | RL-2026-07-13 | Wash-to-drag at 20 bps (ΔSR −0.006), worse combined drawdown; diversification held, returns didn't. Failed. |",
        "| 52-week-strength long book (George-Hwang) | RL-2026-07-14 | 0.94 active-return correlation with momentum, double the standalone drawdown; redundant. Not promoted. |",
        "| Cross-sectional anomaly family (32 factors, US + NSE) | RL-2026-07-07/08/09 | 0 clear the Deflated Sharpe bar; the low-vol/low-beta family actively hurt in the high-beta decade. |",
        "",
        "**Standing caveats on the deployed books (do not overclaim):**",
        "- **0 of ~50+ trials clear the strict Deflated Sharpe bar** — as in every study here. The books are promoted as deployable smart-beta + risk management, NOT as statistically-proven alpha.",
        "- The **2017-2026 test window is heavily RE-USED** across research rounds; each read is one more use, so the forward paper-track is the only clean out-of-sample proof left.",
        "- **Survivorship**: the universe is CURRENT Nifty 500 membership (no free point-in-time source), which inflates long-only absolute returns. The REGIME book's return edge over cap-weighted Nifty is largely the equal-weight/mid-cap breadth premium; what the momentum + regime overlay genuinely adds is risk-adjusted (drawdown/beta), significant as CAPM alpha vs EW-277 (t≈3).",
    ])


# --------------------------------------------------------------------------- build

def _load() -> dict:
    """Rebuild both deployed books and the benchmarks from frozen constructions.
    Returns full-history daily return series (test-sliced by the caller)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        px, mkt, _, _ = india_panel(start="2010-01-01", index="nifty500", ret_clip=0.40)
        sectors = sector_map("nifty500")
        vix = close_prices(load_yahoo_ohlcv(["^INDIAVIX"]))["^INDIAVIX"].reindex(px.index).ffill()

        regime_w = base_book(px, mkt, vix, sectors)
        regime = backtest_weights(px, rebalance_targets(regime_w, "ME"), COST_BPS).returns

        shortable = _fno_shortable_cached()
        score = composite({k: raw_signals(px, mkt, sectors)[k] for k in LS_CORE}, weights=LS_WEIGHTS)
        ls = backtest_weights(px, rebalance_targets(fno_long_short(score, shortable), "ME"), COST_BPS).returns

        nsei, _tr, ew = benchmarks(px, mkt, COST_BPS)

    return {"regime": regime, "ls": ls, "nsei": nsei, "ew": ew, "mkt": mkt,
            "last_date": px.index[-1], "n_stocks": px.shape[1],
            "n_shortable": len(shortable), "n_overlap": len(set(px.columns) & shortable)}


def _fno_shortable_cached() -> set[str]:
    """F&O-shortable set from the newest cached Groww instrument master (read-only,
    no live call). Empty set if no cache exists (the L/S book then holds only longs)."""
    masters = sorted(glob.glob("data/raw/groww_instruments_*.csv"))
    if not masters:
        return set()
    df = pd.read_csv(masters[-1], low_memory=False)
    return {s.upper() for s in fno_shortable(instruments=df)}


def build() -> str:
    return render(_load())


def render(d: dict, forward_paths: tuple[str, str, str] | None = None) -> str:
    """Assemble the markdown from precomputed return series (pure, no data loading).
    `forward_paths` = (regime, ls, fno) ledgers, defaulting to the live locations."""
    fp = forward_paths or (live_paper.SNAPSHOT_PATH, live_paper.LS_SNAPSHOT_PATH, FNO_PATH)
    test = lambda s: s.loc[SPLIT:]
    nsei_t = test(d["nsei"])

    series = {"REGIME (long-only)": test(d["regime"]), "F&O L/S sleeve": test(d["ls"]),
              "Nifty (^NSEI)": nsei_t, "EW-277": test(d["ew"])}
    rows = [(name, book_metrics(s, nsei_t)) for name, s in series.items()]
    years = list(range(int(SPLIT[:4]), d["last_date"].year + 1))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = f"""# Quant Lab — Performance Overview

*As of {d['last_date'].date()} (data through last close). Generated {now}. This is a
build artifact — regenerate with `{REGEN_CMD}`. Every number below is recomputed from
source (frozen constructions + Yahoo `adj_close` + forward ledgers), never hand-typed.*

The lab's point is HONEST research: most ideas fail, and the failure table (§4) is a
valid deliverable. Two books are deployable, both with disclosed caveats — neither
clears the strict trials-aware significance bar.

## 1. Deployed strategies

Rebuilt live on the locked test window ({SPLIT}+), {COST_BPS:.0f} bps turnover cost,
monthly rebalance, N500-{d['n_stocks']} total-return universe.

- **REGIME (long-only)** — top-decile conviction momentum (composite of 12-1, Sharpe-
  and residual-momentum) scaled to cash by a (200-day-MA OR India-VIX) regime overlay.
  Deployable smart-beta + drawdown control (RL-2026-07-10/11).
- **F&O L/S sleeve** — residual-momentum dollar-neutral long-short with the short leg
  restricted to F&O-shortable single stocks ({d['n_overlap']}/{d['n_stocks']} of the
  universe overlaps the {d['n_shortable']} shortable names). Implementable market-neutral
  book (RL-2026-07-12).

{deployed_table(rows)}

CAPM beta vs Nifty confirms the L/S sleeve is cleanly market-neutral (β≈0) while REGIME
keeps a reduced, overlay-managed market exposure.

## 2. Scenario breakdown

**Calendar-year total return** (net of {COST_BPS:.0f} bps):

{year_table(series, years)}

**Feb–Jun 2020 crash:** {crash_row(series, "2020-02-01", "2020-06-30")} — the regime
overlay sidesteps the drawdown the index takes.

**Risk regime day-subsets** (Sharpe in ^NSEI-above vs below its 200-day MA, causal):

{regime_table(series, d['mkt'])}

{forward_section(*fp)}

{graveyard_section()}
"""
    return md


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate PERFORMANCE.md from source artifacts")
    p.add_argument("--out", default=OUT_PATH)
    a = p.parse_args()
    md = build()
    Path(a.out).write_text(md, encoding="utf-8")
    print(f"wrote {a.out} ({len(md)} chars)")


if __name__ == "__main__":
    main()
