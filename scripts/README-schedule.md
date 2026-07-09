# Forward paper-track — REGIME + F&O L/S snapshot (read-only)

Run this whenever you remember — any time after the 15:30 IST cash close, or during
the session for an intraday mark:

```
uv run python scripts/snapshot.py               # refresh Yahoo, snapshot, show record
uv run python scripts/snapshot.py --no-refresh  # skip the slow Yahoo pull
```

One run has **6 ledger-feeding legs then 4 forward reports**. It snapshots **all four**
paper books — the long-only REGIME book (ledger `experiments/paper_trades.jsonl`), the
RL-2026-07-12 F&O-shortable market-neutral L/S sleeve
(`experiments/paper_trades_ls.jsonl`), the RL-2026-07-17 multi-asset trend sleeve
(`experiments/paper_trades_trend.jsonl`), and the RL-2026-07-16 gold_lowbeta risk-off
variant (`experiments/paper_trades_gl.jsonl`) — then runs the RL-2026-07-15 F&O daily
collector (single-stock basis + NIFTY PCR/IV/skew, one row/day to
`experiments/fno_daily.jsonl`), archives 5-minute NSE bars before Groww expires them
(RL-2026-07-25 intraday collector, NIFTY + nifty100 → per-symbol CSVs under
`data/raw/intraday/`, one coverage row/run to `experiments/intraday_archive.jsonl`, and pushes
the archive to the private quantlab-intraday repo) and
marks the RL-2026-07-18 paper NIFTY short-straddle (daily mark/roll rows to
`experiments/paper_options.jsonl`); finally it prints the four accumulated forward
records (one per snapshot ledger). For each book it rebuilds the
current target weights via that study's frozen construction, fetches **read-only** live
Groww LTP for the held names, records the book's intraday move, and appends one row to
that book's ledger. Only the first (REGIME) leg pulls Yahoo when `--refresh` is set; the
other three reuse the warm panel / ETF / gold caches on disk, so a full pull happens at
most once. A failure in any leg is printed and the run continues, so one book's hiccup
never costs the others their snapshot. **It never places a trade** — the only Groww
method it calls is `get_ltp`, through a wrapper that refuses order methods. It works from
any working directory (it fixes its own paths and UTF-8 console), so no env vars are needed.

Requires the Groww `API_KEY`/`API_SECRET` in `.env` (git-ignored) and network. The
default `--refresh` re-pulls the Yahoo panel first (~a few minutes); `--no-refresh`
uses the warm cache. Missed or irregular days are fine — the book drifts between
snapshots and the forward track uses the last row per panel date.

## Optional: register a daily Scheduled Task instead

If you'd rather it run unattended, `paper_snapshot.cmd` wraps the same harness with
`--refresh` and logs to `experiments/paper_snapshot.log`. Register it to run 13:30
local (= 16:00 IST, after the cash close):

```
schtasks /create /tn "QuantLab-REGIME-paper-snapshot" ^
  /tr "C:\Users\ahmad\Downloads\ay\quantlab\scripts\paper_snapshot.cmd" ^
  /sc DAILY /st 13:30 /f
```

## Manage it

```
schtasks /run    /tn "QuantLab-REGIME-paper-snapshot"   REM run once now to test
schtasks /query  /tn "QuantLab-REGIME-paper-snapshot"   REM status
schtasks /delete /tn "QuantLab-REGIME-paper-snapshot" /f  REM remove
```

## Honest caveats

- The same-day `book_intraday_ret` is a daily diagnostic only. The **rigorous** forward
  return (book decided day D, realized close D→D+1, costed) is what the printed forward
  record shows — snapshots log the held weights, and `live_paper --forward` recomputes
  the record from them at any time. Days whose panel_date predates the ledger's first
  snapshot are reconstructed (causal, but not contemporaneous evidence).
- Read-only only. This track record is the cleanest test of the strategy — it is genuine
  out-of-sample, immune to the survivorship / window-reuse caveats in `research_log.md`.
