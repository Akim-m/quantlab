# Forward paper-track — REGIME snapshot (read-only)

Run this whenever you remember — any time after the 15:30 IST cash close, or during
the session for an intraday mark:

```
uv run python scripts/snapshot.py               # refresh Yahoo, snapshot, show record
uv run python scripts/snapshot.py --no-refresh  # skip the slow Yahoo pull
```

It rebuilds the current REGIME book, fetches **read-only** live Groww LTP for the held
names, records the book's move vs Nifty, appends one row to
`experiments/paper_trades.jsonl`, then prints the accumulated forward record. **It never
places a trade** — the only Groww method it calls is `get_ltp`, through a wrapper that
refuses order methods. It works from any working directory (it fixes its own paths and
UTF-8 console), so no env vars are needed.

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
