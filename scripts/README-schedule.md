# Forward paper-track — daily REGIME snapshot (read-only)

`paper_snapshot.cmd` runs the RL-2026-07-10 live paper harness once: it rebuilds the
current REGIME book, fetches **read-only** live Groww LTP for the held names, records
the book's move vs Nifty, and appends one row to `experiments/paper_trades.jsonl`.
Console/errors go to `experiments/paper_snapshot.log`. **It never places a trade** —
the only Groww method it calls is `get_ltp`, through a wrapper that refuses order
methods.

Requires the Groww `API_KEY`/`API_SECRET` in `.env` (git-ignored) and network. Uses
`--refresh`, so it re-pulls the Yahoo panel first (~a few minutes).

## Register the daily task (you run this — it needs your authorization)

Runs 13:30 local (= 16:00 IST, after the 15:30 cash close). In a terminal:

```
schtasks /create /tn "QuantLab-REGIME-paper-snapshot" ^
  /tr "C:\Users\aydhi\OneDrive\Documents\ay\quant\quantlab\scripts\paper_snapshot.cmd" ^
  /sc DAILY /st 13:30 /f
```

## Manage it

```
schtasks /run    /tn "QuantLab-REGIME-paper-snapshot"   REM run once now to test
schtasks /query  /tn "QuantLab-REGIME-paper-snapshot"   REM status
schtasks /delete /tn "QuantLab-REGIME-paper-snapshot" /f  REM remove
```

Or run a snapshot manually any time:
`PYTHONIOENCODING=utf-8 PYTHONPATH=src uv run python -m quantlab.live_paper --refresh`

## Honest caveats

- The book is monthly-rebalanced with slow (12-1) momentum signals, so the recorded
  same-day `book_intraday_ret` is a reasonable daily diagnostic; the **rigorous** forward
  return is the book decided on day D realized over D+1 — read it from the snapshot
  sequence, not a single row. (A weight-logging upgrade for exact forward attribution is
  a small follow-up if you want it.)
- Read-only only. This track record is the cleanest test of the strategy — it is genuine
  out-of-sample, immune to the survivorship / window-reuse caveats in `research_log.md`.
