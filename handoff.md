# HANDOFF — Indian equity strategy research (quantlab)

For the next agent picking up this work. Read `CLAUDE.md` (conventions),
`protocol.md` (honesty rules incl. the bar-idiom lesson), and `research_log.md`
(RL-2026-07-10 → RL-2026-07-25 — the full study record) first; `progress.md` is the
running state log. This file is the operating setup + current state + what to do next.

---

## 1. How to operate (the setup this work used — keep it)

- **Subagent roles:** **Fable** orchestrates/plans/verifies; **Opus** subagents execute.
  **Never more than 3 concurrent subagents.** The orchestrator re-verifies EVERY
  delegated decision number independently in the scratchpad before it enters the
  record (this has caught an inflated DSR, a wrong coverage denominator, a silent
  empty-book skip, and adjudicated a decisive data repair).
- **Pre-register before running** (RL-YYYY-MM-DD id in `research_log.md`): hypothesis,
  locked sample, locked variants (≤4), TRAIN-only design freeze, predicted outcome,
  and a deployment bar **stated in the thesis's own idiom** (Sharpe-difference for
  risk-adjusted claims — receipt: RL-2026-07-19's mis-specified bar). One test read.
  Executors do NOT write research_log/progress — the orchestrator fills results after
  verification.
- **Commit + push after every verified milestone** to the **`fork` remote**
  (github.com/Akim-m/quantlab; `origin` is the read-only upstream — pushing there
  403s). Before each commit: grep staged content for secret VALUES; never stage .env.
- **Groww trade API = DATA ONLY, read-only.** `groww_client.call()` refuses order
  methods; ≤7 req/s enforced by the shared RATE limiter; `get_ltp` batches ≤50.
  Secrets in `.env` (git-ignored, present on this machine); never read/print them.
- **The 2017–2026 test window is heavily RE-USED (~80+ trials).** Strict DSR verdict
  lab-wide: FAILS. New claims should lean on the forward ledgers or the locked
  forward evaluations, not more hold-out reads.

## 2. Environment & checks

- **uv**; dev deps `uv sync --extra dev`. Always `PYTHONIOENCODING=utf-8` (cp1252
  console) and `PYTHONPATH=src` for `-m` runs.
- Tests: `uv run python -m pytest -q` — **224+ passing** (RL-23/24/25 in flight add
  more). Keep green before "done".
- **Data split (owner-confirmed, do not relitigate):** Yahoo `adj_close` = ONLY
  backtest price source (Groww candles are split-adjusted but NOT dividend/demerger-
  adjusted — measured: RELIANCE +8.7% gap = Jio demerger, ITC +3.7% = hotels).
  Groww = live quotes, ALL F&O data, instrument master, intraday, and the ARBITER
  for disputed prints (receipt: RL-17's fabricated 2019-12 Yahoo ETF prints).
  Yahoo ETF series MUST pass `xasset_trend.clean_prices` (transient-spike guard).
- Machine: Windows 11, TZ = UTC+3 (IST = local +2:30). NSE cash close 15:30 IST.

## 3. Research state (13 studies resolved 2026-07-09; see research_log for detail)

**Deployed / promoted:**
- **REGIME long-only** (RL-10/11): top-decile conviction momentum + (200MA OR
  India-VIX) overlay. Test SR 1.865, +35.7%/yr, maxDD −27.2%. Currently risk-OFF.
- **F&O L/S sleeve** (RL-12): resid-mom dollar-neutral, shorts ∩ 210 F&O names.
  SR 0.846, β −0.000, maxDD −16.3%. Shortability costs only 0.02 Sharpe.
- **Multi-asset trend sleeve** (RL-17, promoted diversifier): tsmom+invvol on
  NIFTYBEES/JUNIORBEES/BANKBEES/GOLDBEES/MON100. SR 1.057, corr(REGIME) 0.357.

**Graveyard (do not re-test without new data):** bear-only reversal sleeve (RL-13);
52w-strength (RL-14, 0.94 corr with momentum); risk-off low-beta/gold sleeve (RL-16,
breaches DD cap / gold thesis wrong); three-book blend vs its return-level bar
(RL-19 — frontier alternative, not upgrade); turn-of-month (RL-20, t=1.13);
vol-target overlay (RL-21, binary overlay already owns vol-timing); index band
mean reversion (RL-23, LW z −2.34 — significantly WORSE than B&H; knife-catching);
VIX rebound re-entry (RL-24, harvests recovery return but breaches the DD cap,
z≈0 on 13 episodes); ALL single-stock short-term reversal (RL-11/13, cost-gated);
sector rotation (RL-11). Three overlay studies (RL-16/21/24) all failed to improve
the deployed binary gate — treat further overlay tweaks as low-prior.

**Forward programs (evidence accrues daily, first reads locked):**
- RL-15 F&O collector (basis ~210 names, NIFTY PCR/ATM-IV/skew) — read at 126 days.
- RL-18 paper short straddle (NIFTY weekly ATM, LTP marks) — read at 126 days.
- RL-22 forward evals: E1 trend keeps promotion; E2 gold_lowbeta vs deployed
  (Sharpe-diff z>1 + maxDD); E3 invvol blend from ledgers (Sharpe-diff z>1).
- RL-25 intraday 5-min bar archive (NIFTY + nifty100) — LIVE since 2026-07-09
  with the full ~87-day retention window rescued (454,492 bars). Groww retains
  only ~90 trailing days, so the archive is the ONLY path to future ORB/VWAP
  studies (own registrations, no read before ≥12 months of bars, realistic
  intraday costs). Durability: `data/raw/intraday/` is its own git repo pushing
  to the PRIVATE github.com/Akim-m/quantlab-intraday (auto-commit+push is a
  daily snapshot leg); the main repo still git-ignores the directory.

## 4. Daily operation (the owner's one command)

`uv run python scripts/snapshot.py` (any cwd, no env vars; `--no-refresh` to skip
the Yahoo pull). Runs guarded legs feeding SIX ledgers under `experiments/`:
`paper_trades.jsonl` (REGIME), `paper_trades_ls.jsonl` (L/S),
`paper_trades_trend.jsonl`, `paper_trades_gl.jsonl` (RL-16-flagged variant),
`fno_daily.jsonl`, `paper_options.jsonl` (+ `intraday_archive.jsonl` audit once
RL-25 lands), then prints the forward records. Missed days are safe (books drift;
last row per panel date wins). Ledgers are committed; `data/raw/` bulk is not —
the intraday archive lives ONLY on this machine (back it up externally).
`PERFORMANCE.md` is generated: `uv run python -m quantlab.report` (never hand-edit).

## 5. Open threads (ranked)

1. **In flight (verify → fill log → commit when they land):** RL-2026-07-23 band
   mean reversion (band_mr.py), RL-2026-07-24 VIX rebound overlay (vix_rebound.py),
   RL-2026-07-25 intraday archive (intraday_collect.py + snapshot leg + first
   ~90-day pass).
2. **Keep the daily snapshot running** — the forward ledgers are the lab's only
   clean evidence; every missed day is evidence lost.
3. **~Jan 2027:** the locked first reads (RL-15/18/22 E1-E3) — evaluate exactly as
   registered, BH-FDR across the family, no peeking before the mark.
4. **Paid-data wishlist (owner's wallet):** point-in-time index membership
   (survivorship — the honest ceiling on all long-only claims), earnings
   dates+surprises (PEAD), historical options/intraday depth.
5. Idea triage notes: fundamentals/PEAD data-blocked; ORB/VWAP wait on the archive;
   anything resembling the graveyard needs NEW data to justify a re-test.

## 6. Files

- `research_log.md` — the study record (truth). `protocol.md` — honesty rules.
- `PERFORMANCE.md` — generated overview. `progress.md` — running state log.
- `src/quantlab/`: `india_run.py`/`blend.py` (REGIME construction),
  `india_ls.py` (L/S), `xasset_trend.py` (trend sleeve + `clean_prices` guard +
  `base_returns`), `riskoff_sleeve.py` (RL-16 + gl-variant construction),
  `blend_portfolio.py` (RL-19), `tom_study.py`, `volmgmt_study.py`, `h52_study.py`,
  `bear_sleeve.py`, `live_paper.py` (all paper-tracks + forward_track),
  `fno_collect.py`, `paper_options.py`, `report.py`, `groww_client.py` (read-only).
- `experiments/log.jsonl` — run ledger (every study row carries hypothesis_ref).
