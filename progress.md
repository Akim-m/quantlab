# Progress

Running log of research + build state. Newest first.

## 2026-07-10 (later) — wave 4: backtest-anchored additions (RL-26-08/-19/-20)

- Owner asked "are you backtesting on the available data too? work on more
  strategies" → answer: yes, everything promotable was (TRAIN design → one locked
  2017+ read), but the hold-out is exhausted (~92 trials), so wave 4 takes the
  DUAL-ROT route: TRAIN design reads on available data + forward-only live books,
  ZERO new hold-out spends. Owner also set a global hard rule this session (never
  work on an unverified assumption; verify through implementation) → OPUS.md §1.
- Probes BEFORE registration: `INR=X`/`BZ=F` clean+deep → -08 unblocked; NSE legacy
  MTO delivery archive serves ≥2011 (modern bhavcopy only ~2020+) → new non-price
  source for -20. Pre-registered -19/-20 + wave header, committed before any run.
- METHOD: Fable orchestrator + 3 parallel Opus executors; orchestrator pre-computed
  independent reference numbers BEFORE executor reports (RELIANCE illiq chain, 4
  names' macro-betas via separate lstsq impl, hand-parsed 2013 MTO line), then
  verified every decision number. All three verified exact.
- DONE: **-08 MACRO-BETA LIVE** (forward-only) — bivariate 252d OLS on lagged
  USDINR/Brent returns; TCS betas match independent impl to 4dp; causality test
  proven load-bearing; corr vs VOL-SHOCK +0.086. First row 27L/27S, −0.40%.
- DONE: **-19 ILLIQ LIVE** — TRAIN 2010-16 froze L63 (net SR 1.305@40bps vs 1.239;
  long leg ~5th liquidity pctile flagged; survivorship-inflated design read, NOT
  the claim). SPLPETRO chain exact vs raw-CSV re-derivation; corr vs VOL-SHOCK
  −0.245. First row 27L/27S, +0.34%.
- DONE: **-20 DELIV LIVE** — 1,522 MTO files backfilled (0 errors; 2017-2025
  hold-out never downloaded, physically absent); QC 0.000000% vs modern bhavcopy;
  TRAIN 2011-16 froze LEVEL (0.914, t 2.38 @20bps); SHOCK + SIGNED-SHOCK are
  design-stage NEGATIVES (recorded, count as trials). ABBOTINDIA chain exact to
  6dp via independent parser; corr vs ILLIQ −0.214 (registered confound absent).
  First row 27L/27S, −0.72%.
- Snapshot now runs 11 sleeve legs + MTO collector; suite 413 green (366 + 47 new).
- Lesson worth keeping (executor catch): volshock's dense-ffill weight grid would
  make a BACKTEST trade daily — backtesting sleeves must use the ME-sparse grid
  (h52 pattern). Forward-only sleeves unaffected.
- Family tally: hold-out still ~92 (zero new reads); +5 TRAIN design trials logged
  (-19 ×2, -20 ×3).

## 2026-07-10 — owner-directed autonomous new-strategy wave (RL-2026-07-26 family)

- SETUP: daily snapshot run for 2026-07-10 (REGIME leg recovered after a transient
  Yahoo connection reset; forward record 4 days book −0.42% vs Nifty −1.57%, active
  +1.15%). Intraday storage repo `quantlab-intraday` pulled/cloned and today's +5,754
  bars pushed (99e2e37..7df33a4). Six ledgers committed. F&O live 2026-07-10: NIFTY
  PCR 1.23, ATM IV 10.1.
- METHOD: Fable researcher + Opus auditor (read-only) → orchestrator reconcile →
  Opus implementers → orchestrator independent re-derivation. ≤3 subagents, never
  Sonnet/Haiku. Recovery loop (270s) as a rate-limit backstop. Commit+push per milestone.
- REGISTERED 12 candidates (RL-2026-07-26-01..12), only -02 spends a hold-out read
  (family ~92); rest FORWARD-ONLY or BLOCKED-PENDING-DATA. 7 ideas rejected with receipts.
- DONE: **RL-2026-07-26-02 US-GATE** — ^GSPC US-close gate on NIFTYBEES: FAILED the bar
  (verified to the digit by independent reconstruction; no-look-ahead guard proven
  load-bearing). Test SR 1.043 vs B&H 0.935 but LW z +0.397@10bps / +0.190@20bps (≪1);
  maxDD −20.8% vs −36.3%. Cross-market index timing retired; adds over the local gate but
  doesn't clear. Honest negative (~30% prior). 5 tests.
- DONE: **RL-2026-07-26-01 DUAL-ROT** — 5-ETF dual-momentum rotation, FORWARD-ONLY, LIVE.
  Frozen K2/tsmom (TRAIN 0.693, argmax verified independently); first row 2026-07-09
  GOLDBEES 50% + MON100 50% (12-1 rank verified from primitives). Own ledger
  `paper_trades_dualrot.jsonl` + snapshot leg. First locked read ~2027-07. 20 tests.
- SUITE: full pytest 266 green. Two seed ideas (overnight drift, single-stock TSMOM)
  rejected in wave 1 with receipts. Data-integrity flags surfaced for owner (see below).
- FOR OWNER: (1) RL-2026-07-11 has zero ledger rows — deployed SR 1.865 headline not
  reproducible from experiments/log.jsonl; (2) duplicate REGIME rows with dsr 0.999 vs
  0.01; (3) PERFORMANCE.md §4 "0 of ~50+ trials" stale (family ~92). Not touched — owner
  call.

## 2026-07-09 (evening) — RL-23/24/25 wave

- DONE: RL-2026-07-23 — index band mean reversion: FAILED DECISIVELY (verified
  to the digit + independent z cross-check). Frozen (k=2, mean-touch); test SR
  0.25 vs B&H 0.94; LW z −2.34 (significantly WORSE than B&H); maxDD −36.5% at
  16.8% time invested — uncapped mean-touch exits ride falling knives.
  Graveyard, stronger verdict than turn-of-month.
- DONE: RL-2026-07-24 — VIX rebound overlay: FAILED (verified via independent
  reconstruction, exact). Frozen p90/h10; recovery harvest REAL (+6.3pts/yr
  return) but maxDD worsens 2.4pts and LW z ≈ 0 — re-entering receding panic
  buys the recovery AND its residual drawdown. Third overlay study confirming
  the deployed binary gate is hard to improve. 13-episode power disclosure.
- DONE: RL-2026-07-25 — intraday 5-min bar archive LIVE. First pass rescued
  the full surviving window: 101/101 symbols, 454,492 bars, 2026-04-13→07-09
  (60 sessions × exactly 75 bars; orchestrator spot-verified). Retention floor
  measured ~87 days; first pass ~27 min (latency-bound), dailies ~3-4 min.
  Durability per owner: PRIVATE repo github.com/Akim-m/quantlab-intraday
  (data/raw/intraday is its own git repo; auto-push wired as a snapshot leg).
  ORB/VWAP registrations unlock ≈2027-07.

## 2026-07-09 (close) — forward program v2 (RL-2026-07-22)

- DONE: trend sleeve + gold_lowbeta variant paper-tracks LIVE (orchestrator
  verified recorded weights reproduce the frozen constructions exactly).
  First rows 2026-07-09: trend held 3/5 ETFs (gross 0.50, +0.31% intraday);
  gl variant risk-off book gross 1.0 incl. 50% GOLDBEES leg, +0.55% vs Nifty
  +0.34%. snapshot.py = 6 ledger legs + 4 forward reports, SIX ledgers total.
  Locked forward evaluations E1/E2/E3 (Sharpe-difference idiom per the RL-19
  lesson) first-read ~Jan 2027 alongside RL-15/18. Suite 224.
- Day summary (2026-07-09, Nifty +0.74%): REGIME +0.11% (risk-off, by
  design); L/S +0.49% day one (spread, not direction); straddle day-one
  −315 after an intraday −884; collector PCR 0.876→0.791 intraday.

## 2026-07-09 (late) — RL-19 blend + new study wave

- DONE: RL-2026-07-19 — three-book blend: FAILED the locked bar (verified to
  the digit), and the bar itself was MIS-SPECIFIED — a risk-reduction thesis
  judged by a return-level paired-t is unpassable by construction. Lesson
  graduated to protocol.md. Honest reading: blend SR 1.78 ≈ REGIME 1.87 at
  HALF the drawdown (−12.7% vs −27.2%) and a third of the return — a frontier
  alternative, not an upgrade; formal risk-adjusted comparison deferred to a
  future forward-data registration. Verdict binds: deploy=FALSE.
- Orphans resolved: ensemble.py/test_ensemble.py archived into git history
  (d3d1481) then removed; tree clean.
- DONE: RL-2026-07-20 — turn-of-month: NEGATIVE as predicted-likely (verified
  to the digit). Effect real in sign (9.4 vs 4.2 bps/day) but t=1.13 (ns);
  TOM book SR 0.44 vs B&H 0.94 at 10 bps, collapses at 20. Graveyard.
- DONE: RL-2026-07-21 — vol-managed overlay: NEGATIVE (verified to the digit).
  Frozen 10%/63d; scaled SR 1.737 < baseline 1.865, paired-t −5.4; maxDD
  better (+6.7pts) but ~17pts/yr return given up; cost drag NEGATIVE so costs
  exonerated — the binary overlay already harvests vol-timing. Graveyard.
  Wave complete: 12 studies resolved (2 promotions, 8 negatives, 2 forward
  programs), family tally ~80 trials. `handoff.md` is the operating
setup; `research_log.md` is the study record (truth).

## 2026-07-09 — session: forward paper-track hardening (handoff thread #1)

- Reconciled a divergent prior session: the old `progress.md` here described an
  "RL-2026-07-10 ensembles of the 32 factors" workstream (`ensemble.py`,
  `factor_study.run_ensembles`) that never landed — `run_ensembles` does not exist in
  this tree. The committed handoff/research_log RL-2026-07-10/11 (Indian single-stock
  study) is the real record. Orphans `src/quantlab/ensemble.py` +
  `tests/test_ensemble.py` left untracked pending owner decision.
- This machine lacked the git-ignored data caches from the handoff machine:
  fetched `data/raw/ind_nifty500list.csv` (fixes the env-dependent sector_map test),
  warming the 2010+ nifty500 Yahoo panel. `.env` (Groww keys) MISSING here — live
  quotes unavailable until the owner restores it; `live_paper` degrades gracefully.
- DONE: `live_paper` upgrade (thread #1) — every snapshot logs held WEIGHTS;
  `--forward` computes the rigorous forward record (book decided day D, realized
  close D→D+1, costed via `backtest_weights`, vs ^NSEI from track start). Opus
  executed; orchestrator hand-verified 4 book days on real prices to 1e-12.
  Edge cases locked by tests: dedupe last-per-day, legacy rows skipped, explicit
  all-cash `{}` book rebalances to cash (test proven to fail on unfixed code),
  unpriced symbols dropped WITHOUT renormalizing. Suite 163 green.
- DONE: Groww coverage audit (owner's account, read-only probes) — CORRECTS the
  lab's "~2020+ unadjusted" note: CASH daily to ~2002-07, split/bonus-adjusted,
  NOT dividend/demerger-adjusted; intraday depth ~90d. Live LTP entitlement OK.
  Propagated to CLAUDE.md, handoff.md, research_log.md addendum.
- DONE: first live snapshots on this machine — 2026-07-09, risk_off, 50% cash,
  55 names, 55/55 quotes both runs. First forward numbers: book −0.41% vs Nifty
  −1.43% (active +1.03%) over 3 tracked days, cost drag 0.11%. HONESTY CAVEAT:
  both snapshots were taken 2026-07-09, so the 07-07/07-08 legs are RECONSTRUCTED
  (causal but backfilled) — genuine forward evidence starts 2026-07-09. Ledger
  carries the audit trail (asof_ist vs panel_date). TODO (small): forward_track
  should annotate days whose panel_date predates ledger inception as backfilled.
  Owner still needs to register the daily Windows task (`scripts/paper_snapshot.cmd`).
- DONE: RL-2026-07-12 — implementable market-neutral resid-mom L/S, F&O-only
  shorts (210 shortable, 130/277 overlap). SURPRISE RESULT (verified by
  orchestrator to the digit): the constraint costs only 0.020 Sharpe (0.866 →
  0.846), ann/maxDD IMPROVE (+5.38%, −16.3%), beta −0.000; corr with REGIME
  book +0.371; DSR 0.000 (fails strict bar, as everything here does). Carry
  credit sensitivity: +3%/yr → SR 1.08. TRAIN-frozen THIN short leg (FILL
  rejected — contaminates the long leg). Live L/S paper-track started on its
  own ledger (`paper_trades_ls.jsonl`, 196/196 quotes). Suite 168 green.
- DONE: RL-2026-07-14 — 52w-strength book: NEGATIVE as predicted (verified by
  orchestrator to the digit). off_low frozen on train; test H52 SR 1.57 vs MOM
  1.87, active-corr 0.944, blend fails both promotion bars (1.776 < 1.865,
  paired-t +0.58 < 1). Redundant with momentum; not promoted. Receipt: the
  train-rejected gh_high looked better on test but with negative paired-t —
  freeze prevented test-peeking.
- DONE: PERFORMANCE.md — generated overview (`python -m quantlab.report`):
  deployed books rebuilt live (REGIME 1.865, L/S 0.846 — reproduce logged
  values exactly), per-year + 2020-crash + risk-on/off slices, forward-track
  status, graveyard. Documents-are-code: numbers recomputed, never typed.
- DONE: RL-2026-07-16 — risk-off sleeve: FAILED the bar (near-miss), verified
  to the digit. TRAIN froze lowbeta; test: SR 1.865→1.913, paired-t robustly
  >1, but maxDD worsens 2.2–4.8pts — breaches the 2pt cap at every cost. Book
  keeps CASH in risk-off. Gold-only HURT (stress-bid thesis wrong). 50/50
  gold+lowbeta would have passed but wasn't the train winner — refused
  post-hoc switch (protocol). Flagged for future forward-data confirmation.
  Ledger wart: 3 stale RL-16 rows (frozen_variant=gold placeholder);
  authoritative rows = frozen_variant=lowbeta.
- DONE: RL-2026-07-17 — multi-asset ETF trend sleeve: **PROMOTED** (verified
  to the digit). Frozen tsmom+invvol; test SR 1.057, +11%/yr, maxDD −27.9%,
  corr(REGIME) 0.357, corr(L/S) 0.137 — clears both bars. Decision-critical
  data repair independently adjudicated via GROWW: Yahoo had fabricated
  decimal-shift prints 2019-12-19/20 on the ETFs (129→13, 33.6→0.34); Groww
  shows ~130/33.6 → prints false, causal spike-filter repair justified.
  Follow-up (not yet registered): portfolio-blend study REGIME+L/S+trend.
- DONE: RL-2026-07-18 — paper NIFTY short-straddle harness LIVE (verified:
  ledger arithmetic hand-checked). First position 2026-07-09: short 24000
  straddle exp 07-14, credit 264.65 x65; first mark −884 (premium rose on the
  day's dip — the left tail, honestly recorded from day one). Evaluation
  locked at 126 days. `scripts/snapshot.py` now runs SIX guarded legs feeding
  FOUR ledgers (REGIME, L/S, fno_daily, paper_options). Suite 215.
- DONE: RL-2026-07-15 — F&O forward-collection program. Measured: expired
  contracts unresolvable → basis/PCR/IV are FORWARD-ONLY; hypotheses (H1 basis
  x-section, H2 PCR extremes, H3 IV skew) pre-registered BEFORE day-one data;
  first read locked at 126 collection days. Collector `fno_collect.py` live:
  day-one row 2026-07-09 — 210/210 cash, 210/210 fut1, 208 fut2, chain OK
  (NIFTY PCR 0.876, ATM IV 11.9, skew +4.0; median ann. basis +5.0%/yr).
  Orchestrator re-verified stored basis arithmetic (matches at stored 4-dp
  precision) and PCR against an independent chain pull. Groww API surface
  fully audited → handoff.md (GrowwFeed websocket, quote depth/OI, v2 candle
  API inferior, MCX contract-level only). Suite 184 green.
- DONE: `scripts/snapshot.py` unified — one command runs REGIME + F&O L/S
  snapshots and prints both forward records (guarded legs); F&O collector
  being folded in as a third leg.
- DONE: RL-2026-07-13 — bear-only reversal sleeve: NEGATIVE (wash-to-drag),
  the pre-registered coin-flip's honest side (verified by orchestrator to the
  digit). All 6 configs below base on TRAIN at 20 bps; frozen least-drag
  config on test: dSR −0.006 @20, −0.030 @40 (paired-t −3.91), worse maxDD at
  every cost. Correlation premise held (~0.01) but returns can't fund costs.
  CLOSES handoff thread #4: short-term reversal fully cost-gated in every form.

## State as of RL-2026-07-11 (see research_log.md for full detail)

- Best deployable book: top-decile conviction momentum + (200MA OR India-VIX) regime
  overlay — test SR 1.86, +35.7%/yr, maxDD −27%. PROVISIONAL: 0 of ~50 trials clear
  Deflated Sharpe; the 2017–2026 test window is multi-use. Only clean proof left is
  the forward paper-track (`experiments/paper_trades.jsonl`).
- Honest negatives: sector rotation ties EW; ALL short-term books cost-gated at
  20 bps; bear-only reversal marginal (~break-even), at best a small sleeve.
- Benchmark discipline: EW-277 (test SR 1.35), not Nifty — use paired active-t vs EW.

## Open threads (ranked, from handoff.md)

1. Forward paper-track hardening — THIS SESSION.
2. Deployable market-neutral sleeve: residual-momentum L/S on F&O-shortable names.
3. F&O/options strategies via Groww (walk-forward 2021+ as suggestive only).
4. Regime-conditional bear-only reversal sleeve (small, pre-registered).
5. Survivorship kill — needs point-in-time membership (paid feed); honest ceiling.
