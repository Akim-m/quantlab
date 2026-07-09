# Progress

Running log of research + build state. Newest first.

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
