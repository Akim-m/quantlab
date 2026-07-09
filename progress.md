# Progress

Running log of research + build state. Newest first. `handoff.md` is the operating
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
- IN PROGRESS (Opus): RL-2026-07-13 bear-only reversal sleeve on the
  deployable book (pre-registered; new-files-only isolation).

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
