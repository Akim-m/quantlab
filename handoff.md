# HANDOFF — Indian equity strategy research (quantlab)

For the next agent picking up this work. Read `CLAUDE.md` (conventions) and
`research_log.md` (RL-2026-07-10 and RL-2026-07-11 — the full study) first; this file
is the operating setup + current state + what to do next.

---

## 1. How to operate (the setup this work used — keep it)

- **Subagent roles:** use **Fable** subagents for planning/orchestration, **Opus**
  subagents for execution. **Never run more than 3 subagents concurrently.** You are the
  orchestrator: keep the top-level plan, delegate labor, and **verify every delegated
  number yourself** (re-run the headline metric in the scratchpad before trusting it —
  this caught a subagent's inflated DSR of 0.999 that was really 0.006).
- **Commit + push after every major task** to `origin` (github.com/Akim-m/quantlab), on
  `main` (solo repo, main-based history). Each commit message says *why*.
- **Secrets NEVER go to git and are never read/printed.** Groww `API_KEY`/`API_SECRET`
  live in `.env` (git-ignored). Before every commit, grep the staged files for
  `\.env|API_|secret` and abort if hit.
- **Groww trade API = DATA ONLY, read-only.** NEVER place/modify/cancel an order.
  `groww_client.call()` refuses order methods — go through it, don't build a raw client.
  Self-throttle ≤7 req/s (the shared `RATE` limiter enforces it). `get_ltp` caps at 50
  symbols/call (batch). Live/data entitlement depends on the account's API plan.
- **Honesty protocol (Arnott–Harvey–Markowitz, see `protocol.md`):** pre-register in
  `research_log.md` before running; the test window is touched once; realistic costs;
  BH-FDR + Deflated Sharpe (against the FULL trial family, not just the winners);
  negatives are deliverables. Design blends/overlays on TRAIN, freeze, then one test read.
- **The 2017–2026 test window is now heavily RE-USED.** Every new test on it is one more
  use; the strict DSR verdict stays "fails," and the only clean proof left is the FORWARD
  paper-track. Prefer forward validation or a genuinely fresh hold-out for new claims.

## 2. Environment & checks

- Managed by **uv**. Dev deps: `uv sync --extra dev`.
- ALWAYS set `PYTHONIOENCODING=utf-8` (console is cp1252 — unicode crashes) and
  `PYTHONPATH=src` for `-m` runs.
- Tests: `uv run python -m pytest -q` (currently **147 passing**). Keep green before "done".
- Data: **Yahoo `adj_close` is the backtest price source** (total return; deep history).
  Groww is read-only for the authoritative NSE universe/instruments, F&O flags, live
  quotes, and validation — NOT backtest prices (its daily history is ~2020+ and unadjusted).

## 3. Research state — what's been found

Universe: current Nifty 500 → 277 total-return stocks from 2010; test 2017-01-01+; 20 bps;
monthly; `ret_clip=0.40` (winsorize glitches). Survivorship: current-membership → inflates
long-only; disclosed. **The key benchmark is EQUAL-WEIGHT-277 (test Sharpe 1.35), not just
cap-weighted Nifty (0.89)** — any broad long book beats Nifty on the breadth premium, so
"beats Nifty" alone is NOT selection skill; use a PAIRED active-return t-test vs EW.

**Best deployable strategy (long-only):** top-decile **conviction** momentum
(`blend.conviction_topq` on `composite{mom_12_1, sharpe_mom, resid_mom}`) + a **regime
overlay** that de-risks when Nifty < 200-day MA **OR** India-VIX is in its top 20%
(`blend.regime_on` / `blend.vix_calm`). Test Sharpe **1.86**, +35.7%/yr, maxDD **−27%**,
sidesteps 2020. See `india_run.py` (the frozen family) and RL-2026-07-11.

**Honest bounds (do not overclaim):**
- Beats Nifty, but the RETURN edge over EW-277 is the breadth/size premium. Concentration
  recovers a REAL but not-quite-significant momentum selection edge (paired active-t vs EW
  ~0 at the quintile → ~1.3 at the decile; still < the t≥2 bar). The overlay's genuine
  add is risk-adjusted (drawdown/beta), significant as CAPM alpha vs EW (t≈3).
- **0 of ~50 trials clear the strict Deflated Sharpe bar** — consistent with the whole lab.
- Cleanest PURE alpha = market-neutral **residual-momentum long-short** (Sharpe 0.86,
  uncorrelated, ~5%/yr, DSR-fails).

**Documented negatives:** sector rotation ties EW (not promoted); ALL short-term
(daily/weekly) books are cost-gated — real gross reversal edge (~0.65) killed by ~130%/wk
turnover; even bear-only reversal is ~break-even at 20 bps. See `short_term*.py`.

## 4. Live / forward track (read-only)

- `live_paper.py` — reconstructs the current REGIME book, fetches live Groww LTP
  (read-only, get_ltp only, batched ≤50), records book vs Nifty, appends to
  `experiments/paper_trades.jsonl`. Verified live 2026-07-09 (risk-OFF, 55/55 quotes).
- Run manually: `PYTHONIOENCODING=utf-8 PYTHONPATH=src uv run python -m quantlab.live_paper --refresh`.
- Daily scheduling: `scripts/paper_snapshot.cmd` + `scripts/README-schedule.md`. The user
  registers the Windows task themselves (OS persistence needs their authorization) at
  13:30 local (16:00 IST). Machine TZ = UTC+3; IST = UTC+5:30.

## 5. Open threads (ranked)

1. **Forward paper-track hardening:** upgrade `live_paper` to log the held WEIGHTS so the
   rigorous forward return (book decided day D, realized D+1) is computable exactly, not
   just the same-day diagnostic. Then accumulate the OOS record — the only clean test left.
2. **Deployable market-neutral sleeve:** make residual-momentum L/S implementable — short
   leg restricted to F&O-shortable names (Groww instrument master has the flags), price the
   futures basis carry. Measures how much shortability destroys the 0.86 Sharpe.
3. **F&O / options strategies (Groww-enabled):** futures basis cross-section, put-call
   ratio, IV skew (`get_option_chain`/`get_greeks`). Groww history is shallow (~2020+) and
   unadjusted → validate walk-forward on 2021+ as SUGGESTIVE only, plus forward paper. See
   RL-2026-07-11 planning notes and the Fable batch (B1–B8, G1–G5) referenced there.
4. **Regime-conditional reversal sleeve:** bear-only reversal is marginal standalone but
   uncorrelated with the long-only book (active only when it's defensive) — test as a small
   combined sleeve, pre-registered.
5. **Survivorship kill:** the honest ceiling on long-only claims. Needs point-in-time index
   membership or a delisted-inclusive universe — no free source; flag as needing a paid feed.

## 6. Files

- `research_log.md` — the study (RL-2026-07-10 result + RL-2026-07-11 refinements). Truth.
- `CLAUDE.md` — conventions + module map. `protocol.md` — the honesty rules.
- `src/quantlab/`: `india.py` (universe/panel), `india_run.py` (frozen 9-strategy family +
  sweep), `blend.py` (composite/conviction/overlays/regime), `india_scenarios.py`
  (`evaluate2`: paired-t + CAPM vs Nifty AND EW, regime/sub-period slices), `short_term.py`
  (short-term family), `groww_client.py` (read-only client), `live_paper.py` (forward track).
- `experiments/log.jsonl` — run ledger; `experiments/paper_trades.jsonl` — forward snapshots.
