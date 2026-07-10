# Scoring — strategy performance on historical data

*Generated 2026-07-10. Numbers are transcribed from verified `research_log.md` entries and
the auto-generated `PERFORMANCE.md` (which recomputes the deployed books live — regenerate
with `uv run python -m quantlab.report`). Where a row says "verified", the number was
independently re-derived from primitives this session, not taken on trust.*

**Read this first (the honest frame):**
- Historical test window = **2017-01-01 → present**, net of realistic costs (20 bps equity,
  10 bps ETF), benchmarked vs Nifty and EW-277.
- **~92 trials have touched this window. 0 clear the strict Deflated-Sharpe bar** — every
  historical Sharpe below is promoted as *deployable smart-beta / risk management*, never
  as statistically-proven alpha.
- **Forward-only strategies have NO old-data performance by design** — the window is
  exhausted, so new strategies accrue live paper-track evidence instead (§3). Blank ≠
  missing; blank = honest.
- Survivorship: the universe is current Nifty-500 membership → long-only absolute returns
  are inflated; risk-adjusted comparisons vs EW-277 are the honest lens.

---

## 1. Deployed / promoted books (historical test window, 2017+)

| Strategy | RL | Test Sharpe | Ann. ret | MaxDD | Beta | Verdict |
|---|---|---:|---:|---:|---:|---|
| **REGIME long-only** (conviction momentum + 200MA-OR-VIX gate) | RL-10/11 | **1.865** | +35.7% | −27.2% | +0.36 | Deployed |
| **F&O L/S sleeve** (resid-mom, F&O-shortable shorts) | RL-12 | **0.846** | +5.4% | −16.3% | −0.00 | Deployed (market-neutral) |
| **Multi-asset trend sleeve** (5 ETFs, tsmom+invvol) | RL-17 | **1.057** | +11.0% | −27.9% | +0.49 | Promoted diversifier |
| *Nifty (^NSEI) benchmark* | — | 0.790 | +12.3% | −38.4% | +1.00 | — |
| *EW-277 benchmark* | — | 1.347 | +25.5% | −47.7% | +0.91 | — |

Notes: REGIME's return edge over EW-277 is largely the equal-weight/mid-cap breadth
premium; what it genuinely adds is risk-adjusted (drawdown/beta; CAPM alpha-t vs EW ≈ 3).
Feb–Jun 2020 crash: REGIME −0.5% vs Nifty −13.9%.

## 2. RL-2026-07-26 wave — new strategies (this session)

### 2a. Tested on historical data (the wave's single hold-out spend)

| Strategy | RL | TRAIN SR (frozen) | Test SR | vs B&H | MaxDD | LW z @10/20bps | Bar | Verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| **US-GATE** (^GSPC 200MA gates NIFTYBEES) | -02 | 0.390 (ma200) | **1.043** | 0.935 | −20.8% vs −36.3% | **+0.397 / +0.190** | z>1 | **FAIL — honest negative (verified)** |

The gate cuts drawdown hard and edges Sharpe above buy-and-hold, but the risk-adjusted
edge is statistically indistinguishable from B&H. Disclosure: it *does* beat the
purely-local 200MA gate (SR 0.768, z −0.551). Cross-market index timing retired.

### 2b. Built + LIVE, forward-only (no historical read by design)

| Strategy | RL | Design evidence (TRAIN 2010-16) | First live row (2026-07-10) | First locked read |
|---|---|---|---|---|
| **DUAL-ROT** (5-ETF dual momentum, top-2 + tsmom gate) | -01 | TRAIN SR **0.693** (K2/tsmom, argmax of 4 — verified) | GOLDBEES 50% + MON100 50%, intraday +0.33% | ~2027-07 vs trend sleeve (LW z) |
| **VRP-gate** (short straddle only when IV−RV fat) | -06 | n/a (options: no history exists) | VRP −8.29 → gate OFF (warm-up), flat | ~2027-01+ gated-vs-ungated (LW z) |
| **DIV-CARRY** (dividend-yield decile L/S, non-price signal) | -13 | extraction verified (ITC/COALINDIA/ONGC textbook dividends) | 27L/27S dollar-neutral, intraday −0.07% | 252 fwd days, spread t>1.5 |

### 2c. Registered, awaiting build or data (no numbers yet — honest blanks)

| Strategy | RL | Class | Status |
|---|---|---|---|
| Weekly cash-secured put-write | -03 | Options/VRP | Registered, forward-only |
| Crowded-short basis filter (L/S short leg) | -04 | F&O conditioning | Registered, observational |
| Futures basis-momentum x-sec | -05 | F&O x-sec | Registered, forward-only |
| IV term-structure slope gate | -07 | Options/vol | Blocked: needs far-expiry IV collection |
| USDINR+Brent macro-beta x-sec | -08 | Macro x-sec | Registered; confirm `INR=X`/`BZ=F` cache |
| Same-sector cointegration pairs | -09 | Relative value | Registered, forward-only |
| Futures OI positioning x-sec | -10 | F&O x-sec | Blocked: needs per-name OI collection |
| F&O ban-list (MWPL) crowding events | -11 | Event | Blocked: needs NSE ban-list feed |
| Basis-dispersion L/S conditioning | -12 | F&O conditioning | Registered, observational |
| Jump+volume news-proxy drift | -14 | Event | Registered, forward-only |
| Turnover-shock x-sec | -15 | X-sec | Registered; needs volume QC |
| F&O eligibility-change events | -16 | Event | Registered, forward-only |
| Index-reconstitution flows | -17 | Event | Blocked: needs NSE Indices scrape |
| Expiry-cycle settlement structure | -18 | Calendar/options | Registered, observational |

## 3. Forward paper-track (the only clean out-of-sample evidence)

| Book | Ledger | Days | Record so far |
|---|---|---:|---|
| REGIME long-only | `paper_trades.jsonl` | 4 | book −0.42% vs Nifty −1.57% → **active +1.15%** |
| F&O L/S | `paper_trades_ls.jsonl` | 2 | awaiting ≥2-day forward return |
| Trend sleeve | `paper_trades_trend.jsonl` | 2 | awaiting |
| gold_lowbeta variant | `paper_trades_gl.jsonl` | 2 | awaiting |
| DUAL-ROT | `paper_trades_dualrot.jsonl` | 1 | day one |
| DIV-CARRY | `paper_trades_divcarry.jsonl` | 1 | day one |
| Short straddle (RL-18) | `paper_options.jsonl` | 2 | cum P&L **−403** (left tail, honestly recorded) |
| VRP-gated straddle | `paper_options_vrp.jsonl` | 1 | flat (gate OFF, warm-up) |
| F&O collector (RL-15) | `fno_daily.jsonl` | 2 | PCR 0.83→1.23, ATM IV 12.4→10.1 |

## 4. Graveyard — tested on historical data and retired (verdict numbers)

| Idea | RL | Decisive number | One-line verdict |
|---|---|---|---|
| 32-factor anomaly family (US+NSE) | 07-07/08/09 | 0 of 32 pass DSR (best 0.43) | Best-of-N luck, not alpha |
| Sector rotation | 07-11 | SR 1.35 ≈ EW 1.34, maxDD −45% | Ties equal-weight |
| Short-term reversal (13 books) | 07-11 | 0/13 survive 20 bps | Real gross edge, cost-dead |
| Bear-only reversal sleeve | 07-13 | ΔSR −0.006 @20bps | Wash-to-drag |
| 52-week strength | 07-14 | 0.94 corr w/ momentum | Redundant |
| Low-beta/gold risk-off sleeve | 07-16 | maxDD worsens 2.2–4.8 pts | Breaches DD cap |
| Three-book blend | 07-19 | SR 1.78 at −12.7% maxDD | Frontier alternative, not upgrade |
| Turn-of-month | 07-20 | t = 1.13 (ns) | Real sign, noise significance |
| Vol-target overlay | 07-21 | SR 1.865→1.737, paired-t −5.4 | Binary gate already owns vol-timing |
| Index band mean reversion | 07-23 | LW z **−2.34** (worse than B&H) | Knife-catching |
| VIX rebound re-entry | 07-24 | LW z ≈ 0, maxDD +2.4 pts | Buys recovery + its drawdown |
| **US-GATE (this session)** | **26-02** | **LW z +0.397 (needs >1)** | Cuts DD, edge not significant |

---

*Regeneration: the deployed-book numbers in §1 recompute live via `python -m quantlab.report`
(PERFORMANCE.md). §2–4 numbers cite their `research_log.md` entries — that file is the
source of truth; on any discrepancy, research_log wins.*
