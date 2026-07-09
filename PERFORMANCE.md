# Quant Lab — Performance Overview

*As of 2026-07-09 (data through last close). Generated 2026-07-09 10:17 UTC. This is a
build artifact — regenerate with `PYTHONIOENCODING=utf-8 PYTHONPATH=src uv run python -m quantlab.report`. Every number below is recomputed from
source (frozen constructions + Yahoo `adj_close` + forward ledgers), never hand-typed.*

The lab's point is HONEST research: most ideas fail, and the failure table (§4) is a
valid deliverable. Two books are deployable, both with disclosed caveats — neither
clears the strict trials-aware significance bar.

## 1. Deployed strategies

Rebuilt live on the locked test window (2017-01-01+), 20 bps turnover cost,
monthly rebalance, N500-277 total-return universe.

- **REGIME (long-only)** — top-decile conviction momentum (composite of 12-1, Sharpe-
  and residual-momentum) scaled to cash by a (200-day-MA OR India-VIX) regime overlay.
  Deployable smart-beta + drawdown control (RL-2026-07-10/11).
- **F&O L/S sleeve** — residual-momentum dollar-neutral long-short with the short leg
  restricted to F&O-shortable single stocks (130/277 of the
  universe overlaps the 210 shortable names). Implementable market-neutral
  book (RL-2026-07-12).
- **Multi-asset trend sleeve** *(promoted RL-2026-07-17, not yet in the daily paper-
  track)* — five NSE ETFs (NIFTYBEES / JUNIORBEES / BANKBEES / GOLDBEES / MON100), each
  leg long its inverse-vol share while its own 12-1 trend is up, else cash. A low-
  correlation diversifier to the equity book, promoted as a blend candidate — not one of
  the two deployed books.

| Strategy | Test Sharpe | Ann. return | Max drawdown | Beta vs Nifty |
|---|---:|---:|---:|---:|
| REGIME (long-only) | 1.865 | +35.7% | -27.2% | +0.36 |
| F&O L/S sleeve | 0.846 | +5.4% | -16.3% | -0.00 |
| Multi-asset trend sleeve (promoted, not yet in the daily paper-track) | 1.057 | +11.0% | -27.9% | +0.49 |
| Nifty (^NSEI) | 0.790 | +12.3% | -38.4% | +1.00 |
| EW-277 | 1.347 | +25.5% | -47.7% | +0.91 |

CAPM beta vs Nifty confirms the L/S sleeve is cleanly market-neutral (β≈0) while REGIME
keeps a reduced, overlay-managed market exposure.

## 2. Scenario breakdown

**Calendar-year total return** (net of 20 bps):

| Year | REGIME (long-only) | F&O L/S sleeve | Nifty (^NSEI) | EW-277 |
|---|---:|---:|---:|---:|
| 2017 | +117.9% | +19.9% | +28.6% | +61.5% |
| 2018 | +1.8% | +2.0% | +3.2% | -11.9% |
| 2019 | -7.1% | +7.6% | +12.0% | -4.2% |
| 2020 | +31.5% | -7.8% | +14.9% | +37.0% |
| 2021 | +160.3% | +18.3% | +24.1% | +62.7% |
| 2022 | +4.4% | +0.1% | +4.3% | +15.9% |
| 2023 | +74.6% | +11.6% | +20.0% | +60.5% |
| 2024 | +31.3% | +6.6% | +8.8% | +31.7% |
| 2025 | +1.9% | -5.3% | +10.5% | +5.8% |
| 2026 YTD | -0.0% | +0.5% | -7.9% | +4.9% |

**Feb–Jun 2020 crash:** **REGIME (long-only)** -0.5% · **F&O L/S sleeve** -8.7% · **Nifty (^NSEI)** -13.9% · **EW-277** -8.7% — the regime
overlay sidesteps the drawdown the index takes.

**Risk regime day-subsets** (Sharpe in ^NSEI-above vs below its 200-day MA, causal):

| Strategy | Risk-ON Sharpe | Risk-OFF Sharpe |
|---|---:|---:|
| REGIME (long-only) | 2.13 | -0.05 |
| F&O L/S sleeve | 1.60 | -1.64 |
| Nifty (^NSEI) | 0.81 | 0.88 |
| EW-277 | 1.49 | 1.13 |

## 3. Forward paper-track (out-of-sample, the only clean proof left)

**REGIME long-only book** — 2 snapshot day(s) recorded (panel dates 2026-07-07..2026-07-09).
- Latest (2026-07-09 15:05:24 IST): regime **risk_off**, cash 50%, 55 names, live quotes 55/55 (Groww ok); intraday book -0.0% vs Nifty +0.4%.
- Cumulative forward (rigorous close-to-close, 20 bps) over 3 tracked day(s): book **-0.4%** vs Nifty -1.4% → active **+1.0%**.

**F&O-shortable L/S sleeve** — 1 snapshot day(s) recorded.
- Latest (2026-07-09 15:05:40 IST): gross 1.00, net +0.0000, 138 long / 58 short, intraday +0.5% (target market-neutral ~0).
- Cumulative forward return: needs ≥ 2 snapshot days.

**F&O basis/PCR/IV collector** — 1 collect day(s). Latest 2026-07-09: 210 underlyings, NIFTY OI-PCR 0.825, ATM IV 12.4, skew 3.0. Forward-only (expired contracts unresolvable); first read after ≥126 days.

> Caveat: pre-inception panel dates are RECONSTRUCTED — the genuine forward clock started **2026-07-09** (first live snapshot). Intraday numbers are same-day diagnostics; the rigorous forward return is the close-to-close line above.

## 4. The graveyard — honest negatives

This lab reports where things do NOT win. Ideas tested and shelved (see `research_log.md`):

| Idea | Ref | One-line verdict |
|---|---|---|
| Sector rotation (top-5 industries by 6m momentum) | RL-2026-07-11 | Ties equal-weight (SR 1.35 vs 1.34), worse drawdown (−45%); a 200MA overlay hurts it. Not promoted. |
| Short-term reversal family (13 daily/weekly books) | RL-2026-07-11 | Real gross edge (~0.67 SR) but ~130%/wk turnover; 0 of 13 survive 20 bps. Cost-gated. |
| Bear-only reversal sleeve on the REGIME book | RL-2026-07-13 | Wash-to-drag at 20 bps (ΔSR −0.006), worse combined drawdown; diversification held, returns didn't. Failed. |
| 52-week-strength long book (George-Hwang) | RL-2026-07-14 | 0.94 active-return correlation with momentum, double the standalone drawdown; redundant. Not promoted. |
| Risk-off sleeve — low-beta defensive half instead of cash | RL-2026-07-16 | Return lift is real and robust (combined SR 1.91, +42%/yr, paired-t +1.7) but maxDD worsens 2–5 points, breaching the 2-point cap at every cost. Book keeps cash. |
| Three-book blend (REGIME + F&O L/S + trend sleeve) | RL-2026-07-19 | Frozen inverse-vol blend: SR 1.78 vs REGIME 1.87 at half the drawdown (−12.7% vs −27.2%) but a third of the return — a frontier alternative, not an upgrade; failed its (disclosed) mis-specified return-level bar. |
| Turn-of-month tilt on NIFTYBEES | RL-2026-07-20 | Real in sign — window days earn ~2.2× the off-window daily return — but noise in significance (two-sample t 1.13); TOM-only SR 0.44 trails buy-and-hold 0.94. Not a strategy. |
| Continuous vol-target overlay on the REGIME book | RL-2026-07-21 | Stacking a vol-target on the binary (200MA OR VIX) overlay only de-levers a positive-drift book: SR falls 1.865→1.737, return 35.7%→18.8%. The regime overlay already owns the vol-timing. |
| Cross-sectional anomaly family (32 factors, US + NSE) | RL-2026-07-07/08/09 | 0 clear the Deflated Sharpe bar; the low-vol/low-beta family actively hurt in the high-beta decade. |

**Standing caveats on the deployed books (do not overclaim):**
- **0 of ~50+ trials clear the strict Deflated Sharpe bar** — as in every study here. The books are promoted as deployable smart-beta + risk management, NOT as statistically-proven alpha.
- The **2017-2026 test window is heavily RE-USED** across research rounds; each read is one more use, so the forward paper-track is the only clean out-of-sample proof left.
- **Survivorship**: the universe is CURRENT Nifty 500 membership (no free point-in-time source), which inflates long-only absolute returns. The REGIME book's return edge over cap-weighted Nifty is largely the equal-weight/mid-cap breadth premium; what the momentum + regime overlay genuinely adds is risk-adjusted (drawdown/beta), significant as CAPM alpha vs EW-277 (t≈3).
