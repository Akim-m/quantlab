# Research Log

Pre-registration journal for every research idea, per `protocol.md` (Arnott-Harvey-Markowitz).

**Discipline:** fill the top half of an entry *before* running. The runner appends the
result row to `experiments/log.jsonl`; reference its `hypothesis_ref` here. Never write a
result without a prior hypothesis. Iterating on a model after seeing hold-out results turns
it into in-sample fitting.

## Ideas tried (multiple-testing tally)

One line per idea - including abandoned ones that never ran. This count is needed to adjust
significance for implicit search.

- `(DLTIS-PSTKR)/MRC4` - abandoned before running: arbitrary Compustat-field combo, no
  economic rationale, no fundamental data in project.
- `efficiency_gated_trend` on FX+gold - RL-2026-06-28-01. Grid of 360 configs searched on
  the dev window; this is the implicit-search count to adjust significance against.
- `betting_against_beta` on FX+gold - RL-2026-06-28-02. Single fixed rule, no parameter
  search, so 1 test only (no split needed).
- `beta_long_short` on 10 US stocks - RL-2026-06-28-03. Single fixed rule, daily
  rebalance, no search, no split.
- `beta_timing` per asset (go WITH beta) on 10 stocks + FX/gold/silver -
  RL-2026-06-28-04. Single fixed rule per asset, no search, no split.
- `order_flow_entropy` magnitude predictor on SPY second-bars (Singha 2025,
  arXiv:2512.15720) - RL-2026-06-28-05. Faithful replication of a published
  rule; thresholds frozen on train, walk-forward OOS. The paper itself reports
  20 parameter perturbations + a hand-tuned payoff rule, so treat its 2.89x
  headline as already search-inflated when judging our reproduction. Status:
  engine + walk-forward runner built and tested; faithful run data-blocked (no
  second-resolution SPY source), see entry below.
- **Anomaly-replication study, 24 price/OHLC/volume factors** on a survivorship-
  bias-free ETF cross-section - RL-2026-07-07-01..24. One canonical, un-searched
  parameterization per factor (so the implicit-search count is 24 tests, not a
  grid). Judged on the LOCKED test window under a Benjamini-Hochberg FDR control
  and a Deflated Sharpe Ratio (trials=24). Prior: most fail after correction
  (McLean-Pontiff decay + coarse cross-section). See the RL-2026-07-07 section.
- **8 more factors (ids 25-32)** extending the study to 32 - RL-2026-07-08. Same
  ETF24 universe/split/cost/construction; correction re-run over all 32 (a harsher
  bar). One fixed spec each, so +8 to the implicit-search count. See RL-2026-07-08.
- **Same factor set on Indian markets (NSE sector indices)** - RL-2026-07-09. The
  32-factor family re-run on a different, less-arbitraged market; a NEW sample, so
  a separate pre-registration (not a re-run of the US locked window). See RL-2026-07-09.
- **Broad Indian SINGLE-STOCK study + blends + regime switch** - RL-2026-07-10. The
  32-factor family on a real Nifty 500 cross-section with total-return (adj_close)
  data and long history - the two limits RL-2026-07-09's own conclusion named - PLUS
  new pre-registered strategies (long-only top-quintile variants, a momentum+low-vol
  composite blend, causal trend/vol overlays, a 200-day-MA regime switch). A NEW
  sample => separate pre-registration. Aim: map which strategy wins in which situation
  (constraint / universe breadth / cost / causal market regime) and ship a deployable
  long-only blend. Implicit-search count for this family recorded with the results.

---

## RL-2026-06-28-05 - Order-flow entropy as an intraday magnitude predictor (SPY)

- **Date (pre-registration):** 2026-06-28
- **Economic hypothesis:** Kyle (1985) / Glosten-Milgrom (1985) microstructure:
  informed traders generate persistent, structured order flow that moves price
  toward fundamentals. That persistence lowers the entropy of the trade-state
  sequence *without revealing the sign* of the information (entropy is invariant
  under buy/sell relabelling). So low order-flow entropy should predict the
  MAGNITUDE of the next move (a big move is coming) but NOT its direction. This
  is a volatility-state variable, not a directional alpha. Source: Singha 2025,
  arXiv:2512.15720, which reports |5-min return| 2.89x higher when entropy is
  below its 5th percentile, with directional accuracy at chance (45%).
- **Sample (locked):** instrument=SPY, second-resolution bars (close+volume).
  Target replication window = the paper's 2025-10-01..2025-11-19 (36 trading
  days) if obtainable; otherwise the nearest contiguous ~36-day SPY tick window
  our data vendor provides, stated explicitly before running. Walk-forward:
  10 trading-day train / 5-day test, 5 non-overlapping folds. cost=0.57 bps
  round-trip (paper's calibration). No grid search on our side - the rule and
  all four parameters are taken verbatim from the paper; thresholds (H 5th pct,
  volume 95th pct, take-profit) fit on each train fold only, frozen for test.
- **Preprocessing (locked):** per-second close = last trade price in the second,
  volume = total traded size in the second. State = (sgn(dP) in {-1,0,+1}) x
  (volume quintile 1..5 via trailing-120s empirical CDF) = 15 states. Transition
  matrix over trailing 120s; empty rows -> uniform 1/15; stationary dist via
  eigendecomposition; H normalized by log 15. Regular trading hours only; no
  winsorization. Each fold's thresholds from that fold's train days only.
- **Specification:** entry when H_t < H_0.05(train) AND second-volume > 95th pct
  (train) AND trailing 5-min return in [5,20] bps; enter in the SIGN of the
  trailing 5-min return (momentum heuristic); exit on 5 bps stop-loss, 300 s
  timeout, or ex-ante take-profit chosen on train. Primary test = magnitude
  ratio E[|r| | low-H] / E[|r|] and its t-stat (the paper's actual claim).
  Directional accuracy reported as a falsification check: it SHOULD be ~50%.
  PnL of the trading rule is secondary and treated skeptically.
- **Predicted outcome:** Honest priors. (1) The magnitude effect is the paper's
  real, theory-backed claim and is the most likely to reproduce directionally
  (ratio > 1, significant) - but our ratio will probably be well below 2.89:
  our tick source/feed differs (consolidated vs single-venue), 2.89 is the
  <5th-pct tail not the Q1/Q5 2.17, and the paper already concedes one day
  (Oct 29) was 38.5% of profits. (2) Directional accuracy ~50% - if we ever see
  a real directional edge, suspect a look-ahead bug, not alpha. (3) The +1,126
  bps trading-rule PnL is the least trustworthy number (concentration, optimistic
  fills, tiny n=240); I expect it not to survive realistic intraday costs/fills
  out of sample and will report it as such. Expect "magnitude effect partially
  reproduces, monetization does not" as the most probable verdict.

<!-- filled in AFTER the run -->
- **Result:** DATA-BLOCKED, not run at the paper's resolution. The engine
  (`microstructure.py`) and the pre-registered walk-forward OOS runner
  (`walk_forward`: low-entropy threshold fit on train days, frozen for test) are
  built and covered by 13 tests, including a causality/no-look-ahead guard
  (`entropy_series` is causal to 1e-12) and a leakage guard (each fold's
  threshold uses train rows only, verified non-vacuously). The claim needs SPY
  second-resolution bars over ~2025-10..11; the project's free Yahoo path tops
  out at 1-minute for only ~5 recent days (1s -> HTTP 400, 1m beyond ~5d -> HTTP
  422). A plumbing-only run on the obtainable 1-min SPY (5 days, horizon=5 bars,
  1 fold of 3 train / 2 test) gave OOS magnitude ratio **0.97**, t **-0.48**,
  dir_acc **0.458** - an honest null at the WRONG resolution and sample size,
  NOT a test of the paper's claim (`experiments/log.jsonl`; the run's note field
  flags every caveat). Known fidelity gap to fix before any real-data run:
  fwd/trail returns and the entropy window use positional shift across day
  boundaries, contaminating ~horizon rows at each session open/close; a faithful
  second-resolution run must compute returns and reset transition counts
  within-day.
- **Conclusion:** Unresolved / data-blocked - neither pass nor fail. The
  theory-backed magnitude claim is still untested at the paper's resolution; the
  machinery to test it faithfully now exists and is guarded against look-ahead.
  Awaiting a second-resolution SPY tick source. Do NOT read the 1-min null as
  refutation: wrong resolution, and n far too small for inference.

---

## RL-2026-06-28-04 - Going WITH beta, per asset (beta-scaled exposure)

- **Date (pre-registration):** 2026-06-28
- **Economic hypothesis:** CAPM says expected excess return rises with beta, so
  "going with beta" - scaling each asset's exposure up when its beta to the market
  is high - should harvest the market-risk premium. HONEST CAVEAT: this is the exact
  bet the BAB anomaly says is mispriced - empirically the beta/return line is too
  flat, so leveraging high beta historically delivers high raw return but weak
  risk-adjusted return. So the prior is LOW Sharpe, not high.
- **Sample (locked):** universe run individually = 10 stocks (NVDA,TSLA,AMD,META,
  AAPL,KO,PG,JNJ,WMT,DUK) + EUR,GBP,AUD,CAD,JPY,gold,silver; each over its own full
  history vs SPY from 2006-01 (stocks from IPO); beta lookback=252d; cost=5.0 bps
  (FX would be cheaper); daily rebalance. No split (fixed rule, no search).
- **Preprocessing (locked):** per-asset pairwise inner-join with SPY, dropna. Beta
  capped at +/-3 to bound estimation noise; no other normalization.
- **Specification:** weight_t = clip(beta_t, +/-3) per asset, single-asset book.
  Buy-and-hold Sharpe reported alongside to isolate what the beta scaling adds.
- **Predicted outcome:** For each stock beta hovers ~1, so beta-timing ≈ lightly
  leveraged buy-and-hold; high-beta names (NVDA/TSLA) show big raw returns and big
  drawdowns. FX/gold have small/negative equity beta -> tiny or short positions that
  do little. Expect beta-timing to NOT beat buy-and-hold on Sharpe.

<!-- filled in AFTER the run -->
- **Result:** 17 assets run individually, 2006-2026, beta_lb=252d, cost=5bps
  (`experiments/log.jsonl`, RL-2026-06-28-04). For the high-beta equities that
  drive the book, buy-and-hold Sharpe >= strategy Sharpe: NVDA 0.90 vs 0.76,
  TSLA 0.89 vs 0.77, AAPL 0.92 vs 0.84, META 0.69 vs 0.68; gold worst hit
  (0.64 -> 0.19). Only AMD (0.52 -> 0.60), silver, and a couple of near-zero FX
  pairs nominally improved. Max drawdowns on the high-beta names -67% to -99%.
- **Conclusion:** As predicted - beta-timing is ~lightly leveraged buy-and-hold
  and does NOT beat buy-and-hold on risk-adjusted return for the names that
  carry it. Leveraging high beta buys raw return, not Sharpe (the BAB caveat);
  low-equity-beta FX/gold get tiny positions that do little. Not promoted.

---

## RL-2026-06-28-03 - Betting-against-beta long/short on 10 US stocks

- **Date (pre-registration):** 2026-06-28
- **Economic hypothesis:** Frazzini-Pedersen (2014) BAB, in its native habitat (an
  equity cross-section): leverage-constrained investors overpay for high-beta stocks,
  so a dollar-neutral book long low-beta and short high-beta earns a positive
  risk-adjusted return. This is the textbook construction, unlike the transplanted
  FX/gold tilt of RL-2026-06-28-02.
- **Sample (locked):** universe=10 US stocks chosen for beta spread (NVDA,TSLA,AMD,
  META,AAPL high-beta; KO,PG,JNJ,WMT,DUK low-beta); market=SPY; window=2012-06 ->
  2026-06; rebalance=daily (no monthly snapshot); beta lookback=252d; cost=5.0 bps.
  No train/test split: single pre-committed rule, no search.
- **Preprocessing (locked):** adj_close panel, inner-join stocks+SPY+QQQ to common
  dates (META IPO sets ~2012-06 start). No floor on beta.
- **Specification:** w_i proportional to -(beta_i - cross-sectional mean beta),
  scaled to unit gross leverage; dollar-neutral long/short. Reports vs SPY and QQQ.
- **Predicted outcome:** Low net-market exposure, equity-like idiosyncratic risk.
  Honest prior: hand-picking 10 names for beta spread biases this favorably and 10
  names is a tiny, undiversified book - treat any positive Sharpe with heavy
  skepticism; the stock selection is itself an unaccounted degree of freedom.

<!-- filled in AFTER the run -->
- **Result:** Full sample 2012-2026: total return **-93.1%**, annual -17.3%, Sharpe
  **-0.91**, max drawdown -94.0%, avg daily turnover 2.5%. Reports:
  `reports/beta_long_short/vs_sp500/...`, `.../vs_nasdaq/...`.
- **Conclusion:** FAILED, badly. Over 2012-2026 the high-beta leg (NVDA/TSLA/AAPL/
  AMD/META) massively outperformed the low-beta staples, so shorting high-beta =
  shorting the biggest winners of the decade -> near-total loss. The textbook BAB
  premium did not appear in this sample; high-beta momentum dominated. This is a
  regime + selection artifact, not a clean refutation of BAB: 10 hand-picked names
  with no diversification, a tech-led bull market, and a dollar-neutral book with no
  beta-neutralization (so it carried a persistent short-tech bet). Not promoted.

---

## RL-2026-06-28-02 - Betting-against-beta (inverse-beta weighting) on FX + gold

- **Date (pre-registration):** 2026-06-28
- **Economic hypothesis:** Frazzini-Pedersen (2014): leverage-constrained investors
  bid up high-beta assets, depressing their risk-adjusted returns, so overweighting
  low-beta assets earns a premium. Here applied as a long-only tilt: weight each
  asset inversely to its rolling beta vs the equity market (SPY), rebalanced monthly.
  HONEST CAVEAT: classic BAB is an equity cross-section result; FX/gold are not an
  equity cross-section, so this is really a "low-equity-beta tilt / risk reduction"
  application of the idea, not the textbook premium. Expectation is therefore modest.
- **Sample (locked):** universe=EUR,GBP,AUD,CAD,JPY,gold,silver; market=SPY for beta;
  window=2006-05 -> 2026-06; rebalance=monthly (ME); beta lookback=63d; cost=2.0 bps.
  No train/test split: a single pre-committed rule with no search is not over-fit, so
  the full sample is an honest single-path estimate (only live trading is true OOS).
- **Preprocessing (locked):** adj_close panel, inner-join FX+gold+SPY+QQQ to common
  dates. Beta floored at 0.1 so near-zero/negative betas get bounded max weight.
- **Specification:** w_i proportional to 1/clip(beta_i, 0.1), normalized to sum 1,
  long-only. Reports vs SPY and QQQ via quantstats.
- **Predicted outcome:** A defensive, low-vol long book. Likely lower return than
  equities in bull runs but smaller drawdowns; gold/JPY (low/neg equity beta) carry
  most of the weight. Report whatever the full-sample numbers are.

<!-- filled in AFTER the run -->
- **Result:** Full sample 2006-2026: total return +54.3%, annual +2.2%, Sharpe
  **0.36**, max drawdown -22.6%, avg daily turnover 0.9%. Reports:
  `reports/beta_scaled/vs_sp500/beta_scaled.html`, `.../vs_nasdaq/beta_scaled.html`.
- **Conclusion:** As predicted - a low-turnover defensive tilt, not an alpha engine.
  Sharpe 0.36 is unremarkable and below buy-and-hold equities over the period; the
  book leans on low/neg-beta gold and JPY. Mild positive result, consistent with the
  transplanted (non-equity) rationale. Keep as a defensive building block, not a
  standalone strategy.

---

## RL-2026-06-28-01 - Efficiency-gated time-series momentum on FX + gold

- **Date (pre-registration):** 2026-06-28
- **Economic hypothesis:** Time-series momentum is well documented in FX and
  commodities (Moskowitz-Ooi-Pedersen 2012) - trends persist because information
  diffuses slowly and traders under-react then herd. The *added* claim here: a
  trend is more likely to persist when it travelled *smoothly* (high signal-to-noise)
  than when the same net move came from choppy back-and-forth, which is noise that
  mean-reverts. Gating momentum by Kaufman's efficiency ratio should therefore cut
  whipsaw regimes and the momentum-crash tail, improving risk-adjusted return over
  ungated TSMOM. Overnight/intraday effects were rejected for this universe: FX and
  gold trade ~24h, so there is no overnight session for that mechanism to exist in.
- **Sample (locked):** universe=EUR,GBP,AUD,CAD,JPY (vs USD) + gold, silver futures;
  window=2006-05 -> 2026-06; dev=...2018-12-31, test=2019-01-01...(touched once);
  rebalance in {W-FRI, ME}; cost=2.0 bps per unit turnover.
- **Preprocessing (locked):** adj_close panel, inner-join to common dates (dropna).
  No winsorization. Efficiency ratio and vol are trailing-window only.
- **Specification:** `efficiency_gated_trend` = sign(trailing trend) gated by
  (efficiency_ratio >= threshold), inverse-vol weighted, scaled to unit gross
  leverage, long/short. Search grid (360 configs): trend_lb{42,63,126,189,252} x
  er_window{21,42,63} x er_threshold{0,0.2,0.3,0.4} x vol_lb{21,42,63} x rebalance.
  er_threshold=0 is the ungated TSMOM baseline the gate must beat.
- **Predicted outcome:** Best gated config (threshold>0) beats the best ungated
  (threshold=0) on dev Sharpe; the dev edge survives, attenuated, on the locked test
  window. Honest prior: most of any dev edge is selection across 360 configs and the
  test number will be materially lower - report it whatever it is.

<!-- filled in AFTER the run -->
- **Result:** 360 configs searched on dev. Best dev config (trend_lb=126,
  er_window=42, er_threshold=0.3, vol_lb=21, ME): dev Sharpe **+0.90**, OOS test
  Sharpe **-0.21** (ann -2.5%, maxDD -35%). Pre-registered gated-vs-ungated check:
  gated beat ungated on dev (+0.90 vs +0.55) but BOTH collapse to ~-0.20 on test.
  Trial log: `experiments/autoresearch_fx-trend.tsv`; run in `experiments/log.jsonl`.
- **Conclusion:** FAILED / hypothesis not supported. The efficiency gate's dev
  advantage was selection across 360 configs, not signal - it vanishes out of
  sample and the whole strategy loses money on 2019-2026. The strong dev Sharpe is
  exactly the false positive the protocol guards against. Not promoted. Note the
  single test slice (2019-26) was a structurally weak regime for FX/gold trend, so
  the verdict is "no evidence it works," not "proven worthless"; the correct next
  test is walk-forward (re-select params each window, concatenate OOS), not another
  single split - and that counts as a new, separately-tallied idea.

---

## RL-2026-07-07 - Anomaly-replication study (24 price/OHLC/volume factors)

An honest replication of 24 published anomalies, each pre-registered with an ex
ante economic rationale and ONE canonical, un-searched parameterization. This is
not a hunt for winners: with 24 simultaneous tests the significance bar is raised
for all of them, and the prior (McLean-Pontiff post-publication decay + a coarse
cross-section of ~22 ETFs) is that MOST fail out of sample after correction. The
failure table is the deliverable.

### Shared, locked before running

- **Universe (ETF24):** SPY, QQQ, IWM, EFA, EEM, VNQ, XLB, XLE, XLF, XLI, XLK,
  XLP, XLU, XLV, XLY, TLT, IEF, LQD, HYG, GLD, SLV, DBC (22 liquid ETFs, common
  history from ~2007-06; survivorship among major ETFs over this window is
  minimal - noted as a small residual bias). Adj-close panel, inner-join, dropna.
- **Window / split:** 2007-06 -> present; train ...2015-12-31, **test 2016-01-01
  onward, touched exactly once** at the end. No iterating on the hold-out.
- **Cost:** 5 bps per unit turnover. **Rebalance:** monthly (ME) except where a
  factor's horizon demands otherwise (fixed per factor below). No winsorization.
- **Construction:** cross-sectional factors are rank-demeaned, dollar-neutral,
  scaled to unit gross (uses all names - more robust than decile sorts on ~22
  assets); time-series factors are per-asset, inverse-vol sized, unit gross;
  portfolio-construction factors are long-only, sum-to-one.
- **No parameter search:** one fixed parameterization per factor, taken from its
  source. Implicit-search count = **24 tests** (not a grid).
- **Multiple-testing correction:** Benjamini-Hochberg FDR (q=0.10) on the TEST-
  window Sharpe t-stats across all 24, plus the Deflated Sharpe Ratio
  (Bailey-Lopez de Prado, trials=24). A factor "passes" only if it clears the
  corrected bar on the locked test window.
- **Cross-asset caveat:** several factors below are equity single-name effects
  applied to a cross-ASSET ETF set; that generalization is expected to weaken
  them, and the predictions say so.

### Factors (id, signal, fixed params, rebalance, predicted)

Cross-sectional (rank-demeaned L/S over ETF24; long high signal unless noted):
- **01 short-term reversal** (Lehmann '90): signal = -ret(5d); W-FRI. *Pred:
  weak +, costs bite at weekly turnover; low conf.*
- **02 12-1 momentum** (Jegadeesh-Titman '93): ret(t-252..t-21); ME. *Pred:
  modest +, best cross-sectional shot; medium.*
- **03 long-term reversal** (DeBondt-Thaler '85): -ret(t-1260..t-252); ME.
  *Pred: ~none on ETFs (single-name effect); low.*
- **04 low-volatility** (Baker-Haugen): -vol(252d); ME. *Pred: + risk-adjusted,
  maybe - raw; medium.*
- **05 idiosyncratic vol** (Ang et al '06): -residual_vol vs SPY(252d); ME.
  *Pred: weak on ETFs; low.*
- **06 MAX / lottery** (Bali et al '11): -max(daily ret,21d); ME. *Pred: weak on
  ETFs; low.*
- **07 52-week high** (George-Hwang '04): close/high(252d); ME. *Pred: momentum-
  correlated, modest; low-medium.*
- **08 return skewness** (Boyer et al): -skew(252d); ME. *Pred: weak; low.*
- **09 residual momentum** (Blitz '11): 12-1 momentum of SPY-residual returns; ME.
  *Pred: momentum-like, maybe cleaner; low-medium.*
- **10 same-month seasonality** (Heston-Sadka '08): mean same-calendar-month ret
  over prior years; ME. *Pred: noisy, weak; low.*
- **11 downside beta** (Ang-Chen-Xing '06): -downside_beta vs SPY(252d); ME.
  *Pred: overlaps low-vol; low.*

Time-series / trend (per-asset, inverse-vol sized):
- **12 TSMOM** (Moskowitz-Ooi-Pedersen '12): sign(ret 252d), directional L/S; ME.
  *Pred: +, strong cross-asset prior; medium-high - best shot.*
- **13 Donchian breakout**: +1 at 252d-high, -1 at 252d-low; ME. *Pred: TSMOM-
  correlated +; medium.*
- **14 dual momentum** (Antonacci '14): long if 12m ret > 0 AND > cross-median,
  else flat; ME. *Pred: +, defensive; medium.*
- **15 vol-managed exposure** (Moreira-Muir '17): long, exposure ~ 1/vol(21d) to
  a constant-vol target; ME. *Pred: Sharpe > buy&hold, weak standalone; medium.*
- **16 crash-scaled momentum** (Daniel-Moskowitz '16): TSMOM scaled down when its
  own recent realized vol spikes; ME. *Pred: better tail than 12; medium.*
- **17 overnight-vs-intraday** (Lou-Polk-Skouras '19): long overnight (close->open)
  minus intraday (open->close) component; daily. *Pred: overnight drift real but
  execution-fragile on ETFs; low-medium.*
- **18 Bollinger mean reversion**: long < lower band, short > upper (20d, 2s);
  W-FRI. *Pred: weak at this horizon; low.*
- **19 fixed-pairs mean reversion**: z-scored spread reversion on economically
  linked pairs {SPY-QQQ, GLD-SLV, TLT-IEF, XLE-XLB}, averaged; ME. *Pred:
  marginal after costs; low.*

Portfolio construction (long-only over ETF24, ME):
- **20 risk parity / ERC** (Maillard '10). *Pred: solid Sharpe / low vol; works as
  construction, not alpha; medium.*
- **21 max diversification** (Choueifaty-Coignard '08). *Pred: ERC-like; medium.*
- **22 HRP** (Lopez de Prado '16). *Pred: ERC-like, better OOS stability; medium.*
- **23 min-correlation portfolio**. *Pred: weak ERC variant; low.*
- **24 dual-momentum + ERC overlay** (composite of 14 + 20). *Pred: trend timing +
  RP sizing, best composite; medium.*

<!-- filled in AFTER the single locked test-window evaluation -->
- **Result:** Ran on ETF24, 22 assets, 2007-06 -> 2026-07, test = 2016-01 onward,
  5 bps (`experiments/log.jsonl`, RL-2026-07-07; runner `factor_study.py`).
  **7 of 24 clear naive BH-FDR (q=0.10)** on the test window - and every one is a
  risk-based construction / trend-timing method, NOT a cross-sectional alpha:
  risk_parity_erc (test Sharpe 0.97, t 3.12), min_correlation (0.91, 2.96),
  vol_managed (0.91, 2.93), dual_momentum (0.77, 2.51), dual_mom_erc (0.76, 2.45),
  max_diversification (0.75, 2.44), hrp (0.73, 2.35). **0 of 24 clear the Deflated
  Sharpe Ratio (>0.95); best DSR = 0.42** (risk_parity_erc). The classic equity
  anomalies were mostly NEGATIVE on test: low_volatility -0.54, max_lottery -0.72,
  downside_beta -0.49, long_term_reversal -0.37, short_term_reversal -0.28,
  overnight_intraday -0.29; momentum_12_1 ~0 (0.08), tsmom +0.33 (t 1.07, ns).
- **Conclusion:** Exactly the pre-registered prior. Naive per-test significance
  "finds" 7 strategies; the trials-aware Deflated Sharpe finds **none** - the 7 are
  not distinguishable from the best of 24 lucky draws (and their negative skew
  further deflates the DSR). The cross-sectional equity anomalies show clear
  post-publication + cross-asset-transplant decay (McLean-Pontiff): the low-risk
  family (low-vol, lottery, downside-beta) actively HURT over a high-beta-tech-led
  2016-2026. **None promoted.** Honest headline: after correcting for 24
  simultaneous tests, no anomaly shows convincing OOS skill on this universe.
  Caveats: ~22-ETF cross-section is coarse (weakens single-name effects), and the
  DSR is stringent by construction. Risk-parity/ERC stays a reasonable *construction*
  default (best raw test Sharpe, lowest turnover 0.5%/day) - but as construction,
  not a discovered edge. Any follow-up (e.g. an equity single-name universe) is a
  new, separately-tallied idea, not a re-run on this locked test window.

---

## RL-2026-07-08 - Anomaly study extension (8 more factors, ids 25-32)

Eight further published factors, distinct from the 24, added to the RL-2026-07-07
family under the SAME locked methodology (ETF24 universe, 2007-06 -> present, test
2016-01 onward, 5 bps, one un-searched spec each). The correction is re-run over
all 32 factors (BH-FDR q=0.10 + Deflated Sharpe, trials=32) - a strictly harsher
bar. The original 24 are fixed and un-tweaked; adding pre-registered new tests is
not iterating on the hold-out. Prior: unchanged - most fail.

Cross-sectional (rank-demeaned L/S over ETF24, ME):
- **25 bab_beta** (Frazzini-Pedersen '14): signal = -CAPM beta vs SPY(252d), long
  low beta. *Pred: low-beta bonds/gold get longs -> defensive; low-medium.*
- **26 sharpe_momentum**: signal = (ret t-252..t-21)/vol(126d), smoother trends.
  *Pred: momentum-like, maybe cleaner; low-medium.*
- **27 kurtosis**: signal = -kurtosis(252d), short crash-prone. *Pred: weak on
  ETFs; low.*
- **28 low_52w**: signal = price/min(252d), distance above the 52w low. *Pred:
  strength/anti-distress, momentum-correlated; low.*
- **29 parkinson_lowrange**: signal = -mean(log(high/low)^2, 21d), long low-range
  (OHLC range vol). *Pred: low-vol cousin, likely hurts like low-vol 2016-26; low.*

Time-series / timing (per-asset, inverse-vol sized):
- **30 turn_of_month** (Ariel '87): long-all inverse-vol only on the ToM window
  (last trading day of month + first 3), else cash; daily. *Pred: real seasonal
  but thin and cost-heavy; low-medium.*
- **31 ma_trend**: direction = sign(price - SMA(200d)), classic MA crossover; ME.
  *Pred: TSMOM-correlated +; medium.*
- **32 volume_momentum**: sign(ret 252d) gated by rising volume (vol SMA up); ME.
  *Pred: participation may sharpen trend slightly; low-medium.*

<!-- filled in AFTER the single locked test-window evaluation over all 32 -->
- **Result:** 32-factor run on ETF24, test 2016-2026, 5 bps (`experiments/log.jsonl`,
  strategy `anomaly_replication_32`). The original 24 reproduced bit-identically
  (refactor validated). Of the 8 new factors none is significant: best volume_momentum
  test Sharpe 0.46 (t 1.47, DSR 0.035), then low_52w 0.40, ma_trend 0.36,
  turn_of_month 0.30; sharpe_momentum -0.10, bab_beta -0.47 and parkinson_lowrange
  -0.54 went NEGATIVE. Over all 32: **7 clear naive BH-FDR** (the same risk-based
  construction/timing set as RL-2026-07-07, unchanged), **0 clear DSR>0.95** (best 0.43).
- **Conclusion:** Extending to 32 changed nothing - still 0 survive the trials-aware
  bar, and the harsher trials=32 deflation keeps the 7 BH-passers below DSR 0.95. The 8
  new anomalies add no discovery; bab-beta / parkinson-range / the low-risk family remain
  casualties of the high-beta-tech-led 2016-2026 regime. Reinforces RL-2026-07-07. None
  promoted.

---

## RL-2026-07-09 - Same 32 factors on Indian markets (NSE sector indices)

- **Date (pre-registration):** 2026-07-07
- **Economic hypothesis:** India is a less-arbitraged, higher-retail-participation
  market than the US, so anomalies eroded in the US (short-term reversal, momentum,
  low-vol) may be stronger there - a genuine cross-market out-of-sample test of the
  same 32 factors, not a re-run of the US locked window.
- **Sample (locked):** universe = 12 NSE sector indices (^NSEBANK, ^CNXIT, ^CNXAUTO,
  ^CNXPHARMA, ^CNXFMCG, ^CNXMETAL, ^CNXENERGY, ^CNXREALTY, ^CNXINFRA, ^CNXPSUBANK,
  ^CNXMEDIA, ^CNXPSE); market/benchmark = ^NSEI (Nifty 50, NOT traded); window
  common history ~2011-08 -> present; **test = 2018-01-01 onward**, touched once;
  cost = **20 bps** (higher than US 5 bps: STT + wider spreads + lower index-proxy
  liquidity). Construction/rebalance identical to RL-2026-07-07/08.
- **Preprocessing (locked):** adj_close panel, inner-join, dropna. CAVEAT: ^ indices
  are PRICE indices (no dividend reinvestment), so ~1-1.5%/yr dividend yield is
  missing and long-biased returns are understated. `pairs` (US-symbol-specific) is
  inapplicable here and reports flat - excluded from the verdict, not a real test.
- **Specification:** the same 32 pre-registered factors, one fixed spec each, judged
  on the locked test window under BH-FDR (q=0.10) + Deflated Sharpe (trials = the
  applicable factors). No new search.
- **Predicted outcome:** Honest prior. A less-efficient market MIGHT lift a few
  (short-term reversal, momentum, low-vol) above their dismal US test numbers, but a
  coarse 12-sector cross-section, ~6.5y train / 8.5y test, 20 bps costs, and the same
  harsh multiple-testing bar make broad survival unlikely; risk-based construction
  (ERC/HRP) is again the most probable "best raw Sharpe." Report whatever it is.

<!-- filled in AFTER the single locked test-window evaluation -->
- **Result:** NSE12 sector indices, 2011-08 -> 2026-07, test 2018-01, 20 bps
  (`experiments/log.jsonl`, universe `NSE12`). Best raw: hrp 0.79, risk_parity_erc 0.75,
  max_diversification 0.74 (t 2.13-2.26) - but the shorter, noisier sample gives lower
  t-stats than the US, so **0 clear BH-FDR** (smallest p 0.012 vs the rank-1 bar 0.003)
  and **0 clear DSR** (best 0.016). Strongly NEGATIVE and cost-driven: overnight_intraday
  -2.54 (t -7.3), short_term_reversal -2.14 (t -6.2), bollinger -1.24 (t -3.6) - all
  high-turnover (0.22-0.27) mean-reversion crushed by the 20 bps cost. Momentum mildly
  positive (momentum_12_1 0.19, residual_momentum 0.28, sharpe_momentum 0.29) but ns;
  low_volatility +0.13 (vs -0.54 in the US). `pairs` inapplicable (US symbols) -> flat,
  excluded from the DSR trial set.
- **Conclusion:** The "less-arbitraged market -> stronger anomalies" prior is NOT
  supported - India gives the SAME verdict as the US: 0 survive correction. If anything
  the classic anomalies fare worse here once realistic 20 bps costs hit the high-turnover
  reversal/intraday books; only the small cross-market flip in low-vol (negative in the US,
  mildly positive in India) hints at a regime difference, and it is not significant.
  Risk-based construction (HRP/ERC) is again best raw Sharpe but insignificant on this
  ~8.5y test. Caveats: coarse 12-sector cross-section, price-index (dividends missing,
  returns understated), short history. None promoted; a longer or single-stock Indian
  sample is a separate, future, separately-tallied idea.

---

## RL-2026-07-10 - Broad Indian single-stock strategy search (factors + blends + regime switch)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** RL-2026-07-09 found nothing survived on 12 NSE sector
  PRICE indices and named the two reasons: a coarse cross-section and missing
  dividends. On a BROAD single-stock cross-section (Nifty 500) with TOTAL-return
  data (Yahoo adj_close, corporate-action adjusted) and long history, the anomalies
  best documented in Indian equities - cross-sectional momentum (strong, persistent
  in India), low-volatility (the NSE Low-Vol-30 index has outperformed), and their
  blend - should show detectable OOS skill. Different strategy TYPES are expected to
  win in different SITUATIONS: momentum in risk-on trends, low-vol/defensive in
  drawdowns, construction (ERC/HRP) for stable sizing, and a causal regime switch to
  move between them. Deliverable = that situation->winner map + a deployable long-only
  blend that beats Nifty net of costs OOS.
- **Sample (locked):** universe = current Nifty 500 constituents (primary), with
  Nifty 200 and Nifty 50 as universe-perturbation robustness. Prices = Yahoo
  adj_close total return. Benchmark = ^NSEI (not traded) and NIFTYBEES (tradeable,
  dividend-bearing). Groww is wired read-only for the authoritative NSE
  universe/instrument master, F&O-shortability flags, and recent-price validation -
  NOT as the backtest price source (Groww daily history is ~2020+ and unadjusted for
  corporate actions). Inclusion rule: a name is kept iff listed on/before `start` and
  >= 95% coverage of the Nifty trading calendar over [start,end], small gaps ffilled
  <= 3 days. start = 2010-01-01; window 2010-01 -> present; **test = 2017-01-01
  onward, touched exactly once.** cost = 20 bps/unit turnover primary (10 and 40 bps
  as robustness). rebalance = monthly (ME) primary. SURVIVORSHIP/OLDER-LISTING bias:
  current-membership lists + a listing filter bias results upward; disclosed and
  mitigated by using relative/cross-sectional signals and reporting universe-breadth
  robustness (a broader list dilutes, not removes, the bias). No free point-in-time
  membership source exists.
- **Preprocessing (locked):** adj_close panel per the inclusion rule; no
  winsorization; valid corporate-action-adjusted prices kept as-is. The composite
  BLEND and the regime switch are designed using TRAIN-window (< 2017) evidence and
  economic priors ONLY, then FROZEN before the test window is evaluated once. Looking
  at test-window results before freezing would convert this into in-sample fitting.
- **Specification:**
  1. The 32 pre-registered factors (RL-2026-07-07/08 specs, reused unchanged),
     dollar-neutral long-short. `pairs` uses US symbols -> inapplicable, excluded
     from the DSR trial set.
  2. Long-only top-quintile variants of the cross-sectional signals (the deployable
     form; single-stock cash-market shorting is F&O-only in India).
  3. A composite blend: cross-sectionally z-scored, equal-weighted combination of the
     economically-motivated signals selected on TRAIN evidence (priors: momentum +
     low-vol, plus any train-robust addition), in long-short and long-only forms,
     with causal overlays (trend filter = de-risk when Nifty < 200-day MA; vol-target
     to 15% annualized, no leverage).
  4. A causal regime-switch ensemble: risk-on book (momentum) when Nifty > 200-day MA,
     risk-off book (low-vol or cash) otherwise - a pre-committed rule, not fitted to
     which book won a slice.
  Multiple-testing: BH-FDR (q=0.10) on test-window Sharpe t-stats + Deflated Sharpe
  (trials = applicable strategies) over the whole family. Situations evaluated:
  constraint (L/S vs long-only), universe (Nifty 50/200/500), cost (10/20/40 bps),
  causal market regime (bull/bear via 200-day MA, high/low vol), and sub-periods.
- **Predicted outcome:** Honest priors. Cross-sectional momentum is the most likely
  to show real OOS skill in India (strongest prior); low-vol is defensive (better
  risk-adjusted, may lag raw in bull runs); the momentum+low-vol blend should have the
  best Sharpe of the alpha strategies; ERC/HRP give the best raw Sharpe as CONSTRUCTION
  (not alpha), as in every prior study here. The long-only blend most likely BEATS
  Nifty buy-and-hold net of 20 bps OOS (the primary pragmatic bar) while FEW or none
  clear the strict Deflated Sharpe (trials-aware) bar - so the honest headline will
  probably be "wins the deployable benchmark comparison, does not clear the strict
  multiple-testing bar." High-turnover reversal/overnight/bollinger expected NEGATIVE
  after 20 bps (as on the sector indices). The 200-MA regime switch expected to cut
  drawdown materially vs static momentum (its main value is tail protection, not raw
  Sharpe). Report every number honestly, including where a strategy does NOT win.

<!-- filled in AFTER the single locked test-window evaluation -->
- **Result:** _pending run_
- **Conclusion:** _pending run_

---

## Template

```markdown
## RL-YYYY-MM-DD-NN - <short title>

- **Date (pre-registration):** YYYY-MM-DD
- **Economic hypothesis:** <why this should work, in plain econ terms, BEFORE any results>
- **Sample (locked):** universe=<>, window=<start>→<end>, rebalance=<>, cost=<>bps
- **Preprocessing (locked):** <winsorization / normalization / none>
- **Specification:** <strategy/model + params>
- **Predicted outcome:** <what you expect and why>

<!-- filled in AFTER the run -->
- **Result:** <key metrics / experiments/log.jsonl ids>
- **Conclusion:** worked / failed / shelved - <one line, honest>
```
