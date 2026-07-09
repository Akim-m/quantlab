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
- **Phase-2 refinements (concentration, VIX regime, sector rotation, 13 short-term books)** -
  RL-2026-07-11. Exploratory refinements on the SAME 2017-2026 window (now multi-use, flagged);
  ~4 concentration levels + VIX/MA overlay variants + sector rotation + 13 short-term trials.
  Cumulative India trial count now ~50+; strict DSR bar rises accordingly. See RL-2026-07-11.

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

### Frozen design (2026-07-10, from a planning pass; frozen BEFORE the single test run)

Data cleaning (pre-registered): daily returns winsorized at +/-40% and the price panel
rebuilt (ret_clip=0.40). NSE circuit limits make >40%/day almost always a bad print;
audit found 13 such events in 277x4053, 3 of them >100% (WHIRLPOOL +570% etc.).
Winsorizing barely moves the top strategies (dual_momentum 1.48->1.51, hrp 1.41->1.42,
residual_momentum 0.94->0.97), so results are not glitch-driven.

Benchmark: ^NSEI is price-only; primary benchmark = a TR proxy (^NSEI + 1.4%/yr dividend
accrued daily). Crucially, alpha is ALSO measured vs the EQUAL-WEIGHT-277 book (gross
test Sharpe 1.34 vs Nifty 0.80): any broad long book beats cap-weighted Nifty on the
size/breadth premium, so "beats Nifty" alone is NOT signal skill - the headline must say
whether the strategy beats EW-277 on a PAIRED active-return t-test.

Frozen family (9 new trials; monthly ME, 20 bps, N500-277 primary). Long books EXCLUDE
the low-vol/low-beta/reversal/lottery family (train+test negative) except the regime
risk-off book:
- LO-CORE (frozen long-only): composite{mom_12_1, sharpe_mom, resid_mom} equal ->
  long_only_topq(top=0.2, invvol) -> trend_overlay(200d MA).
- LO-CORE-NOOVERLAY: same, no overlay (attribution + bull book).
- LO-EXT: add mom_6_1, off_low (=low_ratio 252d), sector_mom (mom_12_1 demeaned within
  NSE industry); else as LO-CORE.
- LO-ERC: top-20% by the composite, ERC-weighted (erc_weights_fast) + trend overlay.
- LO-VT: LO-CORE book -> vol_target_overlay(target = train realized vol of no-overlay
  book, cap=1.0).
- REGIME: regime_switch(risk_on = no-overlay book, risk_off = 50% cash + 50% defensive
  low-vol book, 200d MA) - tests if low-vol earns its keep ONLY conditional on bear.
- LO-BAND: LO-CORE with hysteresis banded rebalance (enter rank<=15%, hold<=35%); primary
  cell = the 40 bps column.
- LS-CORE (frozen long-short): composite{resid_mom, sharpe_mom, mom_12_1} equal ->
  long_short(); judged as factor evidence, NOT vs Nifty.
- LS-RESID2: LS-CORE with weights resid_mom:2, sharpe_mom:1, mom_12_1:1 (ex-ante deviation
  justified by resid_mom's train t=2.9).

Success criterion (frozen). PRIMARY WIN (LO-CORE): net test Sharpe > Nifty-TR proxy AND
paired monthly active-return t >= 2 AND beats benchmark in >=2/3 costs and >=2/3 universes
AND (positive active return in both bull and bear slices OR better maxDD than Nifty).
STRICT verdict (co-reported, expected to fail, not spun): BH-FDR q<0.05 within the 9-trial
family AND DSR>0.95 at the CUMULATIVE India trial count (prior 32 + 9 = 41). Noise floor:
Sharpe SE ~0.3-0.45, so gaps < ~0.4 are unrankable.

Disclosed biases: (1) survivorship + older-listing bias UP, worst in N500 - N50 is the
honesty anchor; (2) OOS window is SECOND-USE for these families - graded via cumulative
DSR N=41 and stated plainly; (3) cash earns 0, biasing overlay/regime books DOWN ~6.5%/yr
T-bill; (4) 200MA at month-ends exits AFTER the Feb-Mar 2020 crash - 2020 reported
explicitly.

<!-- filled in AFTER the single locked test-window evaluation -->
- **Result:** Ran the frozen 9-trial family on NIFTY500-277, test 2017-01-01 -> 2026-07-07,
  20 bps, monthly (`india_run.py`; `experiments/log.jsonl`, hypothesis_ref RL-2026-07-10).
  Benchmarks (test Sharpe): raw ^NSEI 0.80, TR-proxy (^NSEI + 1.4%/yr) 0.89, equal-weight-
  277 net **1.35**. Long-only books, all beating Nifty decisively: **REGIME** (momentum
  risk-on / 50%-cash-50%-low-vol risk-off via 200d MA) test Sharpe **1.70**, ann +29.9%,
  maxDD **-21.9%**, turnover 2.8%/mo, Feb-Jun 2020 **-2.2%**; LO-ERC 1.70; LO-CORE 1.63;
  LO-BAND 1.63; LO-EXT 1.62; LO-CORE-NOOVERLAY 1.51; LO-VT 1.44; hrp 1.42. Long-short:
  LS-RESID2 0.86 (t 2.61), LS-CORE 0.80 (t 2.45), maxDD ~-19%, near-zero/negative bear beta.
  - **Alpha attribution (paired monthly active-return t):** vs Nifty-TR, LO-CORE t 1.92,
    REGIME t 2.73 -> beat the index. vs EQUAL-WEIGHT-277, NONE significant on RAW return
    (act_t_ew 0.06-0.52): the index-beating is the mid-cap/breadth premium (EW-277 alone
    = 1.35), NOT momentum stock-selection. But on RISK-ADJUSTED (CAPM) alpha vs EW-277 the
    overlay books ARE significant (alpha_t_ew 2.86-3.17) - the trend/regime overlay adds
    Sharpe via beta/drawdown reduction, not extra return.
  - **Strict verdict:** BH-FDR (q=0.10) passes the 7 long-only books; both long-short pass
    BH. Deflated Sharpe against the FULL 40-trial search (factor Sharpes -2.3..+1.6):
    **0 of 9 clear DSR>0.95** (LO-CORE 0.006, REGIME 0.010, LS 0.000). [A DSR computed on
    only the 9 self-similar winners inflates to ~1.0 - a narrow-trial-set artifact that was
    caught and corrected; the full-family DSR is the honest bar and matches RL-07-08/09's 0.]
  - **Situation -> winner:** long-only deployable / bear / high-vol / 2020-crash -> REGIME;
    max bull exposure -> LO-CORE-NOOVERLAY (bull Sharpe 1.88); universe breadth -> N500 >
    N200 > N50 for every strategy; low-turnover/passive -> HRP (1.0%/mo); market-neutral,
    uncorrelated, calm-market -> LS-RESID2. No strategy beats its own-universe EW on the
    raw paired t in any cell.
- **Conclusion:** Partial WIN, honestly bounded - and the pre-registered prior was right.
  A DEPLOYABLE long-only strategy that beats the Nifty net of costs OOS exists and is robust
  across universes, costs, and sub-periods: **REGIME** (broad momentum book + inverse-vol
  sizing + causal 200-day-MA regime switch) - test Sharpe 1.70 vs Nifty 0.89, drawdown -22%
  vs -38%, and it sidesteps 2020 (-2% vs -13%). Honest attribution: the RETURN edge over the
  cap-weighted index is the equal-weight/mid-cap breadth premium (a naive EW-277 book also
  beats Nifty; paired active-t vs EW ~ 0), NOT momentum alpha; what the momentum + regime
  overlay genuinely adds is RISK-ADJUSTED improvement (higher Sharpe, half the drawdown),
  significant as CAPM alpha vs EW-277 (t~3). The cleanest PURE alpha is the market-neutral
  residual-momentum blend (Sharpe 0.86, uncorrelated) - but only ~5%/yr and it fails DSR.
  NONE clears the strict trials-aware Deflated-Sharpe bar (as in every prior study here).
  Caveats realized: current-membership universe inflates absolute long-only returns via
  survivorship (the less-exposed N50 column and long-short books are more modest); the OOS
  window is second-use for these signal families; cash earns 0, biasing overlay books DOWN
  ~6.5%/yr. **Promoted as a deployable smart-beta + risk-management strategy (REGIME /
  LO-CORE) with disclosed caveats; NOT promoted as statistically-proven alpha.** The winning
  answer for "different situations": momentum-tilted broad book for return, a 200-MA regime
  overlay for drawdown/bear protection, and a market-neutral residual-momentum sleeve as the
  uncorrelated diversifier.

---

## RL-2026-07-11 - Phase-2 refinements: concentration, VIX regime, sector rotation, short-term

- **Date:** 2026-07-09. Exploratory refinement round on the RL-2026-07-10 universe/split
  (N500-277, test 2017+, 20 bps, ret_clip=0.40). HONESTY FLAG: the 2017-2026 window is now
  heavily RE-USED across these rounds; every number below is one more use, so the strict
  statistical verdict is unchanged (fails DSR) and the real proof is FORWARD paper-tracking
  (`live_paper.py` -> `experiments/paper_trades.jsonl`). Each refinement's design is frozen
  before its single test read; parameter searches are disclosed.
- **Experiment zero (TRAIN 2010-2016 only, no test use):** decile rank-response vs EW-277.
  Momentum has a real, near-monotonic TOP-decile gradient on train (top momentum decile
  +13.7%/yr active vs EW; sector-mom +12.9%, 6-1 +10.4%, 52w-strength +6.7%). Low-vol
  NEGATIVE (high-vol names beat EW in the bull decade). The "exclude losers" thesis is weak
  - the loser decile is ~flat, not strongly negative - so the alpha is in OWNING winners
  (concentration), not excluding trash.
- **Concentration recovers momentum alpha over EW (the key phase-2 finding).** The RL-07-10
  top-QUINTILE inverse-vol book had paired active-t vs EW-277 ~ 0 (the momentum selection
  added nothing over the breadth premium). Concentrating recovers it: top-quintile t 0.06 ->
  top-decile inverse-vol t 1.08 -> top-decile CONVICTION (weight ~ +z-score) t 1.34 (test SR
  1.75, ann 38%, maxDD -31%) -> top-5% conviction t 1.57 (SR 1.68, ann 42%, maxDD -45%). So
  the momentum gradient is real and lives in the extremes; the quintile averaged it away. The
  cost is drawdown (concentration raises maxDD). 4 concentration levels searched (disclosed);
  still below the pre-registered t>=2 bar. Codified as `blend.conviction_topq`.
- **India-VIX regime filter improves REGIME (real upgrade).** OR-ing a fast ^INDIAVIX filter
  (risk-off when VIX in its trailing top 20%) with the slow 200d-MA trend filter (risk-off if
  EITHER fires) beat the MA-only overlay on the top-decile book: test Sharpe **1.86** (vs 1.75
  MA-only), maxDD **-27%** (vs -30%), Feb-Jun 2020 -0.5% (VIX-only slice +26%). VIX catches
  spikes the slow MA is late to. ^INDIAVIX is on Yahoo to 2010 so this is fully backtestable.
  Codified as `blend.vix_calm` / `blend.regime_on`. VIX percentile (80th, 252d) is a chosen
  param (disclosed). **New best deployable book: top-decile conviction momentum + (200MA OR
  VIX) regime overlay - test SR 1.86, ann 35.7%, maxDD -27%.**
- **Sector rotation (B3): NEGATIVE.** Long stocks in the top-5 industries by 6m industry
  momentum ties EW (test SR 1.35 vs 1.34), paired active-t vs EW +0.87 (ns), worse maxDD
  (-45%); a 200MA overlay hurts it. Not promoted (as the planning prior predicted).
- **Short-term (daily/weekly) family, 13 books on nifty100: NO cost-survivor.** Short-term
  reversal has a REAL gross edge (5-day reversal ~0.67 gross Sharpe, residual/vol-gated
  similar) that costs DESTROY: net Sharpe ~break-even at 5 bps, negative by 10, badly negative
  at 20 (~130%/wk turnover). Long-only short-horizon tilts show high net Sharpe (0.8-1.1) but
  it is pure market beta - every one has NEGATIVE paired active-t vs EW. Faster 2-day reversal
  stronger gross but hopeless turnover; overnight-intraday negative at all costs; nifty50
  (5 bps) rescues none. 0 of 13 clear BH-FDR + DSR. THE ONE LEAD, then TESTED: reversal's edge
  concentrates in the 200-MA BEAR regime. Gating reversal to bear-only (Nifty < 200MA, ~21% of
  test days) flips it from net@10 -0.43 to **+0.32** (resid-reversal +0.38), but it is
  ~break-even at realistic 20 bps (-0.11) and only +1.3%/yr - better than always-on, NOT
  deployable standalone; at best a small uncorrelated bear-market sleeve. Short-term Indian
  equity is cost-gated: nothing survives 20 bps as a standalone winner.
- **Live paper harness (`live_paper.py`): READ-ONLY, built and verified** (only get_ltp via the
  order-refusing wrapper; method-spy test). As of 2026-07-09 the strategy is risk-OFF (^NSEI
  24,014 vs 200d MA 24,853), 50% cash + defensive book. Groww live/data calls currently return
  HTTP 403 (token authenticates but lacks the data entitlement/quota) - an account issue, not
  code. Forward track record accrues once Groww access is restored.
- **Data addendum (2026-07-09, measurement — not a strategy trial, no test-window use):**
  Groww API coverage audited live on the owner's account (read-only
  `get_historical_candle_data` probes, RELIANCE/ITC). CORRECTS the note above (and
  RL-2026-07-10) that said "~2020+ and unadjusted for splits/bonuses": CASH daily
  candles actually reach back to ~2002-07-18 and ARE split/bonus-adjusted (pre-2024
  RELIANCE prices are halved for the 1:1 bonus). They are NOT dividend-adjusted and
  NOT demerger-adjusted (Groww ≈ raw traded price: +8.7% vs Yahoo close on pre-2023
  RELIANCE = Jio Financial demerger; +3.7% on pre-2025 ITC = ITC Hotels; Yahoo
  close-vs-adj gap = dividends). So the verdict stands — Yahoo `adj_close` remains
  the only valid backtest price source — but Groww daily is now a usable
  cross-VALIDATION source to ~2002 for price levels on non-demerged names.
  Per-request caps: daily ≤730d; intraday depth ~90 days total (60min ≤90d/request,
  10min ≤30d, 5min ≤15d, 1min ≤7d). Live LTP entitlement confirmed working.
- **Conclusion:** Two genuine improvements to the deployable strategy - CONCENTRATION (top
  decile, conviction-weighted) recovers real momentum alpha the diluted quintile hid (paired
  active-t vs EW ~0 -> ~1.3), and a VIX-augmented regime filter sharpens the crash exit (SR
  1.86, maxDD -27%). Neither reaches the strict t>=2 / DSR bar, and the test window is now
  multi-use, so both are PROVISIONAL pending forward validation. Sector rotation and every
  short-term book FAIL to beat EW / survive costs - honest negatives that tighten "EW-277 is
  near the efficient long-only frontier for this universe/decade." Best deployable book updated
  to: top-decile conviction momentum + (200MA OR India-VIX) regime overlay.

---

## RL-2026-07-12 - Implementable market-neutral sleeve: resid-mom L/S with F&O-only shorts

- **Date (pre-registration):** 2026-07-09
- **Economic hypothesis:** The RL-2026-07-10 residual-momentum L/S (test SR 0.86,
  ~5%/yr, market-neutral, uncorrelated with the long book) is deployable only if the
  short leg is implementable. In India, practical single-name shorting beyond intraday
  is via single-stock futures, so the NSE F&O list is the true short universe
  (measured live 2026-07-09 from the Groww instrument master: 948 FUT instruments,
  segment FNO; distinct single-stock underlyings to be reported by the run). Restricting
  shorts to F&O names should DEGRADE but not destroy the edge: the short pool loses
  breadth and the worst losers (small caps) are exactly the excluded names, but a short
  futures position also EARNS basis carry (≈ repo − div yield > 0), offsetting part of
  the lost alpha.
- **Sample (locked):** identical to RL-2026-07-10 — N500-277 total-return universe from
  2010 (Yahoo adj_close), TRAIN 2010→2016-12-31, TEST 2017-01-01→now (ONE read; the
  window is heavily RE-USED — honesty flag; strict DSR verdict expected unchanged),
  monthly rebalance, ret_clip=0.40.
- **Preprocessing (locked):** as RL-2026-07-10 (same panel, same resid-mom signal from
  `india_blend_study.raw_signals`).
- **Specification:** the frozen resid-mom L/S construction from `india_run.py`, changed
  ONLY in the short-leg universe: shorts drawn from the bottom signal ranks INTERSECTED
  with current F&O single-stock-futures underlyings. Any choice needed to rebuild the
  short leg (fill to full decile depth within F&O names vs accept a thinner leg) is
  decided on TRAIN evidence only, frozen, then one test read. Headline costs: 20 bps
  turnover on both legs (comparability with RL-07-10). Disclosed sensitivities, not
  counted as extra trials: short-leg carry credit +0/+3%/+5% ann. on short gross;
  futures roll drag 12 bps/yr; 10/40 bps cost check.
- **Predicted outcome:** test SR degrades from 0.86 to ~0.5–0.7 F&O-only; carry credit
  adds back roughly +1.5–2.5%/yr at 0.5 short gross. Expect real-but-below the strict
  t≥2 / DSR bar; the deliverable is the DEGRADATION measurement plus a live forward
  paper-track for the sleeve (separate ledger, `experiments/paper_trades_ls.jsonl`).
- **Known limitation (disclosed):** current F&O membership applied through history (no
  point-in-time F&O list) — overstates early-sample shortability; bias direction on
  performance is ambiguous and stated as such.

<!-- filled in AFTER the run -->
- **Result:** Ran on N500-277, 2010-01-04 -> 2026-07-09, test 2017-01-01, 20 bps both
  legs (`india_ls.py`; `experiments/log.jsonl`, hypothesis_ref RL-2026-07-12). The
  Groww instrument master (144,089 rows, cached read-only to `data/raw/`) gives **210
  F&O-shortable single stocks** (NSE FNO single-stock FUTURES underlyings that resolve
  to an NSE cash-equity row; index futures NIFTY/BANKNIFTY/MIDCPNIFTY/NIFTYNXT50/... and
  the synthetic *NSETEST underlyings excluded), of which **130 overlap** the 277-stock
  universe.
  - **Frozen short-leg choice = THIN** (mask the unrestricted book's shorts to F&O
    names, rescale short gross to long gross), decided on TRAIN (2010->2016) only. The
    two candidates were statistically indistinguishable on TRAIN (Sharpe **0.951 thin
    vs 0.975 fill**, delta 0.024 << the ~0.3 Sharpe SE), so the tie broke on principle:
    THIN changes ONLY the short leg (long leg bit-identical to the unrestricted book,
    diff 5e-18), whereas the re-rank "fill" alternative CONTAMINATES the long leg on
    1625/4055 dates - a full-universe long can fall in the bottom half of the F&O
    sub-universe and get a short weight, cancelling part of its long. THIN is also the
    honest/conservative book (short exactly the worst names actually shortable, no
    breadth padding). FILL rejected; not shipped.
  - **One test read (20 bps):** unrestricted baseline LS-RESID2 (apples-to-apples, same
    code/dates) test Sharpe **0.866**, t 2.64, ann +4.68%, maxDD -19.0%, CAPM beta vs
    Nifty 0.017. F&O-only headline (LS-FNO-THIN) test Sharpe **0.846**, t 2.58, ann
    **+5.38%**, maxDD **-16.3%**, CAPM beta **-0.000** (cleanly market-neutral), CAPM
    alpha-t 2.58. **Degradation from the F&O constraint = only 0.020 Sharpe** - far less
    than the pre-registered 0.5-0.7; the constrained book's return and drawdown actually
    IMPROVED (the excluded small-cap shorts were adding risk without alpha).
  - Correlation with the deployable REGIME long book (test window) = **+0.371** (partial
    diversifier, not fully uncorrelated). **Deflated Sharpe 0.000** against the full
    searched family (n_trials=32 factor+book Sharpes; cumulative India trials ~51) -
    fails the strict bar, as every book in this lab does.
  - **Disclosed sensitivities on the frozen weights** (short gross ~0.5): carry credit
    +3%/yr -> Sharpe **1.081** (ann +6.99%), +5%/yr -> **1.237** (+8.07%); futures roll
    drag 12 bps/yr -> 0.837; carry +3% net of roll -> 1.071; cost 10 bps -> 0.921, cost
    40 bps -> **0.695**. Basis carry is the dominant lever; roll drag is negligible.
  - **Forward paper-track STARTED:** `experiments/paper_trades_ls.jsonl` (separate ledger,
    signed weights). First live snapshot 2026-07-09: gross 1.00, net 0.0000, 138 long /
    58 short, all shorts F&O large-caps (WIPRO, TCS, ITC, TRENT, ...), 196/196 live Groww
    quotes, intraday +0.48%. `live_paper.py run_ls` (read-only; get_ltp + get_all_instruments).
- **Conclusion:** WORKED, better than predicted - and honestly bounded. Restricting the
  residual-momentum L/S short leg to F&O-shortable names costs almost nothing (Sharpe
  0.866 -> 0.846, a 0.02 haircut vs the feared 0.5-0.7), and the constrained sleeve is
  a genuinely IMPLEMENTABLE market-neutral book: ~+5.4%/yr, CAPM beta ~0, drawdown -16%,
  with single-stock-futures basis carry a large upside lever (+3-5% carry -> Sharpe
  1.08-1.24) and survival even at 40 bps (0.70). The F&O universe is the liquid large/
  mid-cap set, and resid-mom's short alpha lives there, not in the excluded small caps -
  so shortability is NOT the binding constraint it was assumed to be. BUT it is NOT
  statistically-proven alpha: DSR 0.000 (cumulative ~51 trials), t 2.58 clears a naive
  bar but not the trials-aware one, the 2017-26 window is multi-use, and current F&O
  membership applied through history overstates early-sample shortability (bias direction
  on performance ambiguous). Promoted as a deployable, implementable market-neutral sleeve
  with disclosed caveats; the forward paper-track (now live) is the only clean proof left.

---

## RL-2026-07-13 - Bear-only reversal as a small sleeve on the deployable book (thread #4)

- **Date (pre-registration):** 2026-07-09
- **Economic hypothesis:** Short-horizon reversal is a liquidity-provision premium that
  concentrates in bear/panic regimes (RL-2026-07-11: bear-gated reversal net Sharpe
  +0.32/+0.38 at 10 bps, ~break-even at 20; active ~21% of test days). Standalone it
  fails costs — but it is active EXACTLY when the deployable REGIME book is de-risked
  and holding ~50% idle cash, and its returns are uncorrelated with the long book.
  Allocating a small fixed slice of the defensive cash to the bear-only reversal book
  should improve the COMBINED book's risk-adjusted return without adding correlated risk.
- **Sample (locked):** as RL-2026-07-10/11 — panel from 2010, TRAIN 2010→2016-12-31,
  TEST 2017-01-01→now (one read; window heavily RE-USED, honesty flag). Sleeve on the
  nifty100 subset (liquidity, as the RL-07-11 short-term family), weekly rebalance;
  combined book rebalances as its components do. Costs 20 bps headline (10/40 checks).
- **Specification:** combined book = REGIME conviction book + sleeve, where the sleeve
  is ACTIVE only on days ^NSEI < its 200d MA, sized at a fixed fraction of NAV funded
  from the defensive cash. Frozen inputs: existing `short_term.py` books (`rev5` /
  `resid_rev`, vol-gated variants). TRAIN-only choices, then frozen: sleeve size (10%
  vs 20% of NAV) and reversal variant. One test read of the combined book vs the
  REGIME book alone: Sharpe, ann, maxDD, paired t on the return DIFFERENCE, sleeve
  standalone contribution, correlation. DSR against the family.
- **Predicted outcome:** modest improvement — combined Sharpe ~1.86→1.9±0.05, maxDD
  equal or slightly better, sleeve adds ~+0.5–1.0%/yr; at 20 bps a wash is entirely
  plausible and a NEGATIVE (no reliable improvement) verdict is the expected honest
  outcome roughly half the time. Not deployable unless the paired t on the difference
  is positive and the sleeve survives the 40 bps check without flipping sign.
- **Result:** (run 2026-07-09, `bear_sleeve.py`; log.jsonl RL-2026-07-13; orchestrator
  re-verified the 20/40 bps decision rows independently — exact match, and confirmed
  the gated sleeve is structurally flat on all risk-on rebalance rows.) Base book
  reproduced bit-for-bit (test SR 1.865/35.7%/−27.2%). TRAIN (2011→2016, disclosed
  1-yr warmup deviation) already showed ALL 6 configs (3 variants × sizes 0.10/0.20)
  BELOW base at 20 bps (dSR −0.013…−0.036); frozen the least-drag/most-diversifying
  config (resid_rev, s=0.10) for the one test read. TEST: @10 bps dSR +0.006
  (paired-t +1.12); @20 bps headline dSR **−0.006** (sleeve SR −0.23, paired-t
  −0.69); @40 bps dSR −0.030 with sleeve return flipping hard negative (−4.9%/yr,
  paired-t −3.91). Combined maxDD WORSE than base at every cost. Base–sleeve
  correlation 0.006–0.026 — the diversification premise held; the returns don't.
  Sleeve DSR 0.14 vs its variant family. Both pre-registered deployment gates fail.
- **Conclusion:** failed (wash-to-drag) — the ~50%-likely negative the registration
  flagged. Bear-only reversal's real gross edge cannot fund a sleeve at realistic
  Indian costs even when deployed only from idle defensive cash; it doesn't even buy
  drawdown protection. Closes handoff thread #4: short-term reversal is now fully
  cost-gated in every tested form (always-on, bear-only standalone, and combined
  sleeve). +6 disclosed configs to the trial tally; strict DSR verdict unchanged.

---

## RL-2026-07-14 - 52-week strength long book (anchoring/underreaction)

- **Date (pre-registration):** 2026-07-09
- **Economic hypothesis:** Proximity to the 52-week high predicts returns (George &
  Hwang 2004): investors anchor on the high and underreact to good news near it. The
  lab's TRAIN-only experiment zero (RL-2026-07-11) already measured a real top-decile
  gradient for the existing 52-week strength signal (`off_low`, +6.7%/yr active vs
  EW-277 on TRAIN) — weaker than momentum's +13.7% but never taken to a book. The
  open question is NOT whether it beats EW (momentum is stronger); it is whether a
  52w-strength book DIVERSIFIES the deployed momentum book (different anchor, lower
  active-return correlation) enough to improve a blend.
- **Sample (locked):** identical panel/split to RL-2026-07-10/11; monthly; 20 bps
  (10/40 checks); ret_clip=0.40. ONE test read (window re-use honesty flag).
- **Specification:** same construction as the deployed book — `blend.conviction_topq`
  top decile + (200MA OR India-VIX) regime overlay — on a 52w-strength signal.
  TRAIN-only choice, then frozen: signal variant `off_low` (existing, ratio off the
  252d low) vs George-Hwang `px / rolling_max(px, 252)` (proximity to the high).
  One test read: standalone vs EW-277 (paired active-t) and vs the momentum conviction
  book (active-return correlation); then a 50/50 signal-blend variant vs the momentum
  book alone (pre-registered as the ONLY blend tried). DSR against the family.
- **Predicted outcome:** standalone beats EW but below the momentum book (paired
  active-t vs EW ~0.5–1.0, under the t≥2 bar); active-return correlation with momentum
  HIGH (~0.6+) — the likely honest verdict is "real gradient, insufficient
  diversification, not promoted," with the 50/50 blend roughly matching, not beating,
  momentum alone. Promotion requires the blend to beat the momentum book on Sharpe
  AND the paired t of the improvement > 1 — a bar we predict it misses.
- **Result:** (run 2026-07-09, `h52_study.py`; log.jsonl RL-2026-07-14; orchestrator
  re-verified every decision number independently — exact match.) TRAIN froze
  `off_low` (train book SR 0.941, decile active +5.45%/yr) over George-Hwang
  `gh_high` (0.890, decile active NEGATIVE −4.55% on train). Construction validated:
  the momentum book under this code reproduces the deployed RL-07-11 headline
  exactly (test SR 1.865, ann 35.7%, maxDD −27.2%). One test read @20 bps: H52 SR
  1.571 (ann 33.8%, maxDD −50.6%, active-t vs EW 0.87), BLEND 1.776, MOM 1.865,
  EW-277 1.347. Active-return correlation H52-vs-MOM = 0.944 (predicted ~0.6+;
  realized far higher — same names, different label). Blend fails BOTH promotion
  conditions: Sharpe 1.776 < 1.865 and paired-t of the improvement +0.58 < 1;
  robust across 10/40 bps. Honesty receipt: the train-rejected gh_high variant
  would have LOOKED better on test blend Sharpe (1.94) but with negative paired
  mean-return t (−0.74) — the freeze prevented exactly that test-peeking trap.
- **Conclusion:** failed (not promoted) — as predicted. 52w-strength is a real but
  redundant expression of the momentum book (0.94 active correlation, double the
  drawdown standalone). Tightens the finding that the deployed conviction-momentum
  book already spans this anchor. +1 trial to the family; DSR verdict unchanged.

---

## RL-2026-07-15 - F&O forward-collection program (basis / PCR / IV) — forward-only

- **Date (pre-registration):** 2026-07-09
- **Data feasibility (measured, this account):** live F&O contracts serve daily candles
  over their ~3-month life; EXPIRED contracts are not resolvable via the API (symbol
  validation against the live master). Therefore NO historical basis/IV/PCR series can
  be reconstructed — these signals are FORWARD-ONLY on this stack. The deliverable is
  a daily read-only collector (`fno_collect.py` → `experiments/fno_daily.jsonl`) that
  starts the dataset now, plus the hypotheses below, pre-registered BEFORE any data
  exists so the eventual first read is clean.
- **Collected daily (read-only, ≤7 req/s):** per F&O underlying (~210): cash LTP,
  current- and next-month futures LTP → annualized basis; NIFTY option chain (nearest
  expiry): OI put-call ratio, ATM IV, and a fixed-moneyness IV skew.
- **Pre-registered forward hypotheses (first read ONLY after ≥126 trading days of
  collection; evaluation protocol locked now):**
  - **H1 basis cross-section:** stocks in deep backwardation (most negative annualized
    basis) are crowded shorts / hedged names; long bottom-decile basis vs short
    top-decile, monthly, 20 bps — predicted small positive spread (~3-6%/yr gross),
    market-neutral.
  - **H2 NIFTY PCR extremes:** OI-PCR above its trailing 90th percentile is a
    contrarian risk-ON signal (hedging washout). Predicted: modest timing value,
    additive to the 200MA/VIX overlay or nothing; a null is likely and acceptable.
  - **H3 IV skew:** steepening index skew (puts richening vs calls) leads drawdowns;
    predicted weak-positive as a de-risking confirm, redundant with VIX ~50% likely.
- **Evaluation (locked):** each hypothesis gets ONE read at the 126-day mark (then
  quarterly), paired against the deployed overlay where applicable; BH-FDR across the
  three; no peeking before the mark. Collection gaps (missed days) are tolerated and
  logged, never backfilled.

<!-- filled in AFTER the first 126-day read -->
- **Result:**
- **Conclusion:**

---

## RL-2026-07-16 - Risk-off sleeve: what should the defensive half own instead of cash?

- **Date (pre-registration):** 2026-07-09
- **Economic hypothesis:** The deployable REGIME book parks ~50% in zero-return cash
  whenever the (200MA OR India-VIX) overlay fires (~1/5 of days, including TODAY).
  Flight-to-safety assets historically outperform cash exactly then: gold in INR
  (global risk-off bid + rupee depreciation in stress; GOLDBEES.NS total-return
  history to 2009) and low-beta defensive stocks (betting-against-beta pays in
  drawdowns even though unconditional low-vol failed on train in the bull decade —
  the CONDITIONAL claim is different and untested here). Replacing risk-off cash
  with a defensive sleeve should raise the combined Sharpe without materially
  worsening drawdown.
- **Sample (locked):** the RL-2026-07-10/11 panel (N500-277 from 2010, ret_clip
  0.40) + GOLDBEES.NS adj_close; TRAIN 2010→2016-12-31, TEST 2017-01-01→now (ONE
  read; window re-use honesty flag); monthly where applicable, overlay switches as
  the deployed book does; 20 bps headline (10/40 checks).
- **Specification:** base = the deployed book (top-decile conviction momentum +
  overlay). The ONLY change: on risk-off days the freed weight (1 − book gross)
  goes to a sleeve instead of cash. Variants (all locked now, ONE chosen on TRAIN
  then frozen for a single test read): (a) cash [baseline = deployed], (b) gold —
  GOLDBEES held only while its own 200d trend is positive, else cash (avoids
  holding falling gold), (c) low-beta decile — inverse-vol weights on the panel's
  lowest-beta decile, (d) 50/50 gold+low-beta. Causality: all switches use
  prior-day signals, as the deployed overlay does.
- **Predicted outcome:** modest improvement — combined test SR 1.86 → 1.90–2.00
  with equal-or-better maxDD if gold carries its historical stress bid; a wash is
  plausible (~40%); low-beta-only likely adds equity beta back in crashes and
  underperforms gold. Deployment bar: paired-t of the daily difference vs the
  deployed book > 1 AND maxDD not worse by more than 2 points, robust at 10/40 bps.

<!-- filled in AFTER the run -->
- **Result:** (run 2026-07-09, `riskoff_sleeve.py`; orchestrator re-verified the
  frozen-variant decision numbers independently — exact match.) Base/cash variant
  reproduces the deployed book exactly (1.865/35.74%/−27.18%). TRAIN froze
  `lowbeta` (train combined SR 1.599 > gold_lowbeta 1.462 > cash 1.284 > gold
  1.136). Test read (all four PRE-SPECIFIED variants shown for disclosure; the
  verdict binds ONLY to the frozen one): frozen lowbeta combined SR 1.913
  (+42.1%/yr), paired-t vs deployed +1.83/+1.69/+1.41 at 10/20/40 bps — the
  return lift is real and robust — but maxDD worsens 2.2/3.1/4.8 points,
  breaching the 2-point cap at EVERY cost. **deploy = FALSE.** Attribution
  (transparent, not acted on): gold-only HURT (SR 1.836, maxDD −32%; its 2020/
  2022 own-drawdowns swamped the predicted stress bid — the prediction's gold
  thesis was wrong); the 50/50 gold_lowbeta would have cleared the bar (t 2.04,
  dMaxDD −0.4pt) but was NOT the train winner — adopting it post-hoc would be
  iterating on the hold-out, so the fail verdict stands. Ledger note: 3 stale
  RL-16 rows in log.jsonl from a placeholder first run (frozen_variant=gold);
  authoritative rows carry frozen_variant=lowbeta.
- **Conclusion:** failed the pre-registered bar (near-miss) — the honest ~40%
  outcome. Low-beta in the defensive half buys +6%/yr at the price of the very
  drawdown protection the overlay exists for; the book keeps CASH as its risk-off
  asset. gold_lowbeta is a legitimate candidate for a FUTURE, separately
  registered confirmation on forward data only — flagged, not promoted. +4
  disclosed variants to the family tally (~64); DSR verdict unchanged.

---

## RL-2026-07-17 - Multi-asset trend sleeve on Indian-accessible ETFs (long-term)

- **Date (pre-registration):** 2026-07-09
- **Economic hypothesis:** Time-series momentum is the best-documented cross-asset
  anomaly (Moskowitz–Ooi–Pedersen); a retail-implementable Indian version uses NSE
  ETFs spanning distinct return sources: NIFTYBEES (large-cap equity), JUNIORBEES
  (next-50), BANKBEES (banks), GOLDBEES (gold in INR), MON100 (Nasdaq-100 in INR,
  adds USD + global tech). Trend-gating each asset (long when its own trend is up,
  else cash) should produce positive risk-adjusted returns with only moderate
  correlation to the deployed equity book (gold + USD legs diversify INR equity
  stress). The sleeve's VALUE is as a portfolio diversifier, not a Sharpe contest.
- **Sample (locked):** the five ETFs above, Yahoo adj_close (depths measured
  2026-07-09: four from 2009, MON100 from 2011-03 — pre-inception = no position,
  disclosed); TRAIN 2010→2016-12-31, TEST 2017-01-01→now (ONE read; window re-use
  flag); monthly rebalance; 20 bps (10/40 checks).
- **Specification:** per-asset trend gate chosen on TRAIN from exactly two locked
  variants: 12-1 TSMOM sign vs price>200d-MA (`trend.py` primitives). Weighting
  chosen on TRAIN from exactly two locked variants: equal-weight vs inverse-vol
  (126d). Gated assets earn cash (0) when off. FREEZE both choices on TRAIN
  combined Sharpe, then one test read: standalone stats + correlation with the
  deployed REGIME book and with the F&O L/S sleeve.
- **Predicted outcome:** test SR 0.8–1.2, maxDD −10…−20%, corr(deployed book)
  0.3–0.5 (equity legs dominate it). Promotion bar (to a later, separately
  registered portfolio-blend study): standalone SR ≥ 0.8 AND corr < 0.5 at 20 bps.
  A miss on either is a valid negative.

<!-- filled in AFTER the run -->
- **Result:** (run 2026-07-09, `xasset_trend.py`; orchestrator re-verified headline,
  correlations, and base reproduction independently — exact match.) DATA REPAIR,
  decision-critical and independently adjudicated: Yahoo adj_close carries
  fabricated decimal-shift prints on 2019-12-19/20 (NIFTYBEES 129→13→130,
  GOLDBEES 33.6→0.34→33.7; BANKBEES likewise) that create a fake −54% sleeve day.
  GROWW CANDLES (independent source) show ~130.2/33.6 on those dates — the prints
  are false. Repair: causal transient-spike filter (>50% off trailing 5d median →
  drop + ffill), threshold set by physics not outcome; ret_clip=0.40 cannot fix it
  (clipped round-trip leaves a permanent ~16% level shift). Raw/glitched SR would
  be 0.416 (fail); repaired = the honest series. TRAIN froze tsmom + inverse-vol
  (SR 0.889 over ma/equal variants). ONE test read: SR 1.057, +10.97%/yr, maxDD
  −27.9% (10/40 bps: 1.074/1.023 — cost-insensitive at monthly turnover);
  corr(REGIME book) 0.357, corr(F&O L/S) 0.137. Bar: SR ≥ 0.8 ✓, corr < 0.5 ✓ —
  **PROMOTED** as a diversifier candidate. Prediction miss disclosed: maxDD −27.9%
  vs predicted −10…−20% (monthly trend cannot dodge the fast Mar-2020 crash).
- **Conclusion:** worked — first promotion since RL-2026-07-12. A retail-
  implementable 5-ETF trend sleeve earns ~11%/yr at SR ~1.06 with only 0.36
  correlation to the deployed equity book (gold + Nasdaq legs diversify).
  Next step is NOT deployment into the book: a separately registered
  portfolio-blend study (sizing REGIME + L/S + trend sleeve), preferably
  leaning on forward data. +4 trials to the family (~68); DSR verdict unchanged
  lab-wide. Data lesson graduated: Yahoo ETF series need the transient-spike
  guard; Groww is the adjudicating source for disputed prints.

---

## RL-2026-07-18 - Paper options book: NIFTY weekly short straddle (short-term, forward-only)

- **Date (pre-registration):** 2026-07-09
- **Economic hypothesis:** Index options embed a variance risk premium — implied vol
  exceeds subsequently realized vol on average, compensating crash insurance
  sellers. Indian weekly NIFTY options are liquid and the RL-15 collector already
  records the chain daily. A systematically SHORT ATM straddle harvests the premium
  with a fat LEFT tail — expected: positive median day, occasional large losses.
  Options history is unavailable (expired contracts unresolvable) so this is a
  FORWARD-ONLY PAPER book from day one; no backtest exists or will be claimed.
- **Specification (locked):** each run day, maintain ONE paper position: short 1
  ATM straddle (CE+PE at the strike nearest spot) on the nearest weekly expiry
  with ≥2 days to expiry; enter at chain LTPs; hold to expiry settlement (intrinsic
  at expiry-day spot) or roll when dte<2; notional = 1 lot; daily mark-to-market
  from chain LTPs appended to `experiments/paper_options.jsonl` (positions +
  daily P&L + ATM IV + trailing 5d realized vol so the premium itself is recorded).
  Marking at LTP (bid/ask not reliably available) is DISCLOSED as optimistic; no
  delta-hedging in v1 (the book is a variance + gamma-path bet, disclosed). Paper
  only by construction — the repo has no order path.
- **Evaluation (locked):** first read at 126 collection days, alongside RL-15:
  mean/median daily P&L, Sharpe of daily marks, worst day/week, realized-vs-implied
  premium capture. Predicted: positive median, negative skew, Sharpe 0.5–1.5 in a
  calm regime and severe drawdown in any vol spike — the tail is the point of
  measuring before sizing.

<!-- filled in AFTER the first 126-day read -->
- **Result:**
- **Conclusion:**

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
