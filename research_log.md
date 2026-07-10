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

## RL-2026-07-19 - Portfolio blend: sizing REGIME + F&O L/S + multi-asset trend

- **Date (pre-registration):** 2026-07-09
- **Economic hypothesis:** The lab now holds three promoted books with low mutual
  correlation (REGIME long-only SR 1.87; F&O L/S SR 0.85 at β≈0, corr 0.37 to
  REGIME; multi-asset trend SR 1.06, corr 0.36 to REGIME / 0.14 to L/S).
  Diversification math says a blend can carry the same return per unit risk with
  a shallower worst drawdown than REGIME alone. The open question is sizing —
  answered here ONLY with a-priori rules, never test-window optimization.
- **Sample (locked):** the three books rebuilt via their frozen constructions
  (india_run/blend REGIME; india_ls L/S; xasset_trend sleeve incl. its data
  repair); TRAIN 2010→2016-12-31 for the sizing choice (MON100 partial, L/S
  short universe uses the current F&O list — both disclosed as-is from their
  studies); TEST 2017-01-01→now, ONE read; 20 bps headline (10/40 checks).
  Returns-level blend, weights sum to 1 (fully-funded, conservative — the L/S
  sleeve's margin efficiency is ignored; disclosed).
- **Specification:** four locked sizing rules, ONE chosen on TRAIN combined
  Sharpe then frozen: (a) 100% REGIME [baseline], (b) equal thirds,
  (c) inverse-TRAIN-vol weights, (d) ERC on the TRAIN covariance of the three
  books' daily returns. One test read: frozen blend vs REGIME-alone — Sharpe,
  ann, maxDD, paired-t of the daily difference; all four rules' test rows shown
  for disclosure, verdict binds to the frozen one.
- **Predicted outcome:** blend SR 1.95–2.15 with maxDD improving 3–8 points on
  REGIME's −27% (the two diversifiers cushion equity drawdowns); ann return
  lower than REGIME alone (~25–30%/yr vs 35.7%) — the trade is return level for
  risk-adjusted quality. Deployment bar: paired-t > 1 AND maxDD better or equal,
  robust at 10/40 bps. Honest prior ~50% — REGIME's own Sharpe is a high hurdle.

<!-- filled in AFTER the run -->
- **Result:** (run 2026-07-09, `blend_portfolio.py`; orchestrator re-verified the
  frozen rule's weights and all decision numbers independently — exact match.)
  All three books reproduced their frozen headlines (1.865 / 0.846 / 1.05).
  TRAIN froze `invvol` (weights regime 0.21 / L/S 0.48 / trend 0.31; train SR
  1.564 over erc 1.561, thirds 1.512, regime-alone 1.290). Test read @20 bps:
  blend SR 1.781, ann +13.3%, maxDD **−12.65%** vs REGIME-alone 1.865 / +35.7% /
  −27.18%. maxDD half of the bar PASSES massively (+14.5 points shallower,
  ~2x the predicted improvement); paired-t half FAILS hard (−4.83 at 20 bps,
  −4.90/−4.70 at 10/40) because the blend gives up ~60% of REGIME's return
  level. **deploy = FALSE at all costs** — the frozen verdict binds.
- **Conclusion:** failed the locked bar — AND the bar was MIS-SPECIFIED, which
  is this study's real lesson (now in protocol.md): the registration predicted
  a LOWER return (that's what diversification does) yet demanded a paired-t on
  mean return > 1 — a risk-reduction thesis judged by a return-level test is
  unpassable by construction. The verdict stands (no post-hoc bar-switching);
  the honest reading of the data is that the blend is a DIFFERENT point on the
  frontier, neither dominated nor dominating: Sharpe ≈ REGIME (1.78 vs 1.87)
  at half the drawdown and a third of the return — a legitimate alternative
  for a risk-averse allocation, not an upgrade. Any formal risk-adjusted
  comparison (Sharpe-difference test, vol-matched idiom) belongs to a NEW
  registration evaluated on FORWARD data, not a re-read of this window.
  +4 sizing rules to the family tally (~72). DSR verdict unchanged.

---

## RL-2026-07-20 - Turn-of-month effect on the Indian index (short-term, cost-light)

- **Date (pre-registration):** 2026-07-09
- **Economic hypothesis:** Equity returns concentrate around month boundaries
  (documented in US/global data since Ariel 1987; in India plausibly AMPLIFIED by
  structural month-start flows: SIP auto-debits cluster in the first trading days,
  plus institutional window-dressing at month-end). Unlike the cost-gated
  short-term graveyard (reversal ~130%/wk turnover), a turn-of-month book trades
  ~24 one-way legs a YEAR on a liquid index ETF — the cost tax cannot kill it by
  construction; the only question is whether the concentration is real out of
  sample.
- **Sample (locked):** NIFTYBEES.NS adj_close (total-return, tradeable; cache to
  2009, through the ETF spike-repair from RL-2026-07-17), TRAIN 2010→2016-12-31,
  TEST 2017-01-01→now, ONE read. Cost 10 bps per side headline for the index ETF
  (5/20 sensitivity).
- **Specification:** hold the ETF only over the turn-of-month window [last N
  trading days of the month, first M of the next], cash (0) otherwise. Exactly
  four locked (N, M) variants: (3,2), (2,3), (1,3), (3,1). ONE chosen on TRAIN
  Sharpe, frozen, one test read. Diagnostics reported alongside the strategy
  read: mean daily return inside vs outside the frozen window with a
  two-sample t (the effect itself), and Sharpe vs buy-and-hold.
- **Predicted outcome:** the effect exists but has weakened globally post-2000;
  honest prior: TOM window captures 50–80% of buy-and-hold's return in ~25% of
  the days → TOM-only test Sharpe 0.9–1.3 vs B&H ~0.8, inside-vs-outside t
  1.5–2.5. Promotion bar: TOM Sharpe > B&H Sharpe AND inside-outside t ≥ 2 at
  10 bps, surviving 20 bps. A miss is a valid negative (global evidence says
  decay is likely).

<!-- filled in AFTER the run -->
- **Result:** (run 2026-07-09, `tom_study.py`; orchestrator re-verified all
  decision numbers independently — exact match.) TRAIN froze (N=2, M=3)
  (train SR 0.953 over (3,2) 0.816, (3,1) 0.814, (1,3) 0.553). Test read:
  TOM-only SR 0.436 (+3.1%/yr) at 10 bps, collapsing to 0.11 at 20; B&H
  NIFTYBEES 0.937. The effect is directionally present — window days earn
  9.4 bps/day vs 4.2 outside (~2.2x) — but the two-sample t is 1.13 (p≈0.26),
  far under the pre-registered t≥2, and the cash-heavy book gives up too much
  upside sitting out ~75% of days. Every bar clause fails.
- **Conclusion:** failed (not promoted) — the predicted-likely decay outcome.
  Turn-of-month in Indian large-caps 2017–2026 is a real-signed but
  statistically noisy tilt, not a strategy. Slots into the graveyard beside
  the other calendar/short-horizon effects. +4 variants to the family (~76);
  DSR verdict unchanged.

---

## RL-2026-07-21 - Vol-managed (continuous vol-target) overlay on the deployed book

- **Date (pre-registration):** 2026-07-09
- **Economic hypothesis:** Moreira–Muir (2017): scaling exposure by inverse
  realized variance raises Sharpe for momentum-type strategies, because
  volatility is highly persistent while expected return is not proportional to
  it. The deployed book's regime overlay is BINARY (fully in / cash); a
  CONTINUOUS vol-target is a different mechanism that also de-risks smoothly
  into vol clusters instead of at a threshold. Applied ON TOP of the deployed
  book (scale its daily weights), it should shave drawdown/vol more than return.
- **Sample (locked):** the deployed REGIME book rebuilt via its frozen
  construction; TRAIN 2010→2016-12-31, TEST 2017-01-01→now, ONE read; 20 bps
  headline (10/40) — the daily scale-factor trading is REAL turnover and must be
  costed through backtest_weights.
- **Specification:** scaled weights w_t · s_t with s_t = min(1, σ_target /
  σ̂_t) (leverage capped at 1 — long-only, implementable), σ̂_t = trailing
  realized vol of the BOOK's own returns, prior-day information only. Exactly
  four locked variants: σ_target ∈ {10%, 15%} × estimator window ∈ {21d, 63d}.
  ONE chosen on TRAIN Sharpe, frozen, one test read vs the deployed baseline:
  Sharpe, ann, maxDD, paired-t of the daily difference, plus the turnover cost
  drag attributable to scaling.
- **Predicted outcome:** test SR 1.87 → 1.95–2.10 with maxDD improving 3–6
  points and ann return giving up 3–8 points; the scaling turnover drag ~0.5–1%/
  yr. Deployment bar (same family as RL-16/19): paired-t > 1 AND maxDD
  better-or-equal, robust at 10/40 bps. Honest risk: the binary overlay already
  harvests most of the vol-timing gain, leaving this redundant (~40% prior of a
  wash).

<!-- filled in AFTER the run -->
- **Result:** (run 2026-07-09, `volmgmt_study.py`; orchestrator re-verified all
  decision numbers independently — exact match.) Base reproduced exactly
  (1.865, hard-asserted in-module). σ̂ from a FIXED causal reference (the
  unscaled book's gross returns — not the scaled book, no circularity;
  gross-vs-net choice disclosed). TRAIN froze σ_target 10% / 63d window
  (train SR 1.529 over the other three). Test read: scaled SR 1.737 vs
  baseline 1.865 (dSR −0.128), ann 18.8% vs 35.7% (~17 points given up),
  maxDD −20.5% vs −27.2% (+6.7 points better), paired-t ≈ −5.4 at every cost.
  Incremental cost drag NEGATIVE (−0.12…−0.50%/yr — de-levering shrinks
  trades), so costs are exonerated: the loss is pure de-levering of a
  positive-drift book. deploy = FALSE everywhere.
- **Conclusion:** failed — the SR-rises thesis is rejected out of sample, and
  the pre-registered redundancy risk realized worse than a wash: the binary
  (200MA OR VIX) overlay already harvests the vol-timing gain, and stacking a
  continuous vol-target on top only de-levers the book's drift. Moreira–Muir
  does not stack on a regime-switched book here. +4 variants to the family
  (~80); DSR verdict unchanged.

---

## RL-2026-07-22 - Forward program v2: trend-sleeve paper-track + locked forward evaluations

- **Date (pre-registration):** 2026-07-09
- **Purpose:** the multi-asset trend sleeve (RL-2026-07-17, PROMOTED) has zero
  forward evidence; and two flagged candidates from failed studies deserve cheap
  forward confirmation instead of another hold-out read. This entry adds the
  paper-tracks and LOCKS their evaluation protocols now, applying the RL-2026-07-19
  lesson: every bar below is stated in the risk-adjusted idiom (Sharpe-difference /
  drawdown), never a mean-return paired-t for a risk thesis.
- **New paper-tracks (read-only, same safety properties as the existing ledgers):**
  - (a) Multi-asset trend sleeve — frozen tsmom/invvol construction on the five
    ETFs, live Groww LTPs, ledger `experiments/paper_trades_trend.jsonl`.
  - (b) The RL-2026-07-16-flagged gold_lowbeta RISK-OFF VARIANT of the deployed
    book (50/50 trend-gated GOLDBEES + low-beta decile filling the freed weight on
    risk-off days) — observation only, no deployment claim; ledger
    `experiments/paper_trades_gl.jsonl`.
- **Locked forward evaluations (first read after ≥126 forward trading days from
  2026-07-10, alongside RL-15/RL-18; then quarterly; no peeking):**
  - **E1 (trend sleeve keeps promotion):** forward Sharpe > 0 AND realized
    corr(REGIME ledger) < 0.5. A miss demotes it.
  - **E2 (gold_lowbeta candidacy):** vs the deployed book's ledger — Ledoit-Wolf
    Sharpe-difference z > 1 AND forward maxDD not worse. Pass → eligible for a
    deployment registration; fail → the RL-16 verdict is confirmed and the flag
    is retired.
  - **E3 (blend, from ledgers):** the RL-19 frozen invvol blend (REGIME 0.21 /
    L/S 0.48 / trend 0.31) COMPUTED from the component ledgers (no new ledger) —
    Ledoit-Wolf Sharpe-difference z vs the REGIME ledger > 1. Pass → the blend is
    a certified frontier alternative; fail → retired.
- **Predicted outcomes:** E1 pass ~65% (trend is robust but 6 months is short);
  E2 genuine coin-flip (~50%) — that is why it gets forward data, not another
  hold-out read; E3 pass ~55% (the diversification is real; 126 days of Sharpe
  estimation is noisy).

<!-- filled in AFTER the first 126-day read -->
- **Result:**
- **Conclusion:**

---

## RL-2026-07-23 - Index-level band (Bollinger) mean reversion on NIFTYBEES

- **Date (pre-registration):** 2026-07-09
- **Economic hypothesis:** multi-day overreaction at the INDEX level — when price
  stretches far below its trailing mean, forced/panicked selling overshoots and
  reverts. Distinct from the cost-gated single-stock reversal family (index ETF,
  ~a handful of round trips per year, 10 bps). The lab's own trend evidence cuts
  the OTHER way at 12-month horizon; this tests the 1-month horizon where
  reversion classically lives.
- **Sample (locked):** NIFTYBEES.NS adj_close through the spike-repair guard,
  TRAIN 2010→2016-12-31, TEST 2017-01-01→now, ONE read; cost 10 bps per side
  headline (5/20 checks).
- **Specification:** z = (price − 20d MA) / 20d σ, prior-day info. LONG-ONLY
  timing book: enter when z < −k, exit at mean touch (z ≥ 0) or a time stop.
  Exactly four locked variants: k ∈ {1.5, 2.0} × exit ∈ {mean-touch, 10-day
  stop}. ONE frozen on TRAIN Sharpe; one test read. Report entries/yr and days
  invested (power disclosure).
- **Predicted outcome:** weak — likely 10–25% time invested, test Sharpe 0.4–0.9
  vs B&H ~0.94; the bar (proper idiom per RL-19 lesson): Ledoit-Wolf
  Sharpe-difference z vs B&H > 1 at 10 bps, surviving 20. Honest prior ~30%
  pass; a negative retires index mean-reversion timing alongside turn-of-month.

<!-- filled in AFTER the run -->
- **Result:** (run 2026-07-09, `band_mr.py`; orchestrator re-verified all decision
  numbers — exact, and cross-checked the LW z with an independent iid statistic,
  −2.51 vs −2.34, same sign/scale.) TRAIN froze (k=2.0, mean-touch) (train SR
  0.511 over the other three). Test: band book SR 0.253 (+2.0%/yr) at 10 bps,
  maxDD −36.5% while invested only 16.8% of days (3.9 entries/yr); B&H 0.937.
  Ledoit-Wolf z = −2.34/−2.66 at 10/20 bps — the book is SIGNIFICANTLY WORSE
  than buy-and-hold, below even the weak prediction. Mechanism: entering at −2σ
  with an uncapped mean-touch exit rides falling-knife crashes down to the mean
  (2020-style) — the book concentrates exactly the drawdowns B&H spreads out.
- **Conclusion:** failed, decisively — index mean-reversion timing joins
  turn-of-month in the graveyard, and with a stronger verdict: not merely
  insignificant but significantly harmful. Consistent with the lab's trend
  evidence (Indian index risk premium rewards holding/trend, punishes
  knife-catching). +4 variants to the family (~84); DSR verdict unchanged.

---

## RL-2026-07-24 - VIX spike-and-recede re-entry overlay on the deployed book

- **Date (pre-registration):** 2026-07-09
- **Economic hypothesis:** India-VIX spikes overshoot and mean-revert; equity
  returns in the weeks after a spike RECEDES are abnormally high (the risk
  premium being realized). The deployed book sits in CASH during high-VIX
  risk-off; a re-entry overlay that overrides risk-off and re-holds the
  momentum book while a spike is receding would harvest the recovery the binary
  overlay currently waits out.
- **POWER DISCLOSURE (locked up front):** ~15–20 independent spike episodes in
  2010–2026. Whatever the outcome, it rests on few events; a pass is promoted
  only to FORWARD confirmation (paper-track), never straight deployment, and a
  null is weak evidence — both stated now so neither can be oversold later.
- **Sample (locked):** the deployed book's frozen construction + ^INDIAVIX,
  TRAIN 2010→2016-12-31, TEST 2017-01-01→now, ONE read; 20 bps (10/40).
- **Specification:** spike = VIX above its trailing-252d p-th percentile;
  receding = VIX below its own trailing 5-day maximum. While spike-and-receding
  (both flags, prior-day info), the overlay overrides `regime_on` to TRUE for h
  days. Exactly four locked variants: p ∈ {90, 95} × h ∈ {10, 21}. ONE frozen
  on TRAIN combined Sharpe; one test read vs the deployed baseline.
- **Predicted outcome:** small positive — combined SR 1.87 → 1.88–1.95 by
  re-entering recoveries earlier (2020-04, 2022-06 style episodes); bar:
  Ledoit-Wolf Sharpe-difference z > 1 AND maxDD not worse by >2 pts, robust
  10/40. Prior ~40% pass given the episode count.

<!-- filled in AFTER the run -->
- **Result:** (run 2026-07-09, `vix_rebound.py`; orchestrator re-verified via a
  fully independent reconstruction from primitives — exact match on every
  decision number.) Base reproduced bit-identically (1.865). TRAIN froze
  p90/h10 (train SR 1.743 over the other three; base-alone 1.284). Test read
  @20 bps: overlaid SR 1.842 vs base 1.865 (LW z −0.114 ≈ 0), ann +42.0% vs
  +35.7% (the recovery harvest is REAL, +6.3 pts/yr), maxDD −29.6% vs −27.2%
  (worse by 2.4 pts — breaches the 2-pt cap); same shape at 10/40. Power:
  verdict rests on 13 re-entry episodes (374 override days) — the locked
  disclosure applies. Temptation refused and disclosed: non-frozen p90/h21
  showed z +0.69 but fails both clauses anyway; frozen verdict binds.
  **deploy = FALSE at all costs.**
- **Conclusion:** failed (the ~40%-prior honest negative) — but with a sharp
  mechanism finding: re-entering while VIX recedes captures the recovery's
  RETURN yet also its residual drawdown; risk-adjusted it's a wash and the
  drawdown cap breaks. The binary overlay's patience (wait for calm, not for
  receding panic) is again vindicated — third overlay study (RL-16, RL-21,
  RL-24) to confirm the deployed gate is hard to improve. +4 trials (~88);
  DSR verdict unchanged.

---

## RL-2026-07-25 - Intraday bar archive (infrastructure; enables future ORB/VWAP studies)

- **Date (pre-registration):** 2026-07-09
- **Purpose & measured constraint:** Groww serves intraday candles only for the
  trailing ~90 days (measured 2026-07-09: 60-min probes before 2026-04-10 return
  EMPTY) and no free source has Indian intraday history — so opening-range
  breakout, VWAP mean reversion, and any microstructure study are impossible to
  backtest today. The only path is ARCHIVE-BEFORE-EXPIRY: each run fetches bars
  since the last archived session and stores them locally. History then accrues
  at 1:1 real time.
- **Collector (locked):** 5-minute OHLCV bars for a locked universe — NIFTY
  index + current nifty100 constituents (list frozen in the module; membership
  drift disclosed as a limitation) — via `get_historical_candle_data` (5-min cap
  15d/request, so any gap ≤~90 days self-heals on the next run). Storage:
  `data/raw/intraday/` (git-ignored bulk data; DURABILITY RISK disclosed — this
  archive lives only on this machine; back it up externally). A coverage row per
  run goes to `experiments/intraday_archive.jsonl` (committed) so gaps are
  auditable in git even though the bars are not.
- **Locked future-study rule:** ORB and VWAP-reversion designs get their own
  pre-registrations BEFORE any read, and no first read occurs before ≥12 months
  of archived bars (≈2027-07). This entry locks the data program only — no
  strategy claims are made or implied now. Cost prior stated for the record:
  intraday turnover is exactly where this lab's cost evidence is most hostile;
  these studies must clear realistic intraday costs (spread + STT + slippage),
  not 20 bps.

<!-- filled in when the archive program is evaluated -->
- **Result (program go-live, 2026-07-09):** collector built, tested (7 tests), and
  wired into the daily snapshot. First pass archived the FULL surviving retention
  window: 101/101 symbols, 0 failures, 454,492 five-minute bars spanning
  2026-04-13→2026-07-09 (60 sessions × 75 bars/session exactly; orchestrator
  spot-verified bar counts, dedupe, monotonicity, no impossible bars). Measured
  retention floor ~87 days. Probe fact: index candles use trading_symbol 'NIFTY'
  (the NSE_ prefix is get_ltp-only); index bars carry volume=None. First-pass
  runtime ~27 min (latency-bound, ~2s/request serial — the registered ~90s
  estimate was wrong); daily incremental runs ~3–4 min. Audit-row undercount on
  the split first run disclosed (247,492 logged vs 454,492 on disk — the run
  timed out mid-pass and the incremental design self-healed on re-run).
- **Durability (owner-directed, 2026-07-09):** the archive is git-backed in the
  PRIVATE repo github.com/Akim-m/quantlab-intraday (data/raw/intraday is itself
  a git repo; the main repo still ignores it). Daily snapshot auto-commits and
  pushes new bars there — interim home until the owner picks a permanent store.
- **Conclusion:** live and accruing. ORB/VWAP registrations unlock ≈2027-07
  (≥12 months of bars), per the locked rule above.

---

## RL-2026-07-26 wave - new-strategy slate (5 pre-registrations, 2026-07-10)

Owner-directed autonomous research round. Five economically-grounded candidates
spanning long- and short-term. **MTC honesty:** only **-02** spends a hold-out
read (+4 variants → family tally ~92); **-01/-03/-04/-05 consume ZERO test-window
uses** (forward-only / observational). All five are registered here BEFORE any run.
Idea-selection was cross-checked against a full graveyard/data/MTC audit; two seed
ideas were rejected with receipts (overnight drift = cost-dead and already a logged
negative as anomaly factor 17; single-stock TSMOM = graveyard-equivalent, ≳0.8
return-corr with the deployed REGIME book).

## RL-2026-07-26-01 - ETF dual-momentum defensive rotation (DUAL-ROT)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** Relative momentum across five economically distinct
  asset-class sleeves (large-cap equity, next-50, banks, gold-INR, Nasdaq-INR)
  plus an absolute-momentum gate to cash (Antonacci 2014 dual momentum;
  Moskowitz-Ooi-Pedersen 2012 TSMOM). Concentrating in the strongest 1-2 of the
  five distinct return streams should earn more per unit risk than holding ALL
  up-trending sleeves at inverse-vol (the promoted RL-17 trend sleeve), because
  cross-asset relative strength persists at 3-12m while the absolute gate keeps
  crash protection. Gold/Nasdaq legs enter only via relative strength — no
  stress-bid timing thesis (that died in RL-16).
- **Sample (locked):** universe = NIFTYBEES/JUNIORBEES/BANKBEES/GOLDBEES/MON100
  (Yahoo adj_close). TRAIN 2010-01-01→2016-12-31 for DESIGN FREEZE ONLY (MON100
  from 2011-03; that month ineligible, disclosed). **No test-window read.** Forward
  clock from registration; first read ≥252 forward trading days (~2027-07), then
  quarterly, alongside RL-22. rebalance = monthly (ME), prior-day signals. cost =
  20 bps (10/40 sens).
- **Preprocessing (locked):** `xasset_trend.clean_prices` spike guard (mandatory,
  RL-17 lesson); no winsorization.
- **Specification:** top-K ∈ {1,2} × absolute gate ∈ {12-1 tsmom sign, px>200d MA}.
  Selected sleeves equal-weight (K=2 → 50/50); a selected sleeve failing its
  absolute gate holds cash. ONE variant frozen on TRAIN combined Sharpe. Maps to
  `trend.dual_momentum` + `xasset_trend` primitives.
- **Predicted outcome:** frozen-variant TRAIN Sharpe ~0.8-1.1; forward pass prior
  ~35-40%. Concentration cuts the diversification that made RL-17 work; a
  statistical TIE with the trend sleeve is the modal outcome. Bar (head-to-head
  risk-adjusted idiom): at first read, Ledoit-Wolf Sharpe-difference z vs
  `paper_trades_trend.jsonl` > 1 AND forward maxDD not worse by >2 pts AND
  corr(REGIME ledger) < 0.5. Pass → replacement/augmentation registration; fail →
  retired, trend sleeve keeps the slot. **Classification: FORWARD-ONLY** (new
  snapshot leg + `paper_trades_dualrot.jsonl`).

<!-- filled in when the forward leg goes live / at first locked read -->
- **Result (go-live 2026-07-10, `dualrot.py`):** construction built + wired into the
  daily snapshot; 20 dedicated tests + full suite 266 green. Orchestrator independently
  verified: the four TRAIN Sharpes reproduce EXACTLY — **K2/tsmom 0.693 (argmax)** over
  K2/ma 0.669, K1/tsmom 0.583, K1/ma 0.476 — and the live book matches an
  independent-from-primitives 12-1 momentum rank (MON100 +0.694, GOLDBEES +0.495 = the
  top-2 positive; JUNIORBEES +0.019 not top-2; BANKBEES −0.027, NIFTYBEES −0.078 gated
  out). Frozen: **top_k=2, gate=tsmom** (12-1 relative momentum for selection, 12-1 sign
  as the absolute crash gate). First forward row (panel 2026-07-09): GOLDBEES 50% +
  MON100 50%, cash 0%, live intraday +0.33% (quotes 2/2), ledger
  `experiments/paper_trades_dualrot.jsonl`. Note: weights are lagged one trading day (no
  look-ahead) — stricter than the RL-17 trend sleeve; freeze and live use the identical
  lagged construction, so what is frozen equals what trades.
- **Conclusion:** live and accruing (FORWARD-ONLY, zero hold-out use). First locked read
  ≥252 forward trading days (~2027-07): Ledoit-Wolf Sharpe-difference z vs the RL-17 trend
  ledger, maxDD-delta, corr(REGIME) — per the pre-registered head-to-head bar.

## RL-2026-07-26-02 - US-close trend spillover gate on NIFTYBEES (US-GATE)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** US equity returns lead non-US markets at weekly-monthly
  horizons and not vice versa (Rapach-Strauss-Zhou 2013, JF — gradual diffusion of
  information from the world's price-setting market). India's deployed gate is
  purely local (200MA/VIX); a US-trend gate carries information local prices have
  not fully impounded, especially at global risk transitions (2018Q4, 2020-02,
  2022). `^GSPC` has never been used as a signal in this lab — a genuinely new
  input series.
- **Causality clock (locked):** at India close day t (15:30 IST) the latest
  COMPLETE US session is t−1 (US close 01:30/02:30 IST on calendar day t). The
  close-t→close-t+1 position uses `^GSPC` through US day t−1 (shift one US trading
  day, ffill onto the NSE calendar). ≥13h information buffer; close-to-close book
  (no unadjusted-open problem).
- **Sample (locked):** universe = NIFTYBEES.NS adj_close (spike guard). TRAIN
  2010-01-01→2016-12-31, TEST 2017-01-01→run date. **ONE read.** rebalance = daily
  decision, multi-day holds. cost = 10 bps/leg (ETF convention RL-20/23; 5/20 sens).
- **Preprocessing (locked):** adj_close panel, spike guard, no winsorization;
  holiday mismatches ffilled on the LAGGED signal only.
- **Specification:** long NIFTYBEES when US signal ON else cash; signal ∈ {(a)
  `^GSPC` 21d ret>0, (b) 63d ret>0, (c) px>200d MA, (d) a AND c}. ONE frozen on
  TRAIN Sharpe; one test read. Disclosure arm (NO bearing on the verdict):
  NIFTYBEES gated by the INDIA 200MA — shows whether US info adds over the local gate.
- **Predicted outcome:** pass prior ~30%. The lab's index-timing-vs-B&H record is
  bad (TOM t=1.13; band-MR LW z −2.34, *significantly worse* than B&H) and three
  overlay studies say the local gate is hard to beat. Modal outcome: shallower
  maxDD, Sharpe ≈ B&H, LW z<1 — a valid negative retiring cross-market timing. Bar
  (timing idiom, RL-19/23 lesson): LW Sharpe-difference z vs B&H NIFTYBEES > 1 at
  10 bps, surviving 20 bps, AND maxDD better than B&H; promotion only to a forward
  paper-track and additionally requires reported corr(REGIME) < 0.8.
  **Classification: TEST-WINDOW-RUN** — the family's single hold-out spend (+4
  variants → tally ~92). Justified under the audit's 3-part rule: cross-market
  crisis-transition gate needs the 2018/20/22 episodes (only in history), so a
  forward track cannot evaluate it for years; new signal series; idiom-correct bar.

<!-- filled in AFTER the run -->
- **Result:** (run 2026-07-10, `us_gate_study.py`, 5 tests green; orchestrator
  re-verified via a FULLY INDEPENDENT reconstruction from primitives — rebuilt the
  signal + alignment without the study's helpers; EXACT match on every decision
  number.) TRAIN froze **ma200** (`^GSPC` px>200d MA), TRAIN SR 0.390 — argmax over
  ret63 0.199 / ret21 0.189 / ret21_ma200 0.156. TEST 2017-01-01→2026-07-09 (frozen
  ma200, 2353 days) @10 bps: net SR **1.043** vs B&H 0.935, ann **+11.5%** vs +13.5%,
  maxDD **−20.8%** vs B&H **−36.3%**, turnover 56, **LW Sharpe-diff z +0.397**. @5 bps
  z +0.501; @20 bps SR 0.987, z +0.190. Bar = LW z>1 @10 bps AND surviving 20 bps AND
  maxDD better than B&H → z@10 0.397, z@20 0.190 (both ≪ 1), maxDD_better TRUE → **FAIL**.
  Disclosure arm (NIFTYBEES gated by its OWN India 200d MA): SR 0.768, ann +7.8%, maxDD
  −18.6%, LW z −0.551 — the US gate DOES add over the purely-local gate (higher Sharpe,
  positive-though-subthreshold z where the local gate's is negative), but neither clears
  the bar. No-look-ahead guard proven load-bearing: removing the one-US-day shift leaks a
  same-day US close (position flips ON the flip date); the test asserts shift=1 acts only
  the next NSE session and that shift=0 leaks. **deploy = FALSE at all costs.**
- **Conclusion:** failed (the ~30%-prior honest negative) with a clean mechanism
  finding: the US-trend gate genuinely reduces drawdown (−20.8% vs −36.3%) and edges
  Sharpe above buy-and-hold, but the risk-adjusted improvement is statistically
  indistinguishable from B&H (LW z < 0.5) — cross-market INDEX timing does not clear the
  bar. It does dominate the purely-local 200MA gate (a minor positive worth recording).
  Cross-market index timing retired. **+4 trials (family ~92); DSR verdict unchanged.**
  This is the RL-26 wave's ONLY hold-out spend — the other candidates are forward-only.

## RL-2026-07-26-03 - NIFTY weekly cash-secured put-write paper book (PUT-W)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** harvest the index variance-risk premium as EQUITY
  REPLACEMENT — systematically short a ~2%-OTM weekly NIFTY put, fully
  cash-secured (CBOE PUT index; Ungar-Moran 2009). Distinct risk shape from
  RL-18's delta-neutral straddle: long-delta with a premium cushion vs a pure vol
  bet. Indian weekly index options are maximally liquid.
- **Sample (locked):** NIFTY weekly chain from the RL-15 collector; strike nearest
  0.98×spot, nearest expiry ≥2 DTE, hold to settlement / roll at DTE<2, 1 lot, no
  leverage. **No backtest exists or will be claimed** (expired contracts
  unresolvable — measured RL-15). First read at 126 collection days jointly with
  RL-18 under BH-FDR across the options family; sizing only after ≥252 days.
- **Preprocessing (locked):** marks at chain LTP (disclosed optimistic, RL-18
  convention); daily rows to `paper_options.jsonl` under a new book tag.
- **Specification:** exactly ONE variant (0.98 ratio, weekly, hold-to-expiry) —
  zero search dimensions.
- **Predicted outcome:** positive median week, negative skew; a 2020-style episode
  sinks the read. Prior it beats NIFTYBEES risk-adjusted over year one ~45% —
  observing the tail BEFORE sizing is the point. Bar (risk-adjusted vs the equity
  it replaces): at ≥252 forward days, LW Sharpe-difference z vs synchronized
  NIFTYBEES B&H > 1, worst-week + skew disclosed. Fail → retired; no interim
  claims. **Classification: FORWARD-ONLY.**

<!-- filled in when the harness lands / at the locked read -->
- **Result (go-live 2026-07-10, `paper_options_putw.py`):** harness LIVE + wired into the
  daily snapshot; 9 dedicated tests + RL-18's 9 regression tests green. Follows the
  RL-26-06 pattern (own module + own ledger `experiments/paper_options_putw.jsonl`,
  reusing paper_options helpers — only the put payoff is new arithmetic, pinned by a
  hand-computed ITM settle test: credit 100, strike 24000, settle 23800 → (100−200)×lot).
  Rules as registered: strike nearest 0.98×spot from the listed chain, nearest weekly
  ≥2 DTE, hold to European cash settlement or roll at DTE<2, marks at PE LTP (disclosed
  optimistic). Schema mirrors RL-18 + strike_ratio/otm_pct/cash_secured_notional; the
  `atm_iv` key holds the HELD-STRIKE IV (~2% OTM), not true ATM — schema-mirroring
  choice, documented. First position 2026-07-10: SHORT 23700 PE exp 2026-07-14
  (strike_ratio 0.9791, OTM 2.09%), credit 9.15 × lot 65, cash-secured 1,540,500;
  day-one mark 9.15, P&L 0.
- **Conclusion:** live and accruing (FORWARD-ONLY). First read at 126 collection days
  jointly with RL-18 under the options-family BH-FDR; sizing decisions only after ≥252
  days, LW Sharpe-diff z vs synchronized NIFTYBEES B&H per the registered bar.

## RL-2026-07-26-04 - Crowded-short (basis) filter on the F&O L/S short leg

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** deep single-stock-futures backwardation marks
  crowded/expensive-to-short names; crowded shorts are squeeze-prone, so their
  forward returns are LESS negative than the momentum signal implies. Excluding
  the deepest-backwardation decile from the short leg should cut short-leg tail
  risk without losing spread. **Observational first — the DEPLOYED L/S book is NOT
  touched** (protocol §5, no tweaking deployed models).
- **Sample (locked):** RL-15 basis rows × the live L/S ledger
  (`paper_trades_ls.jsonl`) — both lab-generated forward data. Read at 126 days
  (power thin, disclosed) and 252 days (the real read).
- **Preprocessing (locked):** stored annualized basis (4dp), daily cross-sectional
  deciles; no other transforms.
- **Specification:** ONE split — short-leg names in the bottom basis decile vs all
  other shorts; forward 21d returns from entry.
- **Predicted outcome:** prior ~35% the squeeze differential is material at 252
  days. Bar (short-leg RISK-claim idiom): mean forward 21d return of
  bottom-decile-basis shorts minus other shorts > 0 with t>1.5, worst-case
  contribution disclosed. Pass → separate deployment registration for the filtered
  sleeve; fail → retired. **Classification: FORWARD-ONLY** (observational;
  consumes no hold-out).

<!-- filled in at the locked reads -->
- **Result:** (filled at 126/252-day reads.)
- **Conclusion:** pending forward evidence.

## RL-2026-07-26-05 - Futures basis-momentum cross-section (F&O H4)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** basis-momentum (front minus second-contract momentum)
  predicts futures returns beyond basis and price momentum (Boons-Moreira da Prado
  2019, commodities), proxying positioning pressure on the curve. In single-stock
  futures, curve-slope shifts reflect short-demand/financing pressure. Long high /
  short low basis-momentum across the 210 F&O names.
- **Sample (locked):** RL-15 collector (cash/fut1/fut2 LTP daily since
  2026-07-09) — forward-only by measurement. Registered now, BEFORE any usable
  formation window exists. First read at the RL-15 126-day mark if ≥63 formation
  days exist, else 252; joins RL-15's BH-FDR family (H1-H3 + this).
- **Preprocessing (locked):** 21d basis-momentum; cross-sectional winsorization at
  ±3 MAD; names missing fut2 excluded that day.
- **Specification:** ONE variant — decile L/S, equal-weight, monthly; paper signal
  portfolio only.
- **Predicted outcome:** prior ~25% — equity-futures basis is mostly mechanical
  (rate − dividend), leaving less curve signal than commodities; a null retires it
  at near-zero cost. Bar (dollar-neutral return-spread idiom): net spread t>1.5
  within the family BH-FDR; promotion only to a paper-track. cost = 20 bps (10/40);
  ~2-5%/yr drag vs literature 5-10%/yr — margin thin, said plainly.
  **Classification: FORWARD-ONLY.**

<!-- filled in at the locked read -->
- **Result:** (filled at the locked read.)
- **Conclusion:** pending forward evidence.

---

## RL-2026-07-26 wave 2 - seven more (forward-only / blocked-pending-data, 2026-07-10)

Second research wave (owner: "don't stop at 5, add and find more"). Every proposal is
FORWARD-ONLY or BLOCKED-PENDING-DATA — **zero test-window reads** (family tally stays
~92; only wave-1's -02 spent one). Idea space cross-checked in CODE against what the
collector actually logs. Five ideas rejected with receipts: covered-call/buy-write
(put-call-parity duplicate of -03), US ^VIX index gate (same slot as -02), FIP momentum
(RL-14 redundancy), largecap→midcap lead-lag (special case of -01), NIFTY index-futures
basis timing (hostile index-timing record + input not collected). Forward BH-FDR
families: options {RL-18, -03, -06, -07}; F&O collector {H1-H3, -05, -04, -10, -12};
equity-forward {-08, -09}; events {-11}.

## RL-2026-07-26-06 - VRP-gated short volatility (conditional variance-premium harvest)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** the variance risk premium is time-varying and concentrates
  when implied vol is rich vs realized (Bollerslev-Tauchen-Zhou 2009 — VRP is the
  *conditional* compensation for variance risk). RL-18 (straddle) and -03 (put-write)
  harvest it UNCONDITIONALLY; the claim is the harvest is materially better risk-adjusted
  when the measured premium (ATM_IV − RV5d) is fat, and skipping thin-premium weeks avoids
  the worst gamma losses. First conditioning study on the OPTIONS family (the RL-16/21/24
  "overlay low-prior" receipt applies to the deployed equity gate, not here).
- **Sample (locked):** NIFTY weekly ATM straddle = RL-18's frozen construction; signal
  VRP_t = ATM_IV (fno/paper ledgers) − RV5d (`paper_options.realized_vol_5d`). Forward
  clock from registration; first read ≥252 days, jointly with RL-18/-03 under BH-FDR. No
  backtest (expired contracts unresolvable — RL-15).
- **Preprocessing (locked):** VRP percentile on trailing 126 collection days, prior-day
  info; marks at chain LTP (disclosed optimistic, RL-18 convention).
- **Specification:** ONE variant, zero search — hold the RL-18 short straddle only while
  VRP > its trailing-126d median, else flat. New book tag in `paper_options.jsonl`.
- **Predicted outcome:** prior ~40-45% (best of wave 2, still sub-coin-flip — the gate
  halves the sample; a wash is plausible and itself informative). Bar (conditional
  risk-adjusted idiom): LW Sharpe-diff z of gated-vs-ungated synchronized daily marks > 1
  AND gated worst-week not worse. **Classification: FORWARD-ONLY** (all inputs collected
  daily; zero hold-out). Nearest: -03 (same premium source, but -03 is unconditional
  equity-replacement; this is a timing claim gated-vs-ungated) / graveyard RL-21 (vol
  conditioning, but that scaled an equity book by realized vol; here signal is IV−RV and
  the asset is the premium). **BUILD NEXT.**

<!-- filled in at the locked read -->
- **Result (go-live 2026-07-10, `paper_options_vrp.py`):** harness LIVE + wired into the
  daily snapshot; 10 dedicated tests + full suite 276 green. Reuses RL-18's option
  arithmetic (bit-consistent). Gate = VRP_t (ATM_IV − RV5d) > median(≤126 prior non-null
  VRP), strictly greater; today's VRP is structurally excluded from its own median window
  (no look-ahead — tested with prior [1,2], today 100 → median 1.5). Warm-up: 0 prior → gate
  defaults OFF (don't short vol with no evidence the premium beats its own median). DAILY
  gate (can flatten mid-week) per the "synchronized daily marks" bar idiom. Data gap → carry
  a held position, flat stays flat. Own ledger `experiments/paper_options_vrp.jsonl`
  (SEPARATE file — co-mingling into `paper_options.jsonl` would corrupt RL-18's `_last_row`
  state; matches the per-book convention of the other four sleeves; `book="vrp_gated_
  straddle"` tag on every row). First row 2026-07-10: VRP −8.29 (ATM_IV 10.01 − RV5d 18.30),
  gate OFF (warmup, n_hist=0), flat.
- **Conclusion:** live and accruing (FORWARD-ONLY, zero hold-out). First read ≥252 forward
  days jointly with RL-18/-03 under BH-FDR: LW Sharpe-difference z of gated-vs-ungated
  synchronized daily marks. Note: the median needs ~126 collection days, so the gate is in
  warm-up (OFF on negative-VRP days) until ~2027-01.

## RL-2026-07-26-07 - NIFTY IV term-structure slope as a short-vol stress gate

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** the vol term structure inverts under stress (near-IV > far-IV)
  and the slope prices the variance premium's term structure (Johnson 2017, JF): inversion
  predicts elevated near-term realized vol and poor short-vol returns. The MATURITY axis is
  untouched here — RL-15 H3 covers only moneyness (skew).
- **Sample (locked):** NIFTY nearest-weekly ATM IV (already logged) + **next-monthly-expiry
  ATM IV (NEW — one extra chain fetch/day in `fno_collect.py`)**. Slope = IV_far − IV_near.
  First read ≥252 days after the extension goes live.
- **Preprocessing (locked):** ATM strike nearest spot per expiry; slope in annualized IV
  pts; no winsorization; prior-day info for gating.
- **Specification:** ≤2 locked claims, one variant each — (i) signal validity: forward 5d
  NIFTY realized vol higher after inversion (slope<0) days, two-sample t>2; (ii)
  application: RL-18 straddle gated to slope≥0 improves worst-week/maxDD vs ungated, LW z
  not worse than −0.5.
- **Predicted outcome:** prior ~35% (i), ~25% (ii). Power disclosed up front: inversions
  rare (~5-15% of days → ~12-40 events yr 1); a null at 252 days is weak, stated as such.
  Bar: both clauses inside the options-family BH-FDR. **Classification: BLOCKED PENDING
  DATA — confirm collector extension** (one added chain fetch/day, ~1 request, inside the
  7 req/s budget); forward-only thereafter. Every day before the extension is unrecoverable
  (archive-before-expiry logic, RL-25). Nearest: RL-15 H3 (moneyness vs maturity axis) /
  graveyard RL-24 (VIX-shape timing of equity re-entry, not the short-vol book).
  **BUILD NEXT — collector extension is time-urgent.**

<!-- filled in at the locked read -->
- **Result (collector extension LIVE, 2026-07-10):** `fno_collect.py` now logs the
  NIFTY next-monthly-expiry ATM IV each day — strictly additive schema (`far_expiry`,
  `atm_iv_far`, `iv_slope` = far − near in annualized IV pts; every old field
  byte-compatible, `chain_ok` still near-leg-only). Far expiry = nearest monthly
  strictly after the near expiry, monthly = last listed expiry in its calendar month;
  one extra chain fetch/day; far-leg failure writes the near row intact with
  `atm_iv_far=None`. 11 new tests + 8 existing fno tests green. Live validation
  (orchestrator): near 2026-07-14 ATM_IV 9.49, **far 2026-07-28 ATM_IV_far 11.26,
  slope +1.77** — far chain populated, monthly detection correct (weekly→monthly gap
  14d), no inversion on day one. Note: today's ledger carries two rows (the pre-
  extension morning row + this richer one); read-time last-per-date dedup applies.
  **The -07 forward clock starts 2026-07-10**; days before today are unrecoverable
  (disclosed at registration).
- **Conclusion:** data flowing; strategy claims (i)/(ii) pending the ≥252-day read
  (~2027-07) inside the options-family BH-FDR.

## RL-2026-07-26-08 - Macro-sensitivity alignment cross-section (USDINR + Brent betas)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** India is a net oil importer with a persistently depreciating
  managed-float currency; single-stock USDINR/crude exposures are economically
  heterogeneous (IT/pharma exporters gain from INR weakness; oil-marketers/aviation/paints
  lose from crude spikes), and exchange-rate exposure is under-reacted-to at monthly
  horizons (Adler-Dumas; Hong-Stein diffusion). Long names whose macro beta ALIGNS with
  the prevailing macro trend, short the misaligned — dollar-neutral STANDALONE book, not an
  index overlay. Macro series are genuinely NEW inputs to this lab.
- **Sample (locked):** N500-277 (Yahoo adj_close, ret_clip 0.40, survivorship disclosed);
  macro USDINR (`INR=X`), Brent (`BZ=F`; fallback `CL=F`). Forward-only paper signal
  portfolio; history used for estimation warm-up ONLY, no historical performance read;
  first read ≥252 forward days.
- **Preprocessing (locked):** 252d rolling OLS betas on LAGGED macro returns; causality
  clock: Brent settles after NSE close → macro returns enter at t−1 (the -02 discipline);
  betas winsorized ±3 MAD.
- **Specification:** ONE variant — alignment = β_INR·sign(USDINR 63d trend) +
  β_oil·sign(−Brent 63d trend); decile L/S, equal-weight, monthly; paper ledger.
- **Predicted outcome:** prior ~25-30% (daily-OLS betas noisy; the alignment interaction
  compounds two estimations). Hold-out temptation (2013/2018/2022 in-window) considered and
  REJECTED — prior too low for the 3-part test; monthly x-sec accrues forward adequately.
  Bar (L/S spread idiom): net spread t>1.5 at 252 days, inside a wave-2 forward BH-FDR
  family. **Classification: FORWARD-ONLY** (confirm `INR=X`/`BZ=F` depth in the Yahoo cache
  before go-live; if either fails → BLOCKED PENDING DATA). Nearest: -02 (macro/cross-market,
  but -02 times the INDEX; this is dollar-neutral stock selection, zero index-timing) /
  graveyard 32-factor family (distinct — needs NEW input series, not another price transform).

<!-- filled in at go-live / the locked read -->
- **Result (data-confirm 2026-07-10 — PASS):** `INR=X` 2003-12→2026-07-10 (n=5863, 0 NaN,
  zero |ret|>10% prints; TRAIN coverage from 2010-01-01). `BZ=F` 2007-07→2026-07-10
  (n=4715, 0 NaN; 24 |ret|>10% days spot-checked = documented oil history — 2020-04
  COVID, 2022-03 Ukraine, 2026-03/04 — genuine moves, NOT vendor spikes; the
  `clean_prices` transient-spike guard applies at load per the RL-17 convention).
  **UNBLOCKED → build proceeding.** No fallback to `CL=F` needed.
- **Result (go-live 2026-07-10, `macrobeta.py`):** construction built per the locked spec;
  14 dedicated tests green. Implementation decisions (disclosed): betas from ONE
  bivariate 252d rolling OLS per name (intercept + β_INR + β_oil, closed-form
  rolling-covariance algebra, min_periods=252 → full-window-only estimates); macro
  series `clean_prices`-guarded on their native calendar, THEN reindexed to the panel
  with ffill(limit=5); trend path carries two causal lags (63d trend sign reads levels
  through t−1, weights lag a further day); β_INR/β_oil winsorized ±3 MAD independently
  per date. Orchestrator verification: repro command reproduces the report EXACTLY;
  TCS.NS raw betas from the module (+0.45008/−0.04540) match an independently-written
  lstsq implementation on a different calendar alignment (+0.4501/−0.0454) to 4dp;
  causality test proven load-bearing (bar-T macro injection: byte-identical ≤T with
  the lag, moves bar-T without it). First forward row (panel 2026-07-09): 27L/27S,
  gross 1.0, net 0, quotes 54/54, intraday −0.40%. Macro state: USDINR 63d trend +1
  (weakening), Brent 63d trend −1 → longs = INR-weakness winners + oil-fall winners
  (#1 KPRMILL β_INR +0.704 — textile exporter, economically sane); shorts led by
  SONATSOFTW (noisy negative β_INR on an IT name — the registered estimation-noise
  risk, visible from day one and disclosed). **Registered disclosure:** Spearman
  (alignment, -15 turnover-shock) = +0.086, n=277 — a distinct bet from the existing
  volume sleeve. Ledger `experiments/paper_trades_macrobeta.jsonl`.
- **Conclusion:** live and accruing (FORWARD-ONLY). First read ≥252 forward days: net
  spread t>1.5 inside the equity-forward BH-FDR family.

## RL-2026-07-26-09 - Same-sector F&O cointegration pairs (relative value)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** economically twinned large-caps (HDFCBANK/ICICIBANK, TCS/INFY,
  MARUTI/M&M, JSWSTEEL/TATASTEEL, …) share cash-flow drivers; idiosyncratic order-flow
  shocks open temporary spread dislocations that revert (Gatev-Goetzmann-Rouwenhorst 2006).
  HEDGED relative value — genuinely untested here (the 32-family `pairs` factor used US
  symbols, excluded in every India run). Both legs ∩ the 210 F&O names (actually shortable,
  RL-12).
- **Sample (locked):** F&O ∩ N500 (130, RL-12), pairs formed WITHIN NSE industry; formation
  (Engle-Granger + half-life<60d + spread-vol filter, top 10 pairs by in-formation
  stability) FROZEN at registration, forward paper-track only — no historical P&L (formation
  window contaminated by construction).
- **Preprocessing (locked):** log-price spreads on adj_close; spread z on 63d rolling,
  prior-day info; ret_clip 0.40.
- **Specification:** ONE variant — enter |z|≥2 (long cheap/short rich, equal gross/leg),
  exit z=0 or 30-day time stop; ≤10 concurrent pairs, equal capital.
- **Predicted outcome:** prior ~20-25% (lab MR record uniformly hostile; pairs profitability
  decayed post-2000s, Do-Faff; formation-selection overfit channel). Modal = cost-eaten
  wash (~80 bps of pair gross per round trip vs 1.5-3% expected convergence). Bar
  (hedged-spread idiom): at ≥252 days, combined pair-book net Sharpe>0 AND per-trade net
  convergence t>1.5. **Classification: FORWARD-ONLY** (inputs exist today; formation frozen
  at registration, trading strictly forward). Nearest: graveyard short-term reversal /
  RL-23 band-MR (outright price-level bets; this is cross-hedged spread reversion between
  cointegrated twins, weeks not days, market/sector risk netted) / -05 (futures curve vs
  cash spreads).

<!-- filled in at the locked read -->
- **Result (go-live 2026-07-10, `pairs_rv.py`):** formation run once and FROZEN as module
  constants; 20 dedicated tests + full suite 356 green. Universe = the RL-12 F&O∩N500
  130 names → 701 same-industry candidates over 2018-01-02→2026-07-09; three filters
  (Engle-Granger ADF < −3.34 [minimal lag-1 ADF, MacKinnon ~5% cv, approximation
  disclosed], AR(1) half-life < 60d, spread σ ≥ 0.5%) pass 59; top 10 frozen by
  |ADF t|/half-life — economically coherent twins (JSWSTEEL/TATASTEEL, IOC/ONGC,
  M&M/MARUTI, HDFCBANK/KOTAKBANK, BPCL/PETRONET, …). Orchestrator independently
  re-derived JSWSTEEL/TATASTEEL from primitives: β 0.957 (exact), ADF −4.93 vs −4.90,
  half-life ~41d, σ 1.51% (exact) — formation is real. NO historical P&L computed (the
  formation window is contaminated by construction — only forward trading is evidence).
  Live state machine: enter |z|≥2 (63d rolling z, frozen β, prior-day info), exit z=0
  or 30d time stop, ≤10 concurrent, per-pair dollar-neutral gross 0.1. First row (panel
  2026-07-09): **0/10 open — all spreads inside ±2** (closest HDFCBANK/KOTAKBANK z
  +1.96), book all cash. Honest flags recorded: formation-selection overfit is the
  registered risk; equal-dollar legs ≠ β-hedged (BAJFINANCE/KOTAKBANK β 2.56 carries
  residual exposure); shared names (MARUTI, KOTAKBANK ×2) can stack to ±0.10; time-stop
  re-entry adds churn (implemented literally per the registered spec); round trip ≈
  80 bps of pair gross vs 1.5-3% expected convergence.
- **Conclusion:** live and accruing (FORWARD-ONLY). Read at ≥252 forward days: combined
  book net Sharpe > 0 AND per-trade net convergence t > 1.5, per the registered bar —
  the modal predicted outcome remains a cost-eaten wash (~20-25% prior).

## RL-2026-07-26-10 - Single-stock futures OI-positioning cross-section

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** open interest measures NET NEW POSITIONING, not activity: rising
  OI + rising price = levered longs entering (continuation); rising OI + falling price =
  shorts building (continuation); falling OI = unwinds (exhaustion). Hong-Yogo (JFE 2012):
  futures OI growth predicts returns beyond price momentum in commodities; the standard
  Indian F&O desk heuristic, never tested honestly here.
- **Sample (locked):** ~210 F&O underlyings; signal = rank of sign(21d cash return) ×
  21d fut1-OI growth; forward-only from the day the collector logs OI.
- **Preprocessing (locked):** OI growth winsorized ±3 MAD cross-sectionally; names missing
  fut1 OI excluded that day (-05 convention).
- **Specification:** ONE variant — decile L/S, equal-weight, monthly; paper portfolio in the
  RL-15 family.
- **Predicted outcome:** prior ~30% (equity-SSF OI is dominated by hedging/arb inventory,
  not speculative positioning — the commodity result may not transplant). Bar (L/S spread):
  net spread t>1.5 in the RL-15 BH-FDR family (grows by one, disclosed). **Classification:
  BLOCKED PENDING DATA — confirm source:** per-name futures OI is NOT in `fno_daily.jsonl`
  today (verified in code — only NIFTY OI-PCR exists). Probe Groww futures `get_quote` for
  an OI field, then extend the collector (piggyback the existing LTP calls if possible).
  Forward-only thereafter; uncollected days lost. Nearest: -04/-05 (basis = a PRICE on the
  curve; OI = a QUANTITY/positioning) / graveyard factor-32 volume_momentum (cash activity
  vs open positioning — different object).

<!-- filled in once OI is collected / at the read -->
- **Result:** (filled after the OI-field probe + collector extension.)
- **Conclusion:** pending data + forward evidence.

## RL-2026-07-26-11 - F&O ban-list (MWPL) crowding events

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** NSE bans new F&O positions in a stock when aggregate OI crosses
  95% of the market-wide position limit — the exchange itself flagging extreme crowding.
  Ban entry after a run-up marks speculative saturation with a BINDING constraint (no new
  longs; levered longs must eventually unwind) → predicted negative abnormal drift
  post-ban-entry + elevated vol through the ban window. India-specific, structural, outside
  the exhausted price-transform space.
- **Sample (locked):** all names entering the ban list; forward collection from go-live
  (NSE publishes daily; a clean historical archive, if confirmed, would allow a SEPARATELY
  registered event study — not assumed).
- **Preprocessing (locked):** abnormal return = stock − its NSE-industry EW peer basket
  (survivorship-light event design); events overlapping <3 days deduped to first entry.
- **Specification:** ONE variant — event = first day in the ban list; forward 5d/21d
  abnormal returns + realized vol vs trailing-63d baseline; directional sub-hypothesis
  (locked): events preceded by a positive 21d run underperform.
- **Predicted outcome:** prior ~30% vol claim, ~20-25% directional. Power: ~30-80 events/yr;
  a one-year read is thin, graded as such. Bar (event idiom): mean 21d abnormal return t>2
  (directional) + vol elevation t>2 (risk), BH-FDR across the two. **Classification:
  BLOCKED PENDING DATA — confirm source** (NSE daily ban-list feed + a scrape leg in the
  snapshot; zero Groww dependency). Observational first; tradable form (avoid/underweight on
  the L/S long leg, or a short screen) registered separately on a pass. Nearest: -04 (both
  short-side crowding — but -04 uses a CONTINUOUS basis proxy on the sleeve's own shorts;
  this uses the exchange's DISCRETE binding constraint) / graveyard short-term reversal
  (structural forced-unwind, not generic price reversal; observational so the cost gate
  doesn't apply yet).

<!-- filled in once the ban-list feed is wired / at the read -->
- **Result (data collector LIVE, 2026-07-10):** `nse_events.collect_ban_list` wired as a
  daily snapshot leg → `experiments/nse_ban_list.jsonl` (committed). Source probing
  (honest): the nseindia.com JSON API is Akamai-blocked from this machine (403 warmup /
  404 API); the authoritative working source is the CSV archive
  `nsearchives.nseindia.com/content/fo/fo_secban.csv` (HTTP 200, header carries the
  official trade date). Stdlib urllib + browser headers, matching the india.py
  convention; JSON path kept as first-try with the failure recorded per row. Day-one
  row: trade_date 2026-07-10, **n_banned=1 (KAYNES)**. 17 tests green (fixtures, no
  live HTTP in tests). **The -11 event clock starts 2026-07-10**; earlier ban events
  are unrecoverable (disclosed).
- **Conclusion:** data flowing; event claims pending accrual (~30-80 events/yr expected)
  and the registered t>2 bars at the read.

## RL-2026-07-26-12 - Basis-dispersion (limits-to-arbitrage) conditioning of the L/S sleeve

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** cross-sectional dispersion of single-stock basis (std of `b1`
  across ~210 names — computable from day one of the existing collector) proxies
  arbitrage-capital health: when carry arbitrageurs are impaired, dispersion blows out
  (limits-to-arbitrage; Pasquariello dislocations). The F&O L/S sleeve is itself a
  convergence-type book; its forward returns should be worse / more volatile in
  high-dispersion states. Observational only — **the deployed sleeve is NOT touched**
  (protocol §5), the -04 pattern.
- **Sample (locked):** `fno_daily.jsonl` basis × the live `paper_trades_ls.jsonl` ledger —
  both lab-generated forward data; reads at 126 days (thin, disclosed) and 252 days.
- **Preprocessing (locked):** daily dispersion = cross-sectional std of `b1` over `fut1_ok`
  names; trailing-126d percentile, prior-day info.
- **Specification:** ONE split — sleeve forward 21d ledger return + vol in
  top-quartile-dispersion states vs the rest.
- **Predicted outcome:** prior ~25-30% (one year gives few independent 21d windows — power
  disclosed). Bar (conditioning idiom): difference in mean forward 21d sleeve return
  (top-quartile vs rest) t>1.5, vol difference reported. Pass → a separately registered
  sizing rule; fail → retired. **Classification: FORWARD-ONLY** (zero new data, zero
  hold-out). Nearest: -04 (observational conditioning of the same sleeve — but -04 filters
  WHICH shorts at name level; this times HOW MUCH sleeve from an aggregate state) /
  graveyard RL-16/21/24 overlay family (the reason the prior is low; distinct mechanism =
  arbitrage-capital state, and a target book never studied for conditioning).

<!-- filled in at the 126/252-day reads -->
- **Result:** (filled at the locked reads.)
- **Conclusion:** pending forward evidence.

---

## RL-2026-07-26 wave 3 - six more (2026-07-10)

Third research wave. All FORWARD-ONLY or BLOCKED-PENDING-DATA — **zero test-window reads**
(family tally stays ~92; none passes the 3-part hold-out rule, as expected). Six ideas
rejected with receipts: trend-strength-scaled sleeve (RL-06-28-01 OOS death + RL-21
de-lever), breadth/dispersion gate (4th gate study vs the RL-16/21/24+-02 wall),
PCR-change momentum (RL-15 H2 owns the series), skew-change (RL-15 H3 duplicate), ETF
NAV-gap MR (no iNAV feed), historical expiry read (declined hold-out spend). Highlight:
**-13 is the lab's FIRST non-price cross-sectional signal** — dividend yield mined from
the `close`-vs-`adj_close` gap already cached, self-validated against F&O basis-implied
dividends.

## RL-2026-07-26-13 - Dividend-carry cross-section from the adj-vs-unadj gap (DIV-CARRY)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** trailing cash-distribution yield — the lab's first NON-PRICE
  cross-sectional dimension (every prior signal is a price/volume transform). Dividend
  yield proxies payout discipline + carry (Fama-French yield spreads; Litzenberger-
  Ramaswamy tax-clientele); high-yield India names (PSUs/energy/utilities) are
  structurally distinct from the momentum book. Mined at zero collection cost: Yahoo keeps
  `close` (unadjusted) and `adj_close` (dividend-adjusted) side by side, so each ex-div
  event is a permanent step in the close/adj ratio → a per-name dividend history.
- **Sample (locked):** N500-277 (adj_close + close, ret_clip 0.40, survivorship disclosed).
  History = formation warm-up only (no historical performance read, the -08 convention).
  Forward paper portfolio from registration; first read ≥252 forward days.
- **Preprocessing (locked):** ex-div events from day-over-day changes in the close/adj
  adjustment factor; trailing-252d dividend sum ÷ price = yield. **Any single event
  implying a distribution >5% of price EXCLUDED as a non-dividend action** (demergers/
  specials — receipts RELIANCE +8.7% Jio, ITC +3.7% Hotels). Yields winsorized ±3 MAD.
- **Specification:** ONE variant + ONE disclosure arm — decile L/S on trailing yield,
  equal-weight, monthly paper portfolio. Disclosure arm (validity, not a trial):
  rank-corr of realized yield vs basis-implied dividend (−b1) for the 130 F&O∩N500 names
  from `fno_daily.jsonl` — an internal cross-validation.
- **Predicted outcome:** prior ~25-30% (yield-decile spread thin everywhere; 2022-24 PSU
  run may not persist). Turnover ~5-15%/mo → ~0.25-0.7%/yr drag vs ~2-4%/yr gross spread —
  survives 20/40 bps; the risk is INSIGNIFICANCE, not cost. Bar (L/S spread idiom): net
  spread t>1.5 at 252 forward days, inside the equity-forward BH-FDR family {-08,-09,-13}.
  **Classification: FORWARD-ONLY.** Nearest: graveyard 32-factor family (all price
  transforms — this is a cash-flow quantity) / -08 (same new-input pattern, but payouts vs
  macro betas, no estimated-beta noise). Data CONFIRMED: close+adj_close in every cached
  file; b1 per name in fno_daily.jsonl. **BUILD NEXT.**

<!-- filled in at go-live / the locked read -->
- **Result (go-live 2026-07-10, `divcarry.py`):** construction built + wired into the daily
  snapshot; 11 dedicated tests + full suite 287 green. Extraction: f = adj_close/close is a
  pure dividend factor (splits cancel); 1 − f[t−1]/f[t] recovers each distribution; noise
  floor 5e-4 (measured non-event noise ~5e-7, 1000× margin); >5% steps zeroed as
  non-dividend actions. Orchestrator independently re-extracted ITC.NS with separate code:
  textbook dividends recovered (median 1.95% vs module 2.00%; Rs 6.25-8.00 interim+final
  cadence; count differs only by history window). Validation names: COALINDIA ~2.0 ev/yr
  1-4% (9 specials >5% correctly excluded), ITC ~1.33 ev/yr median 2.00%, ONGC ~2.6 ev/yr.
  Book verified dollar-neutral: gross 1.0000, net +0e+00, 27 long / 27 short. Longs = the
  predicted PSU-bank/energy/utility cohort (BANKINDIA, BANKBARODA, BPCL, CANBK, ITC, ...);
  shorts = zero-payout names (SUZLON, IDEA, YESBANK, ...). First forward row (panel
  2026-07-09): live intraday −0.07% ≈ 0, quotes 54/54, ledger
  `experiments/paper_trades_divcarry.jsonl`. **Disclosure arm (honest):** Spearman(realized
  trailing yield, −b1) = **−0.057, n=130 — indistinguishable from zero** (±0.17 2σ band).
  Mechanical reason: near-expiry basis (~19 dte) prices IMPENDING dividends, largely
  orthogonal to a trailing-252d yield (VEDL/BPCL/IOC yield 7-10% yet show contango, no
  ex-date before expiry). Group means do lean the right way (top-30-yield b1 +0.026 vs
  bottom-30 +0.044). The basis cross-check is a WEAK same-day validator; extraction
  validity rests on the per-name event logs above. Caveats disclosed: live-book monthly
  rebalance (ME-held, not daily churn); shorts NOT F&O-restricted (research signal, not an
  implementable short book — an implementable variant would need the RL-12 intersection);
  demerger-type events below 5% of price would pass the filter (none observed on the
  validation names).
- **Conclusion:** live and accruing (FORWARD-ONLY, zero hold-out). First read ≥252 forward
  days: net spread t>1.5 inside the equity-forward BH-FDR family {-08, -09, -13}.

## RL-2026-07-26-14 - Jump + volume-confirmed news-proxy drift (JUMP-MOM)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** the lab has no earnings dates (PEAD blocked), but a price JUMP
  with a VOLUME spike is a free news-event proxy; post-news drift (Chan 2003) predicts
  continuation in the jump's direction at 1-3 months. Counter-evidence disclosed:
  frog-in-the-pan (Da-Gurun-Warachka) says discrete info prices FAST — the sign is the
  hypothesis under test, not assumed.
- **Sample (locked):** N500-277; events from registration forward only; first read ≥252
  days (jumps plentiful — several hundred/yr, best power on this slate).
- **Preprocessing (locked):** event = daily |ret| ≥ 3× trailing-63d σ; abnormal return =
  stock − NSE-industry EW basket (-11 convention); overlapping events per name deduped to
  first (21d blackout).
- **Specification:** 2 variants — (a) price-only events; (b) also require volume >95th
  trailing-63d pct (needs the volume QC of -15). Forward 21d/63d abnormal returns split by
  jump sign; observational (tradable form registered separately on a pass).
- **Predicted outcome:** prior ~25% (sign ambiguity real; Indian retail lottery-chasing
  could flip positive jumps to reversal). Bar (event idiom): mean 21d abnormal return t>2
  per sign, BH-FDR in the equity-forward family. **Classification: FORWARD-ONLY.** Nearest:
  graveyard short-term reversal (unconditional 5d, cost-dead — this is conditional,
  monthly, event-based) / -11 (shared abnormal-return design; different event object).

<!-- filled in at the locked read -->
- **Result (collector LIVE, 2026-07-10, `event_studies.py`):** jump detector + 21d
  maturity measurement wired as a snapshot leg; 12 tests green (3σ boundary pinned,
  σ strictly pre-event — a leak would flip the constructed test verdict, blackout
  suppress/re-arm, peer basket excludes self). Day-one scan (latest completed session
  2026-07-09, 277 names): **1 event — DRREDDY.NS ret −5.89%, z −3.45, DOWN,
  vol_confirmed=True**. DISCLOSED: the first scan covers session 2026-07-09, one
  session before this registration's date; the event is kept (the hypothesis was not
  formed by looking at it) and the 252-day read will report with and without it.
  Winsorized (ret_clip 0.40) panel used for detection — attenuates only the >40%
  tail, disclosed. Abnormal returns measure vs the NSE-industry EW peer basket
  excluding the event stock; idempotent by (event_date, symbol).
- **Conclusion:** collecting; expect several hundred events/yr; claims at the
  252-day read (t>2 per sign, BH-FDR in the equity-forward family).

## RL-2026-07-26-15 - Turnover-shock (visibility premium) cross-section (VOL-SHOCK)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** abnormally high volume raises visibility and subsequent returns
  (Gervais-Kaniel-Mingelgrin 2001, attention channel; Amihud-adjacent). Volume as a
  cross-sectional LEVEL signal is untouched here (used only as a time-series trend gate).
- **Sample (locked):** N500-277; rupee turnover = volume × close. Forward paper portfolio;
  first read ≥252 days.
- **Preprocessing (locked):** shock = (5d mean turnover)/(126d mean), log, winsorized ±3
  MAD; zero/missing-volume names excluded.
- **Specification:** ONE variant — decile L/S (long high-shock), equal-weight, monthly.
- **Predicted outcome:** prior ~25% (attention decays fast; monthly transplant of a weekly
  result dilutes). Turnover ~40-70%/mo → 1-2%/yr drag vs ~3-6%/yr gross — margin thin.
  Bar: net spread t>1.5 at 252 days, equity-forward BH-FDR. **Classification: FORWARD-ONLY,
  conditional on a pre-flight VOLUME QC** (Yahoo `.NS` volume unverified — cross-validate
  ~20 names × recent dates vs Groww daily-candle volume, read-only; QC fail → BLOCKED
  PENDING DATA). Nearest: graveyard factor-32 volume_momentum (time-series gate vs x-sec
  level) / -10 (positioning quantity — futures OI vs cash volume; -10 blocked, this needs
  nothing new if QC passes).

<!-- filled in after the volume QC + at the read -->
- **Result (pre-flight volume QC, 2026-07-10 — PASS):** cross-validated Yahoo `.NS`
  daily volume vs Groww daily candles (read-only), 20 diverse names (large-cap PSU,
  private, mid, distressed), last ~15 common sessions. Median |yahoo/groww − 1| =
  **0.0% on 20/20 names** (volumes identical in level); median rank-corr 0.989
  (14/20 > 0.95 — the misses are rank ties among near-identical values, not data
  disagreement). Yahoo volume is trustworthy for the shock signal. VOL-SHOCK is
  UNBLOCKED (and -14's volume-confirmed variant (b) with it); the strategy itself
  remains registered-not-built.
- **Conclusion:** QC passed; pending build + forward evidence (252-day read).
- **Result (go-live 2026-07-10, `volshock.py`):** construction built + wired into the
  daily snapshot; 10 dedicated tests + full suite 366 green. Shock = log(5d/126d mean
  rupee turnover), signal-day zero/missing-volume mask (0/277 excluded at the current
  read — coverage clean, consistent with the QC), ±3 MAD winsorize, weights lagged one
  day, ME-held decile L/S. Orchestrator independently recomputed TRENT.NS's shock from
  raw volume×close (+0.620 → long decile, matches) and verified the book (gross 1.0000,
  net +0e+00, 27L/27S). First forward row (panel 2026-07-09): live intraday −0.02% ≈ 0,
  quotes 54/54, ledger `experiments/paper_trades_volshock.jsonl`. Disclosed:
  min_periods=1 rolling means (a historical gap doesn't black a name out 126 sessions);
  panel's ≤3-day volume ffill inherited.
- **Final conclusion (go-live):** live and accruing (FORWARD-ONLY). Read at ≥252 forward
  days: net spread t>1.5 inside the equity-forward BH-FDR family.

## RL-2026-07-26-16 - F&O eligibility-change events (SSF-LIST)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** when NSE ADDS a stock to F&O, single-name shorting becomes
  possible for the first time — Miller (1977) divergence-of-opinion says binding short
  constraints inflate prices, so constraint RELEASE predicts negative drift (Chang-Cheng-Yu
  2007, JF: HK short-list adds fall ~−1% to −5%); deletions re-impose it (opposite sign).
  India-specific, structural, outside the price-transform space. Free detection: the
  collector's daily basis-dict keys ARE the F&O underlying set (archived since 2026-07-09)
  → adds/deletes are a one-line set-diff.
- **Sample (locked):** names entering/leaving the collector's underlying set from go-live;
  forward-only (historical F&O-list changes not archived anywhere in the lab).
- **Preprocessing (locked):** abnormal return vs NSE-industry EW basket (-11 convention);
  batch adds on one review date treated as ONE clustered event (cross-sectional dependence
  disclosed).
- **Specification:** ONE variant — event = first appearance/disappearance; forward 21d/63d
  abnormal returns, adds vs deletes separate.
- **Predicted outcome:** prior ~20-25% (set-diff catches the EFFECTIVE date not the earlier
  announcement → attenuation; batch clustering cuts effective n). Bar (event idiom): mean
  21d abnormal return t>2 (adds negative primary), BH-FDR in the events family {-11,-16,-17}.
  **Classification: FORWARD-ONLY** (zero new collection; an announcement-date circular
  scrape is an optional BLOCKED-PENDING-DATA upgrade). Nearest: -11 (temporary OI-limit ban
  vs permanent eligibility change) / graveyard none. Complements RL-12's no-PIT-F&O-list
  limitation by starting that archive. Data CONFIRMED: underlying set implicit in
  fno_daily.jsonl.

<!-- filled in at the locked read -->
- **Result (collector LIVE, 2026-07-10, `event_studies.py`):** SSF eligibility-change
  detector wired as a snapshot leg (runs after the F&O collect): set-diffs the per-name
  basis keys of the last two `fno_daily.jsonl` rows (~210 underlyings). Day-one:
  universe 210, no change (baseline established). Events keyed by the collector's
  snapshot date; batch adds/deletes share one event_date (clustered inference, as
  registered). 21d abnormal-return maturity measurement shared with -14 (peer basket
  excludes self; SSF names resolve bare→.NS; names outside the 277-panel stay pending —
  disclosed gap). Same 12-test suite.
- **Conclusion:** collecting; NSE F&O list revisions are episodic (reviews + ad-hoc);
  claims at the registered t>2 bar inside the events-family BH-FDR.

## RL-2026-07-26-17 - Index-reconstitution flow events, Nifty 50 / Next 50 (RECON)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** semi-annual index reviews force mechanical passive flows at the
  effective date (Harris-Gurel 1986; Shleifer 1986). US add-premia have decayed to ~0
  (Greenwood-Sammon, disclosed) — but India's passive AUM (EPFO/index funds) is young and
  growing, so the price-pressure channel is plausibly still live. New event data; nothing
  in the lab touches membership changes.
- **Sample (locked):** adds/deletes announced for Nifty 50 + Next 50 from go-live (NSE
  Indices releases; ~Feb/Aug announce, Mar/Sep effective); ~10-30 events/yr — thin yr 1.
- **Preprocessing (locked):** abnormal return vs industry EW basket; adds/deletes separate;
  anticipation (announce→effective) + post-effective 21d reversal windows locked.
- **Specification:** 2 claims — (i) anticipation drift (adds outperform announce→effective,
  t>2); (ii) post-effective reversal (t>2) — the two-phase flow signature.
- **Predicted outcome:** prior ~30% (mechanically grounded but US decay precedent + thin n).
  Bar (event idiom): per-claim t>2 inside the events family BH-FDR. **Classification:
  BLOCKED PENDING DATA — confirm source** (NSE Indices press-release scrape leg; free, no
  Groww; collected forward so no PIT history needed). Serves handoff wishlist #4 (starts a
  free forward point-in-time membership archive). Nearest: graveyard sector rotation
  (return-chasing allocation vs mechanical-flow event) / -11 (event sibling).

<!-- filled in once the scrape leg is wired + at the read -->
- **Result (data collector LIVE, 2026-07-10):** `nse_events.collect_index_changes` wired
  as a daily snapshot leg → `experiments/nse_index_changes.jsonl` (committed). Design:
  fetch the official Nifty 50 / Next 50 constituent CSVs (nsearchives endpoint family
  already used by this repo, HTTP 200) and set-diff vs the last stored membership — each
  row stores the FULL sorted member set, so the committed ledger is a self-contained
  forward point-in-time membership archive (serves the handoff paid-data wishlist #4 for
  free, going forward). Error rows carry members=null and are skipped by the differ, so
  a failed fetch never corrupts the next diff (tested). Day-one baselines: nifty50 n=50,
  niftynext50 n=50. 17 tests green. Events are caught at the EFFECTIVE date;
  announcement-date capture noted as a future upgrade (anticipation-window claim (i)
  requires it — until then only claim (ii), post-effective reversal, is measurable).
- **Conclusion:** archive accruing; the ~Feb/Aug 2027 reviews are the first observable
  events; claims judged at the registered t>2 bars inside the events-family BH-FDR.

## RL-2026-07-26-18 - Expiry-cycle settlement structure (EXP-DAY)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** derivative settlement forces flow into the cash close on expiry
  days (closing-VWAP settlement; delta/gamma unwinds) — documented Indian expiry-day
  volume/vol effects (Vipul 2005-era). Distinct from turn-of-month (settlement mechanics,
  not month-boundary flows).
- **Sample (locked):** NIFTY daily (^NSEI/NIFTYBEES) + `nifty_expiry` from the collector +
  the RL-18/-03 options ledgers; forward from registration (~50 weekly + 12 monthly
  expiries/yr).
- **Preprocessing (locked):** expiry days from the collector's logged expiry dates;
  realized-vol baseline = trailing 63d.
- **Specification:** 2 claims — (i) expiry-day realized vol > non-expiry (two-sample t>2);
  (ii) decomposition of the short-vol paper books' daily marks by day-in-cycle (measurement
  only; informs a SEPARATELY registered roll-timing rule — RL-18/-03 NOT touched, §5).
- **Predicted outcome:** prior ~25% (ToM t=1.13 argues thin; value is mostly decision
  support for the options program). Bar: claim (i) t>2; (ii) descriptive, both inside the
  options-family BH-FDR. **Classification: FORWARD-ONLY.** Nearest: graveyard turn-of-month
  (SIP/month-boundary vs derivative-settlement mechanics; observational) / -06/-07 (they
  gate the short-vol book on vol-state; this measures cycle-time structure). Data CONFIRMED:
  nifty_expiry logged daily.

<!-- filled in at the locked read -->
- **Result:** (filled at the read.)
- **Conclusion:** pending forward evidence.

## RL-2026-07-26 wave 4 - backtest-anchored additions (2026-07-10)

Fourth wave, owner-directed ("work on more strategies; keep backtesting on available
data"). Discipline: the 2017+ hold-out stays CLOSED (family ~92, zero new reads) — new
candidates take the DUAL-ROT route instead: TRAIN design read → freeze → forward-only
live book. Pre-registration data probes (quality checks only, no performance reads):
(1) `INR=X` clean 2003→ (0 NaN, no >10% prints); `BZ=F` 2007→ (24 |ret|>10% days, all
documented oil history — spike-guard applies) → **-08 UNBLOCKED**; (2) NSE
`sec_bhavdata_full` 404s before ~2020, but the legacy MTO delivery files serve back to
≥2011 (HTTP 200 at 2011-07-01, 53,888 bytes) → a TRAIN-depth delivery-% archive is
buildable (-20). Considered and REJECTED without runs (implicit-search tally):
seasonality / MAX-lottery / idio-vol / LT-reversal / BAB x-sec (all 32-family graveyard
— no new data, no re-test per handoff §5); covered-call book (put-call parity makes it
-03 PUT-W's twin); AMFI monthly-flow regime gate (monthly series → hopeless n for
years); NIFTYBEES/JUNIORBEES index-pair MR (RL-23 knife-catching precedent + -09
already owns hedged MR).

## RL-2026-07-26-19 - Amihud illiquidity level cross-section (ILLIQ)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** Amihud (2002) — investors demand a premium for holding names
  where a rupee of flow moves price more; |ret|/rupee-turnover LEVEL is the classic
  proxy. Volume enters the lab's x-sec space only as a 5d/126d SHOCK (-15, attention
  channel); the persistent-level premium (compensation channel) is untouched, and Yahoo
  `.NS` volume passed the -15 QC (0.0% dev vs Groww, 20/20). India's retail-heavy
  microstructure plausibly widens the premium; post-publication crowding argues against
  — genuine uncertainty.
- **Sample (locked):** N500-277 (adj_close returns, ret_clip 0.40; rupee turnover =
  volume × unadjusted close). TRAIN design read 2010-01-01→2016-12-31 (freeze only —
  NOT promotable evidence); the 2017+ hold-out is NOT read; forward paper book from
  go-live, first read ≥252 forward days. Survivorship DISCLOSED as acute here:
  current-membership bias inflates an illiquid-leg TRAIN spread; the forward read is
  the only clean number.
- **Preprocessing (locked):** daily illiq = |ret| / rupee turnover; zero/missing-volume
  days masked; rolling mean over the variant lookback; log; winsorize ±3 MAD; names
  with <60% valid days in the lookback excluded.
- **Specification:** decile L/S long HIGH illiquidity, equal-weight, monthly,
  dollar-neutral. TWO variants — lookback **63d vs 252d**; freeze = argmax TRAIN net
  Sharpe at **40 bps** (the harsher cost arm is the honest default when the long leg is
  by construction the least-liquid decile); 20/80 bps sensitivities disclosed at the
  freeze.
- **Predicted outcome:** prior ~25% (premium widely published and likely
  size/liquidity-confounded; TRAIN number survivorship-inflated regardless of sign).
  Illiquidity is persistent → turnover ~5-15%/mo, so cost-death is not the modal risk;
  insignificance is. Bar (L/S spread idiom): forward net spread t>1.5 at ≥252 days
  inside the equity-forward BH-FDR family {-08, -09, -13, -15, -19, -20}.
  **Classification: TRAIN-design + FORWARD-ONLY** (the DUAL-ROT route; zero hold-out
  spend). Nearest: -15 VOL-SHOCK (same raw input, orthogonal channel — level-premium vs
  shock-attention; signal corr reported at go-live) / graveyard 32-family (price
  transforms only, no volume x-sec).

<!-- filled at the TRAIN freeze + go-live + the locked read -->
- **Result (TRAIN freeze + go-live 2026-07-10, `illiq.py`):** TRAIN 2010-01-04→2016-12-30
  (n=1709 daily returns; truncation PHYSICAL — panels sliced before any weight/return;
  test-asserted). Design table (net Sharpe @40/20/80 bps | ann | maxDD | 1-sided mo.
  turnover): **L63 1.305/1.375/1.165 | +10.6% | −17.7% | 11.2%**; L252
  1.239/1.282/1.152 | +10.1% | −16.0% | 7.5%. **FROZEN = L63** (registered argmax
  @40bps; margin +0.066 — thin, but the freeze rule is mechanical and was locked
  first). Cost-realism disclosure: long-leg median rupee-turnover percentile ≈ 5%,
  i.e. the long book lives in the least-liquid twentieth — 40 bps is the honest
  headline arm and even it may understate; the TRAIN Sharpe is additionally
  survivorship-inflated (registered as acute). Turnover 7-11%/mo confirms
  cost-death is not the modal risk. Orchestrator verification: TRAIN table
  reproduced to the digit on re-run; SPLPETRO.NS daily chain re-derived from the
  raw CSV independently — |ret| 0.017952, turnover 5.4253e6, daily illiq
  3.308965e-9 EXACT match (63d-mean differs benignly by window composition:
  panel-calendar vs own-row rolling on a name trading 7.5k sh/day); RELIANCE.NS
  −28.36 deep in the liquid tail, consistent with mega-caps clipping at the −3 MAD
  floor. Implementation catch worth recording: the volshock dense-ffill weight
  pattern would have made the BACKTEST rebalance daily — the study uses the h52
  ME-sparse grid instead (volshock itself unaffected: forward-only, no backtest
  path). First forward row (panel 2026-07-09): 27L/27S, gross 1.0, net 0, quotes
  54/54, intraday +0.34%; longs led by SPLPETRO/EIHOTEL/DCMSHRIRAM, shorts =
  mega-caps (BHARTIARTL, HDFCBANK, ICICIBANK). **Registered disclosure:**
  Spearman(ILLIQ, -15 VOL-SHOCK) = **−0.245, n=277** — negative as mechanics
  predict (a turnover surge deflates measured illiquidity); distinct books. Ledger
  `experiments/paper_trades_illiq.jsonl`; 6 design rows in `experiments/log.jsonl`;
  14 tests.
- **Conclusion:** frozen L63, live and accruing (TRAIN-design + FORWARD-ONLY; zero
  hold-out spend). First read ≥252 forward days: net spread t>1.5 inside the
  equity-forward BH-FDR family — the TRAIN 1.3 Sharpe is design evidence only and
  is NOT the claim.

## RL-2026-07-26-20 - Delivery-percentage conviction cross-section (DELIV)

- **Date (pre-registration):** 2026-07-10
- **Economic hypothesis:** NSE publishes per-name DELIVERABLE quantity daily — the
  slice of volume actually settled and taken home rather than round-tripped intraday.
  High delivery share = conviction/accumulation flow vs speculative churn
  (Llorente-Michaely-Saar-Wang 2002: returns continue after informed-trading volume;
  delivery is India's direct observable of it). A genuinely NON-PRICE input — the lab's
  second after -13 dividends — with real TRAIN-window depth (MTO archive ≥2011, probe
  receipt in the wave header).
- **Sample (locked):** N500-277 ∩ MTO coverage (bare-symbol→`.NS` join on current
  names; renames/gaps = disclosed coverage loss, coverage % reported at the freeze).
  TRAIN design read 2011-07-01→2016-12-31 (start = probed archive floor; actual floor
  measured during backfill and disclosed); the 2017+ hold-out is NOT read; forward from
  go-live, first read ≥252 forward days. Archive under `data/raw/nse_mto/`
  (git-ignored) + a daily collector leg.
- **Preprocessing (locked):** delivery ratio = deliverable_qty / traded_qty ∈ (0,1];
  days missing from MTO (bans/halts/fetch failures) left missing — no ffill beyond 3d;
  ±3 MAD winsorize on the transformed signal; names with <60% valid days in the
  lookback excluded.
- **Specification:** decile L/S, equal-weight, monthly, dollar-neutral. THREE variants
  — (a) LEVEL: 63d mean ratio, long high; (b) SHOCK: log(5d mean / 126d mean), long
  high; (c) SIGNED-SHOCK: sign(5d return) × shock, long high (conviction-confirmed
  moves continue). Freeze = argmax TRAIN net Sharpe at 20 bps; 40 bps sensitivity
  disclosed.
- **Predicted outcome:** prior ~25-30% (practitioner-popular in India → possibly
  crowded; LEVEL may proxy inverse liquidity → signal corr vs -19 reported at go-live
  and the weaker flagged). Bar (L/S spread idiom): forward net spread t>1.5 at ≥252
  days, equity-forward BH-FDR family. **Classification: TRAIN-design + FORWARD-ONLY**
  (zero hold-out spend). Nearest: -15 (volume QUANTITY vs volume QUALITY — delivery
  splits the same day's volume by settlement outcome) / -19 (liquidity level; corr
  check locked) / graveyard ToM (calendar mechanics, unrelated).

<!-- filled at the backfill QC + TRAIN freeze + go-live + the locked read -->
- **Result:** (backfill + TRAIN freeze pending.)
- **Conclusion:** pending.

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
