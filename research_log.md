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
  headline as already search-inflated when judging our reproduction.

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
- **Result:**
- **Conclusion:**

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
- **Result:**
- **Conclusion:**

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
