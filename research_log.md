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
- RL-2026-06-24-01 - multi-asset trend + autoresearch loop (pre-registered, see below).

---

## RL-2026-06-24-01 - Multi-asset trend, vol-targeted, autoresearch-tuned

- **Date (pre-registration):** 2026-06-24
- **Economic hypothesis:** A diversified multi-asset ETF book that tilts toward positive
  medium-term trend, weights by risk contribution, and targets constant volatility earns a
  high, consistent risk-adjusted return. Basis: time-series momentum (underreaction/flows),
  risk-based diversification (estimation-error robustness), and vol persistence. Established
  edges; novelty is in the combination/tuning, not a new signal.
- **Sample (locked):** universe = SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, HYG, DBC, GLD.
  Train/tune 2007-04 -> 2019-12-31 (loop sees only this). OOS 2020-01-01 -> 2026-06-18
  (touch ONCE). Rebalance monthly or bi-monthly. Cost 10 bps per unit turnover.
- **Preprocessing (locked):** returns from adjusted close; expanding-window walk-forward
  with 1-month purge/embargo; book vol-targeted to 10%/yr. No winsorization.
- **Specification:** autoresearch loop edits `src/quantlab/auto_strategy.py` only; tunes
  trend lookback {63,126,189,252}, gate vs tilt, weighting {inverse-vol, risk-parity,
  min-var}, vol target {8,10,12,15}%, vol window, rebalance {1,2 mo}. Objective
  J = mean(fold Sharpe) - 0.5*std(fold Sharpe), net of cost; reject if maxDD > 25% or
  turnover > 12x. Scorecard: Sortino, Calmar, IR vs 60/40, IC, return, vol, maxDD, turnover.
  Final judgment: Deflated Sharpe on OOS using trial count N from experiments/log.jsonl.
- **Predicted outcome:** OOS deflated Sharpe materially > 0 and > the 60/40 and
  equal-weight benchmarks, with maxDD < 25%. Expect many loop iterations to fail; that is
  normal.

<!-- filled in AFTER the run -->
- **Result:** N=22 train trials (experiments/log.jsonl). Champion: lookback=63, gate,
  min_variance, vol_target=0.10, vol_window=63, max_gross=1.5, ME. Train Sharpe ~1.07,
  J=0.812, Deflated Sharpe=0.998 (in-sample edge survives multiple testing). OOS 2020-06..2026:
  Sharpe 0.66, ann 5.7%, vol 9.0%, maxDD -14.3%, IR vs 60/40 = -0.33. OOS benchmarks:
  60/40 Sharpe 0.80, equal-weight-11 Sharpe 0.82.
- **Conclusion:** SHELVED. In-sample edge is statistically real (DSR 0.998) but OOS it does
  NOT beat naive 60/40 or equal-weight; a 2020-driven, one-regime result. High DSR != economic
  value. Process worked (OOS firewall held); strategy rejected per benchmark test.

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
