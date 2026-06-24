# Multi-Asset Trend + Autoresearch Loop - Design

Date: 2026-06-24
Pre-registration: RL-2026-06-24-01 (research_log.md)
Status: design approved, pending spec review

## Objective

Find the best medium-term, multi-asset allocation model for high risk-adjusted return
at a controlled ("average") risk level, by running a Karpathy-style autoresearch loop that
hill-climbs an institutional objective metric - inside a strict anti-overfitting firewall
required by protocol.md.

"High return at average risk" is achieved the way pod shops do it: maximize Sharpe/IR under
a hard drawdown limit, then vol-target the book to the desired risk level. We do not
optimize raw return directly.

## Economic hypothesis (ex ante)

A diversified multi-asset book that (1) tilts toward assets with positive medium-term trend,
(2) weights by risk contribution, and (3) targets constant volatility should earn a high,
consistent risk-adjusted return.

- Time-series momentum: assets up over 3-12 months tend to continue (behavioral
  underreaction + flows; Moskowitz-Ooi-Pedersen). One of the most robust cross-asset effects.
- Risk-based diversification: weighting by risk contribution beats dollar-weighting out of
  sample (covariance / estimation-error argument).
- Volatility targeting: volatility is persistent, so scaling exposure to a vol target
  stabilizes risk and historically improves Sharpe and cuts drawdowns.

These are established, documented edges. Novelty is in the *combination and tuning*, not in
any new unmotivated signal (protocol rules 1-2, and "constrain ML inputs with domain
knowledge").

## Universe (locked)

Liquid ETFs spanning the asset classes, all free from Yahoo:

- Equities: SPY, QQQ, IWM, EFA, EEM
- Bonds: TLT, IEF, LQD, HYG
- Commodities: DBC, GLD

Binding history constraint is HYG (IPO 2007-04), so the common window starts ~2007.

## Sample firewall (locked)

- Train / tune (loop sees this only): 2007-04 -> 2019-12-31
- Out-of-sample (touch ONCE, at the very end): 2020-01-01 -> 2026-06-18
- Inside train: expanding-window walk-forward with a 1-month purge/embargo between train
  and validation fold so the target window never leaks.

The loop NEVER reads the OOS era. Neither do I, until the final scorecard.

## Model and search space (what the loop edits)

A single strategy module, the only file the loop edits. The loop tunes, within
economically sensible bounds:

- trend lookback: {63, 126, 189, 252} trading days
- trend mapping: binary gate vs continuous tilt
- weighting backbone: inverse-vol vs risk-parity (ERC) vs min-variance
- volatility target: {8%, 10%, 12%, 15%} annualized
- vol estimation window
- rebalance: monthly or bi-monthly

The loop may NOT: add new asset classes, add unmotivated signals, change the sample, or
touch the evaluator.

## Metrics

Objective the loop maximizes (inner walk-forward folds, net of 10 bps turnover cost, book
vol-targeted to 10%/yr):

    J = mean(fold_Sharpe) - 0.5 * std(fold_Sharpe)

Hard constraints (candidate rejected if violated on inner folds):

- max drawdown <= 25%
- annualized turnover <= 12x

Reported scorecard (measured, not optimized - avoids metric-hacking):

- Sortino, Calmar, Information Ratio vs 60/40, Information Coefficient of the trend signal,
  annual return, annualized vol, max drawdown, turnover

Final judgment (locked OOS, once):

- Deflated Sharpe Ratio (Bailey & Lopez de Prado) using trial count N from
  experiments/log.jsonl, plus the full scorecard on OOS.

## Autoresearch scaffold (mapped to karpathy/autoresearch)

| autoresearch | here |
|---|---|
| prepare.py (read-only data) | data layer + frozen universe/sample - read-only |
| train.py (only edited file) | src/quantlab/auto_strategy.py |
| frozen eval | frozen walk-forward harness that prints J + scorecard |
| results.tsv | experiments/log.jsonl (already built; richer) |
| program.md | program.md with protocol guardrails in constraints + stopping criteria |

program.md stopping criteria: stop after K iterations with no improvement in J, or when a
pre-set deflated-Sharpe target is hit. Prefer simpler models when J ties (protocol 17-19,
and autoresearch's own "prefer simpler" rule).

## Guardrails (why this is not an overfitting machine)

- Loop optimizes only inner walk-forward folds; OOS is touched once.
- Every iteration is logged -> exact trial count -> deflated Sharpe.
- Search space constrained by economics; no signal mining.
- Objective penalizes cross-fold inconsistency, not raw return.

## Risks / limitations

- Survivorship: today's ETF tickers (documented caveat; defer fix).
- ~19 years total, one OOS regime block - limited independent OOS.
- Vol/covariance estimates are noisy (the MVO fragility lesson) - mitigated by risk-parity
  and weight caps.
- Loop can still overfit the *train* era; only the deflated OOS number is trusted.
