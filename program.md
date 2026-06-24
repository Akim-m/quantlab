# Autoresearch Program - Multi-Asset Trend (RL-2026-06-24-01)

Autonomous hill-climb to maximize the objective J for the strategy in `auto_strategy.py`,
inside the protocol.md firewall. Adapted from karpathy/autoresearch.

## Objective

Maximize, on the TRAIN window only:

    J = mean(fold_Sharpe) - 0.5 * std(fold_Sharpe)   (folds = calendar years)

net of 10 bps turnover cost, book vol-targeted to its CONFIG vol_target.

Reject any candidate that violates a hard constraint:
- max drawdown < -25%
- annualized turnover > 12x

## What you may edit

`src/quantlab/auto_strategy.py` - the CONFIG dict and the weighting logic ONLY.
Tune within these bounds (domain-constrained, protocol rule "constrain inputs"):
- lookback {63,126,189,252}, mode {gate,tilt}, backbone {equal,inverse_vol,min_variance},
  vol_target {0.08,0.10,0.12,0.15}, vol_window {42,63,126}, max_gross {1.0,1.5,2.0},
  rebalance {ME,2ME}.

## What you may NOT do

- Read or evaluate the OOS window (2020-01-01..2026-06-18). It is touched once, at the end.
- Edit `auto_eval.py` (the scorer) to change the metric or the sample.
- Add asset classes, add unmotivated signals, or change the universe / cost.

## Loop

1. Evaluate the current CONFIG via `auto_eval.run_and_log` (logs to experiments/log.jsonl).
2. Propose a change within bounds; evaluate; keep if J improves and constraints hold, else revert.
3. Prefer the simpler config when J ties (fewer moving parts; protocol 17-19).
4. Log EVERY trial - the count N feeds the final Deflated Sharpe.

## Stopping criteria

Stop after one full coordinate-ascent pass with no J improvement, or when a clear champion
is stable. Then write the champion into CONFIG and report. Only after stopping, run
`auto_eval.oos_report` ONCE for the final scorecard + Deflated Sharpe(N).
