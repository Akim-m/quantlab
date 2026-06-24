# program.md - Autoresearch: Multi-Asset Alpha (RL-2026-06-24-01)

You are an autonomous research agent. Your job is to discover the best possible medium-term
multi-asset allocation model and improve it relentlessly, adapted from karpathy/autoresearch
to this quant repo, inside the protocol.md firewall.

## Setup (once)

- Work on git branch `research/multi-asset-trend`.
- Read context: `protocol.md`, `docs/superpowers/specs/2026-06-24-multi-asset-trend-autoresearch-design.md`,
  `src/quantlab/auto_strategy.py`, `src/quantlab/auto_eval.py`, `src/quantlab/backtest.py`.
- Trials are logged to `experiments/log.jsonl` (commit hash, config, objective J, metrics,
  status keep/discard/crash, note). This is the multiple-testing ledger.

## Objective

Maximize, on the TRAIN window only (2007-04-11 .. 2019-12-31):

    J = mean_folds( strat_Sharpe - benchmark_Sharpe ) - 0.5 * std_folds( ... )

i.e. consistently BEAT the benchmarks (equal-weight-11 and 60/40), net of 10 bps cost,
book vol-targeted. Folds = calendar years. A model that does not beat the benchmarks has
J <= 0 and has failed. The point is to defeat everyone.

Reject any candidate that violates a hard constraint:
- max drawdown < -25%
- annualized turnover > 12x

## What you may edit

`src/quantlab/auto_strategy.py` only - the CONFIG and the model/signal/construction logic.
Everything is fair game: the signal (trend vs ML), the feature set, the ML model and its
hyperparameters, the portfolio construction, vol targeting, rebalance. Constrain choices
with domain knowledge (protocol); do not bolt on unmotivated signals.

## What you may NOT do

- Read or evaluate the OOS window (2020-01-01 .. 2026-06-18). It is the final test, touched
  once, only when the human interrupts. NEVER peek at it inside the loop.
- Edit `auto_eval.py` (the scorer), the objective, the sample, the universe, or the cost.
- Add a result without a prior hypothesis (protocol).

## The loop

1. Evaluate current CONFIG via `auto_eval.run_and_log` (logs the trial).
2. Form a hypothesis. Change `auto_strategy.py`. Evaluate.
3. Keep the change if J improves and constraints hold; else revert.
4. Prefer the simpler model when J ties (protocol 17-19).
5. Repeat.

## NEVER STOP

Once the loop has begun, do NOT pause to ask the human whether to continue. Do NOT ask
"should I keep going?" or "is this a good stopping point?". The human may be asleep or away
and expects you to keep working until manually stopped. You are autonomous. If you run out
of ideas, think harder: re-read the in-scope files for new angles, read the papers and
methods referenced in protocol.md and the spec, combine previous near-misses, try more
radical changes - different signals, an ML model, different feature sets, different portfolio
construction. The loop runs until the human interrupts, period.

## Stopping criteria

There is no automatic stop. "Best" = the highest validation J that beats every benchmark with
the constraints satisfied. Keep climbing until the human interrupts. ONLY when interrupted,
write the champion into CONFIG and run `auto_eval.oos_report` exactly ONCE for the final
scorecard + Deflated Sharpe(N).
