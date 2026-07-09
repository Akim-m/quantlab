
# Ordered Backtesting Protocol (Avoiding False Positives)

Source: *A Backtesting Protocol in the Era of Machine Learning* (Arnott, Harvey, Markowitz)

## 1. Research Motivation
1. Establish an ex ante economic foundation before testing.
2. Never create an ex post narrative to justify empirical findings.

## 2. Multiple Testing & Statistical Methods
3. Track every model, variable, and failed attempt.
4. Account for every interaction/combination tested.
5. Avoid the "parallel universe" problem—adjust significance for all implicit alternatives.

## 3. Data & Sample Choice
6. Define the sample before research begins.
7. Verify data quality before modeling.
8. Predefine and document all preprocessing and transformations.
9. Do not arbitrarily remove outliers.
10. Decide winsorization policy before model construction.

## 4. Cross-Validation
11. Treat historical out-of-sample as imperfect; only live trading is truly out-of-sample.
12. Never iterate on the hold-out set.
13. Include realistic trading costs, implementation shortfall, and data revisions.

## 5. Model Dynamics
14. Test robustness to structural change.
15. Consider crowding and reflexivity after publication.
16. Avoid continual tweaking of deployed models.

## 6. Model Complexity
17. Minimize dimensionality.
18. Prefer the simplest model that achieves comparable performance; regularize.
19. Require interpretability rather than black-box predictions.

## 7. Research Culture
20. Reward scientific rigor rather than positive backtests.
21. Expect most research ideas to fail.
22. Closely supervise delegated research and incentivize truth-seeking.

## Expanded Guidance

### Research Motivation
- Every predictor should have an economic rationale established before looking at results.
- Avoid post-hoc explanations.
- Constrain ML inputs using domain knowledge.

### Multiple Testing
- Record all attempted specifications.
- Adjust statistical thresholds for multiple comparisons.
- Count implicit searches, including feature interactions and abandoned paths.

### Data
- Lock the sample window beforehand.
- Clean and validate datasets.
- Pre-register preprocessing.
- Keep valid outliers unless exclusion has an economic justification.
- Fix winsorization rules in advance.

### Cross-Validation
- Historical holdouts are influenced by researcher knowledge.
- Modifying a model after observing holdout performance converts it into in-sample fitting.
- Simulate realistic execution costs.

### Model Dynamics
- Markets evolve.
- Alpha decays through publication and crowding.
- Resist reactive model adjustments.

### Complexity
- Finance datasets are small relative to ML requirements.
- Prefer parsimonious models with regularization.
- Inspect model logic.

### Research Culture
- Incentives should reward sound methodology.
- Failed experiments are expected.
- Monitor delegated research to avoid confirmation bias.
- The pre-registered bar must be stated in the thesis's own idiom: a
  risk-reduction thesis cannot be judged by a return-level test (a
  diversifying blend that halves drawdown will always fail a paired-t on
  mean return — receipt: RL-2026-07-19). When a bar is later found
  mis-specified, the frozen verdict still binds; the fix is a NEW
  registration on forward data, never a re-read under a new bar.
