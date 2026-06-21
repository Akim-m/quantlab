# Quant Lab

Quantitative algorithms for trading capital markets and investment portfolio optimization.

> Our message on the use of machine learning in backtests is one of caution and is consistent with the admonitions of López de Prado (2018). Machine learning techniques have been widely deployed for uses ranging from detecting consumer preferences to autonomous vehicles, all situations that involve big data. The large amount of data allows for multiple layers of cross-validation, which minimizes the risk of overfitting. We are not so lucky in finance. Our data are limited. We cannot flip a 4TeV switch at a particle accelerator and create trillions of fresh (not simulated) out-of-sample data. But we are lucky in that finance theory can help us filter out ideas that lack an ex ante economic basis.
> We also do well to remember that we are not investing in signals or data; we are investing in financial assets which represent partial ownership of a business, or of debt, or of real properties, or of commodities. The quantitative community is sometimes so focused on its models that we seem to forget that these models are crude approximations of the real world, and cannot possibly reflect all of the nuances of the assets that actually comprise our portfolios. The amount of noise may dwarf the signal. Finance is a world of human beings, with emotions, herding behavior, and short memories. And market anomalies – opportunities that are the main source of intended profit for the quantitative community and our clients – are hardly static. They change with time and are often easily arbitraged away. We ignore the gaping chasm between our models and the real world at our peril.
>
> — _A Backtesting Protocol in the Era of Machine Learning_, Rob Arnott (Research Affiliates), Campbell R. Harvey (Fuqua School of Business, Duke University), Harry Markowitz (Harry Markowitz Company)

## Protocols to avoid False Positive
Followed strictly across the research - [research protocol](protocol.md)

## Running experiments

Every experiment must be logged. This is not optional - it is how we count implicit
searches and keep the research honest (see [protocol.md](protocol.md), rules 1–5).

Do this for **any** run, in order:

1. **Pre-register first.** Add an entry to [research_log.md](research_log.md) *before*
   running - economic hypothesis, locked sample window, preprocessing, and spec. Get its
   id (e.g. `RL-2026-06-21-01`). Never write down a result without a prior hypothesis.
2. **Run with the id.** Pass `--hypothesis-ref` so the run links back to the journal:

   ```
   python -m quantlab.experiments --universe nasdaq50 --no-split --hypothesis-ref RL-2026-06-21-01
   ```

   Each run appends one row per strategy to `experiments/log.jsonl` (params, sample,
   metrics, git commit, and a `git_dirty` flag for reproducibility).
3. **Annotate the outcome** back in `research_log.md` - worked / failed / shelved.

Also log ideas you **abandon** before running, in the journal's multiple-testing tally.
A ledger row with no matching journal entry is a protocol breach.