# Research Log

Pre-registration journal for every research idea, per `protocol.md` (Arnott-Harvey-Markowitz).

**Discipline:** fill the top half of an entry *before* running. The runner appends the
result row to `experiments/log.jsonl`; reference its `hypothesis_ref` here. Never write a
result without a prior hypothesis. Iterating on a model after seeing hold-out results turns
it into in-sample fitting.

## Ideas tried (multiple-testing tally)

One line per idea — including abandoned ones that never ran. This count is needed to adjust
significance for implicit search.

- `(DLTIS-PSTKR)/MRC4` — abandoned before running: arbitrary Compustat-field combo, no
  economic rationale, no fundamental data in project.

---

## Template

```markdown
## RL-YYYY-MM-DD-NN — <short title>

- **Date (pre-registration):** YYYY-MM-DD
- **Economic hypothesis:** <why this should work, in plain econ terms, BEFORE any results>
- **Sample (locked):** universe=<>, window=<start>→<end>, rebalance=<>, cost=<>bps
- **Preprocessing (locked):** <winsorization / normalization / none>
- **Specification:** <strategy/model + params>
- **Predicted outcome:** <what you expect and why>

<!-- filled in AFTER the run -->
- **Result:** <key metrics / experiments/log.jsonl ids>
- **Conclusion:** worked / failed / shelved — <one line, honest>
```
