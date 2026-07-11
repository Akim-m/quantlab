# CLAUDE.md — Quant Lab

Quantitative research lab for trading strategies and portfolio optimization. The
point of this repo is **honest** research: most ideas are expected to fail, and the
failure table is a valid deliverable. Read `protocol.md` (Arnott–Harvey–Markowitz)
and `research_log.md` before proposing or running anything.

## Golden rules (these override convenience)

- **Pre-register before running.** Add a `RL-YYYY-MM-DD-NN` entry to `research_log.md`
  — economic hypothesis, locked sample window, preprocessing, spec, predicted outcome
  — *before* the run. A result without a prior hypothesis is a protocol breach. Log
  abandoned ideas too (they count toward the multiple-testing tally).
- **Never iterate on the hold-out.** The test window is touched once. Blends/overlays
  are designed on TRAIN-window evidence only, then frozen before test.
- **Realistic costs, always.** Turnover-based bps costs on every backtest (US 5 bps;
  India 20 bps for STT + spreads; sensitivity-check 10/40).
- **Correct for multiple testing.** BH-FDR + Deflated Sharpe (trials-aware) over the
  family. A naive per-test t-stat overstates significance.
- **Report where it does NOT win.** Negative and "beats benchmark but fails the strict
  bar" results are the honest headline more often than not.

## Data sources

- **US / long-history backtest prices: Yahoo `adj_close`** via `data.load_yahoo_ohlcv`
  (total return — corporate-action + dividend adjusted; deep history to ~1996 for
  large caps). Cached under `data/raw/yahoo/` (git-ignored). Indian single stocks
  use the `.NS` suffix; NSE index membership via `india.nse_index_symbols("nifty500"|...)`.
- **Indian market backtests: Groww backtesting API** (owner directive 2026-07-11):
  `get_historical_candles` / `get_expiries` / `get_contracts` — NSE+BSE, CASH+FNO
  (OHLC, volume, OI for derivatives). History from **2020 only**; per-request window
  caps (1–5 min: 30 d, 10–30 min: 90 d, 1 h+/daily: 180 d) — chunk multi-year fetches.
  Dividend/corp-action adjustment of this endpoint is **UNVERIFIED**: before the first
  equity total-return study on it, compare vs Yahoo `adj_close` on a high-yield name
  (e.g. COALINDIA.NS) across an ex-date. If unadjusted, use it for F&O and
  short-horizon work and keep Yahoo for long-horizon total-return.
- **Groww trade API: DATA ONLY, read-only** (`groww_client.py`). Also the
  authoritative NSE instrument master (`get_all_instruments`), F&O-shortability flags,
  live LTP/quotes, and recent-price validation. Its plain historical-data endpoint's
  daily history (measured 2026-07-09: back to ~2002-07, split/bonus-adjusted) is
  **NOT dividend- or demerger-adjusted** — returns from it are wrong on high-yield
  and demerged names. Intraday candles there: trailing ~90 days only.
  - **NEVER place/modify/cancel an order.** `groww_client.call()` refuses order
    methods; do not bypass it. There is no trading path in this repo, by design.
  - **Secrets:** `API_KEY`/`API_SECRET` live in `.env` (git-ignored). Never print,
    log, commit, or hardcode them. `load_env()` reads them into the process env only.
  - **Rate limit:** self-throttle to ≤6 req/s — 4 below Groww's 10/s live-data
    ceiling (non-trading allows 20/s). The shared `RATE` token-bucket enforces it.

## Environment & checks

- Managed by **uv**. Install dev deps: `uv sync --extra dev`.
- Tests: `uv run python -m pytest -q` — keep the suite green before claiming done.
  Every engine/signal has tests, including causality/no-look-ahead and leakage guards.
- Run a study: `uv run python -m quantlab.<module>` (e.g. `factor_study`,
  `india_study`). Studies append one row per strategy to `experiments/log.jsonl` with
  params, sample, metrics, git commit, and a `git_dirty` flag.

## Module map (`src/quantlab/`)

- `data.py` — Yahoo OHLCV loader + `close_prices` panel.
- `features.py` — rolling primitives (vol, beta, residuals, skew/kurt, ranges).
- `xsec.py` — cross-sectional anomaly factors (dollar-neutral, unit gross).
- `trend.py` — time-series/trend signals (TSMOM, Donchian, dual-mom, vol-managed, …).
- `optimization.py` — construction (ERC, HRP, max-div, min-corr, MVO). NOTE: ERC and
  max-div use SLSQP — costly at large universe size; prefer HRP/min-corr at scale.
- `portfolio.py` — weighting + `rebalance_targets(weights, freq)`.
- `backtest.py` — `backtest_weights(prices, weights, cost_bps)` → returns/equity/
  turnover; supports long, short, and partial-cash (sum<1) books.
- `evaluation.py` — Sharpe t-stat, BH-FDR, Deflated Sharpe.
- `factor_study.py` — the 32-factor anomaly-replication study (US ETF24 + NSE sectors).
- `india.py` / `india_study.py` / `blend.py` / `india_blend_study.py` — RL-2026-07-10
  broad Indian single-stock study, composite blends, causal overlays/regime switch.
- `groww_client.py` — read-only Groww data client (see rules above).

## Commit discipline

- Commit + push after each milestone. **Never commit `.env`, secrets, `data/raw/`, or
  `graphify-out/`** (all git-ignored). Messages say *why*, not just *what*.
