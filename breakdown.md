# Quant Lab Breakdown

This file is the running explanation log for what we build, why it exists, and what each piece means.

## Stage 1: Research Lab Skeleton

Quant trading is a pipeline:

```text
market data -> features/signals -> portfolio weights -> backtest -> metrics
```

We built the first small version of that pipeline.

### Market Data

The raw input is price data.

Example:

```text
Date        SPY price
Jan 1       100
Jan 2       101
Jan 3       103
Jan 4       102
```

Before using ML, we need to answer simpler questions:

- What is the asset return?
- What signal would we have seen at the time?
- What position would we take?
- What would that position earn after costs?
- Did we accidentally use future data?

That is why we started with the core plumbing.

### Features

A feature is information computed from raw market data.

In `src/quantlab/features.py`, we added three basic features.

`pct_returns(prices)` turns prices into percentage returns.

If SPY goes from `100` to `105`, return is:

```text
(105 / 100) - 1 = 0.05 = 5%
```

Returns matter more than raw prices because strategies make money from percentage changes, not absolute price levels.

`rolling_momentum(prices, lookback)` asks:

```text
How much has this asset gone up over the last N days?
```

Example:

```text
SPY price 20 days ago = 100
SPY price today       = 110

20-day momentum = 10%
```

A momentum strategy says assets that have recently gone up may continue going up.

`rolling_vol(prices, lookback)` measures how jumpy or risky an asset has been.

Example:

```text
Asset A daily moves: +1%, -1%, +1%
Asset B daily moves: +5%, -6%, +4%
```

Asset B is more volatile. In portfolio construction, we often give less weight to more volatile assets.

### Signals

A signal is a feature interpreted as a trading opinion.

Example:

```text
SPY 20-day momentum = +5%
TLT 20-day momentum = -2%
GLD 20-day momentum = +8%
```

A simple signal says:

```text
GLD is strongest, SPY is second, TLT is weakest.
```

This does not yet say how much to buy. It only ranks or scores assets.

### Portfolio Weights

A portfolio weight means what percentage of capital goes into each asset.

Example:

```text
SPY: 50%
GLD: 50%
TLT: 0%
```

This is what `src/quantlab/portfolio.py` handles.

`equal_weight(signals)` splits capital equally across active assets.

Example:

```text
Active assets: SPY, GLD

SPY: 50%
GLD: 50%
TLT: 0%
```

`inverse_vol_weight(vol)` gives less weight to riskier assets and more weight to calmer assets.

Example:

```text
SPY volatility = 10%
TLT volatility = 20%
```

TLT is twice as volatile, so inverse-vol weighting gives SPY more capital.

The intuition:

```text
Riskier asset -> smaller position
Calmer asset  -> larger position
```

This is one of the first real portfolio optimization ideas.

### Strategies

A strategy combines features and portfolio rules.

In `src/quantlab/strategies.py`, we added two simple strategies.

`long_top_momentum(prices, lookback, count)` does:

```text
1. Calculate momentum for each asset.
2. Rank assets from strongest to weakest.
3. Pick the top N.
4. Equal-weight them.
```

Example:

```text
Asset   Momentum
GLD     +8%
SPY     +5%
TLT     -2%
```

If `count = 2`, we hold:

```text
GLD: 50%
SPY: 50%
TLT: 0%
```

That is a basic momentum strategy.

`inverse_vol(prices, lookback)` does:

```text
1. Calculate volatility for each asset.
2. Allocate more to lower-volatility assets.
3. Allocate less to higher-volatility assets.
```

This is not predicting returns. It is a risk-balanced allocation.

### Backtest

The backtester is the simulation engine.

In `src/quantlab/backtest.py`, the main function is:

```python
backtest_weights(prices, weights, cost_bps=0.0)
```

It takes:

```text
prices   = historical prices
weights  = what we wanted to hold each day
cost_bps = trading cost
```

Then it simulates what would have happened.

Example:

```text
Day 1:
SPY price = 100
Weight = 100% SPY

Day 2:
SPY price = 110
Return = +10%

Portfolio return = +10%
```

If we started with `$1`, we now have:

```text
$1 * 1.10 = $1.10
```

This growing account value is called the equity curve.

### Prior-Day Weights

The backtester uses yesterday's portfolio weights to earn today's returns.

Why?

Because if today's price movement has already happened, we cannot go back in time and trade before it.

Bad backtest:

```text
Use today's close price to decide today's trade.
Earn today's close-to-close return.
```

That leaks future information.

Better version:

```text
Use today's information to decide position.
Earn return from next period onward.
```

In code, this is why we do:

```python
shifted = weights.shift().fillna(0.0)
```

That means:

```text
today's return is earned using yesterday's weights
```

This protects us from a very common form of fake performance.

### Turnover Costs

Turnover means how much the portfolio changed.

Example:

Yesterday:

```text
SPY: 100%
QQQ: 0%
```

Today:

```text
SPY: 0%
QQQ: 100%
```

We sold 100% SPY and bought 100% QQQ.

Total turnover:

```text
100% sold + 100% bought = 200%
```

Trading is not free. Even if commissions are zero, there is spread, slippage, market impact, taxes, and execution drag.

So the backtester subtracts cost:

```python
costs = turnover * cost_bps / 10_000
```

Basis points are finance units:

```text
1 bp = 0.01%
10 bps = 0.10%
100 bps = 1.00%
```

If turnover is `2.0` and cost is `10 bps`:

```text
cost = 2.0 * 10 / 10000
     = 0.002
     = 0.20%
```

This matters because many strategies look profitable before costs and die after costs.

### Backtest Result

The backtester returns a `BacktestResult`.

It includes:

`returns`: daily strategy returns.

Example:

```text
Day 1: 0.00%
Day 2: 1.00%
Day 3: -0.50%
```

`equity`: the account growth curve.

Example:

```text
Start: $1.00
Day 2: $1.01
Day 3: $1.00495
```

`weights`: what the strategy held each day.

`turnover`: how much trading happened each day.

It also calculates:

`total_return`: how much money the strategy made overall.

`max_drawdown`: worst peak-to-trough loss.

Example:

```text
Portfolio grows to $1.50
Then falls to $1.20

Drawdown = -20%
```

This matters because a strategy can make money but still be psychologically or financially hard to hold.

`sharpe`: a rough risk-adjusted return metric.

Simplified:

```text
Sharpe = average return / volatility of return
```

Higher Sharpe means more return per unit of volatility.

Very rough interpretation:

```text
Sharpe < 0: bad
Sharpe ~ 0.5: weak
Sharpe ~ 1: decent
Sharpe ~ 2: very strong
Sharpe > 3: be suspicious unless we deeply understand why
```

### Tests

The tests are not decoration. They protect us from lying to ourselves.

We added tests for:

- Does the backtester use prior-day weights?
- Does it charge turnover cost?
- Does momentum select the strongest assets?
- Do portfolio weights sum to 100%?

One test caught a real issue immediately: the first version did not charge the initial cost of entering a portfolio. That was wrong, so we fixed it.

This is why tests matter in quant work. Tiny accounting bugs can create fake alpha.

### Big Picture

Right now, we have:

```text
prices
  -> compute momentum or volatility
  -> convert that into weights
  -> simulate portfolio returns
  -> subtract trading costs
  -> calculate performance
```

This is not yet a full trading business. It is the first trusted kernel.

Before ML, DL, or RL, we need this kernel to be correct because every advanced model will sit on top of it.

If the backtester is wrong, the model can look genius while actually being garbage.

## Stage 2: Local Data Cache and First ETF Baseline

In this stage we added real daily market data and ran our first baseline experiment.

The new pipeline is:

```text
download daily OHLCV data
  -> cache it locally
  -> extract adjusted close prices
  -> create strategy weights
  -> backtest each strategy
  -> compare metrics
```

### Why We Need a Data Layer

Quant research depends on repeatable data.

If every experiment downloads fresh data in a slightly different way, we can get inconsistent results. So we added a local cache:

```text
data/raw/yahoo/
```

That folder stores downloaded CSV files for each symbol.

The cache matters because:

- experiments run faster after the first download
- we can rerun the same strategy without hitting the network every time
- we have a local copy of the exact data used for a result
- later we can add data validation before trusting a file

The code lives in `src/quantlab/data.py`.

### OHLCV Data

OHLCV means:

```text
O = open price
H = high price
L = low price
C = close price
V = volume
```

Example:

```text
Date        Open   High   Low    Close   Volume
2024-01-02  100    101    99     100.5   1000
```

For this stage, the backtest uses adjusted close prices rather than raw close prices.

### Adjusted Close

Adjusted close tries to account for splits and dividends.

Example:

If an ETF pays a dividend, the raw close price may drop even though an investor did not really lose money. Adjusted close smooths that out so historical returns are closer to what a total-return investor experienced.

For medium-term ETF backtests, adjusted close is usually a better first input than raw close.

### Data Source

We first tried Stooq, but its CSV endpoint returned a browser-verification page in this environment. That is not reliable for automated research.

So we switched to Yahoo Finance's chart endpoint and cache the normalized result as CSV.

The function is:

```python
load_yahoo_ohlcv(symbols, refresh=False)
```

It returns a dictionary:

```text
{
  "SPY": SPY dataframe,
  "QQQ": QQQ dataframe,
  ...
}
```

Then:

```python
close_prices(data)
```

combines each symbol's adjusted close into one price table.

Example:

```text
Date        SPY     QQQ     TLT
2024-01-02  470.1   405.2   95.4
2024-01-03  468.8   402.9   96.1
```

This wide price table is what our strategies consume.

### ETF Universe

The first universe is:

```text
SPY = S&P 500 ETF
QQQ = Nasdaq 100 ETF
IWM = Russell 2000 ETF
TLT = long-term Treasury bond ETF
GLD = gold ETF
```

This is a useful starter universe because it includes different asset types:

- US large-cap stocks
- US growth/tech-heavy stocks
- US small-cap stocks
- long-term bonds
- gold

That gives us some diversification instead of testing only equity ETFs.

### Baseline Strategies

We tested three simple baselines.

`equal_weight`

Hold every asset equally.

With five ETFs:

```text
SPY: 20%
QQQ: 20%
IWM: 20%
TLT: 20%
GLD: 20%
```

This is the dumb benchmark. Any fancy strategy should be compared against it.

`inverse_vol_63d`

Look at each ETF's trailing 63-day volatility, which is about three trading months.

Then allocate more to calmer assets and less to jumpier assets.

The idea:

```text
lower volatility -> larger weight
higher volatility -> smaller weight
```

This is a simple risk-balanced portfolio.

`momentum_126d_top2`

Look at each ETF's trailing 126-day momentum, which is about six trading months.

Then hold only the top two strongest ETFs equally.

Example:

```text
Asset   126-day momentum
QQQ     +15%
GLD     +10%
SPY     +7%
TLT     -2%
IWM     -5%
```

The strategy holds:

```text
QQQ: 50%
GLD: 50%
```

This is a basic cross-sectional momentum strategy.

### Correcting Turnover

This stage exposed an important backtest issue.

The first version charged turnover only when target weights changed.

That misses a real effect: portfolio weights drift when assets move.

Example:

Start with:

```text
SPY: 50%
TLT: 50%
```

Then SPY rises and TLT is flat.

Before rebalancing, the portfolio might become:

```text
SPY: 52.4%
TLT: 47.6%
```

If the strategy still wants 50/50, we must sell some SPY and buy some TLT. That is turnover, even though the target weights did not change.

The backtester now does this:

```text
1. Use yesterday's weights to earn today's asset returns.
2. Let the weights drift based on those returns.
3. Compare drifted weights to today's target weights.
4. Charge transaction cost on the trade needed to rebalance.
```

This is more realistic than the first version.

### Baseline Result

We ran:

```text
python -m quantlab.experiments
```

with:

```text
symbols: SPY, QQQ, IWM, TLT, GLD
start: 2005-01-01
cost: 5 bps per unit of turnover
cached data range: 2004-11-18 to 2026-06-18
```

Result:

```text
strategy            total_return  annual_return  sharpe  max_drawdown  avg_daily_turnover
equal_weight        8.2527        0.1093         0.8762  -0.3164       0.0066
inverse_vol_63d     8.3604        0.1099         1.0141  -0.2407       0.0119
momentum_126d_top2  4.4441        0.0822         0.6057  -0.2871       0.1214
```

### How To Read These Metrics

`total_return`

If total return is `8.2527`, then `$1` grew to about `$9.25`.

Why?

```text
ending equity = 1 + total_return
              = 9.2527
```

`annual_return`

The average compounded yearly return.

Example:

```text
0.1093 = 10.93% per year
```

`sharpe`

Return per unit of volatility.

In this run, inverse volatility had the best Sharpe:

```text
inverse_vol_63d Sharpe = 1.0141
```

That means it delivered a smoother return stream than the other two baselines.

`max_drawdown`

Worst peak-to-trough loss.

Example:

```text
equal_weight max drawdown = -31.64%
```

That means the strategy was down 31.64% from a previous high at its worst point.

`avg_daily_turnover`

Average amount traded per day.

Momentum had much higher turnover:

```text
momentum_126d_top2 avg daily turnover = 0.1214
```

That means it traded about 12.14% of the portfolio per day on average. That is high for a simple ETF strategy and can make it sensitive to costs and taxes.

### First Interpretation

This was not an alpha discovery stage. It was a baseline stage.

The early lesson:

- equal weight performed surprisingly well
- inverse volatility slightly improved Sharpe and reduced drawdown
- top-2 momentum underperformed these baselines in this setup
- momentum also traded much more, which is a warning sign

This does not mean momentum is useless.

It means this specific version:

```text
126-day lookback
top 2 assets
daily target refresh
5 bps turnover cost
SPY/QQQ/IWM/TLT/GLD universe
2005-2026 period
```

was not better than the simpler baselines.

That is useful. We want to kill weak ideas early.

### What This Stage Gives Us

We now have:

```text
real data -> local cache -> baseline strategies -> corrected backtest -> comparison table
```

This is the beginning of a research process.

Next, we should add:

- monthly or weekly rebalancing
- train/test periods
- parameter sweeps
- benchmark comparison against SPY
- result export so experiments are saved

That will tell us whether a strategy is robust or just lucky.

## Stage 3: QuantStats HTML Reports

In this stage we added simple visual reports using QuantStats.

The pipeline is now:

```text
strategy returns
  -> QuantStats
  -> HTML tear sheet
  -> visual inspection
```

### Why Visualization Matters

A summary table is useful, but it hides the path.

Two strategies can have similar annual returns but feel completely different:

- one may compound smoothly
- one may spend years underwater
- one may crash badly during bear markets
- one may only work in one market regime

Visual reports help us see those patterns faster.

### What QuantStats Does

QuantStats takes a return series and produces a performance report.

Our strategy returns look like:

```text
Date        Return
2024-01-02  0.0010
2024-01-03 -0.0025
2024-01-04  0.0040
```

QuantStats turns that into charts and metrics such as:

- cumulative return
- drawdowns
- monthly returns
- rolling Sharpe
- volatility
- best and worst periods
- comparison against a benchmark

This gives us a quick visual tear sheet for each strategy.

### Benchmark

We compare each strategy against SPY.

SPY is a reasonable first benchmark because it represents broad US large-cap equity exposure.

That does not mean every strategy must beat SPY in raw return.

For example, an asset-allocation strategy may be useful if it has:

- lower drawdown
- smoother returns
- better Sharpe
- better behavior during equity crashes

But if it returns less, has similar drawdown, and adds complexity, it is probably not useful.

### What We Added

We added `quantstats` to `pyproject.toml`.

We added a `--report-dir` option to the ETF baseline experiment:

```text
python -m quantlab.experiments --report-dir reports/etf_baseline
```

That generates one HTML report per strategy:

```text
reports/etf_baseline/equal_weight.html
reports/etf_baseline/inverse_vol_63d.html
reports/etf_baseline/momentum_126d_top2.html
```

The reports are ignored by git because they are generated artifacts.

The code still prints the same summary table:

```text
strategy            total_return  annual_return  sharpe  max_drawdown  avg_daily_turnover
equal_weight        8.2527        0.1093         0.8762  -0.3164       0.0066
inverse_vol_63d     8.3604        0.1099         1.0141  -0.2407       0.0119
momentum_126d_top2  4.4441        0.0822         0.6057  -0.2871       0.1214
```

### How The Code Works

In `src/quantlab/experiments.py`, we keep the normal baseline workflow:

```text
load data -> compute prices -> build strategy weights -> backtest
```

If `report_dir` is provided, we also do:

```python
qs.reports.html(
    res.returns,
    benchmark=benchmark,
    output="reports/etf_baseline/strategy_name.html",
    title="strategy_name vs SPY",
)
```

The strategy return series is our backtested daily returns.

The benchmark return series is SPY's daily adjusted-close return.

### Why This Is Still Not Proof

QuantStats is a reporting tool, not a truth machine.

It visualizes whatever returns we feed it.

So the quality of the report depends on:

- correct data
- correct backtest accounting
- realistic costs
- no lookahead bias
- robust out-of-sample testing

That is why we still need train/test splits, rebalance rules, and parameter sweeps.

### What This Stage Gives Us

We now have:

```text
baseline strategy -> metrics table -> visual HTML report
```

This makes strategy review much easier.

Instead of only asking:

```text
Which strategy has the highest Sharpe?
```

we can also ask:

```text
When did it suffer?
How long did it stay underwater?
Did it only win during one market regime?
How does it behave versus SPY?
```

That is closer to real investment research.
