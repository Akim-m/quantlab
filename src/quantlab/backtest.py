from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] - 1.0)

    @property
    def max_drawdown(self) -> float:
        drawdown = self.equity / self.equity.cummax() - 1.0
        return float(drawdown.min())

    @property
    def sharpe(self) -> float:
        if self.returns.std() == 0:
            return 0.0
        return float((252**0.5) * self.returns.mean() / self.returns.std())


def backtest_weights(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 0.0,
) -> BacktestResult:
    if prices.empty:
        raise ValueError("prices is empty")

    targets = weights.reindex(prices.index)
    targets = targets.reindex(columns=prices.columns)
    rebalance = targets.notna().any(axis=1)
    targets = targets.fillna(0.0)

    asset_returns = prices.pct_change().fillna(0.0)
    prev = pd.Series(0.0, index=prices.columns)
    returns = []
    turnover = []
    actual_weights = []

    for date, row in asset_returns.iterrows():
        gross = float((prev * row).sum())
        capital = 1.0 + gross
        if capital <= 0:
            raise ValueError("portfolio capital fell to zero")

        drifted = prev * (1.0 + row) / capital
        if rebalance.loc[date]:
            target = targets.loc[date]
            traded = float((target - drifted).abs().sum())
            prev = target
        else:
            traded = 0.0
            prev = drifted

        returns.append(gross - traded * cost_bps / 10_000)
        turnover.append(traded)
        actual_weights.append(prev)

    strategy_returns = pd.Series(returns, index=prices.index)
    turnover = pd.Series(turnover, index=prices.index)
    actual = pd.DataFrame(actual_weights, index=prices.index, columns=prices.columns)
    equity = (1.0 + strategy_returns).cumprod()

    return BacktestResult(
        returns=strategy_returns,
        equity=equity,
        weights=actual,
        turnover=turnover,
    )
