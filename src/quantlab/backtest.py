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

    weights = weights.reindex(prices.index).ffill().fillna(0.0)
    weights = weights.reindex(columns=prices.columns, fill_value=0.0)

    asset_returns = prices.pct_change().fillna(0.0)
    prev = pd.Series(0.0, index=prices.columns)
    returns = []
    turnover = []

    for date, row in asset_returns.iterrows():
        target = weights.loc[date]
        gross = float((prev * row).sum())
        capital = 1.0 + gross
        if capital <= 0:
            raise ValueError("portfolio capital fell to zero")

        drifted = prev * (1.0 + row) / capital
        traded = float((target - drifted).abs().sum())
        returns.append(gross - traded * cost_bps / 10_000)
        turnover.append(traded)
        prev = target

    strategy_returns = pd.Series(returns, index=prices.index)
    turnover = pd.Series(turnover, index=prices.index)
    equity = (1.0 + strategy_returns).cumprod()

    return BacktestResult(
        returns=strategy_returns,
        equity=equity,
        weights=weights,
        turnover=turnover,
    )
