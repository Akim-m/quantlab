"""RL-2026-07-10: situation -> winning-strategy sweep on the Indian universe.

Evaluates a set of named strategy books across SITUATIONS - universe breadth
(Nifty 50/200/500), cost regime, and causal market regime (bull/bear via the
200-day MA) - and reports, on the locked TEST window, net-of-cost Sharpe, annual
return, max drawdown, turnover, whether it beats the Nifty benchmark, and its
Sharpe conditioned on the bull/bear regime.

The regime split is CAUSAL (the market's 200-MA state at t-1 buckets day t's
return), so a strategy that trades on it is deployable, not hindsight. Per-regime
Sharpe is still descriptive; the deployable claim is the regime-SWITCH strategy's
own overall number (see blend.regime_switch).
"""

import numpy as np
import pandas as pd

from .backtest import backtest_weights
from .blend import market_on
from .evaluation import sharpe_tstat
from .portfolio import rebalance_targets


def _ann(r: pd.Series) -> float:
    return float((1 + r).prod() ** (252 / max(len(r), 1)) - 1)


def _maxdd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def regime_conditional_sharpe(returns: pd.Series, market: pd.Series, ma_lb: int = 200) -> tuple[float, float]:
    """(bull_sharpe, bear_sharpe): day t's return bucketed by the market's 200-MA
    state at t-1 (causal)."""
    on = market_on(market, ma_lb).shift(1, fill_value=False).reindex(returns.index, fill_value=False).astype(bool)
    bull, _ = sharpe_tstat(returns[on])
    bear, _ = sharpe_tstat(returns[~on])
    return bull, bear


def evaluate(
    strategies: dict[str, tuple[pd.DataFrame, str | None]],
    px: pd.DataFrame,
    mkt: pd.Series,
    bench_ret: pd.Series,
    split: str,
    cost_bps: float = 20.0,
) -> pd.DataFrame:
    """Backtest each named (weights, rebalance_freq) and return a metrics table."""
    bench_test = bench_ret.loc[split:]
    bench_sr, _ = sharpe_tstat(bench_test)
    rows = []
    for name, (weights, freq) in strategies.items():
        res = backtest_weights(px, rebalance_targets(weights, freq), cost_bps)
        test = res.returns.loc[split:]
        sr, t = sharpe_tstat(test)
        bull, bear = regime_conditional_sharpe(test, mkt.loc[split:])
        rows.append({
            "strategy": name,
            "test_sharpe": round(sr, 3),
            "test_tstat": round(t, 2),
            "ann_return": round(_ann(test), 4),
            "max_dd": round(_maxdd(test), 4),
            "turnover": round(float(res.turnover.mean()), 4),
            "beats_nifty": bool(sr > bench_sr),
            "bull_sharpe": round(bull, 3),
            "bear_sharpe": round(bear, 3),
        })
    table = pd.DataFrame(rows).sort_values("test_sharpe", ascending=False).reset_index(drop=True)
    table.attrs["bench_sharpe"] = round(bench_sr, 3)
    return table


_SUBPERIODS = [
    ("sr_17_19", "2017-01-01", "2019-12-31"),
    ("sr_20", "2020-01-01", "2020-12-31"),
    ("sr_21_23", "2021-01-01", "2023-12-31"),
    ("sr_24_26H1", "2024-01-01", "2026-06-30"),
]


def _paired_t(a: pd.Series, b: pd.Series) -> float:
    """t-stat of the paired monthly active-return series (H0: mean active = 0)."""
    d = (a - b).dropna()
    if len(d) < 2 or d.std(ddof=1) == 0:
        return 0.0
    return float(d.mean() / (d.std(ddof=1) / len(d) ** 0.5))


def _capm(strat: pd.Series, bench: pd.Series) -> tuple[float, float, float]:
    """(beta, annualized alpha, alpha t-stat) from an OLS of strat on bench (daily)."""
    df = pd.concat([strat, bench], axis=1).dropna()
    if len(df) < 3:
        return 0.0, 0.0, 0.0
    y, x = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    dof = len(x) - 2
    if dof <= 0:
        return beta, alpha * 252, 0.0
    resid = y - X @ coef
    s2 = float(resid @ resid) / dof
    alpha_se = (s2 * np.linalg.inv(X.T @ X)[0, 0]) ** 0.5
    # a series regressed on itself gives alpha ~ 0 with ~0 residuals; the resulting
    # 0/0 t-stat is numerical noise, so report it as 0 for such degenerate fits.
    alpha_t = alpha / alpha_se if alpha_se > 0 and abs(alpha) > 1e-10 else 0.0
    return beta, alpha * 252, alpha_t


def _row_metrics(
    name: str, test: pd.Series, turnover: float, mkt_full: pd.Series, bench_sr: float,
    tr_test: pd.Series, ew_test: pd.Series, hi_mask: pd.Series,
) -> dict:
    sr, t = sharpe_tstat(test)
    bull, bear = regime_conditional_sharpe(test, mkt_full)  # MA warmed on full history
    to_m = lambda r: (1.0 + r).resample("ME").prod() - 1.0
    sm, trm, ewm = to_m(test), to_m(tr_test), to_m(ew_test)
    beta_tr, alpha_tr, alpha_t_tr = _capm(test, tr_test)
    beta_ew, alpha_ew, alpha_t_ew = _capm(test, ew_test)
    hi = hi_mask.reindex(test.index).fillna(False).astype(bool)
    sr_hi, _ = sharpe_tstat(test[hi])
    sr_lo, _ = sharpe_tstat(test[~hi])
    crash = test.loc["2020-02-01":"2020-06-30"]
    row = {
        "strategy": name,
        "test_sharpe": round(sr, 3),
        "test_tstat": round(t, 2),
        "ann_return": round(_ann(test), 4),
        "ann_vol": round(float(test.std() * np.sqrt(252)), 4),
        "max_dd": round(_maxdd(test), 4),
        "turnover": round(turnover, 4),
        "beats_nifty": bool(sr > bench_sr),
        "act_ret_tr": round(_ann(test) - _ann(tr_test), 4),
        "act_t_tr": round(_paired_t(sm, trm), 2),
        "beta_tr": round(beta_tr, 3),
        "alpha_t_tr": round(alpha_t_tr, 2),
        "act_ret_ew": round(_ann(test) - _ann(ew_test), 4),
        "act_t_ew": round(_paired_t(sm, ewm), 2),
        "beta_ew": round(beta_ew, 3),
        "alpha_t_ew": round(alpha_t_ew, 2),
        "beats_ew": bool(_paired_t(sm, ewm) >= 2.0),
        "bull_sharpe": round(bull, 3),
        "bear_sharpe": round(bear, 3),
        "hivol_sharpe": round(sr_hi, 3),
        "lovol_sharpe": round(sr_lo, 3),
        "feb_jun_2020": round(float((1.0 + crash).prod() - 1.0), 4),
    }
    for col, lo, hi_d in _SUBPERIODS:
        seg_sr, _ = sharpe_tstat(test.loc[lo:hi_d])
        row[col] = round(seg_sr, 3)
    return row


def evaluate2(
    strategies: dict[str, tuple[pd.DataFrame, str | None]],
    px: pd.DataFrame,
    mkt: pd.Series,
    bench_ret: pd.Series,
    split: str,
    tr_ret: pd.Series,
    ew_ret: pd.Series,
    cost_bps: float = 20.0,
    extra_returns: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Full situation table on the TEST window. Backtests each (weights, freq) book
    and also scores any pre-computed `extra_returns` series (e.g. benchmarks) with the
    same metrics: annual vol; active return, paired monthly-active t, CAPM beta and
    annualized-alpha t vs both the TR-proxy and the EW-panel-net benchmark; per-regime
    and sub-period Sharpes; high/low-vol-slice Sharpe (trailing 21d Nifty vol vs its
    train-period median, causal); and the Feb-Jun 2020 crash return."""
    bench_sr, _ = sharpe_tstat(bench_ret.loc[split:])
    tr_test, ew_test = tr_ret.loc[split:], ew_ret.loc[split:]

    vol21 = mkt.pct_change().rolling(21).std()
    train_median = vol21.loc[:split].iloc[:-1].median()
    hi_mask = vol21.shift(1) > train_median  # causal: yesterday's vol buckets today

    rows = []
    for name, (weights, freq) in strategies.items():
        res = backtest_weights(px, rebalance_targets(weights, freq), cost_bps)
        rows.append(_row_metrics(name, res.returns.loc[split:], float(res.turnover.mean()),
                                 mkt, bench_sr, tr_test, ew_test, hi_mask))
    for name, ret in (extra_returns or {}).items():
        rows.append(_row_metrics(name, ret.loc[split:], 0.0, mkt, bench_sr,
                                 tr_test, ew_test, hi_mask))

    table = pd.DataFrame(rows).sort_values("test_sharpe", ascending=False).reset_index(drop=True)
    table.attrs["bench_sharpe"] = round(bench_sr, 3)
    return table
