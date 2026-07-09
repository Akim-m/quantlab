"""RL-2026-07-23: index-level band (Bollinger) mean reversion on NIFTYBEES.NS.

Multi-day overreaction at the INDEX level - when price stretches far below its trailing
mean, panicked selling overshoots and reverts. A long-only timing book on the liquid
index ETF (~a handful of round trips a year, so costs cannot gate it), distinct from the
cost-gated single-stock reversal family. z = (price - 20d MA) / 20d sigma. Enter long when
z < -k while flat; exit at mean touch (z >= 0) or after a 10-day time stop - exactly four
locked variants k in {1.5, 2.0} x exit in {mean-touch, 10d-stop}. TRAIN (2010->2016-12-31)
freezes one variant on Sharpe; the TEST window (2017+) is read once at 5/10/20 bps.

Prior-day info / alignment: the position held DURING day t is backtest_weights' target set
at t-1, and that target is the state-machine decision from z_{t-1} (prices through t-1).
So perturbing price_t cannot change day t's position - proven in the tests. backtest_weights
earns a target set at D on D+1, so the weights frame needs no manual shift.

Bar (idiom-correct per the RL-2026-07-19 lesson: a timing book is judged in the
risk-adjusted idiom, not a mean-return paired-t): promotion iff the Ledoit-Wolf (2008)
HAC-robust Sharpe-difference z of the band book over buy-and-hold NIFTYBEES exceeds 1 at
10 bps AND survives 20 bps. Prices go through the RL-2026-07-17 spike repair - the 2019-12
fabricated prints hit NIFTYBEES inside the test window.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from .backtest import backtest_weights
from .data import close_prices, load_yahoo_ohlcv
from .evaluation import sharpe_tstat
from .tracking import log_run
from .xasset_trend import clean_prices

SYMBOL = "NIFTYBEES.NS"
WINDOW = 20
TIME_STOP = 10
VARIANTS = [(1.5, "mean"), (1.5, "stop10"), (2.0, "mean"), (2.0, "stop10")]
EXIT_LABEL = {"mean": "mean-touch", "stop10": "10d-stop"}
TRAIN0, TRAIN1 = "2010-01-01", "2016-12-31"
HEADLINE_BPS = 10.0
# Frozen on TRAIN (2010->2016-12-31) Sharpe @10 bps, before the single test read:
FROZEN_K, FROZEN_EXIT = 2.0, "mean"


def zscore(prices: pd.Series, window: int = WINDOW) -> pd.Series:
    """z = (price - rolling mean) / rolling std, all from prices up to and including the
    current bar; NaN before warm-up or where the std is zero (no signal)."""
    ma = prices.rolling(window).mean()
    sd = prices.rolling(window).std()
    return ((prices - ma) / sd).where(sd > 0)


def _positions(z: np.ndarray, k: float, exit_rule: str) -> np.ndarray:
    """Long-only band state machine over a z-series -> 0/1 per bar. Enter (1) when z < -k
    while flat; once long, exit (0) at z >= 0 (mean-touch) or after TIME_STOP held bars
    (stop10). No re-entry on an exit bar. Bar d's value is the position to carry into d+1."""
    pos = np.zeros(len(z))
    in_pos = False
    held = 0
    for i, zi in enumerate(z):
        if not np.isfinite(zi):
            in_pos, held = False, 0
            continue
        if in_pos:
            held += 1
            hit = zi >= 0.0 if exit_rule == "mean" else held >= TIME_STOP
            if hit:
                in_pos, held = False, 0
            else:
                pos[i] = 1.0
        elif zi < -k:
            in_pos, held, pos[i] = True, 0, 1.0
    return pos


def band_weights(prices: pd.DataFrame, k: float, exit_rule: str) -> pd.DataFrame:
    col = prices.columns[0]
    pos = _positions(zscore(prices[col]).to_numpy(), k, exit_rule)
    return pd.DataFrame(pos, index=prices.index, columns=[col])


def band_returns(prices: pd.DataFrame, k: float, exit_rule: str, cost_bps: float) -> pd.Series:
    return backtest_weights(prices, band_weights(prices, k, exit_rule), cost_bps).returns


def _metrics(r: pd.Series) -> tuple[float, float, float]:
    """(annualized Sharpe, annualized return, max drawdown) of a daily return series."""
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    ann = float((1.0 + r).prod() ** (252 / max(len(r), 1)) - 1.0)
    return round(sharpe_tstat(r)[0], 3), round(ann, 4), round(dd, 4)


def sharpe_diff_test(r1: np.ndarray, r2: np.ndarray) -> tuple[float, float, float]:
    """Ledoit-Wolf (2008) HAC-robust test of H0: SR1 - SR2 = 0 for two aligned return
    series. Returns (annualized SR1, annualized SR2, z). Delta method on the four moments
    (mu1, mu2, E[r1^2], E[r2^2]) with a Bartlett (Newey-West) HAC long-run covariance and
    the automatic bandwidth floor(4(T/100)^(2/9)). Ledoit-Wolf note any consistent HAC is
    valid; their prewhitened-Parzen refinement is for small samples and immaterial here
    (T~2.4k, L~8). The z is invariant to annualization, so it is computed per-period."""
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    n = len(r1)
    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = (r1**2).mean(), (r2**2).mean()
    s1, s2 = g1 - mu1**2, g2 - mu2**2
    sr1, sr2 = mu1 / s1**0.5, mu2 / s2**0.5
    grad = np.array([g1 / s1**1.5, -g2 / s2**1.5, -mu1 / (2 * s1**1.5), mu2 / (2 * s2**1.5)])

    y = np.column_stack([r1, r2, r1**2, r2**2])
    y = y - y.mean(0)
    lag = int(4 * (n / 100.0) ** (2 / 9))
    psi = y.T @ y / n
    for j in range(1, lag + 1):
        gamma = y[j:].T @ y[:-j] / n
        psi += (1.0 - j / (lag + 1.0)) * (gamma + gamma.T)

    var = float(grad @ psi @ grad / n)
    z = (sr1 - sr2) / var**0.5 if var > 0 else 0.0
    ann = 252**0.5
    return float(ann * sr1), float(ann * sr2), float(z)


def load_prices(refresh: bool = False) -> pd.DataFrame:
    return clean_prices(close_prices(load_yahoo_ohlcv([SYMBOL], refresh=refresh))[[SYMBOL]])


def _power(prices: pd.DataFrame, k: float, exit_rule: str, split: str) -> tuple[float, float]:
    """(entries per year, fraction of days invested) over the test window, on the
    position actually held each day = the weights frame shifted into D+1."""
    held = band_weights(prices, k, exit_rule)[SYMBOL].shift(1).fillna(0.0).loc[split:]
    entries = int(((held.to_numpy()[1:] > 0) & (held.to_numpy()[:-1] == 0)).sum())
    years = len(held) / 252.0
    return round(entries / years, 2), round(float((held > 0).mean()), 4)


def run(split: str = "2017-01-01", refresh: bool = False) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        px = load_prices(refresh=refresh)

        train_rows = []
        for k, ex in VARIANTS:
            sr, ann, dd = _metrics(band_returns(px, k, ex, HEADLINE_BPS).loc[TRAIN0:TRAIN1])
            train_rows.append({"k": k, "exit": EXIT_LABEL[ex], "train_sharpe": sr,
                               "train_ann": ann, "train_maxdd": dd})
        train = pd.DataFrame(train_rows).sort_values("train_sharpe", ascending=False).reset_index(drop=True)
        argmax = (float(train.iloc[0]["k"]), train.iloc[0]["exit"])

        bh = px[SYMBOL].pct_change().loc[split:].dropna()
        bh_sr = round(sharpe_tstat(bh)[0], 3)
        entries_yr, pct_inv = _power(px, FROZEN_K, FROZEN_EXIT, split)

        test_rows, z_by_cost = [], {}
        for cost in (5.0, 10.0, 20.0):
            book = band_returns(px, FROZEN_K, FROZEN_EXIT, cost).loc[split:].dropna()
            common = book.index.intersection(bh.index)
            book_sr, _, z = sharpe_diff_test(book.loc[common].to_numpy(), bh.loc[common].to_numpy())
            z_by_cost[int(cost)] = round(z, 3)
            sr, ann, dd = _metrics(book)
            test_rows.append({"cost_bps": int(cost), "test_sharpe": sr, "ann_return": ann,
                              "max_dd": dd, "lw_z_vs_bh": round(z, 3)})
        test = pd.DataFrame(test_rows)
        promoted = bool(z_by_cost[10] > 1.0 and z_by_cost[20] > 1.0)

    print(f"BAND-MR NIFTYBEES  {px.index[0].date()}->{px.index[-1].date()}  test>={split}")
    print(f"\n[TRAIN {TRAIN0}->{TRAIN1} @{int(HEADLINE_BPS)}bps | 4 variants -> FREEZE on Sharpe]")
    print(train.to_string(index=False))
    print(f"FROZEN: (k={FROZEN_K}, exit={EXIT_LABEL[FROZEN_EXIT]})  (TRAIN argmax = "
          f"(k={argmax[0]}, exit={argmax[1]}); "
          f"{'matches' if argmax == (FROZEN_K, EXIT_LABEL[FROZEN_EXIT]) else 'MISMATCH - freeze held per protocol'})")
    print(f"\n[TEST {split}+ ONE read | frozen (k={FROZEN_K}, exit={EXIT_LABEL[FROZEN_EXIT]}) | 5/10/20 bps]")
    print(test.to_string(index=False))
    print(f"B&H test Sharpe = {bh_sr}  ({len(bh)} days)   entries/yr = {entries_yr}   "
          f"days invested = {pct_inv*100:.1f}%")
    print(f"\n[PROMOTION]  bar: Ledoit-Wolf Sharpe-diff z(book - B&H) > 1 @10bps, surviving 20bps  |  "
          f"z@10={z_by_cost[10]} z@20={z_by_cost[20]}  -> {'PROMOTED' if promoted else 'NOT PROMOTED'}")

    for row in test_rows:
        log_run({"hypothesis_ref": "RL-2026-07-23", "universe": SYMBOL,
                 "cost_bps": float(row["cost_bps"]), "split": split,
                 "strategy": f"band_mr_k{FROZEN_K}_{FROZEN_EXIT}",
                 "frozen_k": FROZEN_K, "frozen_exit": FROZEN_EXIT, "metrics": row,
                 "bh_sharpe": bh_sr, "lw_z_vs_bh": row["lw_z_vs_bh"],
                 "entries_per_yr": entries_yr, "days_invested": pct_inv,
                 "promoted": promoted, "n_trials_family": len(VARIANTS), "status": "success"})


def main() -> None:
    p = argparse.ArgumentParser(description="RL-2026-07-23 index-band mean reversion on NIFTYBEES")
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(split=a.split, refresh=a.refresh)


if __name__ == "__main__":
    main()
