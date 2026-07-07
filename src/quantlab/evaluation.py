"""Multiple-testing correction for the anomaly-replication study (RL-2026-07-07).

When 24 strategies are tested at once, an unadjusted Sharpe t-stat overstates
significance. Two guards, reported together:
  - Benjamini-Hochberg FDR on the one-sided Sharpe t-stats (controls the expected
    false-discovery rate across the family).
  - Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): the probability the true
    Sharpe exceeds zero, deflated by the number of trials, track length, and the
    returns' skew/kurtosis.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist

GAMMA = 0.5772156649015329  # Euler-Mascheroni


def sharpe_tstat(returns: pd.Series, periods: int = 252) -> tuple[float, float]:
    """Annualized Sharpe and the t-stat of its mean (H0: mean return = 0)."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0, 0.0
    sr = r.mean() / r.std()
    return float(periods**0.5 * sr), float(len(r) ** 0.5 * sr)


def one_sided_p(t_stat: float, n: int) -> float:
    """One-sided p-value for Sharpe > 0 from Student-t (df = n-1)."""
    if n < 2:
        return 1.0
    return float(tdist.sf(t_stat, df=n - 1))


def benjamini_hochberg(pvalues: list[float], q: float = 0.10) -> np.ndarray:
    """Boolean mask of hypotheses rejected under BH-FDR at level q."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, n + 1) / n
    passed = np.where(p[order] <= thresh)[0]
    reject = np.zeros(n, dtype=bool)
    if len(passed):
        reject[order[: passed.max() + 1]] = True
    return reject


def deflated_sharpe_ratio(returns: pd.Series, sr_trials: list[float]) -> float:
    """P(true Sharpe > deflated benchmark), per Bailey-Lopez de Prado.

    `sr_trials` are the per-period (non-annualized) Sharpe estimates of all
    strategies in the family; their spread sets the benchmark the candidate must
    beat to be called skill rather than the best of many lucky draws.
    """
    r = returns.dropna()
    if len(r) < 3 or r.std() == 0:
        return 0.0
    sr = r.mean() / r.std()
    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3.0  # pandas gives excess; formula wants raw

    trials = np.asarray(sr_trials, dtype=float)
    n = len(trials)
    if n < 2:
        return 0.0
    v = np.var(trials, ddof=1)
    sr0 = v**0.5 * ((1 - GAMMA) * norm.ppf(1 - 1 / n) + GAMMA * norm.ppf(1 - 1 / (n * np.e)))

    denom = (1 - skew * sr + (kurt - 1) / 4 * sr**2) ** 0.5
    return float(norm.cdf((sr - sr0) * (len(r) - 1) ** 0.5 / denom))
