"""Order-flow entropy magnitude predictor (Singha 2025, arXiv:2512.15720).

Faithful core: build the 15-state trade sequence, the rolling stationary-weighted
entropy, and the test of the paper's actual claim - low entropy predicts the
MAGNITUDE of the next move, not its direction. Input is second-resolution SPY
bars (last price + total volume per second); vendor-agnostic.
"""

import numpy as np
import pandas as pd

K = 15  # 3 sign states x 5 volume quintiles


def second_states(close: pd.Series, volume: pd.Series, vol_win: int = 120) -> pd.Series:
    """Map each second to a state in 0..14 = (sign(dP)+1)*5 + (volume_quintile-1)."""
    q = np.sign(close.diff().fillna(0.0)).astype(int) + 1  # {0,1,2}
    f = volume.rolling(vol_win, min_periods=1).rank(pct=True)  # CDF of current vol
    quintile = np.ceil(5.0 * f).clip(1, 5).astype(int)
    return (q * 5 + (quintile - 1)).rename("state")


def entropy_series(states: pd.Series, win: int = 120) -> pd.Series:
    """Stationary-weighted normalized entropy of the trailing-`win` transition matrix."""
    s = states.to_numpy()
    n = len(s)
    out = np.full(n, np.nan)
    counts = np.zeros((K, K))
    inv_log = 1.0 / np.log(K)
    head = 0  # transitions [head, t) are inside the window
    for t in range(1, n):
        counts[s[t - 1], s[t]] += 1.0
        while t - head > win:
            counts[s[head], s[head + 1]] -= 1.0
            head += 1
        out[t] = _entropy(counts, inv_log)
    return pd.Series(out, index=states.index, name="entropy")


def _entropy(counts: np.ndarray, inv_log: float) -> float:
    row = counts.sum(axis=1, keepdims=True)
    P = np.where(row > 0, counts / np.maximum(row, 1.0), 1.0 / K)
    pi = _stationary(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
    row_entropy = -(P * logP).sum(axis=1)
    return float(inv_log * (pi @ row_entropy))


def _stationary(P: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(P.T)
    pi = np.abs(np.real(v[:, np.argmin(np.abs(w - 1.0))]))
    total = pi.sum()
    return pi / total if total > 0 else np.full(K, 1.0 / K)


def magnitude_test(
    close: pd.Series,
    entropy: pd.Series,
    horizon_s: int = 300,
    low_pct: float = 5.0,
) -> dict:
    """Test the paper's claim on one (in-sample) block.

    Returns the magnitude ratio E[|r| | low-H]/E[|r|] with a Welch t-stat (the
    real claim, Theorem 1), directional accuracy of the momentum heuristic
    (should be ~50%, Theorem 2), and per-quintile |r| to mirror Figure 1.
    """
    fwd = (close.shift(-horizon_s) / close - 1.0) * 1e4  # bps
    trail = (close / close.shift(horizon_s) - 1.0) * 1e4
    df = pd.DataFrame({"H": entropy, "fwd": fwd, "trail": trail}).dropna()
    if df.empty:
        raise ValueError("no overlapping entropy/return observations")

    absr = df["fwd"].abs()
    thr = np.percentile(df["H"], low_pct)
    low = df["H"] <= thr

    a, b = absr[low], absr[~low]
    t_stat = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    quintile = pd.qcut(df["H"], 5, labels=False, duplicates="drop")
    hit = np.sign(df["fwd"][low]) == np.sign(df["trail"][low])

    return {
        "n": int(len(df)),
        "n_low": int(low.sum()),
        "uncond_abs_bps": round(float(absr.mean()), 3),
        "low_abs_bps": round(float(a.mean()), 3),
        "ratio": round(float(a.mean() / absr.mean()), 3),
        "t_stat": round(float(t_stat), 2),
        "dir_acc": round(float(hit.mean()), 3),
        "quintile_abs_bps": absr.groupby(quintile).mean().round(3).to_dict(),
    }
