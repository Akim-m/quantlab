"""Order-flow entropy magnitude predictor (Singha 2025, arXiv:2512.15720).

Faithful core: build the 15-state trade sequence, the rolling stationary-weighted
entropy, and the test of the paper's actual claim - low entropy predicts the
MAGNITUDE of the next move, not its direction. Input is second-resolution SPY
bars (last price + total volume per second); vendor-agnostic.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .tracking import log_run

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


def _mag_stats(fwd: pd.Series, trail: pd.Series, low: pd.Series) -> dict:
    absr = fwd.abs()
    a, b = absr[low], absr[~low]
    t = (a.mean() - b.mean()) / np.sqrt(
        a.var(ddof=1) / max(len(a), 1) + b.var(ddof=1) / max(len(b), 1)
    )
    hit = np.sign(fwd[low]) == np.sign(trail[low])
    return {
        "n": int(len(fwd)),
        "n_low": int(low.sum()),
        "uncond_abs_bps": round(float(absr.mean()), 3),
        "low_abs_bps": round(float(a.mean()), 3),
        "ratio": round(float(a.mean() / absr.mean()), 3),
        "t_stat": round(float(t), 3),
        "dir_acc": round(float(hit.mean()), 3),
    }


def walk_forward(close: pd.Series, volume: pd.Series, *, train_days: int = 10,
                 test_days: int = 5, n_folds: int = 5, low_pct: float = 5.0,
                 horizon_s: int = 300, vol_win: int = 120, win: int = 120) -> dict:
    """Pre-registered OOS magnitude test (RL-2026-06-28-05): per fold, fit the
    low-entropy threshold on train days only, freeze it, score test days;
    pool test rows across folds."""
    H = entropy_series(second_states(close, volume, vol_win), win)
    fwd = (close.shift(-horizon_s) / close - 1.0) * 1e4
    trail = (close / close.shift(horizon_s) - 1.0) * 1e4
    df = pd.DataFrame({"H": H, "fwd": fwd, "trail": trail})
    day = df.index.normalize()
    dates = day.unique().sort_values()
    span = train_days + test_days
    folds, pooled = [], []
    for i in range(n_folds):
        block = dates[i * span : (i + 1) * span]
        if len(block) <= train_days:  # no test dates left
            break
        h_tr = df.loc[day.isin(block[:train_days]), "H"].dropna()
        te = df[day.isin(block[train_days:])].dropna()
        if h_tr.empty or te.empty:
            continue
        h_thr = float(np.percentile(h_tr, low_pct))  # frozen: train rows only
        low = te["H"] <= h_thr
        s = _mag_stats(te["fwd"], te["trail"], low)
        folds.append({"fold": i, "h_thr": round(h_thr, 4), "n_test": s["n"],
                      "n_low": s["n_low"], "ratio": s["ratio"], "dir_acc": s["dir_acc"]})
        pooled.append(te.assign(low=low))
    all_te = pd.concat(pooled) if pooled else df.iloc[:0].assign(low=np.array([], dtype=bool))
    return {"n_folds": len(folds),
            "oos": _mag_stats(all_te["fwd"], all_te["trail"], all_te["low"]),
            "folds": folds}


def load_bars(path: str | Path) -> tuple[pd.Series, pd.Series]:
    """Read an intraday CSV (timestamp, close, volume) into aligned series."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df["close"], df["volume"]


def main() -> None:
    p = argparse.ArgumentParser(description="Order-flow entropy walk-forward (RL-2026-06-28-05)")
    p.add_argument("--path", required=True, help="intraday CSV: timestamp,close,volume")
    p.add_argument("--train-days", type=int, default=10)
    p.add_argument("--test-days", type=int, default=5)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--horizon-s", type=int, default=300)
    p.add_argument("--win", type=int, default=120)
    p.add_argument("--vol-win", type=int, default=120)
    p.add_argument("--low-pct", type=float, default=5.0)
    p.add_argument("--hypothesis-ref", default=None)
    p.add_argument("--note", default=None, help="honest caveats recorded with the run")
    args = p.parse_args()

    close, volume = load_bars(args.path)
    days = int(close.index.normalize().nunique())
    print(f"{args.path}: {len(close)} bars over {days} days, "
          f"{close.index[0]} -> {close.index[-1]}")

    res = walk_forward(
        close, volume, train_days=args.train_days, test_days=args.test_days,
        n_folds=args.n_folds, low_pct=args.low_pct, horizon_s=args.horizon_s,
        vol_win=args.vol_win, win=args.win,
    )
    print(f"\nfolds: {res['n_folds']}\nOOS (pooled): {res['oos']}")
    for f in res["folds"]:
        print(" ", f)

    log_run({
        "hypothesis_ref": args.hypothesis_ref,
        "source": args.path,
        "note": args.note,
        "bars": int(len(close)),
        "days": days,
        "strategy": "order_flow_entropy",
        "params": {"train_days": args.train_days, "test_days": args.test_days,
                   "n_folds": args.n_folds, "horizon_s": args.horizon_s,
                   "win": args.win, "vol_win": args.vol_win, "low_pct": args.low_pct},
        "metrics": res,
        "status": "success",
    })


if __name__ == "__main__":
    main()
