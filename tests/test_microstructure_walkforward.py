"""Walk-forward OOS tests for the entropy magnitude predictor (RL-2026-06-28-05)."""

import numpy as np
import pandas as pd
import pytest

from quantlab.microstructure import entropy_series, second_states, walk_forward

CFG = dict(train_days=10, test_days=5, n_folds=5, low_pct=5.0,
           horizon_s=60, vol_win=60, win=60)


def synth(n_days: int, quiet_days=(), n: int = 600, seed: int = 7):
    """Per-second bars: noisy walk everywhere; on quiet_days a frozen 180 s
    stretch drives entropy to ~0 and ends in a ~100 bps jump, so low-entropy
    seconds precede a large |move|."""
    rng = np.random.default_rng(seed)
    px, vol, idx = [], [], []
    for d in range(n_days):
        step = rng.choice([-0.01, 0.0, 0.01], size=n)
        v = rng.integers(1, 100, size=n).astype(float)
        if d in quiet_days:
            step[120:300], v[120:300] = 0.0, 50.0
            step[300] = 1.0
        px.append(100.0 + np.cumsum(step))
        vol.append(v)
        t0 = pd.Timestamp("2026-01-01 09:30") + pd.Timedelta(days=d)
        idx.append(pd.date_range(t0, periods=n, freq="s"))
    ix = idx[0].append(idx[1:])
    return pd.Series(np.concatenate(px), index=ix), pd.Series(np.concatenate(vol), index=ix)


@pytest.fixture(scope="module")
def wf():
    # 30 days -> 2 folds; quiet stretches only on each fold's TEST days
    close, vol = synth(30, quiet_days=set(range(10, 15)) | set(range(25, 30)))
    return close, vol, walk_forward(close, vol, **CFG)


def test_signal_recovery(wf):
    _, _, res = wf
    assert res["n_folds"] == 2
    assert res["oos"]["n_low"] > 0
    assert res["oos"]["ratio"] > 1.0


def test_leakage_guard(wf):
    close, vol, res = wf
    H = entropy_series(second_states(close, vol, CFG["vol_win"]), CFG["win"])
    day = H.index.normalize()
    dates = day.unique().sort_values()
    span = CFG["train_days"] + CFG["test_days"]
    for f in res["folds"]:
        block = dates[f["fold"] * span : (f["fold"] + 1) * span]
        tr = H[day.isin(block[: CFG["train_days"]])].dropna()
        assert f["h_thr"] == round(float(np.percentile(tr, CFG["low_pct"])), 4)
        # non-vacuous: same percentile over train+test rows must differ
        both = H[day.isin(block)].dropna()
        assert f["h_thr"] != round(float(np.percentile(both, CFG["low_pct"])), 4)


def test_fold_plumbing():
    small = dict(train_days=2, test_days=1, n_folds=5, horizon_s=30, vol_win=30, win=30)

    close, vol = synth(15, n=200, seed=1)
    res = walk_forward(close, vol, **small)
    assert res["n_folds"] == len(res["folds"]) == 5

    close, vol = synth(4, n=200, seed=2)
    res = walk_forward(close, vol, **small)
    assert res["n_folds"] == len(res["folds"]) == 1

    close, vol = synth(2, n=200, seed=3)
    res = walk_forward(close, vol, **small)
    assert res["n_folds"] == 0 and res["folds"] == [] and res["oos"]["n"] == 0
