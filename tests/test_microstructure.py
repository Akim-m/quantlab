"""Guards for the order-flow entropy module: above all, no look-ahead."""

import numpy as np
import pandas as pd

from quantlab.microstructure import (
    K,
    _entropy,
    _stationary,
    entropy_series,
    magnitude_test,
    second_states,
)


def test_second_states_hand_example() -> None:
    close = pd.Series([10.0, 11.0, 11.0, 10.0, 12.0])  # flat, up, flat, down, up
    vol_up = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])  # always top quintile
    vol_dn = pd.Series([500.0, 400.0, 300.0, 200.0, 100.0])

    # state = (sign(dP)+1)*5 + (quintile-1); rising vol pins quintile 5
    assert second_states(close, vol_up).tolist() == [9, 14, 9, 4, 14]
    # falling vol: rolling pct-ranks 1, .5, 1/3, .25, .2 -> quintiles 5, 3, 2, 2, 1
    assert second_states(close, vol_dn).tolist() == [9, 12, 6, 1, 10]


def test_second_states_range() -> None:
    rng = np.random.default_rng(42)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 1e-4, 500)))
    volume = pd.Series(rng.integers(1, 1000, 500).astype(float))

    st = second_states(close, volume)

    assert pd.api.types.is_integer_dtype(st)
    assert st.between(0, K - 1).all()


def test_entropy_no_lookahead() -> None:
    # The crown jewel: truncating the future must not change a past value.
    # A failure here is a real look-ahead bug in entropy_series, not test noise.
    rng = np.random.default_rng(7)
    states = pd.Series(rng.integers(0, K, 300))
    full = entropy_series(states)

    for t in [5, 50, 119, 130, 200, 299]:
        trunc = entropy_series(states.iloc[: t + 1])
        assert abs(full.iloc[t] - trunc.iloc[t]) < 1e-12, f"look-ahead at t={t}"


def test_entropy_deterministic_is_zero() -> None:
    # cycle 0..14: every state has one successor -> converges to exactly 0.0
    det = pd.Series(np.tile(np.arange(K), 20))

    assert entropy_series(det).iloc[-1] < 1e-12


def test_entropy_disorder_above_order() -> None:
    rng = np.random.default_rng(7)
    ent = entropy_series(pd.Series(rng.integers(0, K, 300)))

    assert np.isnan(ent.iloc[0])  # no transition yet
    assert 0.5 < ent.iloc[-1] <= 1.0  # measured ~0.68, far above deterministic 0


def test_entropy_window_drops_old_transitions() -> None:
    win = 120
    rng = np.random.default_rng(11)
    a = pd.Series(rng.integers(0, K, 300))
    # same last win+1 states, different prefix -> identical final entropy
    b = pd.Series(np.concatenate([np.zeros(179, dtype=int), a.iloc[-(win + 1) :].to_numpy()]))
    assert not a.iloc[:179].equals(b.iloc[:179])

    assert abs(entropy_series(a, win).iloc[-1] - entropy_series(b, win).iloc[-1]) < 1e-12


def test_entropy_counts_extremes() -> None:
    inv_log = 1.0 / np.log(K)

    assert _entropy(np.ones((K, K)), inv_log) == 1.0  # uniform transitions
    assert _entropy(10.0 * np.roll(np.eye(K), 1, axis=1), inv_log) == 0.0  # deterministic


def test_stationary_doubly_stochastic_is_uniform() -> None:
    P = 0.5 * np.eye(K) + 0.5 * np.roll(np.eye(K), 1, axis=1)

    pi = _stationary(P)

    assert np.allclose(pi, 1.0 / K, atol=1e-12)
    assert (pi >= 0).all()
    assert pi.sum() == 1.0


def test_stationary_two_state_chain() -> None:
    # states 0,1 recurrent with p01=0.1, p10=0.5 -> pi = (5/6, 1/6); rest transient
    P = np.zeros((K, K))
    P[0, 0], P[0, 1] = 0.9, 0.1
    P[1, 0], P[1, 1] = 0.5, 0.5
    P[2:, 0] = 1.0

    pi = _stationary(P)

    assert np.allclose(pi[:2], [5 / 6, 1 / 6], atol=1e-12)
    assert np.allclose(pi[2:], 0.0, atol=1e-12)
    assert pi.sum() == 1.0


def test_magnitude_low_entropy_precedes_big_moves() -> None:
    # magnitude_test intentionally uses FORWARD returns: it evaluates the paper's
    # claim in-sample, it is not a tradable signal. Do not "fix" it into causality.
    n = 400
    low_idx = list(range(8, 392, 8))  # 12% low so the 10th percentile lands on them
    ent = pd.Series(0.9, index=range(n))
    ent.iloc[low_idx] = 0.1

    r = np.full(n, 0.5e-4)
    r[1::2] *= -1
    for j, t in enumerate(low_idx):  # big move right after each low-entropy second
        r[t + 1] = 50e-4 * (1 if j % 2 == 0 else -1)
    close = pd.Series(100 * np.cumprod(1 + r), index=range(n))

    res = magnitude_test(close, ent, horizon_s=1, low_pct=10.0)

    assert res["n"] == n - 2  # first row loses trail, last loses fwd
    assert res["n_low"] == len(low_idx)
    assert res["ratio"] > 1
    assert res["t_stat"] > 0
    assert res["low_abs_bps"] > res["uncond_abs_bps"]
    assert res["dir_acc"] == 0.5  # alternating jump signs: magnitude edge, no direction

    ent_nan = ent.copy()
    ent_nan.iloc[5] = np.nan
    assert magnitude_test(close, ent_nan, horizon_s=1, low_pct=10.0)["n"] == n - 3
