import numpy as np
import pandas as pd
import pytest

from quantlab.ensemble import blend_weights, equal_weight_returns, risk_parity_returns
from quantlab.optimization import erc_weights

RNG = np.random.default_rng(42)
IDX = pd.bdate_range("2023-01-02", periods=260)
RETS = pd.DataFrame(
    {
        "a": RNG.normal(0.0, 0.01, 260),
        "b": RNG.normal(0.0, 0.01, 260),
        "hi": RNG.normal(0.0, 0.05, 260),  # ~5x the vol of the others
    },
    index=IDX,
)
LOOKBACK = 126


def _neutral_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sig = pd.DataFrame(
        rng.normal(size=(6, 4)),
        index=pd.bdate_range("2024-01-01", periods=6),
        columns=list("ABCD"),
    )
    s = sig.sub(sig.mean(axis=1), axis=0)
    w = s.div(s.abs().sum(axis=1), axis=0)
    w.iloc[0] = 0.0  # warmup row
    return w


def test_blend_weights_unit_gross_neutral_zero_row() -> None:
    w = blend_weights([_neutral_frame(1), _neutral_frame(2)])

    assert not w.isna().any().any()
    assert (w.iloc[0] == 0.0).all()
    np.testing.assert_allclose(w.abs().sum(axis=1).iloc[1:], 1.0, atol=1e-12)
    np.testing.assert_allclose(w.sum(axis=1), 0.0, atol=1e-12)


def test_equal_weight_returns_row_mean() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    r = pd.DataFrame({"f": [0.01, 0.02, -0.01], "g": [0.03, 0.00, 0.01]}, index=idx)

    out = equal_weight_returns(r)

    pd.testing.assert_series_equal(out, pd.Series([0.02, 0.01, 0.0], index=idx))


def test_risk_parity_downweights_high_vol() -> None:
    out = risk_parity_returns(RETS, lookback=LOOKBACK)

    assert isinstance(out, pd.Series)
    assert out.index.equals(RETS.index)

    # last rebalance whose weights are actually applied (at index[-1], via shift)
    rebs = RETS.groupby(pd.Grouper(freq="ME")).tail(1).index
    applied = [d for d in rebs if len(RETS.loc[:d]) >= LOOKBACK and d <= RETS.index[-2]][-1]
    w = erc_weights(RETS.loc[:applied].tail(LOOKBACK).cov())

    assert w["hi"] == w.min()
    assert w["hi"] < 0.2  # far below the 1/3 equal-weight share
    assert out.iloc[-1] == pytest.approx(float((w * RETS.iloc[-1]).sum()), abs=1e-12)


# interior, non-rebalance dates. Truncating at exactly t (extra=0) is the case
# verified to flag unshifted and shift(-1) variants — t becomes a rebalance date
# of the truncated frame; extra=3 adds the plain "nothing past t leaks in" check.
@pytest.mark.parametrize("t", [180, 220, 251])
@pytest.mark.parametrize("extra", [0, 3])
def test_risk_parity_no_lookahead(t: int, extra: int) -> None:
    full = risk_parity_returns(RETS, lookback=LOOKBACK)
    trunc = risk_parity_returns(RETS.iloc[: t + 1 + extra], lookback=LOOKBACK)

    assert abs(full.iloc[t] - trunc.iloc[t]) < 1e-12
