import numpy as np
import pandas as pd
import pytest

from quantlab import xsec

RNG = np.random.default_rng(7)
IDX = pd.date_range("2015-01-02", periods=1400, freq="B")
PRICES = pd.DataFrame(
    100.0 * np.exp(np.cumsum(RNG.normal(0.0, 0.01, (1400, 4)), axis=0)),
    index=IDX,
    columns=["AAA", "BBB", "CCC", "DDD"],
)
MARKET = pd.Series(
    100.0 * np.exp(np.cumsum(RNG.normal(0.0, 0.008, 1400))), index=IDX, name="MKT"
)

FACTORS = {
    "short_term_reversal": lambda: xsec.short_term_reversal(PRICES),
    "momentum_12_1": lambda: xsec.momentum_12_1(PRICES),
    "long_term_reversal": lambda: xsec.long_term_reversal(PRICES),
    "low_volatility": lambda: xsec.low_volatility(PRICES),
    "idio_vol": lambda: xsec.idio_vol(PRICES, MARKET),
    "max_lottery": lambda: xsec.max_lottery(PRICES),
    "high_52w": lambda: xsec.high_52w(PRICES),
    "skewness": lambda: xsec.skewness(PRICES),
    "residual_momentum": lambda: xsec.residual_momentum(PRICES, MARKET),
    "seasonality": lambda: xsec.seasonality(PRICES),
    "downside_beta_factor": lambda: xsec.downside_beta_factor(PRICES, MARKET),
}


# interior dates only: truncating at the last row is the identity and can
# never catch a leak. All three points verified to flag leaky variants
# (unshifted seasonality, shift(-21) momentum).
@pytest.mark.parametrize("t", [800, 1050, 1200])
def test_momentum_12_1_no_lookahead(t: int) -> None:
    full = xsec.momentum_12_1(PRICES).iloc[t]
    trunc = xsec.momentum_12_1(PRICES.iloc[: t + 1]).iloc[-1]
    np.testing.assert_allclose(full.to_numpy(), trunc.to_numpy(), atol=1e-9)


@pytest.mark.parametrize("t", [800, 1050, 1200])
def test_seasonality_no_lookahead(t: int) -> None:
    full = xsec.seasonality(PRICES).iloc[t]
    trunc = xsec.seasonality(PRICES.iloc[: t + 1]).iloc[-1]
    np.testing.assert_allclose(full.to_numpy(), trunc.to_numpy(), atol=1e-9)


@pytest.mark.parametrize("name", sorted(FACTORS))
def test_dollar_neutral_unit_gross(name: str) -> None:
    w = FACTORS[name]()
    assert list(w.index) == list(IDX)
    np.testing.assert_allclose(w.sum(axis=1).to_numpy(), 0.0, atol=1e-9)
    gross = w.abs().sum(axis=1).to_numpy()
    active = gross > 0
    assert active[-1]  # warmed up by the end of the panel
    np.testing.assert_allclose(gross[active], 1.0, atol=1e-9)


def test_short_term_reversal_longs_biggest_faller() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    prices = pd.DataFrame(
        {
            "UP": np.linspace(100.0, 110.0, 10),
            "FLAT": np.full(10, 100.0),
            "DOWN": np.linspace(100.0, 80.0, 10),
        },
        index=idx,
    )
    w = xsec.short_term_reversal(prices).iloc[-1]
    assert w["DOWN"] == w.max() > 0
    assert w["UP"] == w.min() < 0


def test_low_volatility_longs_calmest() -> None:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    noise = rng.normal(0.0, 1.0, 80)
    prices = pd.DataFrame(
        {
            "CALM": 100.0 * np.exp(np.cumsum(0.001 * noise)),
            "MID": 100.0 * np.exp(np.cumsum(0.01 * noise)),
            "WILD": 100.0 * np.exp(np.cumsum(0.03 * noise)),
        },
        index=idx,
    )
    w = xsec.low_volatility(prices, lb=60).iloc[-1]
    assert w["CALM"] == w.max() > 0
    assert w["WILD"] == w.min() < 0
