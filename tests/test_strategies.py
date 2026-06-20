import pandas as pd
import pytest

from quantlab.portfolio import inverse_vol_weight
from quantlab.strategies import long_top_momentum


def test_long_top_momentum_selects_best_assets() -> None:
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 121.0],
            "BBB": [100.0, 105.0, 106.0],
            "CCC": [100.0, 90.0, 80.0],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )

    weights = long_top_momentum(prices, lookback=1, count=2)

    assert weights.iloc[-1].to_dict() == pytest.approx(
        {"AAA": 0.5, "BBB": 0.5, "CCC": 0.0}
    )


def test_inverse_vol_weight_normalizes_rows() -> None:
    vol = pd.DataFrame(
        {"AAA": [0.2], "BBB": [0.1]},
        index=pd.date_range("2024-01-01", periods=1),
    )

    weights = inverse_vol_weight(vol)

    assert weights.iloc[0].sum() == pytest.approx(1.0)
    assert weights.iloc[0]["BBB"] > weights.iloc[0]["AAA"]
