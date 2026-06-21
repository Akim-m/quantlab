import pandas as pd

from quantlab.portfolio import rebalance_targets


def test_rebalance_targets_keeps_only_period_end_rows() -> None:
    weights = pd.DataFrame(
        {"AAA": [1.0, 0.8, 0.6, 0.4], "BBB": [0.0, 0.2, 0.4, 0.6]},
        index=pd.date_range("2024-01-01", periods=4),
    )

    res = rebalance_targets(weights, "2D")

    assert res.iloc[0].isna().all()
    assert res.iloc[1].tolist() == [0.8, 0.2]
    assert res.iloc[2].isna().all()
    assert res.iloc[3].tolist() == [0.4, 0.6]
