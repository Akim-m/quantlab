import pandas as pd

from quantlab.data import close_prices, load_yahoo_ohlcv


def test_load_yahoo_ohlcv_reads_cache() -> None:
    data = load_yahoo_ohlcv(["SPY"], cache_dir="tests/fixtures/yahoo")

    assert data["SPY"].index.tolist() == list(
        pd.to_datetime(["2024-01-02", "2024-01-03"])
    )
    assert data["SPY"]["adj_close"].tolist() == [100.0, 101.0]


def test_close_prices_combines_symbols() -> None:
    idx = pd.date_range("2024-01-01", periods=2)
    data = {
        "SPY": pd.DataFrame({"close": [100.0, 101.0]}, index=idx),
        "QQQ": pd.DataFrame({"close": [200.0, 202.0]}, index=idx),
    }

    prices = close_prices(data, field="close")

    assert prices.to_dict("list") == {"SPY": [100.0, 101.0], "QQQ": [200.0, 202.0]}
