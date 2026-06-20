from pathlib import Path
from time import time
import json
from urllib.request import Request, urlopen

import pandas as pd


def load_yahoo_ohlcv(
    symbols: list[str],
    cache_dir: str | Path = "data/raw/yahoo",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    data = {}
    for symbol in symbols:
        path = cache / f"{symbol.lower()}.csv"
        if refresh or not path.exists():
            _download_yahoo(symbol, path)
        data[symbol.upper()] = _read_ohlcv(path)
    return data


def close_prices(data: dict[str, pd.DataFrame], field: str = "adj_close") -> pd.DataFrame:
    prices = {symbol: df[field] for symbol, df in data.items()}
    return pd.DataFrame(prices).sort_index()


def _download_yahoo(symbol: str, path: Path) -> None:
    period2 = int(time())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}"
        f"?period1=0&period2={period2}&interval=1d&events=history"
        "&includeAdjustedClose=true"
    )
    req = Request(url, headers={"User-Agent": "quantlab/0.1"})
    with urlopen(req, timeout=30) as res:
        raw = json.loads(res.read().decode("utf-8"))

    result = raw["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    adj = result["indicators"]["adjclose"][0]["adjclose"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(result["timestamp"], unit="s").date,
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "adj_close": adj,
            "volume": quote["volume"],
        }
    ).dropna()
    df.to_csv(path, index=False)


def _read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.lower() for col in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"missing date column in {path}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {path}: {missing}")
    return df[cols]
