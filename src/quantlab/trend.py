import numpy as np
import pandas as pd

from .features import rolling_momentum, rolling_vol


def _size(direction: pd.DataFrame, prices: pd.DataFrame, vol_lb: int = 63) -> pd.DataFrame:
    inv = 1.0 / rolling_vol(prices, vol_lb).replace(0.0, np.nan)
    raw = direction * inv
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    return raw.div(gross, axis=0).fillna(0.0)  # inverse-vol, unit gross


def tsmom(prices: pd.DataFrame, lb: int = 252, vol_lb: int = 63) -> pd.DataFrame:
    return _size(np.sign(rolling_momentum(prices, lb)), prices, vol_lb)


def donchian(prices: pd.DataFrame, lb: int = 252, vol_lb: int = 63) -> pd.DataFrame:
    # channel over the trailing lb bars EXCLUDING today, else close >= max is trivially true
    prev = prices.shift(1)
    brk = pd.DataFrame(
        np.where(prices >= prev.rolling(lb).max(), 1.0,
                 np.where(prices <= prev.rolling(lb).min(), -1.0, np.nan)),
        index=prices.index,
        columns=prices.columns,
    )
    return _size(brk.ffill().fillna(0.0), prices, vol_lb)


def dual_momentum(prices: pd.DataFrame, lb: int = 252, vol_lb: int = 63) -> pd.DataFrame:
    ret = rolling_momentum(prices, lb)
    qual = (ret > 0) & ret.gt(ret.median(axis=1), axis=0)
    inv = (1.0 / rolling_vol(prices, vol_lb).replace(0.0, np.nan)).where(qual)
    total = inv.sum(axis=1).replace(0.0, np.nan)
    return inv.div(total, axis=0).fillna(0.0)  # long-only, sums to 1 (or 0 = cash)


def vol_managed(
    prices: pd.DataFrame, vol_lb: int = 21, target: float = 0.15, cap: float = 3.0
) -> pd.DataFrame:
    ann = rolling_vol(prices, vol_lb) * np.sqrt(252)
    expo = (target / ann.replace(0.0, np.nan)).clip(0.0, cap)
    gross = expo.sum(axis=1).replace(0.0, np.nan)
    return expo.div(gross, axis=0).fillna(0.0)


def crash_scaled_tsmom(
    prices: pd.DataFrame,
    lb: int = 252,
    vol_lb: int = 63,
    target: float = 0.15,
    cap: float = 3.0,
) -> pd.DataFrame:
    # tsmom vol proxied by same-day sign(ret_lb) * daily ret (path proxy, computable
    # pre-backtest); only its trailing 63d std feeds the scale, so weights at t use
    # nothing past t
    proxy = (np.sign(rolling_momentum(prices, lb)) * prices.pct_change()).mean(axis=1)
    strat_vol = proxy.rolling(63).std() * np.sqrt(252)
    scale = (target / strat_vol.replace(0.0, np.nan)).clip(0.0, cap)
    return tsmom(prices, lb, vol_lb).mul(scale, axis=0).fillna(0.0)


def overnight_intraday(open_px: pd.DataFrame, close_px: pd.DataFrame, lb: int = 21) -> pd.DataFrame:
    # RAW (unadjusted) open/close: a split distorts the overnight return on its ex-date
    overnight = open_px / close_px.shift(1) - 1.0
    intraday = close_px / open_px - 1.0
    sig = overnight.rolling(lb).mean() - intraday.rolling(lb).mean()
    ctr = sig.sub(sig.mean(axis=1), axis=0)  # dollar-neutral
    gross = ctr.abs().sum(axis=1).replace(0.0, np.nan)
    return ctr.div(gross, axis=0).fillna(0.0)


def bollinger(prices: pd.DataFrame, lb: int = 20, k: float = 2.0, vol_lb: int = 63) -> pd.DataFrame:
    # gated mean reversion: long only below the lower band, short only above the
    # upper band (|z| > k), flat inside - so k is the entry threshold
    mid = prices.rolling(lb).mean()
    sd = prices.rolling(lb).std().replace(0.0, np.nan)
    z = (prices - mid) / sd
    direction = (-np.sign(z)).where(z.abs() > k, 0.0)
    return _size(direction, prices, vol_lb)


def pairs(
    prices: pd.DataFrame,
    lb: int = 63,
    pairs: tuple[tuple[str, str], ...] = (
        ("SPY", "QQQ"),
        ("GLD", "SLV"),
        ("TLT", "IEF"),
        ("XLE", "XLB"),
    ),
) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for a, b in pairs:
        if a not in prices.columns or b not in prices.columns:
            continue
        spread = np.log(prices[a]) - np.log(prices[b])
        z = (spread - spread.rolling(lb).mean()) / spread.rolling(lb).std()
        z = z.fillna(0.0)
        w[a] -= z  # fade the spread
        w[b] += z
    gross = w.abs().sum(axis=1).replace(0.0, np.nan)
    return w.div(gross, axis=0).fillna(0.0)
