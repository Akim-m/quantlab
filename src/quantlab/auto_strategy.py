"""The only file the autoresearch loop edits. CONFIG is the current champion.

Signal stage is pluggable: 'trend' (time-series momentum) or 'ml' (gradient-boosted
forward-return predictor). Construction stage gates on positive signal, weights by a risk
backbone, then vol-targets. See program.md.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import pct_returns, rolling_momentum, rolling_vol
from .optimization import min_variance_weights
from .portfolio import equal_weight, inverse_vol_weight

CONFIG = {
    "signal": "trend",          # trend | ml
    "lookback": 63,             # trend lookback (signal=trend)
    "horizon": 21,              # forward target / holding horizon (signal=ml)
    "max_depth": 3,             # ml model
    "learning_rate": 0.05,      # ml model
    "n_estimators": 200,        # ml model
    "mode": "gate",            # gate | tilt
    "backbone": "inverse_vol",  # equal | inverse_vol | min_variance
    "vol_target": 0.12,
    "vol_window": 63,
    "max_gross": 2.0,
    "rebalance": "ME",          # ME | 2ME
}

ML_FEATURES = ["mom21", "mom63", "mom126", "mom252", "vol63", "rev5"]


def strategy_weights(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    signal = _signal(prices, cfg)
    vol = rolling_vol(prices, cfg["vol_window"])
    returns = pct_returns(prices)

    out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for date in _rebalance_dates(prices, cfg["rebalance"]):
        if date not in signal.index:
            continue
        s = signal.loc[date]
        if s.isna().all():
            continue
        base = _base_weights(s, vol.loc[date], returns.loc[:date], cfg)
        out.loc[date] = _scale_to_vol(base, returns.loc[:date], cfg)
    return out


def _signal(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if cfg["signal"] == "trend":
        return rolling_momentum(prices, cfg["lookback"])
    if cfg["signal"] == "ml":
        return _ml_predictions(prices, cfg)
    raise ValueError(f"unknown signal: {cfg['signal']}")


def _ml_predictions(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    h = cfg["horizon"]
    feats = _feature_panel(prices)
    fwd = prices.pct_change(h).shift(-h)
    idx = prices.index
    pos = {d: i for i, d in enumerate(idx)}
    cols = list(prices.columns)

    rdates = _rebalance_dates(prices, cfg["rebalance"])
    preds = pd.DataFrame(np.nan, index=rdates, columns=cols)
    for date in rdates:
        cut = pos[date] - h            # last row whose forward target is realized by `date`
        if cut < 252:
            continue
        x_tr, y_tr = [], []
        for a in cols:
            f = feats[a].iloc[: cut + 1]
            y = fwd[a].iloc[: cut + 1]
            m = f.notna().all(axis=1) & y.notna()
            x_tr.append(f[m].to_numpy())
            y_tr.append(y[m].to_numpy())
        X, Y = np.vstack(x_tr), np.concatenate(y_tr)
        if len(Y) < 500:
            continue
        model = HistGradientBoostingRegressor(
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            max_iter=cfg["n_estimators"],
            random_state=0,
        )
        model.fit(X, Y)
        live = {a: feats[a].loc[date] for a in cols if feats[a].loc[date].notna().all()}
        if live:
            p = model.predict(np.vstack(list(live.values())))
            for a, pi in zip(live, p):
                preds.at[date, a] = pi
    return preds


def _feature_panel(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mom = {h: prices.pct_change(h) for h in (21, 63, 126, 252)}
    vol63 = rolling_vol(prices, 63)
    rev5 = prices.pct_change(5)
    return {
        a: pd.DataFrame({
            "mom21": mom[21][a], "mom63": mom[63][a], "mom126": mom[126][a],
            "mom252": mom[252][a], "vol63": vol63[a], "rev5": rev5[a],
        })
        for a in prices.columns
    }


def _base_weights(
    signal: pd.Series,
    vol: pd.Series,
    hist: pd.DataFrame,
    cfg: dict,
) -> pd.Series:
    selected = signal > 0
    if not selected.any():
        return pd.Series(0.0, index=signal.index)

    if cfg["mode"] == "tilt":
        raw = signal.where(selected).clip(lower=0.0)
        return (raw / raw.sum()).fillna(0.0)

    kind = cfg["backbone"]
    if kind == "equal":
        return equal_weight(selected.where(selected).to_frame().T).iloc[0]
    if kind == "inverse_vol":
        return inverse_vol_weight(vol.where(selected).to_frame().T).iloc[0]
    if kind == "min_variance":
        return _min_var(selected, hist, cfg)
    raise ValueError(f"unknown backbone: {kind}")


def _min_var(selected: pd.Series, hist: pd.DataFrame, cfg: dict) -> pd.Series:
    names = list(selected.index[selected])
    if len(names) < 3:
        return inverse_vol_weight(hist[names].std().to_frame().T).iloc[0].reindex(
            selected.index
        ).fillna(0.0)
    window = max(cfg["vol_window"], 60)
    cov = hist[names].tail(window).cov()
    max_weight = max(0.5, 1.0 / len(names) + 1e-9)
    w = min_variance_weights(cov, max_weight)
    return w.reindex(selected.index).fillna(0.0)


def _scale_to_vol(base: pd.Series, hist: pd.DataFrame, cfg: dict) -> pd.Series:
    port = (hist.tail(cfg["vol_window"]) * base).sum(axis=1)
    realized = float(port.std() * (252**0.5))
    if realized <= 0:
        return base
    scale = min(cfg["vol_target"] / realized, cfg["max_gross"])
    return base * scale


def _rebalance_dates(prices: pd.DataFrame, freq: str) -> pd.Index:
    step = 2 if freq == "2ME" else 1
    dates = prices.groupby(pd.Grouper(freq="ME")).tail(1).index
    return dates[::step]
