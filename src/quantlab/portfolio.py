import numpy as np
import pandas as pd


def equal_weight(signals: pd.DataFrame) -> pd.DataFrame:
    active = signals.notna() & (signals != 0)
    counts = active.sum(axis=1).replace(0, np.nan)
    return active.div(counts, axis=0).fillna(0.0)


def inverse_vol_weight(vol: pd.DataFrame) -> pd.DataFrame:
    inv = 1.0 / vol.replace(0.0, np.nan)
    sums = inv.sum(axis=1).replace(0.0, np.nan)
    return inv.div(sums, axis=0).fillna(0.0)


def rebalance_targets(weights: pd.DataFrame, freq: str | None) -> pd.DataFrame:
    if freq is None:
        return weights

    out = pd.DataFrame(np.nan, index=weights.index, columns=weights.columns)
    dates = weights.groupby(pd.Grouper(freq=freq)).tail(1).index
    out.loc[dates] = weights.loc[dates]
    return out
