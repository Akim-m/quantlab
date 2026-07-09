from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab import india
from quantlab.india_blend_study import _sector_demean, lo_erc, raw_signals

_NIFTY500_CSV = Path("data/raw/ind_nifty500list.csv")


def _prices(n=400, cols=("A", "B", "C", "D", "E", "F"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    steps = rng.normal(0.0005, 0.02, size=(n, len(cols)))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=list(cols))


@pytest.mark.skipif(
    not _NIFTY500_CSV.exists(),
    reason="data/raw/ind_nifty500list.csv absent; fetch via india.nse_index_symbols('nifty500')",
)
def test_sector_map_parses_symbols_and_industries():
    m = india.sector_map("nifty500")
    assert len(m) > 400
    assert all(k.endswith(".NS") for k in m)
    assert m.get("RELIANCE.NS")            # a known large-cap has a non-empty industry


def test_sector_demean_within_group_and_plain_for_unlabeled():
    idx = pd.bdate_range("2020-01-01", periods=2)
    sig = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [10.0, 0.0]}, index=idx)
    sectors = {"A": "X", "B": "X"}          # C has no label
    out = _sector_demean(sig, sectors)
    # A,B share group X -> demeaned within {A,B}: each is +/- half their spread
    assert np.allclose(out["A"].to_numpy(), [-1.0, -1.0])
    assert np.allclose(out["B"].to_numpy(), [1.0, 1.0])
    # C unlabeled -> plain (global) demean = value minus row mean of all three
    plain_c = sig["C"] - sig.mean(axis=1)
    assert np.allclose(out["C"].to_numpy(), plain_c.to_numpy())


def test_sector_demean_no_map_is_plain_demean():
    sig = _prices(n=5)
    out = _sector_demean(sig, None)
    assert np.allclose(out.to_numpy(), sig.sub(sig.mean(axis=1), axis=0).to_numpy())


def test_raw_signals_has_new_keys():
    px = _prices()
    mkt = px.mean(axis=1)
    sigs = raw_signals(px, mkt, sectors={"A": "X", "B": "X"})
    for k in ("mom_6_1", "off_low", "sector_mom", "mom_12_1", "sharpe_mom", "resid_mom"):
        assert k in sigs
        assert sigs[k].shape == px.shape


def test_lo_erc_monthly_weights_sum_to_one_or_cash_and_nonneg():
    px = _prices(n=400)
    score = px.pct_change(120)              # a slow cross-sectional score
    w = lo_erc(px, score, top=0.5, lookback=252)
    rebal = px.groupby(pd.Grouper(freq="ME")).tail(1).index
    filled = w.loc[rebal].dropna(how="all")
    assert len(filled) > 0
    for _, row in filled.iterrows():
        s = row.sum()
        assert (row >= -1e-12).all()
        assert abs(s - 1.0) < 1e-9 or s == 0.0   # invested (sums to 1) or all-cash
    # non-rebalance rows stay NaN (monthly book)
    non_rebal = w.index.difference(rebal)
    assert w.loc[non_rebal].isna().all().all()
