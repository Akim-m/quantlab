"""RL-2026-07-12: F&O-shortable universe parsing + resid-mom L/S sleeve construction."""

import numpy as np
import pandas as pd

from quantlab.blend import fno_long_short
from quantlab.india import fno_shortable


def _instruments() -> pd.DataFrame:
    """Canned instrument master exercising every exclusion path (no network)."""
    return pd.DataFrame([
        # NSE cash equities
        {"instrument_type": "EQ", "exchange": "NSE", "segment": "CASH", "trading_symbol": "RELIANCE", "underlying_symbol": None},
        {"instrument_type": "EQ", "exchange": "NSE", "segment": "CASH", "trading_symbol": "TCS", "underlying_symbol": None},
        {"instrument_type": "EQ", "exchange": "NSE", "segment": "CASH", "trading_symbol": "INFY", "underlying_symbol": None},
        # single-stock futures -> shortable
        {"instrument_type": "FUT", "exchange": "NSE", "segment": "FNO", "trading_symbol": "RELIANCE25JULFUT", "underlying_symbol": "RELIANCE"},
        {"instrument_type": "FUT", "exchange": "NSE", "segment": "FNO", "trading_symbol": "TCS25JULFUT", "underlying_symbol": "TCS"},
        # index futures -> excluded (NIFTY has an IDX row; MIDCPNIFTY has no EQ row)
        {"instrument_type": "FUT", "exchange": "NSE", "segment": "FNO", "trading_symbol": "NIFTY25JULFUT", "underlying_symbol": "NIFTY"},
        {"instrument_type": "FUT", "exchange": "NSE", "segment": "FNO", "trading_symbol": "MIDCPNIFTY25JULFUT", "underlying_symbol": "MIDCPNIFTY"},
        # NSE synthetic test underlying -> no EQ row -> excluded
        {"instrument_type": "FUT", "exchange": "NSE", "segment": "FNO", "trading_symbol": "011NSETEST36DECFUT", "underlying_symbol": "011NSETEST"},
        # commodity (MCX) and BSE index future -> excluded by segment / exchange
        {"instrument_type": "FUT", "exchange": "MCX", "segment": "COMMODITY", "trading_symbol": "GOLD25AUGFUT", "underlying_symbol": "GOLD"},
        {"instrument_type": "FUT", "exchange": "BSE", "segment": "FNO", "trading_symbol": "SENSEX25JULFUT", "underlying_symbol": "SENSEX"},
        # index rows that name the indices
        {"instrument_type": "IDX", "exchange": "NSE", "segment": "CASH", "trading_symbol": "NIFTY", "underlying_symbol": None},
        {"instrument_type": "IDX", "exchange": "NSE", "segment": "CASH", "trading_symbol": "BANKNIFTY", "underlying_symbol": None},
    ])


def test_fno_shortable_excludes_indices_and_test_maps_to_ns():
    out = fno_shortable(_instruments())
    # only real single-stock underlyings with a futures contract, mapped to SYMBOL.NS;
    # INFY has no future, NIFTY/MIDCPNIFTY/TEST/commodity/BSE all excluded.
    assert out == {"RELIANCE.NS", "TCS.NS"}


def _score() -> pd.DataFrame:
    # A,B always rank lowest (the shorts); C,D always highest (the longs).
    idx = pd.bdate_range("2024-01-01", periods=5)
    return pd.DataFrame({"A": [1, 2, 1, 2, 1], "B": [2, 1, 2, 1, 2],
                         "C": [9, 9, 9, 9, 9], "D": [8, 8, 8, 8, 8]}, index=idx)


def test_fno_long_short_short_leg_is_fno_only_and_neutral():
    w = fno_long_short(_score(), shortable={"A", "B"})
    shorted = set(w.columns[(w < -1e-12).any(axis=0)])
    assert shorted <= {"A", "B"}                    # never short a non-F&O name
    assert not (w.loc[:, ["C", "D"]] < 0).any().any()
    g = w.abs().sum(axis=1)
    np.testing.assert_allclose(g[g > 1e-9], 1.0, atol=1e-12)   # unit gross
    np.testing.assert_allclose(w.sum(axis=1), 0.0, atol=1e-12)  # dollar-neutral


def test_fno_long_short_is_causal():
    """Row t is a pure cross-section of score(t): truncating future rows cannot move it."""
    score = _score()
    full = fno_long_short(score, shortable={"A", "B"})
    for t in range(1, len(score)):
        trunc = fno_long_short(score.iloc[: t + 1], shortable={"A", "B"}).iloc[t]
        pd.testing.assert_series_equal(trunc, full.iloc[t], check_names=False)
