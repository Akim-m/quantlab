"""Tests for the RL-2026-07-26-13 dividend-carry (DIV-CARRY) sleeve.

FORWARD-ONLY construction: no performance backtest is exercised. What matters is
that dividends are recovered from a synthetic ex-div adj/close gap, that the >5%
non-dividend exclusion fires, that the decile L/S book is dollar-neutral + unit-gross
and deterministic, and that the weights use only prior-day data. The live snapshot leg
mirrors run_ls: read-only, signed, forward_track-compatible. Fully synthetic - no network.
"""

import json

import numpy as np
import pandas as pd
import pytest

from quantlab import divcarry as dc
from quantlab import groww_client as gc
from quantlab import live_paper as lp


def _bidx(n, start="2018-01-01"):
    return pd.bdate_range(start=start, periods=n)


def _adj_from_schedule(close: pd.DataFrame, dfrac: np.ndarray) -> pd.DataFrame:
    """adj_close implied by a per-date ex-dividend fraction matrix: f[t] = prod_{s>t}
    (1 - dfrac[s]), so adj = close * f is split+dividend adjusted while close is not."""
    one_minus = 1.0 - dfrac
    rev = np.cumprod(one_minus[::-1], axis=0)[::-1]     # rev[t] = prod_{s>=t}
    f = rev / one_minus                                 # prod_{s>t}
    return close * f


def _div_panel(n=700, m=30, seed=0, ex_every=63, max_frac=0.04):
    """m names, each paying a dividend every `ex_every` days at a name-specific fraction
    spanning 0..max_frac -> a clean monotone cross-section of trailing yields."""
    rng = np.random.default_rng(seed)
    idx = _bidx(n)
    cols = [f"S{i:02d}" for i in range(m)]
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(rng.normal(0, 0.008, (n, m)), axis=0)),
                         index=idx, columns=cols)
    dfrac = np.zeros((n, m))
    ex = np.arange(40, n, ex_every)
    dfrac[np.ix_(ex, np.arange(m))] = np.linspace(0.0, max_frac, m)
    return close, _adj_from_schedule(close, dfrac)


# ---- extraction: a known ex-dividend is recovered exactly ----

def test_extracts_known_dividend():
    n, ex, div, price = 300, 150, 2.0, 100.0        # a single 2%-of-price cash dividend
    idx = _bidx(n)
    close = pd.DataFrame({"AAA": np.full(n, price)}, index=idx)
    f = np.where(np.arange(n) < ex, 1.0 - div / price, 1.0)
    adj = pd.DataFrame({"AAA": price * f}, index=idx)

    amt = dc.dividend_amounts(close, adj)["AAA"]
    assert amt.iloc[ex] == pytest.approx(div, abs=1e-9)          # recovered the cash amount
    assert amt.drop(idx[ex]).abs().max() == pytest.approx(0.0, abs=1e-9)   # 0 every other day
    assert dc.dividend_frac(close, adj)["AAA"].iloc[ex] == pytest.approx(div / price, abs=1e-9)


def test_trailing_yield_tracks_distributions():
    # Flat price, four 1% dividends in the trailing year -> yield ~ 4%.
    n, price = 400, 100.0
    idx = _bidx(n)
    close = pd.DataFrame({"AAA": np.full(n, price)}, index=idx)
    dfrac = np.zeros((n, 1))
    dfrac[[60, 120, 180, 240], 0] = 0.01
    adj = _adj_from_schedule(close, dfrac)
    y = dc.trailing_yield(close, adj)["AAA"]
    assert y.iloc[250] == pytest.approx(0.04, abs=1e-6)          # four 1% events in trailing 252d


# ---- the >5% non-dividend exclusion ----

def test_excludes_events_over_five_percent():
    n, price = 400, 100.0
    idx = _bidx(n)
    close = pd.DataFrame({"AAA": np.full(n, price)}, index=idx)
    dfrac = np.zeros((n, 1))
    dfrac[100, 0] = 0.03                              # a real 3% dividend -> kept
    dfrac[250, 0] = 0.08                              # an 8% step (demerger/special) -> excluded
    adj = _adj_from_schedule(close, dfrac)

    amt = dc.dividend_amounts(close, adj)["AAA"]
    assert amt.iloc[100] == pytest.approx(0.03 * price, abs=1e-9)    # dividend survives
    assert amt.iloc[250] == pytest.approx(0.0, abs=1e-12)            # >5% event zeroed out
    # and the excluded step never enters the trailing yield
    assert dc.trailing_yield(close, adj)["AAA"].iloc[260] == pytest.approx(0.03, abs=1e-6)


def test_rounding_noise_below_floor_is_ignored():
    # A sub-min_frac wiggle in the adj factor is treated as rounding, not a dividend.
    n, price = 200, 100.0
    idx = _bidx(n)
    close = pd.DataFrame({"AAA": np.full(n, price)}, index=idx)
    dfrac = np.zeros((n, 1))
    dfrac[100, 0] = dc.MIN_FRAC / 10.0               # far below the noise floor
    adj = _adj_from_schedule(close, dfrac)
    assert dc.dividend_amounts(close, adj)["AAA"].abs().max() == pytest.approx(0.0, abs=1e-12)


# ---- decile L/S: dollar-neutral, unit-gross, longs the highest yield ----

def test_decile_dollar_neutral_unit_gross():
    close, adj = _div_panel()
    w = dc.weights(close, adj, rebalance=None)                    # raw daily decile
    net = w.sum(axis=1).to_numpy()
    gross = w.abs().sum(axis=1).to_numpy()
    active = gross > 0
    assert active[-1]                                            # warmed up by the end
    np.testing.assert_allclose(net[active], 0.0, atol=1e-9)      # dollar-neutral
    np.testing.assert_allclose(gross[active], 1.0, atol=1e-9)    # unit gross
    assert (gross[~active] == 0.0).all()


def test_longs_top_yield_shorts_bottom():
    close, adj = _div_panel()
    y = dc.trailing_yield(close, adj).iloc[-2]                   # yield feeding the last weight (t-1)
    w = dc.weights(close, adj, rebalance=None).iloc[-1]
    longs, shorts = w[w > 0].index, w[w < 0].index
    assert y[longs].min() > y[shorts].max()                     # every long out-yields every short
    assert w[longs].nunique() == 1 and w[shorts].nunique() == 1  # equal-weight within each leg


def test_monthly_rebalanced_book_is_neutral_and_deterministic():
    close, adj = _div_panel()
    w = dc.weights(close, adj)                                   # default ME rebalance
    last = w.iloc[-1]
    assert last.abs().sum() == pytest.approx(1.0, abs=1e-9)      # gross 1
    assert last.sum() == pytest.approx(0.0, abs=1e-9)           # net 0
    pd.testing.assert_frame_equal(w, dc.weights(close, adj))     # determinism
    pd.testing.assert_series_equal(dc.latest_weights(close, adj), last)


# ---- no look-ahead: date-t weight reads only data through t-1 ----

def test_weights_use_prior_day_data_only():
    close, adj = _div_panel(seed=3)
    w = dc.weights(close, adj, rebalance=None)

    # Perturbing the LAST bar moves NO weight: weight[t] never reads (close,adj)[t].
    c2, a2 = close.copy(), adj.copy()
    c2.iloc[-1] *= 1.5
    a2.iloc[-1] *= 1.5
    pd.testing.assert_frame_equal(w, dc.weights(c2, a2, rebalance=None))

    # Perturbing an interior bar t leaves every weight on/before t unchanged (data <= t-1).
    t = 400
    c3, a3 = close.copy(), adj.copy()
    c3.iloc[t] *= 1.5
    a3.iloc[t] *= 1.5
    pd.testing.assert_frame_equal(w.iloc[:t + 1],
                                  dc.weights(c3, a3, rebalance=None).iloc[:t + 1])


# ---- live snapshot leg: read-only, signed, forward_track-compatible (mirrors run_ls) ----

class Spy:
    """Records every method routed through groww_client.call; serves canned LTP."""

    PRICES = {"NSE_A": 110.0, "NSE_B": 105.0, "NSE_C": 90.0, "NSE_D": 95.0}

    def __init__(self):
        self.methods = []

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:                          # the spy honors the real guard
            raise PermissionError(method)
        syms = kwargs.get("exchange_trading_symbols", ())
        return {s: self.PRICES[s] for s in syms if s in self.PRICES}


def _synthetic_divcarry():
    return lp.Book(
        weights=pd.Series({"A.NS": 0.25, "B.NS": 0.25, "C.NS": -0.25, "D.NS": -0.25}),
        regime_on=True, cash_frac=0.0, latest_date=pd.Timestamp("2026-07-08"),
        prev_close=pd.Series({"A.NS": 100.0, "B.NS": 100.0, "C.NS": 100.0, "D.NS": 100.0}),
        nsei_prev_close=float("nan"),
    )


def test_run_divcarry_read_only_schema_and_pnl(tmp_path, monkeypatch):
    spy = Spy()
    monkeypatch.setattr(gc, "call", spy)
    monkeypatch.setattr(lp, "current_divcarry_book", lambda **kw: _synthetic_divcarry())

    out = tmp_path / "paper_trades_divcarry.jsonl"
    row = lp.run_divcarry(path=str(out), write=True)

    # SAFETY: only read-only methods dispatched, none of them order methods.
    assert spy.methods and set(spy.methods) <= set(lp.READ_METHODS)
    assert not any(m in gc._ORDER_METHODS for m in spy.methods)

    assert row["kind"] == "live_paper_divcarry_snapshot"
    assert row["hypothesis_ref"] == "RL-2026-07-26-13"
    assert sum(row["weights"].values()) == pytest.approx(0.0, abs=1e-9)   # dollar-neutral
    assert sum(abs(v) for v in row["weights"].values()) == pytest.approx(1.0)  # unit gross
    assert row["n_long"] == 2 and row["n_short"] == 2
    # book_ret = .25*(.10) + .25*(.05) - .25*(-.10) - .25*(-.05) = 0.075
    assert row["book_intraday_ret"] == pytest.approx(0.075)
    assert row["n_quotes_ok"] == 4 and row["n_names"] == 4

    rec = json.loads(out.read_text().strip().splitlines()[0])
    assert "panel_date" in rec and "weights" in rec              # forward_track-compatible


def test_run_divcarry_records_book_only_when_quotes_unavailable(tmp_path, monkeypatch):
    def boom(method, *a, **k):
        raise RuntimeError("no entitlement")
    monkeypatch.setattr(gc, "call", boom)
    monkeypatch.setattr(lp, "current_divcarry_book", lambda **kw: _synthetic_divcarry())

    out = tmp_path / "d.jsonl"
    row = lp.run_divcarry(path=str(out), write=True)
    assert row["book_intraday_ret"] is None                     # no live P&L invented
    assert row["n_quotes_ok"] == 0 and row["groww_ok"] is False
    assert row["weights"]                                        # book still recorded
    assert out.read_text().strip()                              # snapshot still written


def test_current_divcarry_book_dollar_neutral(monkeypatch):
    close, adj = _div_panel()
    nsei = pd.DataFrame({lp.BENCH: np.full(len(close), 100.0)}, index=close.index)
    monkeypatch.setattr(lp.divcarry, "panels", lambda **kw: (close, adj))
    monkeypatch.setattr(lp, "load_yahoo_ohlcv", lambda syms, refresh=False: {})
    monkeypatch.setattr(lp, "close_prices", lambda data, field="adj_close": nsei)

    book = lp.current_divcarry_book()
    assert book.weights.sum() == pytest.approx(0.0, abs=1e-9)    # net 0
    assert book.weights.abs().sum() == pytest.approx(1.0, abs=1e-9)  # gross 1
    assert (book.weights > 0).any() and (book.weights < 0).any()
    assert set(book.prev_close.index) == set(book.weights.index)  # priced for live P&L
