import numpy as np
import pandas as pd

from quantlab.blend import (
    composite,
    long_only_topq,
    long_short,
    market_on,
    regime_switch,
    trend_overlay,
    vol_target_overlay,
    zscore_xs,
)


def _prices(n=400, cols=("A", "B", "C", "D"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    steps = rng.normal(0.0005, 0.02, size=(n, len(cols)))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=list(cols))


def test_zscore_is_cross_sectional_unit():
    px = _prices()
    z = zscore_xs(px)
    row = z.iloc[-1]
    assert abs(row.mean()) < 1e-9
    assert abs(row.std(ddof=1) - 1.0) < 1e-9


def test_composite_equal_weight_averages_zscores():
    px = _prices()
    a, b = zscore_xs(px), zscore_xs(-px)
    comp = composite({"a": px, "b": -px})
    assert np.allclose(comp.iloc[-1].to_numpy(), ((a + b) / 2).iloc[-1].to_numpy())


def test_long_only_topq_sums_to_one_and_nonneg():
    px = _prices()
    score = px.pct_change(60)
    w = long_only_topq(score, px, top=0.5, weighting="ew").iloc[-1]
    assert (w >= -1e-12).all()
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w > 0).sum() == 2  # top half of 4 names


def test_long_short_dollar_neutral_unit_gross():
    px = _prices()
    w = long_short(px.pct_change(60)).iloc[-1]
    assert abs(w.sum()) < 1e-9
    assert abs(w.abs().sum() - 1.0) < 1e-9


def test_trend_overlay_goes_to_cash_below_ma_and_is_causal():
    px = _prices()
    book = long_only_topq(px.pct_change(60), px, top=0.5)
    # a market that is strictly falling ends below its own MA -> book to cash
    falling = pd.Series(np.linspace(200, 100, len(px)), index=px.index)
    out = trend_overlay(book, falling, ma_lb=50)
    assert out.iloc[60:].abs().sum().sum() == 0.0  # fully de-risked once MA exists

    # causality: a future market spike must not change today's overlay
    m = pd.Series(100.0, index=px.index)
    m2 = m.copy()
    m2.iloc[-1] = 1e6
    t = -5
    assert trend_overlay(book, m, 50).iloc[t].equals(trend_overlay(book, m2, 50).iloc[t])


def test_regime_switch_selects_book_by_causal_market_state():
    px = _prices()
    on_book = long_only_topq(px.pct_change(60), px, top=0.5)      # sums to 1
    off_book = pd.DataFrame(0.0, index=px.index, columns=px.columns)  # cash
    rising = pd.Series(np.linspace(100, 300, len(px)), index=px.index)
    falling = pd.Series(np.linspace(300, 100, len(px)), index=px.index)
    # rising market -> risk-on book held (weights sum ~1 once MA exists)
    up = regime_switch(on_book, off_book, rising, ma_lb=50)
    assert up.iloc[200].sum() > 0.99
    # falling market -> risk-off (cash) held
    down = regime_switch(on_book, off_book, falling, ma_lb=50)
    assert down.iloc[200].abs().sum() == 0.0
    # causal: today's allocation ignores a future market spike
    m = pd.Series(100.0, index=px.index); m2 = m.copy(); m2.iloc[-1] = 1e6
    assert regime_switch(on_book, off_book, m, 50).iloc[-5].equals(
        regime_switch(on_book, off_book, m2, 50).iloc[-5])


def test_vol_target_overlay_is_causal_and_cuts_high_vol():
    px = _prices()
    book = long_only_topq(px.pct_change(60), px, top=0.5)
    rets = px.pct_change().fillna(0.0)
    out = vol_target_overlay(book, rets, target=0.10, lb=21, cap=1.0)
    # never levers above the cap
    assert out.abs().sum(axis=1).max() <= 1.0 + 1e-9
    # a future return cannot change today's scaled weights (uses book.shift(1))
    r2 = rets.copy()
    r2.iloc[-1] = 5.0
    assert np.allclose(vol_target_overlay(book, rets, cap=1.0).iloc[-5].to_numpy(),
                       vol_target_overlay(book, r2, cap=1.0).iloc[-5].to_numpy())
