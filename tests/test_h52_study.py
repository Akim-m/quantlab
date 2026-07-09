import numpy as np
import pandas as pd

from quantlab.blend import composite, zscore_xs
from quantlab.features import high_ratio
from quantlab.h52_study import deployed_book, h52_variants
from quantlab.india_blend_study import raw_signals


def _prices(n=400, cols=("A", "B", "C", "D", "E"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    steps = rng.normal(0.0005, 0.02, size=(n, len(cols)))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=list(cols))


def _market(px):
    return px.mean(axis=1)


def test_h52_signals_are_causal_no_future_high_or_low_leaks():
    px = _prices()
    mkt = _market(px)
    v = h52_variants(px, raw_signals(px, mkt))
    # perturb only a FUTURE price; today's 52w signals must not move
    px2 = px.copy()
    px2.iloc[-1] = px2.iloc[-1] * 5.0
    v2 = h52_variants(px2, raw_signals(px2, mkt))
    t = -5
    for name in ("off_low", "gh_high"):
        assert np.allclose(v[name].iloc[t].to_numpy(), v2[name].iloc[t].to_numpy(), equal_nan=True)


def test_gh_high_is_george_hwang_proximity_to_the_high():
    px = _prices()
    gh = h52_variants(px, raw_signals(px, _market(px)))["gh_high"]
    assert np.allclose(gh.to_numpy(), high_ratio(px, 252).to_numpy(), equal_nan=True)
    # a name AT its 252d high has ratio 1.0 (the maximum); never above.
    assert float(gh.iloc[252:].max().max()) <= 1.0 + 1e-12


def test_deployed_book_is_long_only_and_sums_to_at_most_one():
    px = _prices()
    rising = pd.Series(np.linspace(100, 300, len(px)), index=px.index)  # above MA -> risk-on
    falling = pd.Series(np.linspace(40, 10, len(px)), index=px.index)   # last day below its quantile -> calm
    book = deployed_book(px.pct_change(60), rising, falling)
    gross = book.sum(axis=1)
    assert (book.to_numpy()[~np.isnan(book.to_numpy())] >= -1e-12).all()  # long-only
    assert (gross <= 1.0 + 1e-9).all()                                    # never levers
    # a fully warmed risk-on, calm day holds the decile book fully invested (sum == 1)
    assert abs(gross.iloc[-1] - 1.0) < 1e-9


def test_regime_overlay_de_risks_to_cash_when_vix_spikes():
    px = _prices()
    mkt = _market(px)
    hi = pd.Series(15.0, index=px.index)
    hi.iloc[300:] = 100.0                    # sustained spike -> above trailing quantile
    book = deployed_book(px.pct_change(60), mkt, hi)
    assert book.iloc[-1].abs().sum() == 0.0  # not calm -> fully in cash


def test_blend_is_the_5050_of_standardized_signals():
    px = _prices()
    a, b = px.pct_change(60), high_ratio(px, 252)
    blend = composite({"mom": a, "h52": b})
    expected = 0.5 * zscore_xs(a) + 0.5 * zscore_xs(b)
    row = -1
    assert np.allclose(blend.iloc[row].to_numpy(), expected.iloc[row].to_numpy(), equal_nan=True)
