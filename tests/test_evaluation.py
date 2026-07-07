import numpy as np
import pandas as pd

from quantlab.evaluation import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    one_sided_p,
    sharpe_tstat,
)


def test_sharpe_tstat_relationship():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    sr_ann, t = sharpe_tstat(r)
    # t = annualized Sharpe * sqrt(years); years = n/252
    assert np.isclose(t, sr_ann * np.sqrt(len(r) / 252), rtol=1e-9)
    assert sharpe_tstat(pd.Series([0.0, 0.0, 0.0]))[0] == 0.0


def test_one_sided_p_bounds():
    assert one_sided_p(0.0, 500) == 0.5  # t=0 -> p=0.5
    assert one_sided_p(5.0, 500) < 0.01  # strong signal -> tiny p
    assert one_sided_p(-5.0, 500) > 0.99


def test_benjamini_hochberg_known_vector():
    # BH at q=0.1 on sorted thresholds 0.02,0.04,...,0.10
    p = [0.001, 0.2, 0.03, 0.5, 0.9]
    rej = benjamini_hochberg(p, q=0.10)
    # ranked: 0.001(t=0.02) ok, 0.03(t=0.04) ok, 0.2(0.06) no, 0.5(0.08) no, 0.9 no
    # largest passing rank is 0.03 -> reject {0.001, 0.03}
    assert list(rej) == [True, False, True, False, False]
    assert not benjamini_hochberg([0.5, 0.6, 0.7], q=0.10).any()


def test_deflated_sharpe_monotonic_and_penalizes_trials():
    rng = np.random.default_rng(1)
    strong = pd.Series(rng.normal(0.001, 0.01, 2000))   # ~1.6 daily-ann Sharpe
    weak = pd.Series(rng.normal(0.0001, 0.01, 2000))
    trials_tight = [0.01, 0.02, 0.015, 0.018]           # low cross-trial spread
    trials_wide = list(np.linspace(-0.1, 0.1, 40))      # many, high spread

    dsr_strong = deflated_sharpe_ratio(strong, trials_tight)
    dsr_weak = deflated_sharpe_ratio(weak, trials_tight)
    assert dsr_strong > dsr_weak                        # higher SR -> higher DSR
    assert 0.0 <= dsr_strong <= 1.0

    # a wide, many-trial family raises the bar -> lower DSR for the same returns
    assert deflated_sharpe_ratio(strong, trials_wide) < deflated_sharpe_ratio(strong, trials_tight)
