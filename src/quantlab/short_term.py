"""RL-2026-07-10: short-term (daily/weekly) Indian equity signals and books.

Everything in the RL-2026-07-10 family so far was monthly. This module adds the
SHORT-horizon signals whose central question is whether any gross edge survives
realistic costs (~5-20 bps/side) at weekly/daily turnover. High turnover = costs
dominate, so every book is meant to be read GROSS vs NET side by side.

Two output forms per signal:
  - long-short: cross-sectionally ranked, demeaned, dollar-neutral, unit gross
    (the natural short-term form; factor evidence, not deployable in the Indian
    cash market where single-stock shorts need F&O).
  - long-only top-quintile: the deployable tilt vs an equal-weight book.

Every signal is CAUSAL: it uses only trailing prices / OHLC, and the backtest
applies weights set at close(t) to returns from t+1 (built-in 1-day lag). No-look-
ahead is asserted by truncation-invariance tests in tests/test_short_term.py.
"""

import pandas as pd

from .blend import long_only_topq, long_only_topq_banded
from .features import residual_returns, rolling_vol
from .trend import overnight_intraday
from .xsec import _neutral

WEEKLY = "W-FRI"


def st_raw_signals(px: pd.DataFrame, mkt: pd.Series, beta_lb: int = 252) -> dict[str, pd.DataFrame]:
    """Short-horizon cross-sectional scores; higher = more attractive to hold long.

    rev5/rev2 are raw price reversals (short the recent winner). resid_rev reverses
    the market-residual return so the market bounce is stripped out. mom21 is a
    one-month momentum; wkmom is a 4-week momentum skipping the last week (avoids
    the 1-week reversal contaminating the signal).
    """
    resid = residual_returns(px, mkt, beta_lb)
    return {
        "rev5": -px.pct_change(5),
        "rev2": -px.pct_change(2),
        "resid_rev": -resid.rolling(5).sum(),
        "mom21": px.pct_change(21),
        "wkmom": px.shift(5) / px.shift(20) - 1.0,
    }


def vol_gate(px: pd.DataFrame, lb: int = 21, top_frac: float = 0.5) -> pd.DataFrame:
    """Boolean mask of names in the top `top_frac` by trailing `lb`-day vol (causal).

    Reversal is stronger and cheaper to harvest in liquid, high-vol names; gating
    to the high-vol half concentrates the book where the effect lives."""
    vol = rolling_vol(px, lb)
    return vol.rank(axis=1, ascending=False, pct=True).le(top_frac)


def gated_rev5(px: pd.DataFrame, lb: int = 21, rev_lb: int = 5, top_frac: float = 0.5) -> pd.DataFrame:
    """5-day reversal score, kept only for the top-half-vol names (others NaN)."""
    return (-px.pct_change(rev_lb)).where(vol_gate(px, lb, top_frac))


def long_short(signal: pd.DataFrame) -> pd.DataFrame:
    """Dollar-neutral, unit-gross book from a raw score (long high score)."""
    return _neutral(signal.rank(axis=1))


def build_books(
    px: pd.DataFrame, mkt: pd.Series, ohlcv: dict[str, pd.DataFrame]
) -> dict[str, tuple[pd.DataFrame, str | None]]:
    """The frozen short-term family as {name: (weights, rebalance_freq)}.

    LS-* are dollar-neutral unit-gross (factor evidence). LO-* are top-quintile
    invvol tilts (deployable). Frequency is weekly (W-FRI) except the two genuinely
    daily signals (rev2, overnight), which rebalance every day (freq=None).
    """
    sig = st_raw_signals(px, mkt)
    gated = gated_rev5(px)
    o, c = ohlcv["open"], ohlcv["close"]
    return {
        # ---- long-short (dollar-neutral) ----
        "LS-REV5": (long_short(sig["rev5"]), WEEKLY),
        "LS-REV2": (long_short(sig["rev2"]), None),
        "LS-RESID-REV": (long_short(sig["resid_rev"]), WEEKLY),
        "LS-VOLGATE": (long_short(gated), WEEKLY),
        "LS-MOM21": (long_short(sig["mom21"]), WEEKLY),
        "LS-WKMOM": (long_short(sig["wkmom"]), WEEKLY),
        "LS-OVNIGHT": (overnight_intraday(o, c, lb=5), None),
        # ---- long-only top-quintile tilts (deployable) ----
        "LO-REV5": (long_only_topq(sig["rev5"], px, top=0.2), WEEKLY),
        "LO-RESID-REV": (long_only_topq(sig["resid_rev"], px, top=0.2), WEEKLY),
        "LO-VOLGATE": (long_only_topq(gated, px, top=0.2), WEEKLY),
        "LO-MOM21": (long_only_topq(sig["mom21"], px, top=0.2), WEEKLY),
        "LO-WKMOM": (long_only_topq(sig["wkmom"], px, top=0.2), WEEKLY),
        "LO-REV5-BAND": (long_only_topq_banded(sig["rev5"], px, buy_top=0.15, hold_top=0.35), WEEKLY),
    }


LS_NAMES = ["LS-REV5", "LS-REV2", "LS-RESID-REV", "LS-VOLGATE", "LS-MOM21", "LS-WKMOM", "LS-OVNIGHT"]
LO_NAMES = ["LO-REV5", "LO-RESID-REV", "LO-VOLGATE", "LO-MOM21", "LO-WKMOM", "LO-REV5-BAND"]
