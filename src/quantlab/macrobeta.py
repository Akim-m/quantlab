"""RL-2026-07-26-08: macro-sensitivity alignment cross-section (MACRO-BETA). FORWARD-ONLY.

Economic hypothesis (Adler-Dumas exposure; Hong-Stein slow diffusion): India is a net
oil importer with a managed-float, persistently depreciating currency, and single-stock
USDINR/crude exposures are economically heterogeneous - IT/pharma exporters gain from
INR weakness, oil-marketers/aviation/paints lose from crude spikes - yet that exposure is
under-reacted-to at monthly horizons. LONG the names whose macro beta ALIGNS with the
prevailing macro trend, SHORT the misaligned: a dollar-neutral STANDALONE book, not an
index overlay. The macro series (USDINR `INR=X`, Brent `BZ=F`) are genuinely NEW inputs to
this lab - every prior factor is a price/volume transform of the stocks themselves.

Two estimations compound here. First, per stock, ONE bivariate 252d rolling OLS of its
daily (adj_close) return on the LAGGED aligned macro returns (macro at t-1 vs stock at t -
Brent settles after the NSE close, the -02 causality clock), giving beta_INR and beta_oil
from closed-form 2-regressor algebra (no per-window loops). The macro-macro moments are
shared across the panel, so the whole 277-name x ~4000-day beta panel is a handful of
vectorized rolling means. Betas are winsorized +/-3 MAD cross-sectionally. Second, the
alignment score = beta_INR * sign(USDINR 63d trend) + beta_oil * sign(-Brent 63d trend),
both trend signs read prior-day. A decile L/S book LONGS the top-alignment decile and
SHORTS the bottom, equal-weight, dollar-neutral, held on a monthly (ME) grid.

The macro trades a different calendar than the NSE; its adj_close LEVELS are reindexed to
the panel's date index with a 5-day ffill limit and returns are taken from the aligned
levels, so a macro gap longer than a week leaves those names beta-less rather than
fabricating a stale co-movement. `INR=X`/`BZ=F` depth was confirmed at registration
(INR 2003-12+, Brent 2007-07+, the 24 >10% Brent days are documented oil history, not
vendor spikes); each series is passed through the RL-17 `clean_prices` transient-spike
guard at load.

DESIGN is frozen at registration - no post-registration backtest number is computed or
reported here (protocol). The deliverable is the construction and its live paper-track leg
(live_paper.run_macrobeta -> paper_trades_macrobeta.jsonl). A same-day disclosure arm
rank-correlates the alignment score against the VOL-SHOCK turnover-shock signal to show the
new book is not a re-packaging of an existing sleeve.

The daily decile weights are lagged one trading day, so a date-t target uses only data
through t-1 (prior-day signals, locked spec), then held on a monthly (ME) grid.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from . import volshock
from .data import close_prices, load_yahoo_ohlcv
from .india import india_panel
from .portfolio import rebalance_targets
from .xasset_trend import clean_prices

INR_SYM = "INR=X"       # USDINR spot
OIL_SYM = "BZ=F"        # Brent crude (registered primary; CL=F is the fallback)
OIL_FALLBACK = "CL=F"
WINDOW = 252            # rolling OLS beta window (~1yr)
LAG = 1                 # macro return enters at t-1 vs the stock return at t (-02 clock)
TREND = 63              # macro-trend lookback (~quarter) for the alignment sign
FFILL_LIMIT = 5         # cross-calendar macro carry-forward limit (registered)
DECILE = 0.10           # top/bottom decile long/short
MAD_N = 3.0             # cross-sectional winsorization half-width, in MADs
REBALANCE = "ME"


def align_level(level: pd.Series, index: pd.DatetimeIndex,
                limit: int = FFILL_LIMIT) -> pd.Series:
    """Reindex a macro adj_close LEVEL onto the panel calendar, carrying the last known
    value forward at most `limit` days. A macro gap longer than `limit` (a holiday the NSE
    trades through, a data outage) stays NaN, so returns/betas built on it drop those days
    rather than reusing a week-old price - the registered causal alignment."""
    return level.reindex(index).ffill(limit=limit)


def _clean(level: pd.Series) -> pd.Series:
    """RL-17 transient-spike guard on a single macro series (clean_prices wants a frame)."""
    return clean_prices(level.to_frame("x"))["x"]


def betas(stock_ret: pd.DataFrame, inr_ret: pd.Series, oil_ret: pd.Series,
          window: int = WINDOW, lag: int = LAG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(beta_INR, beta_oil) panels from ONE bivariate 252d rolling OLS per stock.

    Each stock's return at t is regressed on the LAGGED macro returns (macro at t-1), with
    an intercept, via the closed-form 2-regressor normal equations. Because the two
    regressors are the SAME for every stock, their rolling variances/covariance are shared
    scalars per date; only cov(macro, stock) is per-name. min_periods == window means a
    beta exists only on a FULL 252-obs window, which (a) is the registered
    insufficient-history exclusion and (b) guarantees every rolling mean below is taken over
    the identical fully-populated window, so E[ab]-E[a]E[b] is a consistent covariance.
    The population (1/N) normalisation cancels in every ratio, so no bias correction is
    needed."""
    x1, x2 = inr_ret.shift(lag), oil_ret.shift(lag)          # macro at t-1 vs stock at t

    def rmean(s):
        return s.rolling(window, min_periods=window).mean()

    mx1, mx2 = rmean(x1), rmean(x2)                          # shared macro moments (Series)
    vx1 = rmean(x1 * x1) - mx1 * mx1
    vx2 = rmean(x2 * x2) - mx2 * mx2
    cx12 = rmean(x1 * x2) - mx1 * mx2
    det = vx1 * vx2 - cx12 * cx12

    my = rmean(stock_ret)                                    # per-name moments (DataFrame)
    cx1y = rmean(stock_ret.mul(x1, axis=0)) - my.mul(mx1, axis=0)
    cx2y = rmean(stock_ret.mul(x2, axis=0)) - my.mul(mx2, axis=0)

    b_inr = (cx1y.mul(vx2, axis=0) - cx2y.mul(cx12, axis=0)).div(det, axis=0)
    b_oil = (cx2y.mul(vx1, axis=0) - cx1y.mul(cx12, axis=0)).div(det, axis=0)
    inf = [np.inf, -np.inf]
    return b_inr.replace(inf, np.nan), b_oil.replace(inf, np.nan)


def trend_sign(level: pd.Series, window: int = TREND) -> pd.Series:
    """sign of the trailing `window`-day change in the aligned level, read prior-day: the
    value at t reflects level(t-1) - level(t-1-window), so today's book never peeks at
    today's macro close. A flat window (sign 0) contributes nothing to the alignment."""
    return np.sign(level.diff(window).shift(1))


def winsorize(sig: pd.DataFrame, n: float = MAD_N) -> pd.DataFrame:
    """Cross-sectionally clip each row to median +/- n*MAD. Where dispersion collapses
    (MAD == 0) the row is passed through unclipped."""
    med = sig.median(axis=1)
    mad = sig.sub(med, axis=0).abs().median(axis=1)
    bad = ~(mad > 0)
    lo = (med - n * mad).mask(bad, -np.inf)
    hi = (med + n * mad).mask(bad, np.inf)
    return sig.clip(lower=lo, upper=hi, axis=0)


def alignment_score(px: pd.DataFrame, inr_level: pd.Series, oil_level: pd.Series,
                    window: int = WINDOW, lag: int = LAG, n_mad: float = MAD_N,
                    trend_win: int = TREND) -> pd.DataFrame:
    """Per-name alignment score panel: winsorized macro betas dotted with the prevailing
    macro trend. beta_INR loads on sign(USDINR trend); beta_oil on sign(-Brent trend), so
    an oil beta counts as ALIGNED when crude is falling. Both trend signs are prior-day."""
    ret = px.pct_change()
    b_inr, b_oil = betas(ret, inr_level.pct_change(), oil_level.pct_change(), window, lag)
    b_inr, b_oil = winsorize(b_inr, n_mad), winsorize(b_oil, n_mad)
    s_inr = trend_sign(inr_level, trend_win)
    s_oil = trend_sign(oil_level, trend_win)
    return b_inr.mul(s_inr, axis=0) + b_oil.mul(-s_oil, axis=0)


def decile_ls(sig: pd.DataFrame, q: float = DECILE) -> pd.DataFrame:
    """Long the top-`q` alignment decile, short the bottom-`q`: equal-weight within each
    leg, dollar-neutral (net 0), unit gross (|w| sums to 1) - xsec.py conventions."""
    need = int(round(1.0 / q))

    def row(r: pd.Series) -> pd.Series:
        v = r.dropna()
        w = pd.Series(0.0, index=r.index)
        if len(v) < need:                       # too few names to form both deciles
            return w
        k = max(1, int(np.floor(len(v) * q)))
        order = v.sort_values()                 # ascending: misaligned first, aligned last
        w[order.index[-k:]] = 0.5 / k           # top (most-aligned) decile long
        w[order.index[:k]] = -0.5 / k           # bottom (misaligned) decile short
        return w

    return sig.apply(row, axis=1)


def weights(px: pd.DataFrame, inr_level: pd.Series, oil_level: pd.Series,
            window: int = WINDOW, lag: int = LAG, q: float = DECILE, n_mad: float = MAD_N,
            trend_win: int = TREND, rebalance: str | None = REBALANCE) -> pd.DataFrame:
    """Daily signed target weights. The alignment score is lagged one day (date-t weights
    read data through t-1), then decile L/S, then held on a monthly grid (`rebalance`)."""
    sc = alignment_score(px, inr_level, oil_level, window, lag, n_mad, trend_win)
    w = decile_ls(sc.shift(1), q)
    if rebalance:
        w = rebalance_targets(w, rebalance).ffill().fillna(0.0)
    return w


def latest_weights(px: pd.DataFrame, inr_level: pd.Series, oil_level: pd.Series,
                   **kw) -> pd.Series:
    """Frozen target weights on the last panel date (for the live paper book)."""
    return weights(px, inr_level, oil_level, **kw).iloc[-1]


def panels(start: str = "2010-01-01", index: str = "nifty500",
           refresh: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(px, inr_level, oil_level): the N500 adj_close total-return panel over india_panel's
    frozen universe/calendar, plus the two macro adj_close LEVELS cleaned (RL-17 spike
    guard) and aligned onto the panel calendar. Brent falls back to CL=F only if BZ=F is
    empty (registered; not expected - the data-confirm passed on BZ=F)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        px, _, _, _ = india_panel(start=start, index=index, ret_clip=0.40, refresh=refresh)
        inr_raw = close_prices(load_yahoo_ohlcv([INR_SYM], refresh=refresh))[INR_SYM]
        oil = close_prices(load_yahoo_ohlcv([OIL_SYM], refresh=refresh))[OIL_SYM]
        if oil.dropna().empty:
            oil = close_prices(load_yahoo_ohlcv([OIL_FALLBACK], refresh=refresh))[OIL_FALLBACK]
        inr = align_level(_clean(inr_raw), px.index)
        oil = align_level(_clean(oil), px.index)
    return px, inr, oil


def disclosure_corr(score_latest: pd.Series, shock_latest: pd.Series) -> tuple[float, int]:
    """Same-day VALIDITY check: cross-sectional Spearman rank-corr of the current alignment
    score against the VOL-SHOCK turnover-shock over the names both cover. A LOW |rho| is the
    point - it shows this macro-exposure book ranks names differently from the existing
    attention/turnover sleeve, i.e. it is a distinct bet, not a re-labelled one. (rho, n)."""
    common = score_latest.dropna().index.intersection(shock_latest.dropna().index)
    if len(common) < 3:
        return float("nan"), len(common)
    a, b = score_latest.reindex(common), shock_latest.reindex(common)
    return float(a.corr(b, method="spearman")), len(common)


def _print_decile(w: pd.Series, n: int = 10) -> None:
    w = w[w.abs() > 1e-12].sort_values(ascending=False)
    longs, shorts = w[w > 0], w[w < 0]
    print(f"long (aligned) decile: {len(longs)} names   short (misaligned): {len(shorts)} names")
    print(f"gross {w.abs().sum():.4f}  net {w.sum():+.6f}")
    print("TOP longs:")
    for s, wt in longs.head(n).items():
        print(f"  + {s:16s} {wt*100:6.2f}%")
    print("TOP shorts:")
    for s, wt in shorts.tail(n).items():
        print(f"  - {s:16s} {wt*100:6.2f}%")


def run(refresh: bool = False) -> None:
    """FORWARD-ONLY construction print: macro trend states, beta coverage, the same-day
    VOL-SHOCK disclosure rank-corr, and the current top/bottom-decile book. No performance
    number (protocol)."""
    pd.set_option("display.width", 160)
    px, inr, oil = panels(refresh=refresh)
    print(f"MACRO-BETA  N500-{px.shape[1]}  {px.index[0].date()}->{px.index[-1].date()}  "
          f"FORWARD-ONLY")

    s_inr = float(trend_sign(inr).iloc[-1])
    s_oil = float(trend_sign(oil).iloc[-1])
    print(f"\n[macro trend  {px.index[-1].date()}]  USDINR 63d sign {s_inr:+.0f}  "
          f"Brent 63d sign {s_oil:+.0f}  (oil aligns on -sign = {-s_oil:+.0f})")

    sc = alignment_score(px, inr, oil)
    latest = sc.iloc[-1]
    print(f"[coverage] {int(latest.notna().sum())}/{px.shape[1]} names have a full-window "
          f"alignment score on the signal day")

    close, vol = volshock.panels(refresh=refresh)
    shock = volshock.shock(close, vol).iloc[-1]
    rho, k = disclosure_corr(latest, shock)
    print(f"[disclosure arm] Spearman(alignment, VOL-SHOCK turnover-shock) = {rho:+.3f}  "
          f"n={k}  (low |rho| => a distinct bet)")

    print(f"\n[current book  panel {px.index[-1].date()}]")
    _print_decile(latest_weights(px, inr, oil))


def main() -> None:
    p = argparse.ArgumentParser(
        description="RL-2026-07-26-08 MACRO-BETA construction + same-day disclosure print")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        run(refresh=a.refresh)


if __name__ == "__main__":
    main()
