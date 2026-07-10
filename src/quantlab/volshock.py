"""RL-2026-07-26-15: turnover-shock (visibility premium) cross-section (VOL-SHOCK).
FORWARD-ONLY.

Economic hypothesis (Gervais-Kaniel-Mingelgrin 2001): a burst of abnormal trading
volume raises a stock's visibility, and the attention premium predicts positive
subsequent returns. Volume enters here only as a TIME-SERIES trend gate in the rest of
the lab; this is the first cross-sectional LEVEL use of it, expressed as a shock.

Signal = rupee turnover = volume * unadjusted close. The shock is the log ratio of the
trailing 5-day mean turnover to the trailing 126-day mean turnover - a recent surge
relative to a name's own half-year baseline, so it is level-free (each name compared to
itself, not to a mega-cap). A non-traded session (zero or missing volume) contributes
NO turnover to the means, and a name that did not trade on the signal day is dropped
from that day's cross-section (the registered zero/missing-volume rule). The shock is
winsorized +/- 3 MAD cross-sectionally, then a decile long/short book LONGS the top
(highest-shock) decile and SHORTS the bottom.

DESIGN is frozen at registration - no post-registration backtest number is computed or
reported here (protocol). The pre-flight volume QC PASSED 2026-07-10 (Yahoo .NS daily
volume == Groww daily candles on 20/20 names), so the signal input is trusted. The
deliverable is the construction and its live paper-track leg (live_paper.run_volshock ->
paper_trades_volshock.jsonl).

The daily decile weights are lagged one trading day, so a date-t target uses only data
through t-1 (prior-day turnover, locked spec), then held on a monthly (ME) grid.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from .india import india_panel
from .portfolio import rebalance_targets

SHORT = 5               # trailing window for the recent-turnover surge
LONG = 126              # trailing window for the ~half-year turnover baseline
DECILE = 0.10           # top/bottom decile long/short
MAD_N = 3.0             # cross-sectional winsorization half-width, in MADs
REBALANCE = "ME"


def turnover(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Rupee turnover = unadjusted close * traded volume. A non-traded session (zero or
    missing volume) is NaN, so it never counts as a real low-turnover day in the rolling
    means and the name drops out of any cross-section that reads it."""
    return close * volume.where(volume > 0)


def shock(close: pd.DataFrame, volume: pd.DataFrame,
          short: int = SHORT, long: int = LONG) -> pd.DataFrame:
    """log( trailing `short`-day mean turnover / trailing `long`-day mean turnover ).

    Means skip non-traded days (NaN turnover), so the ratio is well defined and positive
    wherever any history exists. A name that did not trade on the signal day (zero or
    missing volume that day) is masked to NaN and thus excluded from that day's
    cross-section - the registered zero/missing-volume exclusion."""
    t = turnover(close, volume)
    short_mean = t.rolling(short, min_periods=1).mean()
    long_mean = t.rolling(long, min_periods=1).mean()
    sh = np.log(short_mean / long_mean)
    return sh.where(volume > 0)


def winsorize(sig: pd.DataFrame, n: float = MAD_N) -> pd.DataFrame:
    """Cross-sectionally clip each row to median +/- n*MAD. Where dispersion collapses
    (MAD == 0) the row is passed through unclipped."""
    med = sig.median(axis=1)
    mad = sig.sub(med, axis=0).abs().median(axis=1)
    bad = ~(mad > 0)
    lo = (med - n * mad).mask(bad, -np.inf)
    hi = (med + n * mad).mask(bad, np.inf)
    return sig.clip(lower=lo, upper=hi, axis=0)


def decile_ls(sig: pd.DataFrame, q: float = DECILE) -> pd.DataFrame:
    """Long the top-`q` shock decile, short the bottom-`q`: equal-weight within each leg,
    dollar-neutral (net 0), unit gross (|w| sums to 1) - xsec.py conventions."""
    need = int(round(1.0 / q))

    def row(r: pd.Series) -> pd.Series:
        v = r.dropna()
        w = pd.Series(0.0, index=r.index)
        if len(v) < need:                       # too few names to form both deciles
            return w
        k = max(1, int(np.floor(len(v) * q)))
        order = v.sort_values()                 # ascending: low shock first, high last
        w[order.index[-k:]] = 0.5 / k           # top (highest-shock) decile long
        w[order.index[:k]] = -0.5 / k           # bottom decile short
        return w

    return sig.apply(row, axis=1)


def weights(close: pd.DataFrame, volume: pd.DataFrame, short: int = SHORT, long: int = LONG,
            q: float = DECILE, n_mad: float = MAD_N,
            rebalance: str | None = REBALANCE) -> pd.DataFrame:
    """Daily signed target weights. The shock is lagged one day (date-t weights read data
    through t-1), then decile L/S, then held on a monthly grid (`rebalance`)."""
    sh = winsorize(shock(close, volume, short, long), n_mad)
    w = decile_ls(sh.shift(1), q)
    if rebalance:
        w = rebalance_targets(w, rebalance).ffill().fillna(0.0)
    return w


def latest_weights(close: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.Series:
    """Frozen target weights on the last panel date (for the live paper book)."""
    return weights(close, volume, **kw).iloc[-1]


def panels(start: str = "2010-01-01", index: str = "nifty500",
           refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(close, volume) panels over india_panel's frozen N500 universe/calendar. `close`
    is the raw split-adjusted (dividend-UNadjusted) price the shock multiplies by volume
    to form rupee turnover; the intraday live baseline is this same raw close."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        _, _, ohlcv, _ = india_panel(start=start, index=index,
                                     ret_clip=0.40, refresh=refresh)
    return ohlcv["close"], ohlcv["volume"]


def _print_decile(w: pd.Series, n: int = 10) -> None:
    w = w[w.abs() > 1e-12].sort_values(ascending=False)
    longs, shorts = w[w > 0], w[w < 0]
    print(f"long (high-shock) decile: {len(longs)} names   short (low-shock): {len(shorts)} names")
    print(f"gross {w.abs().sum():.4f}  net {w.sum():+.6f}")
    print("TOP longs:")
    for s, wt in longs.head(n).items():
        print(f"  + {s:16s} {wt*100:6.2f}%")
    print("TOP shorts:")
    for s, wt in shorts.tail(n).items():
        print(f"  - {s:16s} {wt*100:6.2f}%")


def run(refresh: bool = False) -> None:
    """FORWARD-ONLY construction print: signal-day volume coverage and the current
    top/bottom-decile book. No performance number (protocol)."""
    pd.set_option("display.width", 160)
    close, volume = panels(refresh=refresh)
    print(f"VOL-SHOCK  N500-{close.shape[1]}  {close.index[0].date()}->{close.index[-1].date()}  "
          f"FORWARD-ONLY")

    signal_day = close.index[-1]                        # data feeding the next (t) weight
    traded = (volume.loc[signal_day] > 0)
    excluded = int((~traded).sum())
    print(f"\n[coverage  signal day {signal_day.date()}]  traded {int(traded.sum())}/"
          f"{close.shape[1]}  excluded (zero/missing volume) {excluded}")

    print(f"\n[current book  panel {close.index[-1].date()}]")
    _print_decile(latest_weights(close, volume))


def main() -> None:
    p = argparse.ArgumentParser(
        description="RL-2026-07-26-15 VOL-SHOCK construction print (forward-only)")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        run(refresh=a.refresh)


if __name__ == "__main__":
    main()
