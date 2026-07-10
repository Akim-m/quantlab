"""RL-2026-07-26-13: dividend-carry cross-section (DIV-CARRY). FORWARD-ONLY.

The lab's first NON-price cross-sectional signal - every prior factor is a
price/volume transform; this is a cash-flow quantity. Cash distributions are mined
for free from Yahoo's close-vs-adj_close gap: adj_close is split+dividend adjusted,
close is split-adjusted only, so f = adj_close/close is a PURE dividend adjustment
factor (splits cancel). f steps up across each ex-date by 1/(1 - d/prev_close), so
1 - f[t-1]/f[t] recovers the distribution as a fraction of the prior close. A single
step > 5% of price is a non-dividend corporate action (demerger/special - receipts
RELIANCE +8.7% Jio, ITC +3.7% Hotels) and is EXCLUDED; sub-noise steps are dropped.
Trailing-252d dividend sum / price = yield; decile L/S on the winsorized yield.

DESIGN is frozen at registration - no post-registration backtest number is computed
or reported here (protocol). The deliverable is the construction and its live
paper-track leg (live_paper.run_divcarry -> paper_trades_divcarry.jsonl). A same-day
VALIDITY check (disclosure_corr) rank-correlates realized yield against the F&O
basis-implied dividend (-b1) - an internal cross-validation, not a performance claim.

The daily decile weights are lagged one trading day, so a date-t target uses only
data through t-1 (prior-day signals, locked spec), then held on a monthly (ME) grid.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .data import close_prices, load_yahoo_ohlcv
from .india import india_panel
from .portfolio import rebalance_targets

LOOKBACK = 252          # trailing dividend window (~1yr of ex-dates)
DECILE = 0.10           # top/bottom decile long/short
MAX_FRAC = 0.05         # single step > 5% of price = non-dividend action, excluded
MIN_FRAC = 5e-4         # below this an adj-factor wiggle is rounding noise (p99 ~5e-7)
MAD_N = 3.0             # cross-sectional winsorization half-width, in MADs
REBALANCE = "ME"
FNO_PATH = "experiments/fno_daily.jsonl"


def dividend_frac(close: pd.DataFrame, adj_close: pd.DataFrame,
                  min_frac: float = MIN_FRAC, max_frac: float = MAX_FRAC) -> pd.DataFrame:
    """Per-name distribution as a fraction of the prior close on each ex-date (0 else).

    f = adj/close is the dividend adjustment factor; 1 - f[t-1]/f[t] is the implied
    div/prev_close. Steps <= min_frac are rounding noise -> 0; steps > max_frac are
    corporate actions -> 0 (excluded)."""
    f = adj_close / close
    step = 1.0 - f.shift(1) / f
    frac = step.where(step > min_frac, 0.0)
    return frac.where(frac <= max_frac, 0.0)


def dividend_amounts(close: pd.DataFrame, adj_close: pd.DataFrame, **kw) -> pd.DataFrame:
    """Cash distribution per ex-date in split-adjusted price units (0 elsewhere)."""
    return dividend_frac(close, adj_close, **kw) * close.shift(1)


def trailing_yield(close: pd.DataFrame, adj_close: pd.DataFrame,
                   lookback: int = LOOKBACK, **kw) -> pd.DataFrame:
    """Trailing-`lookback` cash-distribution sum / current price. Both numerator and
    denominator are in split-adjusted units, so splits cancel exactly."""
    div = dividend_amounts(close, adj_close, **kw)
    return div.rolling(lookback, min_periods=1).sum() / close


def winsorize(yields: pd.DataFrame, n: float = MAD_N) -> pd.DataFrame:
    """Cross-sectionally clip each row to median +/- n*MAD. Where dispersion collapses
    (MAD == 0, e.g. before any dividends accrue) the row is passed through unclipped."""
    med = yields.median(axis=1)
    mad = yields.sub(med, axis=0).abs().median(axis=1)
    bad = ~(mad > 0)
    lo = (med - n * mad).mask(bad, -np.inf)
    hi = (med + n * mad).mask(bad, np.inf)
    return yields.clip(lower=lo, upper=hi, axis=0)


def decile_ls(yields: pd.DataFrame, q: float = DECILE) -> pd.DataFrame:
    """Long the top-`q` yield decile, short the bottom-`q`: equal-weight within each
    leg, dollar-neutral (net 0), unit gross (|w| sums to 1) - xsec.py conventions."""
    need = int(round(1.0 / q))

    def row(r: pd.Series) -> pd.Series:
        v = r.dropna()
        w = pd.Series(0.0, index=r.index)
        if len(v) < need:                       # too few names to form both deciles
            return w
        k = max(1, int(np.floor(len(v) * q)))
        order = v.sort_values()                 # ascending: low yield first, high last
        w[order.index[-k:]] = 0.5 / k           # top decile long
        w[order.index[:k]] = -0.5 / k           # bottom decile short
        return w

    return yields.apply(row, axis=1)


def weights(close: pd.DataFrame, adj_close: pd.DataFrame, lookback: int = LOOKBACK,
            q: float = DECILE, n_mad: float = MAD_N, rebalance: str | None = REBALANCE,
            min_frac: float = MIN_FRAC, max_frac: float = MAX_FRAC) -> pd.DataFrame:
    """Daily signed target weights. The yield is lagged one day (date-t weights read
    data through t-1), then decile L/S, then held on a monthly grid (`rebalance`)."""
    y = winsorize(trailing_yield(close, adj_close, lookback,
                                 min_frac=min_frac, max_frac=max_frac), n_mad)
    w = decile_ls(y.shift(1), q)
    if rebalance:
        w = rebalance_targets(w, rebalance).ffill().fillna(0.0)
    return w


def latest_weights(close: pd.DataFrame, adj_close: pd.DataFrame, **kw) -> pd.Series:
    """Frozen target weights on the last panel date (for the live paper book)."""
    return weights(close, adj_close, **kw).iloc[-1]


def panels(start: str = "2010-01-01", index: str = "nifty500",
           refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(close, adj_close) panels over india_panel's frozen N500 universe/calendar -
    the two series whose gap encodes the dividend history. `close` is split-adjusted
    but dividend-UNadjusted (the raw price); `adj_close` is split+dividend adjusted."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        px, _, ohlcv, kept = india_panel(start=start, index=index,
                                          ret_clip=0.40, refresh=refresh)
        data = load_yahoo_ohlcv(kept)           # warm cache; india_panel already pulled
        adj = close_prices(data, "adj_close")[kept].reindex(px.index).ffill(limit=3)
    return ohlcv["close"], adj


def dividend_log(close: pd.DataFrame, adj_close: pd.DataFrame, symbol: str,
                 **kw) -> pd.DataFrame:
    """Extracted (ex_date -> cash_div, pct_of_prev_close) events for one name - the
    extraction-validation view."""
    amt = dividend_amounts(close[[symbol]], adj_close[[symbol]], **kw)[symbol]
    frac = dividend_frac(close[[symbol]], adj_close[[symbol]], **kw)[symbol]
    ev = amt[amt > 0]
    return pd.DataFrame({"cash_div": ev.round(2), "pct_of_prev_close": (frac[ev.index] * 100).round(3)})


def _latest_fno_basis(path: str = FNO_PATH) -> dict[str, float | None]:
    """b1 (annualized near-future basis) per F&O underlying from the last fno_daily row."""
    last = None
    for line in Path(path).read_text().splitlines():
        if line.strip():
            last = line
    if last is None:
        return {}
    rec = json.loads(last)
    return {u: e.get("b1") for u, e in rec.get("basis", {}).items()}


def disclosure_corr(close: pd.DataFrame, adj_close: pd.DataFrame,
                    fno_path: str = FNO_PATH, **kw) -> tuple[float, int]:
    """Same-day VALIDITY check: Spearman rank-corr of trailing realized yield vs the
    basis-implied dividend (-b1) across F&O intersect N500 names. Deep backwardation
    (b1 << 0) implies a large expected dividend before expiry, so a POSITIVE rank-corr
    is evidence the adj-gap extraction captures real dividends. Returns (rho, n)."""
    y = trailing_yield(close, adj_close, **kw).iloc[-1]
    basis = _latest_fno_basis(fno_path)
    pairs = [(float(y[f"{u}.NS"]), -float(b1)) for u, b1 in basis.items()
             if b1 is not None and f"{u}.NS" in y.index and np.isfinite(y[f"{u}.NS"])]
    if len(pairs) < 3:
        return float("nan"), len(pairs)
    yv = pd.Series([p[0] for p in pairs])
    mb = pd.Series([p[1] for p in pairs])
    return float(yv.corr(mb, method="spearman")), len(pairs)


def _print_decile(w: pd.Series, n: int = 10) -> None:
    w = w[w.abs() > 1e-12].sort_values(ascending=False)
    longs, shorts = w[w > 0], w[w < 0]
    print(f"long (top-yield) decile: {len(longs)} names   short (bottom-yield): {len(shorts)} names")
    print(f"gross {w.abs().sum():.4f}  net {w.sum():+.6f}")
    print("TOP longs:")
    for s, wt in longs.head(n).items():
        print(f"  + {s:16s} {wt*100:6.2f}%")
    print("TOP shorts:")
    for s, wt in shorts.tail(n).items():
        print(f"  - {s:16s} {wt*100:6.2f}%")


def run(refresh: bool = False, symbol: str = "COALINDIA.NS") -> None:
    """FORWARD-ONLY construction print: extraction validation, disclosure-arm rank-corr,
    and the current top/bottom-decile book. No performance number (protocol)."""
    pd.set_option("display.width", 160)
    close, adj = panels(refresh=refresh)
    print(f"DIV-CARRY  N500-{close.shape[1]}  {close.index[0].date()}->{close.index[-1].date()}  "
          f"FORWARD-ONLY")

    for sym in dict.fromkeys([symbol, "ITC.NS", "ONGC.NS"]):
        if sym not in close.columns:
            continue
        ev = dividend_log(close, adj, sym)
        yrs = (close.index[-1] - close.index[0]).days / 365.25
        print(f"\n[extraction check] {sym}: {len(ev)} events  (~{len(ev)/yrs:.2f}/yr, "
              f"median {ev['pct_of_prev_close'].median():.2f}% of price)")
        print(ev.tail(8).to_string())

    rho, n = disclosure_corr(close, adj)
    print(f"\n[disclosure arm] Spearman(realized trailing yield, -b1) = {rho:+.3f}  n={n}  "
          f"(positive => extraction captures real dividends)")

    print(f"\n[current book  panel {close.index[-1].date()}]")
    _print_decile(latest_weights(close, adj))


def main() -> None:
    p = argparse.ArgumentParser(
        description="RL-2026-07-26-13 DIV-CARRY construction + same-day validity print")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--symbol", default="COALINDIA.NS", help="dividend-extraction validation name")
    a = p.parse_args()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        run(refresh=a.refresh, symbol=a.symbol)


if __name__ == "__main__":
    main()
