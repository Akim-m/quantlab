"""RL-2026-07-26-09: same-sector F&O cointegration pairs (relative value). FORWARD-ONLY.

Economically twinned large-caps that share cash-flow drivers (HDFCBANK/ICICIBANK,
TCS/INFY, ...) drift apart on idiosyncratic order flow and revert
(Gatev-Goetzmann-Rouwenhorst 2006). We hedge the sector/market beta out and trade the
residual log-price spread when it dislocates.

FORWARD-ONLY BY CONSTRUCTION. The FORMATION step below uses ALL history through the
registration date (2026-07-10) to SELECT the ten pairs - it is contaminated by
construction, so NO historical P&L or Sharpe is computed from it. The pair set is
FROZEN as `FROZEN_PAIRS` module constants at registration; only forward paper-trading
from that date is evidence (live_paper.run_pairs -> paper_trades_pairs.jsonl).

Universe = the RL-12 F&O-shortable intersect Nifty-500 overlap (130 names: names in
india_panel(start=2010) that have an NSE single-stock future). Candidate pairs = every
within-NSE-industry pair in that set. Each candidate is tested over the FORMATION window
(2018-01-01 -> registration) on log adj_close:

  * Engle-Granger cointegration: OLS-regress log P_a on log P_b + const, then an ADF
    unit-root test on the residual spread. ADF is a DELIBERATELY MINIMAL fixed-lag-1
    implementation (regress d spread_t on spread_{t-1}, d spread_{t-1}, const; the
    t-stat on the level coefficient is the ADF stat) rather than pulling statsmodels -
    it is the standard augmented Dickey-Fuller with a single augmentation lag. The
    pass threshold is the MacKinnon ~5% Engle-Granger critical value for one regressor
    (-3.34); with the minimal augmentation the nominal size is approximate (disclosed).
  * half-life of mean reversion from an AR(1) fit on the spread: b = OLS slope of
    spread_t on spread_{t-1}; half-life = -ln 2 / ln b. Require < 60 trading days.
  * spread-vol floor: daily sigma of the spread (std of its first difference) must be
    >= 0.5% or a round trip cannot clear costs.

Passing pairs are ranked by STABILITY = |ADF t| / half-life (larger = stronger and
faster reversion) and the top 10 are frozen. The spread the live book trades is
spread_t = log P_a - beta * log P_b (the OLS intercept folds into the rolling mean and
cancels in the z-score). The frozen (mu, sigma) are the FORMATION-window mean/std of
that spread, kept FOR REFERENCE ONLY: the live z uses a 63-day ROLLING mean/std
(prior-day info), per the registration, NOT the frozen constants.

Cost math (India 20 bps/leg): a pair holds two equal-dollar legs; a round trip trades
each leg in and out = 4 leg-crossings ~= 80 bps of the pair's gross notional, against an
expected ~1.5-3% spread convergence. The modal predicted outcome is a cost-eaten wash.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from .india import fno_shortable, india_panel, sector_map

UNIVERSE_START = "2010-01-01"   # RL-12 universe-resolution window (the 130-name overlap)
FORMATION_START = "2018-01-01"  # locked cointegration formation window start
INDEX = "nifty500"

ADF_CRIT = -3.34        # MacKinnon ~5% Engle-Granger critical value, 1 regressor (approx)
HALF_LIFE_MAX = 60.0    # trading days: reject spreads that revert too slowly
VOL_FLOOR = 0.005       # daily spread sigma floor (0.5%): below it a round trip can't clear cost
TOP_N = 10              # freeze the top-N passing pairs

Z_WINDOW = 63           # rolling window for the live spread z (prior-day info)
ENTRY_Z = 2.0           # enter when |z| >= 2 (long cheap leg, short rich leg)
EXIT_Z = 0.0            # exit on the z = 0 crossing
TIME_STOP = 30          # ... or after 30 trading days, whichever comes first
MAX_PAIRS = 10          # max concurrent pairs; equal capital => per-pair gross 1/MAX_PAIRS
PER_LEG = 1.0 / MAX_PAIRS / 2.0   # each leg's gross weight (0.05): two legs => 0.1 per pair

# Frozen at registration 2026-07-10 over the 2018-01-02 -> 2026-07-09 formation window,
# top 10 of 59 passing pairs (701 same-industry candidates) by stability = |ADF t|/half-life.
# Each entry: (a, b, hedge_beta, spread_mu, spread_sigma). spread = log P_a - beta*log P_b;
# (mu, sigma) are the formation mean/std of that spread FOR REFERENCE ONLY - the live book
# z-scores on a 63-day ROLLING window. Regenerate with `python -m quantlab.pairs_rv`.
FROZEN_PAIRS: tuple[tuple[str, str, float, float, float], ...] = (
    ("BAJFINANCE.NS", "KOTAKBANK.NS", 2.5546, -8.59518, 0.1604),
    ("NHPC.NS", "POWERGRID.NS", 1.1466, -2.1034, 0.10436),
    ("ABB.NS", "KEI.NS", 0.7305, 2.64137, 0.12067),
    ("EICHERMOT.NS", "MARUTI.NS", 1.4163, -4.80456, 0.10798),
    ("M&M.NS", "MARUTI.NS", 2.1566, -12.50773, 0.18225),
    ("BPCL.NS", "PETRONET.NS", 1.5701, -3.23511, 0.14447),
    ("DABUR.NS", "HINDUNILVR.NS", 0.7245, 0.6462, 0.06288),
    ("IOC.NS", "ONGC.NS", 0.9213, -0.15394, 0.09884),
    ("JSWSTEEL.NS", "TATASTEEL.NS", 0.9571, 2.06893, 0.0958),
    ("HDFCBANK.NS", "KOTAKBANK.NS", 1.1639, -0.26128, 0.08789),
)


# ---- least squares + the cointegration / mean-reversion primitives (numpy only) ----

def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS of y on X (X carries its own intercept column). Returns (coef, tstats, resid)
    with classical (homoskedastic) t-stats coef / se."""
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, coef / se, 0.0)
    return coef, t, resid


def adf_tstat(x: np.ndarray, lag: int = 1) -> float:
    """Augmented Dickey-Fuller t-stat with a fixed augmentation `lag` (default 1).

    Regress d x_t = c + rho*x_{t-1} + sum_i gamma_i d x_{t-i} + u; the ADF statistic is
    the t-stat on rho. A strongly negative value rejects the unit root (mean-reverting /
    cointegrated residual). Minimal by design - no p-value interpolation, no lag search."""
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    m = len(dx)
    if m <= lag + 2:
        return 0.0
    y = dx[lag:]                                  # d x_t for t = lag .. m-1
    lvl = x[lag:m]                                # x_{t-1}
    cols = [np.ones_like(lvl), lvl]
    for i in range(1, lag + 1):
        cols.append(dx[lag - i:m - i])            # d x_{t-i}
    X = np.column_stack(cols)
    _, t, _ = _ols(y, X)
    return float(t[1])                            # t-stat on rho (the level coefficient)


def half_life(x: np.ndarray) -> float:
    """Mean-reversion half-life (trading days) from an AR(1) fit x_t = a + b*x_{t-1}:
    half-life = -ln 2 / ln b. Returns +inf when the fit is not mean-reverting (b outside
    (0, 1)) - such a spread reverts too slowly / not at all and fails the < 60d filter."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return float("inf")
    coef, _, _ = _ols(x[1:], np.column_stack([np.ones(len(x) - 1), x[:-1]]))
    b = coef[1]
    if not (0.0 < b < 1.0):
        return float("inf")
    return float(-np.log(2.0) / np.log(b))


def spread_vol(x: np.ndarray) -> float:
    """Daily volatility of the spread = std of its first difference."""
    dx = np.diff(np.asarray(x, dtype=float))
    return float(np.std(dx, ddof=1)) if len(dx) > 1 else 0.0


def engle_granger(la: np.ndarray, lb: np.ndarray) -> tuple[float, float]:
    """OLS log P_a = alpha + beta*log P_b. Returns (beta, alpha) - the hedge ratio and
    intercept of the cointegrating regression."""
    coef, _, _ = _ols(np.asarray(la, float),
                      np.column_stack([np.ones(len(lb)), np.asarray(lb, float)]))
    return float(coef[1]), float(coef[0])


def pair_spread(log_px: pd.DataFrame, a: str, b: str, beta: float) -> pd.Series:
    """The traded spread log P_a - beta*log P_b (the intercept is left in the level; it
    cancels in the rolling z-score)."""
    return log_px[a] - beta * log_px[b]


def zscore(spread: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    """(spread - rolling mean) / rolling std over `window`. Prior-day info: each value
    uses only the trailing window ending on that date."""
    roll = spread.rolling(window)
    return (spread - roll.mean()) / roll.std()


# ---- formation: score every candidate, freeze the top 10 (run once at registration) ----

def _candidates(universe: list[str], industry: dict[str, str]) -> list[tuple[str, str, str]]:
    """Every within-NSE-industry unordered pair (a < b) in the universe, as (a, b, ind)."""
    from collections import defaultdict
    by_ind: dict[str, list[str]] = defaultdict(list)
    for s in universe:
        g = industry.get(s)
        if g:
            by_ind[g].append(s)
    out = []
    for g, names in by_ind.items():
        names = sorted(names)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                out.append((names[i], names[j], g))
    return out


def build_panel(refresh: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """(log adj_close, raw close, Nifty, universe) over the RL-12 130-name overlap.

    Universe resolution matches RL-12 exactly: india_panel(start=2010) intersect the
    F&O-shortable single-stock names. The winsorized (ret_clip 0.40) adj_close is logged
    for the spreads; the raw close is the intraday P&L baseline for the live book."""
    px, mkt, ohlcv, _ = india_panel(start=UNIVERSE_START, index=INDEX,
                                    ret_clip=0.40, refresh=refresh)
    universe = sorted(set(px.columns) & {s.upper() for s in fno_shortable(refresh=refresh)})
    return np.log(px[universe]), ohlcv["close"], mkt, universe


def formation(refresh: bool = False) -> pd.DataFrame:
    """Score every same-industry candidate pair over the formation window and return the
    ranked table (all candidates, with a `pass` flag and the STABILITY score). Contaminated
    by construction - used ONLY to SELECT the frozen set, never for a performance claim."""
    log_full, _, _, universe = build_panel(refresh=refresh)
    log_px = log_full.loc[FORMATION_START:]
    industry = {k.upper(): v for k, v in sector_map(INDEX).items()}

    rows = []
    for a, b, ind in _candidates(universe, industry):
        beta, _ = engle_granger(log_px[a].to_numpy(), log_px[b].to_numpy())
        spread = (log_px[a] - beta * log_px[b]).to_numpy()
        adf = adf_tstat(spread)
        hl = half_life(spread)
        sv = spread_vol(spread)
        ok = (adf < ADF_CRIT) and (hl < HALF_LIFE_MAX) and (sv >= VOL_FLOOR)
        rows.append({"a": a, "b": b, "industry": ind, "beta": round(beta, 4),
                     "adf_t": round(adf, 3), "half_life": round(hl, 2), "sigma": round(sv, 5),
                     "mu": round(float(spread.mean()), 5), "spread_sd": round(float(spread.std()), 5),
                     "pass": ok,
                     "stability": round(abs(adf) / hl, 4) if np.isfinite(hl) and hl > 0 else 0.0})
    df = pd.DataFrame(rows)
    passed = df[df["pass"]].sort_values("stability", ascending=False)
    rest = df[~df["pass"]].sort_values("stability", ascending=False)
    return pd.concat([passed, rest]).reset_index(drop=True)


def frozen_top(df: pd.DataFrame, n: int = TOP_N) -> tuple:
    """The FROZEN_PAIRS literal from a formation table: top-n passing pairs as
    (a, b, beta, mu, spread_sd)."""
    top = df[df["pass"]].sort_values("stability", ascending=False).head(n)
    return tuple((r.a, r.b, float(r.beta), float(r.mu), float(r.spread_sd))
                 for r in top.itertuples())


# ---- live position state machine (pure; derivable from the ledger's last row) ----

def entry_direction(z: float) -> str | None:
    """New-entry direction from today's z: 'short_a' when a is rich (z >= +ENTRY_Z),
    'long_a' when a is cheap (z <= -ENTRY_Z), else None (inside the band -> no trade)."""
    if z >= ENTRY_Z:
        return "short_a"
    if z <= -ENTRY_Z:
        return "long_a"
    return None


def should_exit(direction: str, z: float, days_held: int) -> bool:
    """Exit an open pair on the z = 0 crossing (relative to entry side) or the time stop."""
    if days_held >= TIME_STOP:
        return True
    return z >= EXIT_Z if direction == "long_a" else z <= EXIT_Z


def _days_held(entry_date: str, dates: pd.DatetimeIndex, today) -> int:
    """Trading days elapsed since entry: panel dates in (entry_date, today]."""
    ed, td = pd.Timestamp(entry_date), pd.Timestamp(today)
    return int(((dates > ed) & (dates <= td)).sum())


def update_open_pairs(prev_open: list[dict], z_now: dict[tuple[str, str], float],
                      dates: pd.DatetimeIndex, today) -> list[dict]:
    """Today's open-pair state from the prior ledger state + today's z per frozen pair.

    Carries an open pair unless it hits an exit; opens a flat pair when |z| >= ENTRY_Z and
    there is capacity. When today's z is unavailable (NaN) an open pair is carried unchanged
    (a data gap never force-exits). Deterministic and pure - the ledger's last row is the
    only state, so missed days are safe."""
    prev = {(p["a"], p["b"]): p for p in prev_open}
    out: list[dict] = []
    for a, b, *_ in FROZEN_PAIRS:
        key = (a, b)
        z = z_now.get(key)
        held = prev.get(key)
        if z is None or not np.isfinite(z):
            if held is not None:
                out.append(held)                         # carry through a data gap
            continue
        if held is not None:
            if not should_exit(held["direction"], z, _days_held(held["entry_date"], dates, today)):
                out.append(held)
        elif len(out) < MAX_PAIRS and (d := entry_direction(z)) is not None:
            out.append({"a": a, "b": b, "z_entry": round(float(z), 4),
                        "entry_date": str(pd.Timestamp(today).date()), "direction": d})
    return out


def target_weights(open_pairs: list[dict]) -> pd.Series:
    """Signed target weights per symbol: each open pair puts PER_LEG on the long leg and
    -PER_LEG on the short leg (equal dollars, so each pair is dollar-neutral). Symbols
    shared across pairs are summed; idle pairs leave cash (book gross = PER_LEG*2*n_open)."""
    w: dict[str, float] = {}
    for p in open_pairs:
        signed = -PER_LEG if p["direction"] == "short_a" else PER_LEG   # sign on leg a
        w[p["a"]] = w.get(p["a"], 0.0) + signed
        w[p["b"]] = w.get(p["b"], 0.0) - signed
    s = pd.Series(w, dtype=float)
    return s[s.abs() > 1e-12] if len(s) else s


def current_z(log_px: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Latest-date rolling z per frozen pair, using the FROZEN hedge beta."""
    out: dict[tuple[str, str], float] = {}
    for a, b, beta, *_ in FROZEN_PAIRS:
        if a in log_px.columns and b in log_px.columns:
            out[(a, b)] = float(zscore(pair_spread(log_px, a, b, beta)).iloc[-1])
    return out


# ---- module entry point: print the formation table (does NOT trade) ----

def run(refresh: bool = False) -> pd.DataFrame:
    pd.set_option("display.width", 200, "display.max_columns", 40)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        df = formation(refresh=refresh)
    n_pass = int(df["pass"].sum())
    print(f"PAIRS-RV formation  universe=F&O-cap-N500 (130 names)  "
          f"window {FORMATION_START}->registration  FORWARD-ONLY")
    print(f"candidate same-industry pairs: {len(df)}   passing (ADF<{ADF_CRIT}, HL<{HALF_LIFE_MAX:.0f}d, "
          f"sigma>={VOL_FLOOR:.3f}): {n_pass}")
    cols = ["a", "b", "industry", "beta", "adf_t", "half_life", "sigma", "stability", "pass"]
    print("\n[ALL PASSING PAIRS, ranked by stability = |ADF t| / half-life]")
    print(df[df["pass"]][cols].to_string(index=False))
    print(f"\n[FROZEN TOP {TOP_N}]")
    print(df[df["pass"]].head(TOP_N)[cols].to_string(index=False))
    print("\nFROZEN_PAIRS = (")
    for a, b, beta, mu, sd in frozen_top(df):
        print(f'    ("{a}", "{b}", {beta}, {mu}, {sd}),')
    print(")")
    print("\nFORWARD-ONLY: no historical P&L / Sharpe is computed (formation is contaminated).")
    return df


def main() -> None:
    p = argparse.ArgumentParser(
        description="RL-2026-07-26-09 same-sector cointegration pairs formation print")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(refresh=a.refresh)


if __name__ == "__main__":
    main()
