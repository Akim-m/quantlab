"""RL-2026-07-26-19: Amihud illiquidity level cross-section (ILLIQ).

Economic hypothesis (Amihud 2002): investors demand a premium for holding names where a
rupee of flow moves price more, so the illiquid decile out-earns the liquid one. The
classic proxy is |ret| / rupee-turnover - the daily absolute return per rupee traded. The
lab already uses volume as a 5d/126d SHOCK (-15, the attention channel); this is the
orthogonal persistent-LEVEL premium (the compensation channel).

Signal: daily illiq = |adj-close return| / rupee turnover, where rupee turnover = raw
(dividend-unadjusted, split-adjusted) close * traded volume. A non-traded session (zero or
missing volume) contributes NO turnover -> NaN illiq that day, so it counts as an invalid
day and is skipped by the rolling mean. illiq is averaged over the variant lookback L, a
name with < 60% valid days inside L is excluded that day, the mean is logged, then
winsorized +/- 3 MAD cross-sectionally. A decile long/short book LONGS the top (most
illiquid) decile and SHORTS the bottom, equal-weight, dollar-neutral, held monthly (ME).
The daily weights are lagged one trading day, so a date-t target uses only data through
t-1 (locked spec).

Two arms of the deliverable (the DUAL-ROT route: TRAIN-design + FORWARD-ONLY, zero
hold-out spend):
  (A) `train_study` / `run` - the TRAIN-window design study over 2010-01-01..2016-12-31
      ONLY, backtesting L=63 vs L=252 at 40 bps (the FREEZE criterion - the harsh cost arm
      is honest when the long leg is by construction the least-liquid decile) plus 20/80 bps
      sensitivities. FREEZE = argmax TRAIN net Sharpe at 40 bps. Every panel is physically
      truncated to the TRAIN boundary before a single weight or return is computed, so no
      statistic can read a date after 2016-12-31. Survivorship is acute (current N500
      membership inflates an illiquid-leg TRAIN spread); the forward read is the clean number.
  (B) the frozen live paper book (live_paper.run_illiq -> paper_trades_illiq.jsonl), signal
      computed on the full panel - that full history is warm-up for the rolling mean, not a
      performance read.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from .backtest import backtest_weights
from .evaluation import sharpe_tstat
from .india import india_panel
from .portfolio import rebalance_targets
from .tracking import log_run

LOOKBACKS = (63, 252)   # the two registered variants (trailing illiq windows, trading days)
FROZEN_LOOKBACK = 63    # freeze = argmax TRAIN net Sharpe @40bps (L63 1.101 > L252 1.050; see run)
DECILE = 0.10           # top/bottom decile long/short
MAD_N = 3.0             # cross-sectional winsorization half-width, in MADs
MIN_VALID = 0.60        # a name needs >= 60% valid days inside the lookback to be scored
REBALANCE = "ME"
TRAIN_START = "2010-01-01"
TRAIN_END = "2016-12-31"   # HARD: no performance statistic is read past this date
COST_ARMS = (40.0, 20.0, 80.0)   # 40 = FREEZE criterion; 20/80 = disclosed sensitivities
FREEZE_COST = 40.0


def turnover(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Rupee turnover = raw close * traded volume. A non-traded session (zero or missing
    volume) is NaN, so it is never counted as a real low-turnover day and the name drops
    out of any cross-section that reads it - the registered zero/missing-volume rule."""
    return close * volume.where(volume > 0)


def daily_illiq(px: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Amihud daily illiquidity |ret| / rupee turnover, per name per day.

    `ret` is the adjusted-close daily return (the ret_clip 0.40 winsorization is already
    baked into `px`); numerator and denominator are the SAME day t (Amihud's contemporaneous
    definition). A NaN-turnover (non-traded) day yields NaN illiq -> an invalid day that the
    rolling mean skips and the valid-day count excludes."""
    return px.pct_change().abs() / turnover(close, volume)


def illiq(px: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
          lookback: int = FROZEN_LOOKBACK, min_valid: float = MIN_VALID) -> pd.DataFrame:
    """log of the trailing `lookback`-day MEAN Amihud illiquidity.

    The mean skips invalid (NaN) days; a name with fewer than `min_valid` of the lookback's
    days valid is excluded (NaN) that day. Higher = more illiquid (the long leg)."""
    di = daily_illiq(px, close, volume)
    mean = di.rolling(lookback, min_periods=1).mean()
    valid = di.notna().rolling(lookback, min_periods=1).sum()
    return np.log(mean.where(valid >= min_valid * lookback))


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
    """Long the top-`q` illiquidity decile, short the bottom-`q`: equal-weight within each
    leg, dollar-neutral (net 0), unit gross (|w| sums to 1) - xsec.py conventions."""
    need = int(round(1.0 / q))

    def row(r: pd.Series) -> pd.Series:
        v = r.dropna()
        w = pd.Series(0.0, index=r.index)
        if len(v) < need:                       # too few names to form both deciles
            return w
        k = max(1, int(np.floor(len(v) * q)))
        order = v.sort_values()                 # ascending: liquid (low illiq) first
        w[order.index[-k:]] = 0.5 / k           # top (most illiquid) decile long
        w[order.index[:k]] = -0.5 / k           # bottom (most liquid) decile short
        return w

    return sig.apply(row, axis=1)


def weights(px: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
            lookback: int = FROZEN_LOOKBACK, q: float = DECILE, n_mad: float = MAD_N,
            min_valid: float = MIN_VALID, rebalance: str | None = REBALANCE) -> pd.DataFrame:
    """Signed month-end-rebalanced target weights. The signal is lagged one day (date-t
    weights read data through t-1), then decile L/S, then placed on the `rebalance` grid:
    only the grid rows carry weights, every other row is NaN so a backtest trades solely on
    the grid (true monthly turnover - the house pattern). The live book ffills this to read
    the currently-held decile (see latest_weights)."""
    sig = winsorize(illiq(px, close, volume, lookback, min_valid), n_mad)
    w = decile_ls(sig.shift(1), q)
    return rebalance_targets(w, rebalance) if rebalance else w


def latest_weights(px: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, **kw) -> pd.Series:
    """Currently-held decile on the last panel date (for the live paper book): the ME grid
    carried forward, so a non-rebalance day holds the most recent month-end book."""
    return weights(px, close, volume, **kw).ffill().iloc[-1]


def latest_signal(px: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
                  lookback: int = FROZEN_LOOKBACK) -> pd.Series:
    """Latest cross-sectional winsorized log-illiquidity (higher = more illiquid), for the
    orchestrator's cross-signal correlation. Un-lagged: the freshest cross-section."""
    return winsorize(illiq(px, close, volume, lookback), MAD_N).iloc[-1]


def panels(start: str = TRAIN_START, index: str = "nifty500",
           refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(px_adj, close_raw, volume) over india_panel's frozen N500 universe/calendar.
    `px_adj` is the ret_clip-0.40 adjusted-close panel whose daily return is the |ret|
    numerator; `close_raw` is the dividend-unadjusted close and `volume` the traded volume,
    whose product is the rupee-turnover denominator (the live intraday baseline is
    `close_raw`, matching the raw Groww LTP)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        px, _, ohlcv, _ = india_panel(start=start, index=index,
                                      ret_clip=0.40, refresh=refresh)
    return px, ohlcv["close"], ohlcv["volume"]


# ---- (A) TRAIN-window design study (reads ONLY 2010-01-01..2016-12-31) ----

def _metrics(res) -> tuple[float, float, float, float]:
    """(net Sharpe, ann return, maxDD, avg monthly one-sided turnover). Monthly turnover is
    the mean over rebalance days of the traded fraction halved to one side (the registration's
    idiom: 'turnover ~5-15%/mo' means the fraction of the book replaced each month)."""
    r = res.returns
    sr = sharpe_tstat(r)[0]
    ann = float((1.0 + r).prod() ** (252 / max(len(r), 1)) - 1.0)
    dd = res.max_drawdown
    turn = res.turnover
    onesided = turn[turn > 0] / 2.0
    mt = float(onesided.mean()) if len(onesided) else 0.0
    return sr, ann, dd, mt


def long_leg_liquidity(w: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
                       lookback: int) -> float:
    """Median over the book's active days of the median rupee-turnover percentile of the
    LONG (most-illiquid) leg. Percentile is from the trailing-`lookback` MEAN rupee turnover
    (1.0 = most liquid), lagged one day to match the weight's information set. A low value is
    the cost-realism disclosure: the long leg sits deep in the least-liquid names."""
    mean_to = turnover(close, volume).rolling(lookback, min_periods=1).mean()
    pct = mean_to.rank(axis=1, pct=True).shift(1)
    daily = []
    for date, row in w.ffill().iterrows():          # the held book each day, not just ME rows
        longs = row.index[row > 0]
        if len(longs):
            p = pct.loc[date, longs].dropna()
            if len(p):
                daily.append(float(p.median()))
    return float(np.median(daily)) if daily else float("nan")


def train_study(px: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
                costs: tuple[float, ...] = COST_ARMS
                ) -> tuple[pd.DataFrame, dict[tuple[int, float], pd.Series]]:
    """TRAIN-only design table for L=63 vs L=252 across the cost arms.

    Every panel is physically truncated to [TRAIN_START, TRAIN_END] BEFORE any weight or
    return is computed, so no statistic can read a date after the hold-out boundary. Returns
    (table, returns_by_key), returns_by_key[(L, cost)] the net daily return series whose last
    date is <= TRAIN_END by construction."""
    px = px.loc[TRAIN_START:TRAIN_END]
    close = close.loc[TRAIN_START:TRAIN_END]
    volume = volume.loc[TRAIN_START:TRAIN_END]

    rows, rets = [], {}
    for L in LOOKBACKS:
        w = weights(px, close, volume, lookback=L)
        llp = long_leg_liquidity(w, close, volume, L)
        for cost in costs:
            res = backtest_weights(px, w, cost_bps=cost)
            sr, ann, dd, mt = _metrics(res)
            rets[(L, cost)] = res.returns
            rows.append({"variant": f"L{L}", "lookback": L, "cost_bps": int(cost),
                         "net_sharpe": round(sr, 3), "ann_return": round(ann, 4),
                         "max_dd": round(dd, 4), "monthly_turnover": round(mt, 4),
                         "longleg_to_pctile": round(llp, 4)})
    return pd.DataFrame(rows), rets


def freeze(table: pd.DataFrame) -> int:
    """FREEZE = the lookback with the largest TRAIN net Sharpe at the FREEZE_COST arm."""
    at = table[table["cost_bps"] == int(FREEZE_COST)].sort_values("net_sharpe", ascending=False)
    return int(at.iloc[0]["lookback"])


def run(refresh: bool = False, log: bool = True) -> pd.DataFrame:
    """TRAIN design study: print the L=63/L=252 x {40,20,80}bps table, the freeze, and the
    long-leg liquidity disclosure; append one row per variant x cost to experiments/log.jsonl.
    Reads NO date after TRAIN_END (train_study truncates the panels first)."""
    pd.set_option("display.width", 200, "display.max_columns", 30)
    px, close, volume = panels(refresh=refresh)
    table, rets = train_study(px, close, volume)
    frozen = freeze(table)

    n_assets = int(px.loc[TRAIN_START:TRAIN_END].shape[1])
    last_dates = {k: str(v.index[-1].date()) for k, v in rets.items()}
    at40 = table[table["cost_bps"] == int(FREEZE_COST)].sort_values("net_sharpe", ascending=False)
    margin = float(at40.iloc[0]["net_sharpe"] - at40.iloc[-1]["net_sharpe"])

    print(f"ILLIQ  N500-{n_assets}  TRAIN {TRAIN_START}..{TRAIN_END}  (returns end "
          f"{max(last_dates.values())} <= {TRAIN_END})")
    print(f"\n[TRAIN design | L=63 vs L=252 x 40/20/80 bps | FREEZE = argmax net Sharpe @40bps]")
    print(table.to_string(index=False))
    print(f"\nFROZEN: L{frozen}  (TRAIN argmax @40bps; margin over the loser {margin:+.3f} Sharpe; "
          f"{'matches' if frozen == FROZEN_LOOKBACK else 'MISMATCH vs FROZEN_LOOKBACK - freeze held per protocol'})")
    print(f"long-leg median rupee-turnover percentile (cost realism): "
          + ", ".join(f"L{int(r['lookback'])}={r['longleg_to_pctile']:.3f}"
                      for r in table.drop_duplicates('lookback').to_dict('records')))

    if log:
        for r in table.to_dict("records"):
            log_run({
                "hypothesis_ref": "RL-2026-07-26-19", "universe": "NIFTY500",
                "n_assets": n_assets, "train_window": f"{TRAIN_START}:{TRAIN_END}",
                "cost_bps": float(r["cost_bps"]), "strategy": f"illiq_{r['variant']}",
                "variant": r["variant"], "lookback": int(r["lookback"]),
                "frozen_variant": f"L{frozen}", "train_argmax": f"L{frozen}",
                "freeze_cost_bps": FREEZE_COST,
                "metrics": {"net_sharpe": float(r["net_sharpe"]), "ann_return": float(r["ann_return"]),
                            "max_dd": float(r["max_dd"]), "monthly_turnover": float(r["monthly_turnover"]),
                            "longleg_to_pctile": float(r["longleg_to_pctile"])},
                "returns_last_date": last_dates[(int(r["lookback"]), float(r["cost_bps"]))],
                "n_trials_family": len(LOOKBACKS), "status": "success",
            })
    return table


def main() -> None:
    p = argparse.ArgumentParser(
        description="RL-2026-07-26-19 ILLIQ TRAIN design study (reads only 2010..2016)")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--no-log", action="store_true", help="print but do not append log.jsonl rows")
    a = p.parse_args()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        run(refresh=a.refresh, log=not a.no_log)


if __name__ == "__main__":
    main()
