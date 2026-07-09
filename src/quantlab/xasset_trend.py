"""RL-2026-07-17: multi-asset trend sleeve on five NSE ETFs (long-term).

Time-series momentum is the best-documented cross-asset anomaly (Moskowitz-Ooi-
Pedersen); this is a retail-implementable Indian version on five NSE ETFs spanning
distinct return sources - NIFTYBEES (large-cap equity), JUNIORBEES (next-50),
BANKBEES (banks), GOLDBEES (gold in INR), MON100 (Nasdaq-100 in INR, adds USD +
global tech). Each asset is trend-gated (long its base share when its own trend is
up, else cash); the sleeve's value is as a portfolio diversifier, not a Sharpe
contest.

TRAIN (2010->2016-12-31) picks the gate (12-1 TSMOM sign vs price>200d-MA) and the
weighting (equal vs inverse-vol 126d) from exactly those four combinations on the
sleeve's combined Sharpe, then FREEZES both before the single TEST read (2017+).
Promotion bar (pre-registered): standalone test SR >= 0.8 AND corr with the deployed
REGIME book < 0.5. A miss on either is a valid negative.

Data quality: Yahoo adj_close for NIFTYBEES/BANKBEES/GOLDBEES carries a two-day
bad-print round-trip on 2019-12-19..20 (the price collapses to ~1/10..1/100 then
snaps back) - impossible for an ETF. The lab's ret_clip=0.40 (which clips daily
*returns*) cannot repair a multi-day round-trip: the clipped -40% down and +40%
recovery legs do not offset, leaving a permanent ~16% level shift across the whole
post-2019 test window (and its iloc[0] rebuild nulls MON100's leading-NaN column).
We instead drop transient price spikes vs a trailing 5-day median and forward-fill -
the minimal causal repair that touches only physically impossible prints (>50% off
the local median; real ETF days move <=~15%). Disclosed; the TRAIN freeze is
unaffected (the glitch is post-2016). Groww is never called here - the REGIME/L/S
reconstructions read Yahoo prices and the cached F&O instrument master from disk.
"""

import argparse
import glob
import warnings

import numpy as np
import pandas as pd

from .backtest import backtest_weights
from .bear_sleeve import base_book
from .blend import composite, fno_long_short
from .data import close_prices, load_yahoo_ohlcv
from .evaluation import sharpe_tstat
from .features import rolling_vol
from .india import fno_shortable, india_panel, sector_map
from .india_blend_study import raw_signals
from .india_ls import CORE as LS_CORE, WEIGHTS as LS_WEIGHTS
from .portfolio import rebalance_targets
from .tracking import log_run

ETFS = ["NIFTYBEES.NS", "JUNIORBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "MON100.NS"]
GATES = ("tsmom", "ma")
WEIGHTINGS = ("equal", "invvol")
TRAIN0, TRAIN1 = "2010-01-01", "2016-12-31"
# Frozen on TRAIN (2010->2016-12-31) combined Sharpe, before the single test read:
FROZEN_GATE, FROZEN_WEIGHTING = "tsmom", "invvol"
CUMULATIVE_INDIA_TRIALS = 68  # 64 prior India trials + these 4 combos


def clean_prices(px: pd.DataFrame, win: int = 5, dev: float = 0.5) -> pd.DataFrame:
    """Drop transient price spikes and forward-fill (data-quality repair).

    A bar is a bad print if it sits below `dev`x or above `1/dev`x its trailing
    `win`-day median - impossible for an ETF (real days move <=~15%). This repairs
    the 2019-12 round-trip glitches that ret_clip cannot (a 2-day glitch's clipped
    down/up legs never offset). Causal: the trailing median and the ffill replacement
    use only past bars, so a repaired value never depends on the future."""
    med = px.rolling(win, min_periods=3).median()
    bad = (px / med < dev) | (px / med > 1.0 / dev)
    return px.mask(bad).ffill()


def etf_panel(refresh: bool = False) -> pd.DataFrame:
    """Cleaned adj_close panel for the five ETFs on the NIFTYBEES (NSE) calendar.

    MON100 lists 2011-03; its pre-inception rows stay NaN (no position, disclosed)."""
    px = close_prices(load_yahoo_ohlcv(ETFS, refresh=refresh))[ETFS]
    px = px.reindex(px["NIFTYBEES.NS"].dropna().index).ffill(limit=3)
    return clean_prices(px)


def _gate(px: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Per-asset trend signal (higher = stronger uptrend), from trailing prices only:
    12-1 momentum (px[t-21]/px[t-252]-1) or the distance of price above its 200d MA."""
    if kind == "tsmom":
        return px.shift(21) / px.shift(252) - 1.0
    return px - px.rolling(200).mean()


def sleeve_weights(px: pd.DataFrame, gate: str, weighting: str) -> pd.DataFrame:
    """Long-only trend sleeve: each asset holds its base share when its own trend is
    up, else cash (0). Base shares - equal, or inverse-vol (126d) - are normalized
    over the assets whose gate is computable that day, so the book sums to <=1 (the
    remainder is cash when trends are off or an asset is pre-inception)."""
    sig = _gate(px, gate)
    valid = sig.notna() & px.notna()
    up = (sig > 0) & valid
    if weighting == "equal":
        base = valid.astype(float)
    else:
        base = (1.0 / rolling_vol(px, 126).replace(0.0, np.nan)).where(valid)
    base = base.div(base.sum(axis=1).replace(0.0, np.nan), axis=0)
    return base.where(up, 0.0).fillna(0.0)


def _metrics(r: pd.Series) -> tuple[float, float, float]:
    """(Sharpe, annualized return, max drawdown) of a daily return series."""
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    ann = float((1.0 + r).prod() ** (252 / max(len(r), 1)) - 1.0)
    return round(sharpe_tstat(r)[0], 3), round(ann, 4), round(dd, 4)


def sleeve_ret(px: pd.DataFrame, gate: str, weighting: str, cost_bps: float) -> pd.Series:
    return backtest_weights(px, rebalance_targets(sleeve_weights(px, gate, weighting), "ME"), cost_bps).returns


def _cached_master() -> pd.DataFrame:
    """Newest cached Groww instrument master, read from disk (NO live Groww call)."""
    files = sorted(glob.glob("data/raw/groww_instruments_*.csv"))
    if not files:
        raise FileNotFoundError("no cached data/raw/groww_instruments_*.csv for the F&O master")
    return pd.read_csv(files[-1], low_memory=False)


def base_returns(cost_bps: float, refresh: bool = False) -> tuple[pd.Series, pd.Series]:
    """Deployed REGIME long book (reproduces test SR 1.865) and the F&O-shortable L/S
    sleeve, rebuilt via their frozen constructions for the correlation reads. Yahoo
    prices + the cached F&O master only - no Groww call."""
    px, mkt, _, _ = india_panel(start="2010-01-01", index="nifty500", ret_clip=0.40, refresh=refresh)
    sectors = sector_map("nifty500")
    vix = close_prices(load_yahoo_ohlcv(["^INDIAVIX"], refresh=refresh))["^INDIAVIX"].reindex(px.index).ffill()
    regime = backtest_weights(px, rebalance_targets(base_book(px, mkt, vix, sectors), "ME"), cost_bps).returns
    shortable = {s.upper() for s in fno_shortable(instruments=_cached_master())}
    score = composite({k: raw_signals(px, mkt, sectors)[k] for k in LS_CORE}, weights=LS_WEIGHTS)
    ls = backtest_weights(px, rebalance_targets(fno_long_short(score, shortable), "ME"), cost_bps).returns
    return regime, ls


def run(split: str = "2017-01-01", refresh: bool = False) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 30)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        px = etf_panel(refresh=refresh)

        train_rows = []
        for g in GATES:
            for w in WEIGHTINGS:
                sr, ann, dd = _metrics(sleeve_ret(px, g, w, 20.0).loc[TRAIN0:TRAIN1])
                train_rows.append({"gate": g, "weighting": w, "train_sharpe": sr,
                                   "train_ann": ann, "train_maxdd": dd})
        train = pd.DataFrame(train_rows).sort_values("train_sharpe", ascending=False).reset_index(drop=True)
        argmax = (train.iloc[0]["gate"], train.iloc[0]["weighting"])

        regime, ls = base_returns(20.0, refresh=refresh)
        regime_sr = round(sharpe_tstat(regime.loc[split:])[0], 3)
        ls_sr = round(sharpe_tstat(ls.loc[split:])[0], 3)

        test_rows = []
        for cost in (10.0, 20.0, 40.0):
            r = sleeve_ret(px, FROZEN_GATE, FROZEN_WEIGHTING, cost).loc[split:]
            sr, ann, dd = _metrics(r)
            test_rows.append({"cost_bps": int(cost), "test_sharpe": sr, "ann_return": ann, "max_dd": dd,
                              "corr_regime": round(float(r.corr(regime.loc[split:])), 3),
                              "corr_ls": round(float(r.corr(ls.loc[split:])), 3)})
        test = pd.DataFrame(test_rows)
        head = test[test["cost_bps"] == 20].iloc[0]
        promoted = bool(head["test_sharpe"] >= 0.8 and head["corr_regime"] < 0.5)

    print(f"5-ETF trend sleeve  {px.index[0].date()}->{px.index[-1].date()}  test>={split}")
    print(f"MON100 first valid {px['MON100.NS'].first_valid_index().date()} (pre-inception = no position)")
    print(f"\n[TRAIN {TRAIN0}->{TRAIN1} @20bps | 4 combos -> FREEZE on combined Sharpe]")
    print(train.to_string(index=False))
    print(f"FROZEN: gate={FROZEN_GATE} weighting={FROZEN_WEIGHTING}  "
          f"(TRAIN argmax = {argmax[0]}/{argmax[1]}; {'matches' if argmax == (FROZEN_GATE, FROZEN_WEIGHTING) else 'MISMATCH - freeze held per protocol'})")
    print(f"\n[BASE reconstruction check]  REGIME test SR = {regime_sr} (target 1.865)  |  "
          f"L/S test SR = {ls_sr} (target ~0.846)")
    print(f"\n[TEST {split}+ ONE read | frozen {FROZEN_GATE}/{FROZEN_WEIGHTING} | 10/20/40 bps]")
    print(test.to_string(index=False))
    print(f"\n[PROMOTION]  bar: test SR>=0.8 AND corr(REGIME)<0.5  |  "
          f"SR={head['test_sharpe']} corr(REGIME)={head['corr_regime']} corr(L/S)={head['corr_ls']}  "
          f"-> {'PROMOTED' if promoted else 'NOT PROMOTED'}")

    for row in test_rows:
        log_run({"hypothesis_ref": "RL-2026-07-17", "universe": "NSE-ETF5",
                 "cost_bps": float(row["cost_bps"]), "split": split,
                 "strategy": f"xasset_trend_{FROZEN_GATE}_{FROZEN_WEIGHTING}",
                 "frozen_gate": FROZEN_GATE, "frozen_weighting": FROZEN_WEIGHTING,
                 "n_assets": len(ETFS), "metrics": row, "regime_sharpe": regime_sr,
                 "ls_sharpe": ls_sr, "promoted": promoted, "n_trials_family": 4,
                 "n_trials_cumulative": CUMULATIVE_INDIA_TRIALS, "status": "success"})


def main() -> None:
    p = argparse.ArgumentParser(description="RL-2026-07-17 multi-asset ETF trend sleeve")
    p.add_argument("--split", default="2017-01-01")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    run(split=a.split, refresh=a.refresh)


if __name__ == "__main__":
    main()
