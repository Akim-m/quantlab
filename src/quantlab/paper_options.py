"""Forward-only PAPER options harness (RL-2026-07-18).

Systematically SHORT one NIFTY weekly ATM straddle, marked daily from the live
Groww option chain. The ledger `experiments/paper_options.jsonl` IS the state:
each run reads its last row, then marks / settles / rolls / opens accordingly.

FORWARD-ONLY PAPER by construction. Options history is unavailable (expired
contracts are unresolvable), so nothing here backtests. Every Groww call routes
through `groww_client.call`, which rate-limits and refuses order methods; this
harness reads market data only. Marks use leg LTP (bid/ask not reliably served),
which is OPTIMISTIC - recorded as `mark_basis="ltp"` so the caveat is auditable.

Short-straddle P&L convention (per unit, then x lot): the straddle premium is
S = CE_ltp + PE_ltp; a short gains when premium decays, so
  daily P&L = (prev_mark_S - current_S) * lot.
At expiry the straddle is worth its intrinsic |spot - strike|.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import groww_client as gc
from .data import close_prices, load_yahoo_ohlcv
from .india import groww_instruments
from .tracking import log_run

IST = timezone(timedelta(hours=5, minutes=30))
SNAPSHOT_PATH = "experiments/paper_options.jsonl"
NIFTY = "NIFTY"
NIFTY_SPOT_SYM = "NSE_NIFTY"          # cash index LTP fallback for expiry settlement
BENCH = "^NSEI"                       # Yahoo cache symbol for trailing realized vol
MIN_DTE = 2                           # enter/hold only expiries with >= 2 days to expiry
# read-only Groww methods this harness routes through gc.call (order methods are refused
# by gc.call itself; the method-spy test proves only these are ever dispatched).
READ_METHODS = ("get_expiries", "get_option_chain", "get_ltp", "get_all_instruments")

# every ledger row carries this uniform position/mark block (None when flat / degraded)
POS_KEYS = ("expiry", "strike", "lot_size", "entry_date", "ce_entry", "pe_entry",
            "entry_credit", "spot", "dte", "ce_ltp", "pe_ltp", "mark_value", "atm_iv")


def _num(x) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def _r(x, n: int = 4) -> float | None:
    return None if x is None else round(x, n)


def _parse_date(s) -> date | None:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _iv(leg: dict) -> float | None:
    iv = _num((leg.get("greeks") or {}).get("iv")) if isinstance(leg, dict) else None
    return iv if iv and iv > 0 else None


def chain_legs(oc: dict, strike: float) -> tuple[float | None, float | None,
                                                 float | None, float | None, float | None]:
    """From an option-chain payload, the CE/PE for the grid strike nearest `strike`.
    Returns (spot, matched_strike, ce_ltp, pe_ltp, atm_iv); missing pieces are None.
    Passing the spot as `strike` yields the ATM strike (nearest to spot)."""
    spot = _num(oc.get("underlying_ltp"))
    strikes = {k: v for k, v in ((_num(s), v) for s, v in (oc.get("strikes") or {}).items())
               if k is not None and isinstance(v, dict)}
    if not strikes:
        return spot, None, None, None, None
    k = min(strikes, key=lambda x: abs(x - strike))
    ce, pe = strikes[k].get("CE") or {}, strikes[k].get("PE") or {}
    ivs = [v for v in (_iv(ce), _iv(pe)) if v is not None]
    atm_iv = sum(ivs) / len(ivs) if ivs else None
    return spot, k, _num(ce.get("ltp")), _num(pe.get("ltp")), atm_iv


def nearest_expiry(today: date, min_dte: int = MIN_DTE) -> date | None:
    """Nearest NIFTY expiry with at least `min_dte` days to run (IST dates)."""
    exp = gc.call("get_expiries", exchange="NSE", underlying_symbol=NIFTY).get("expiries", [])
    fut = sorted(d for e in exp if (d := _parse_date(e)) is not None and (d - today).days >= min_dte)
    return fut[0] if fut else None


def option_chain(expiry: date) -> dict:
    return gc.call("get_option_chain", exchange="NSE", underlying=NIFTY,
                   expiry_date=expiry.isoformat())


def nifty_lot_size(instruments: pd.DataFrame) -> int | None:
    """Lot size of NIFTY options from the instrument master (mode over CE/PE rows)."""
    opt = instruments[(instruments["underlying_symbol"] == NIFTY)
                      & (instruments["instrument_type"].isin(["CE", "PE"]))]
    vals = pd.to_numeric(opt.get("lot_size"), errors="coerce").dropna()
    return int(vals.mode().iloc[0]) if len(vals) else None


def realized_vol_5d(refresh: bool = False) -> float | None:
    """Trailing 5-day annualized realized NIFTY vol (%) from the Yahoo ^NSEI cache."""
    try:
        px = close_prices(load_yahoo_ohlcv([BENCH], refresh=refresh))[BENCH.upper()]
        r = px.pct_change().dropna()
        if len(r) < 5:
            return None
        return round(float(r.iloc[-5:].std(ddof=1) * math.sqrt(252) * 100), 4)
    except Exception:
        return None


def _last_row(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    last = None
    for line in p.read_text().splitlines():
        if line.strip():
            last = line
    return json.loads(last) if last else None


def _open_fields(today: date) -> tuple[dict | None, str]:
    """Short a fresh ATM straddle on the nearest weekly expiry (>= MIN_DTE dte).
    Returns (position_fields, note); position_fields is None on any failure so the
    caller degrades gracefully instead of crashing."""
    try:
        expiry = nearest_expiry(today)
        if expiry is None:
            return None, f"no_expiry_ge_{MIN_DTE}dte"
        oc = option_chain(expiry)
        instruments = groww_instruments()
    except Exception as e:
        return None, f"open_failed: {type(e).__name__}: {e}"

    spot = _num(oc.get("underlying_ltp"))
    if spot is None:
        return None, "chain_empty"
    _, strike, ce, pe, iv = chain_legs(oc, spot)   # nearest strike to spot = ATM
    lot = nifty_lot_size(instruments)
    if strike is None or ce is None or pe is None or lot is None:
        return None, "missing_ltp_or_lot"
    ce, pe = round(ce, 2), round(pe, 2)
    credit = round(ce + pe, 4)
    return {
        "expiry": expiry.isoformat(), "strike": strike, "lot_size": lot,
        "entry_date": today.isoformat(), "ce_entry": ce, "pe_entry": pe,
        "entry_credit": credit, "spot": _r(spot, 2), "dte": (expiry - today).days,
        "ce_ltp": ce, "pe_ltp": pe, "mark_value": credit, "atm_iv": _r(iv),
    }, "ok"


def _null_position() -> dict:
    return {k: None for k in POS_KEYS}


def _carry(prev: dict, dte: int | None) -> dict:
    """Hold the existing position through a data gap: entries and last mark unchanged,
    live fields nulled, cumulative untouched (the missed day's move lands in the next
    successful mark)."""
    keep = {k: prev.get(k) for k in ("expiry", "strike", "lot_size", "entry_date",
            "ce_entry", "pe_entry", "entry_credit", "mark_value")}
    return {**keep, "spot": None, "dte": dte, "ce_ltp": None, "pe_ltp": None,
            "atm_iv": None, "action": "carry", "daily_pnl": None,
            "cumulative_pnl": round(prev.get("cumulative_pnl") or 0.0, 2), "settled": None}


def _settle_spot(expiry: date) -> tuple[float | None, str | None]:
    """Spot for expiry settlement: the chain's underlying_ltp, else cash NIFTY LTP."""
    try:
        s = _num(option_chain(expiry).get("underlying_ltp"))
        if s is not None:
            return s, None
    except Exception:
        pass
    try:
        p = gc.call("get_ltp", exchange_trading_symbols=(NIFTY_SPOT_SYM,), segment="CASH")
        return _num(p.get(NIFTY_SPOT_SYM)), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _roll(today: date, cum: float, realized: float, settled: dict,
          action: str) -> tuple[dict, str]:
    """Book the closed position's realized P&L, then open the next straddle."""
    fields, note = _open_fields(today)
    if fields is None:
        return {**_null_position(), "action": action + "_noopen",
                "daily_pnl": round(realized, 2), "cumulative_pnl": round(cum, 2),
                "settled": settled}, f"settled_but_open_failed: {note}"
    return {**fields, "action": action, "daily_pnl": round(realized, 2),
            "cumulative_pnl": round(cum, 2), "settled": settled}, "ok"


def _mark_or_roll(prev: dict, today: date, cum_prev: float) -> tuple[dict, str]:
    expiry = _parse_date(prev["expiry"])
    dte = (expiry - today).days if expiry else None
    strike, lot, prev_mark = prev["strike"], prev["lot_size"], prev["mark_value"]

    if expiry is not None and today >= expiry:                 # settle at intrinsic, roll
        spot, serr = _settle_spot(expiry)
        if spot is None:
            return _carry(prev, dte), f"settle_no_spot: {serr}"
        intrinsic = abs(spot - strike)
        realized = (prev_mark - intrinsic) * lot
        settled = {"strike": strike, "expiry": prev["expiry"], "basis": "intrinsic",
                   "settle_spot": round(spot, 2), "prev_mark": prev_mark,
                   "close_value": round(intrinsic, 4), "realized_pnl": round(realized, 2)}
        return _roll(today, cum_prev + realized, realized, settled, "roll_settle")

    if dte is not None and dte < MIN_DTE:                      # close at LTP, roll
        try:
            _, _, ce, pe, _ = chain_legs(option_chain(expiry), strike)
        except Exception as e:
            return _carry(prev, dte), f"close_failed: {type(e).__name__}: {e}"
        if ce is None or pe is None:
            return _carry(prev, dte), "close_missing_ltp"
        close_val = round(ce + pe, 4)
        realized = (prev_mark - close_val) * lot
        settled = {"strike": strike, "expiry": prev["expiry"], "basis": "ltp",
                   "prev_mark": prev_mark, "close_value": close_val,
                   "realized_pnl": round(realized, 2)}
        return _roll(today, cum_prev + realized, realized, settled, "roll_close")

    try:                                                       # normal mark-to-market
        spot, _, ce, pe, iv = chain_legs(option_chain(expiry), strike)
    except Exception as e:
        return _carry(prev, dte), f"mark_failed: {type(e).__name__}: {e}"
    if ce is None or pe is None:
        return _carry(prev, dte), "mark_missing_ltp"
    mark = round(ce + pe, 4)
    daily = (prev_mark - mark) * lot
    keep = {k: prev[k] for k in ("expiry", "strike", "lot_size", "entry_date",
            "ce_entry", "pe_entry", "entry_credit")}
    return {**keep, "spot": _r(spot, 2), "dte": dte, "ce_ltp": round(ce, 2),
            "pe_ltp": round(pe, 2), "mark_value": mark, "atm_iv": _r(iv),
            "action": "mark", "daily_pnl": round(daily, 2),
            "cumulative_pnl": round(cum_prev + daily, 2), "settled": None}, "ok"


def _try_open(today: date, cum_prev: float) -> tuple[dict, str]:
    fields, note = _open_fields(today)
    if fields is None:
        return {**_null_position(), "action": "degraded", "daily_pnl": None,
                "cumulative_pnl": round(cum_prev, 2), "settled": None}, note
    return {**fields, "action": "open", "daily_pnl": 0.0,
            "cumulative_pnl": round(cum_prev, 2), "settled": None}, "ok"


def snapshot(today: date | None = None, path: str = SNAPSHOT_PATH, write: bool = True,
             refresh: bool = False, verbose: bool = True) -> dict:
    today = today or datetime.now(IST).date()
    prev = _last_row(path)
    cum_prev = float((prev or {}).get("cumulative_pnl") or 0.0)
    has_pos = bool(prev and prev.get("strike") is not None)

    fields, note = (_mark_or_roll(prev, today, cum_prev) if has_pos
                    else _try_open(today, cum_prev))

    row = {
        "hypothesis_ref": "RL-2026-07-18", "kind": "paper_options_snapshot",
        "asof_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "run_date": today.isoformat(), "underlying": NIFTY, "mark_basis": "ltp",
        "realized_vol_5d": realized_vol_5d(refresh=refresh),
        **fields, "note": note,
    }
    if write:
        log_run(row, path=path)
    if verbose:
        _print_summary(row, path, write)
    return row


def _print_summary(row: dict, path: str, write: bool) -> None:
    print(f"[PAPER options] {row['run_date']}  action={row['action']}  "
          f"underlying={row['underlying']}")
    if row["strike"] is not None:
        print(f"  SHORT straddle  expiry={row['expiry']}  strike={row['strike']}  "
              f"lot={row['lot_size']}  dte={row['dte']}  (entered {row['entry_date']})")
        print(f"  entry credit={row['entry_credit']} (CE {row['ce_entry']} + PE {row['pe_entry']})")
        print(f"  spot={row['spot']}  mark={row['mark_value']} "
              f"(CE {row['ce_ltp']} + PE {row['pe_ltp']})  ATM_IV={row['atm_iv']}  "
              f"RV5d={row['realized_vol_5d']}")
    else:
        print("  NO open position")
    if row.get("settled"):
        s = row["settled"]
        print(f"  settled prior {s['expiry']} @ {s['strike']}: "
              f"close({s['basis']})={s['close_value']} vs mark {s['prev_mark']} "
              f"-> realized {s['realized_pnl']}")
    print(f"  daily P&L={row['daily_pnl']}  cumulative P&L={row['cumulative_pnl']}  "
          f"mark_basis={row['mark_basis']}")
    print(f"  note: {row['note']}")
    print(f"  snapshot {'appended to ' + path if write else 'NOT written (dry run)'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Forward-only PAPER NIFTY short-straddle harness")
    p.add_argument("--dry-run", action="store_true", help="print but do not append a row")
    p.add_argument("--refresh", action="store_true",
                   help="refresh the ^NSEI Yahoo cache before computing realized vol")
    p.add_argument("--path", default=SNAPSHOT_PATH)
    a = p.parse_args()
    snapshot(path=a.path, write=not a.dry_run, refresh=a.refresh)


if __name__ == "__main__":
    main()
