"""Forward-only PAPER cash-secured put-write harness (RL-2026-07-26-03).

A PARALLEL book to RL-18's straddle: same NIFTY weekly chain, same daily LTP
marks, same expiry-settle-and-roll lifecycle and ledger discipline - EXCEPT the
position is a SINGLE short put, ~2% out of the money, fully cash-secured. This
harvests the index variance-risk premium as EQUITY REPLACEMENT (CBOE PUT index;
Ungar-Moran 2009): long-delta with a premium cushion, a distinct risk shape from
RL-18's delta-neutral vol bet.

Strike selection: the strikes come from the live chain; each entry picks the
listed strike minimizing |strike - 0.98 * spot| (nearest-to-2%-OTM). Expiry is
the nearest NIFTY weekly with at least MIN_DTE days to run.

Lifecycle (mirrors RL-18 exactly, one put leg instead of two):
  - enter    : short one lot of that put; credit = PE_ltp; cash-secure strike*lot.
  - mark     : daily mark at the put's chain LTP.
  - settle   : at/after expiry, close at intrinsic max(0, strike - settle_spot)
               (European cash settlement), then roll into the next weekly put.
  - roll     : when DTE < MIN_DTE (before expiry), close at the put's LTP and roll.

Short-put P&L convention (per unit, then x lot): mark M = PE_ltp; a short gains as
the premium decays, so
  daily P&L    = (prev_mark - current_mark) * lot
  settle  P&L  = (prev_mark - max(0, strike - settle_spot)) * lot,
which telescopes over the life to (entry_credit - close_value) * lot.

FORWARD-ONLY by construction (RL-15): expired option contracts are unresolvable,
so nothing here backtests and no Sharpe/return is ever claimed - the leg is LIVE,
logging the daily mark from day one. Marks use leg LTP (bid/ask not reliably
served), OPTIMISTIC and disclosed as `mark_basis="ltp"`. Every Groww call routes
through `groww_client.call`, which rate-limits and refuses order methods;
read-only.

The row schema mirrors RL-18's position/mark block plus three put-write fields:
`strike_ratio` (actual strike / entry spot), `otm_pct` ((1 - strike/spot)*100),
and `cash_secured_notional` (strike * lot). There is no call leg, so `ce_entry`
and `ce_ltp` are always None; `pe_entry`/`pe_ltp`/`mark_value` carry the put, and
`atm_iv` carries the put strike's IV (kept under RL-18's key name for schema
consistency, though the strike is ~2% OTM, not ATM). All Groww-touching and P&L
arithmetic is reused from `paper_options` so the book stays bit-consistent.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from . import paper_options as po
from .tracking import log_run

SNAPSHOT_PATH = "experiments/paper_options_putw.jsonl"
BOOK = "cash_secured_putwrite"
HYPOTHESIS = "RL-2026-07-26-03"
STRIKE_RATIO = 0.98                   # target = 0.98 * spot -> ~2% OTM put
MIN_DTE = po.MIN_DTE
READ_METHODS = po.READ_METHODS       # entry + marks dispatch only these read-only methods

# put-write additions to RL-18's POS_KEYS (entry-time constants, carried through marks)
EXTRA_KEYS = ("strike_ratio", "otm_pct", "cash_secured_notional")


def _null_position() -> dict:
    return {**po._null_position(), **{k: None for k in EXTRA_KEYS}}


def _carry(prev: dict, dte: int | None) -> dict:
    """Hold through a data gap: RL-18's carry (entries + last mark unchanged, live
    fields nulled) plus the put-write entry constants carried alongside."""
    return {**po._carry(prev, dte), **{k: prev.get(k) for k in EXTRA_KEYS}}


def _open_fields(today) -> tuple[dict | None, str]:
    """Short a fresh ~2%-OTM put on the nearest weekly expiry (>= MIN_DTE dte).
    Returns (position_fields, note); position_fields is None on any failure so the
    caller degrades gracefully instead of crashing."""
    try:
        expiry = po.nearest_expiry(today)
        if expiry is None:
            return None, f"no_expiry_ge_{MIN_DTE}dte"
        oc = po.option_chain(expiry)
        instruments = po.groww_instruments()
    except Exception as e:
        return None, f"open_failed: {type(e).__name__}: {e}"

    spot = po._num(oc.get("underlying_ltp"))
    if spot is None:
        return None, "chain_empty"
    # nearest listed strike to 0.98*spot; chain_legs also yields that strike's PE/IV
    _, strike, _, pe, iv = po.chain_legs(oc, STRIKE_RATIO * spot)
    lot = po.nifty_lot_size(instruments)
    if strike is None or pe is None or lot is None:
        return None, "missing_ltp_or_lot"
    pe = round(pe, 2)
    credit = round(pe, 4)
    return {
        "expiry": expiry.isoformat(), "strike": strike, "lot_size": lot,
        "entry_date": today.isoformat(), "ce_entry": None, "pe_entry": pe,
        "entry_credit": credit, "spot": po._r(spot, 2), "dte": (expiry - today).days,
        "ce_ltp": None, "pe_ltp": pe, "mark_value": credit, "atm_iv": po._r(iv),
        "strike_ratio": round(strike / spot, 6),
        "otm_pct": round((1 - strike / spot) * 100, 4),
        "cash_secured_notional": round(strike * lot, 2),
    }, "ok"


def _roll(today, cum: float, realized: float, settled: dict,
          action: str) -> tuple[dict, str]:
    """Book the closed put's realized P&L, then open the next weekly put."""
    fields, note = _open_fields(today)
    if fields is None:
        return {**_null_position(), "action": action + "_noopen",
                "daily_pnl": round(realized, 2), "cumulative_pnl": round(cum, 2),
                "settled": settled}, f"settled_but_open_failed: {note}"
    return {**fields, "action": action, "daily_pnl": round(realized, 2),
            "cumulative_pnl": round(cum, 2), "settled": settled}, "ok"


def _mark_or_roll(prev: dict, today, cum_prev: float) -> tuple[dict, str]:
    expiry = po._parse_date(prev["expiry"])
    dte = (expiry - today).days if expiry else None
    strike, lot, prev_mark = prev["strike"], prev["lot_size"], prev["mark_value"]

    if expiry is not None and today >= expiry:                 # settle at intrinsic, roll
        spot, serr = po._settle_spot(expiry)
        if spot is None:
            return _carry(prev, dte), f"settle_no_spot: {serr}"
        intrinsic = max(0.0, strike - spot)                    # European put payoff
        realized = (prev_mark - intrinsic) * lot
        settled = {"strike": strike, "expiry": prev["expiry"], "basis": "intrinsic",
                   "settle_spot": round(spot, 2), "prev_mark": prev_mark,
                   "close_value": round(intrinsic, 4), "realized_pnl": round(realized, 2)}
        return _roll(today, cum_prev + realized, realized, settled, "roll_settle")

    if dte is not None and dte < MIN_DTE:                      # close at LTP, roll
        try:
            _, _, _, pe, _ = po.chain_legs(po.option_chain(expiry), strike)
        except Exception as e:
            return _carry(prev, dte), f"close_failed: {type(e).__name__}: {e}"
        if pe is None:
            return _carry(prev, dte), "close_missing_ltp"
        close_val = round(pe, 4)
        realized = (prev_mark - close_val) * lot
        settled = {"strike": strike, "expiry": prev["expiry"], "basis": "ltp",
                   "prev_mark": prev_mark, "close_value": close_val,
                   "realized_pnl": round(realized, 2)}
        return _roll(today, cum_prev + realized, realized, settled, "roll_close")

    try:                                                       # normal mark-to-market
        spot, _, _, pe, iv = po.chain_legs(po.option_chain(expiry), strike)
    except Exception as e:
        return _carry(prev, dte), f"mark_failed: {type(e).__name__}: {e}"
    if pe is None:
        return _carry(prev, dte), "mark_missing_ltp"
    mark = round(pe, 4)
    daily = (prev_mark - mark) * lot
    keep = {k: prev[k] for k in ("expiry", "strike", "lot_size", "entry_date",
            "ce_entry", "pe_entry", "entry_credit", *EXTRA_KEYS)}
    return {**keep, "spot": po._r(spot, 2), "dte": dte, "ce_ltp": None,
            "pe_ltp": round(pe, 2), "mark_value": mark, "atm_iv": po._r(iv),
            "action": "mark", "daily_pnl": round(daily, 2),
            "cumulative_pnl": round(cum_prev + daily, 2), "settled": None}, "ok"


def _try_open(today, cum_prev: float) -> tuple[dict, str]:
    fields, note = _open_fields(today)
    if fields is None:
        return {**_null_position(), "action": "degraded", "daily_pnl": None,
                "cumulative_pnl": round(cum_prev, 2), "settled": None}, note
    return {**fields, "action": "open", "daily_pnl": 0.0,
            "cumulative_pnl": round(cum_prev, 2), "settled": None}, "ok"


def snapshot(today=None, path: str = SNAPSHOT_PATH, write: bool = True,
             refresh: bool = False, verbose: bool = True) -> dict:
    today = today or datetime.now(po.IST).date()
    prev = po._last_row(path)
    cum_prev = float((prev or {}).get("cumulative_pnl") or 0.0)
    has_pos = bool(prev and prev.get("strike") is not None)

    fields, note = (_mark_or_roll(prev, today, cum_prev) if has_pos
                    else _try_open(today, cum_prev))

    row = {
        "hypothesis_ref": HYPOTHESIS, "book": BOOK,
        "kind": "paper_options_snapshot",
        "asof_ist": datetime.now(po.IST).strftime("%Y-%m-%d %H:%M:%S"),
        "run_date": today.isoformat(), "underlying": po.NIFTY, "mark_basis": "ltp",
        "realized_vol_5d": po.realized_vol_5d(refresh=refresh),
        **fields, "note": note,
    }
    if write:
        log_run(row, path=path)
    if verbose:
        _print_summary(row, path, write)
    return row


def _print_summary(row: dict, path: str, write: bool) -> None:
    print(f"[PAPER put-write] {row['run_date']}  action={row['action']}  "
          f"underlying={row['underlying']}")
    if row["strike"] is not None:
        print(f"  SHORT put  expiry={row['expiry']}  strike={row['strike']}  "
              f"lot={row['lot_size']}  dte={row['dte']}  (entered {row['entry_date']})")
        print(f"  credit={row['entry_credit']} (PE)  strike_ratio={row['strike_ratio']}  "
              f"otm={row['otm_pct']}%  cash_secured={row['cash_secured_notional']}")
        print(f"  spot={row['spot']}  mark={row['mark_value']} (PE {row['pe_ltp']})  "
              f"IV={row['atm_iv']}  RV5d={row['realized_vol_5d']}")
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
    p = argparse.ArgumentParser(
        description="Forward-only PAPER NIFTY cash-secured put-write harness")
    p.add_argument("--dry-run", action="store_true", help="print but do not append a row")
    p.add_argument("--refresh", action="store_true",
                   help="refresh the ^NSEI Yahoo cache before computing realized vol")
    p.add_argument("--path", default=SNAPSHOT_PATH)
    a = p.parse_args()
    snapshot(path=a.path, write=not a.dry_run, refresh=a.refresh)


if __name__ == "__main__":
    main()
