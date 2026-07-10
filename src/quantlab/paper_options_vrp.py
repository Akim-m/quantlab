"""Forward-only VRP-GATED PAPER short-straddle harness (RL-2026-07-26-06).

A PARALLEL book to RL-18: same NIFTY weekly ATM short straddle, same daily LTP
marks, same expiry-settle-and-roll lifecycle - EXCEPT it only holds the straddle
in weeks where the measured variance risk premium is fat. Each collection day:

    VRP_t = ATM_IV_t - RV5d_t
      ATM_IV_t = today's nearest-weekly-expiry ATM implied vol (CE/PE greeks-IV
                 mean) - the same construction `fno_collect` and `paper_options`
                 use, evaluable every day whether or not a position is open.
      RV5d_t   = `paper_options.realized_vol_5d` (trailing 5-day annualized).

    gate_on = VRP_t > median(trailing 126 prior collection-day VRPs)   [prior-day
              info only; today's VRP is NEVER in its own median window].

    gate_on  -> enter (if flat) / hold+mark (if in) the RL-18 straddle.
    gate_off -> flat: close any open straddle at LTP (intrinsic at/after expiry),
                otherwise stay flat and just log VRP + the gate decision.

WARM-UP (critical - only ~2 collection-day VRPs exist at registration, far short
of the 126-day window): the median is taken over WHATEVER prior history exists,
i.e. the most recent up-to-126 prior non-null VRP observations. No history is
fabricated - the series starts empty today and grows one row per run.
  - >=1 prior VRP  -> gate operates normally on the median of what exists.
  - 0 prior VRP    -> the median is undefined; the gate defaults OFF (the harness
                      does NOT short vol with zero evidence the premium is above
                      its own median). `warmup=True` until 126 priors accrue.
This is FORWARD-ONLY (RL-15): expired option contracts are unresolvable, so
nothing here backtests - the leg is LIVE and logging the gated decision + daily
mark from day one, exactly like RL-15/RL-18. Marks use leg LTP (optimistic,
disclosed as `mark_basis="ltp"`). Every Groww call routes through
`groww_client.call`, which rate-limits and refuses order methods; read-only.

All Groww-touching and P&L arithmetic is reused from `paper_options` so this book
stays bit-for-bit consistent with RL-18's construction; only the gate is new.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from . import paper_options as po
from .tracking import log_run

SNAPSHOT_PATH = "experiments/paper_options_vrp.jsonl"
BOOK = "vrp_gated_straddle"
VRP_WINDOW = 126                      # trailing collection-day median window
READ_METHODS = po.READ_METHODS       # gate + marks dispatch only these read-only methods


def _rows(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def current_atm_iv(today) -> tuple[float | None, str | None]:
    """Today's nearest-weekly-expiry ATM implied vol (CE/PE greeks-IV mean) - the
    gate's IV input, available every day whether or not a position is open. Returns
    (atm_iv, note); atm_iv is None with a note on any data gap (never raises)."""
    try:
        expiry = po.nearest_expiry(today)
        if expiry is None:
            return None, f"no_expiry_ge_{po.MIN_DTE}dte"
        oc = po.option_chain(expiry)
    except Exception as e:
        return None, f"iv_failed: {type(e).__name__}: {e}"
    spot = po._num(oc.get("underlying_ltp"))
    if spot is None:
        return None, "chain_empty"
    _, _, _, _, iv = po.chain_legs(oc, spot)     # nearest strike to spot = ATM
    return (iv, None) if iv is not None else (None, "no_atm_iv")


def _gate(vrp: float | None, window: list[float]) -> tuple[bool | None, float | None, int]:
    """(gate_on, median, n) from today's VRP and the trailing prior-VRP window.
    gate_on is True iff VRP strictly exceeds the median of PRIOR observations.
    Zero prior history -> median undefined, gate defaults OFF (warm-up rule).
    VRP itself unavailable (data gap) -> gate_on None (undecidable this day)."""
    n = len(window)
    median = statistics.median(window) if window else None
    if vrp is None:
        return None, median, n
    if median is None:
        return False, None, n
    return vrp > median, median, n


def _flat(cum_prev: float) -> dict:
    """No position this cycle: null the position block, carry cumulative, zero daily."""
    return {**po._null_position(), "action": "flat", "daily_pnl": 0.0,
            "cumulative_pnl": round(cum_prev, 2), "settled": None}


def _close(prev: dict, today, cum_prev: float) -> tuple[dict, str]:
    """Gate turned OFF while holding: flatten and stay flat. Realize at intrinsic
    at/after expiry, else at leg LTP - RL-18's close arithmetic, minus the reopen."""
    expiry = po._parse_date(prev["expiry"])
    dte = (expiry - today).days if expiry else None
    strike, lot, prev_mark = prev["strike"], prev["lot_size"], prev["mark_value"]

    if expiry is not None and today >= expiry:
        spot, serr = po._settle_spot(expiry)
        if spot is None:
            return po._carry(prev, dte), f"gateoff_settle_no_spot: {serr}"
        close_val, basis, extra = abs(spot - strike), "intrinsic", {"settle_spot": round(spot, 2)}
    else:
        try:
            _, _, ce, pe, _ = po.chain_legs(po.option_chain(expiry), strike)
        except Exception as e:
            return po._carry(prev, dte), f"gateoff_close_failed: {type(e).__name__}: {e}"
        if ce is None or pe is None:
            return po._carry(prev, dte), "gateoff_close_missing_ltp"
        close_val, basis, extra = round(ce + pe, 4), "ltp", {}

    realized = (prev_mark - close_val) * lot
    settled = {"strike": strike, "expiry": prev["expiry"], "basis": basis, **extra,
               "prev_mark": prev_mark, "close_value": round(close_val, 4),
               "realized_pnl": round(realized, 2)}
    return {**po._null_position(), "action": "gate_close", "daily_pnl": round(realized, 2),
            "cumulative_pnl": round(cum_prev + realized, 2), "settled": settled}, "ok"


def _act(prev: dict | None, today, cum_prev: float, has_pos: bool,
         gate_on: bool | None) -> tuple[dict, str]:
    if gate_on is None:                              # VRP undecidable (data gap)
        if has_pos:
            expiry = po._parse_date(prev["expiry"])
            return po._carry(prev, (expiry - today).days if expiry else None), "vrp_gap_carry"
        return _flat(cum_prev), "vrp_gap_flat"
    if gate_on:                                      # premium fat: RL-18 lifecycle
        return (po._mark_or_roll(prev, today, cum_prev) if has_pos
                else po._try_open(today, cum_prev))
    if has_pos:                                      # premium thin: flatten
        return _close(prev, today, cum_prev)
    return _flat(cum_prev), "gate_off_flat"          # premium thin, already flat


def snapshot(today=None, path: str = SNAPSHOT_PATH, write: bool = True,
             refresh: bool = False, verbose: bool = True) -> dict:
    today = today or datetime.now(po.IST).date()
    rows = _rows(path)
    prev = rows[-1] if rows else None
    cum_prev = float((prev or {}).get("cumulative_pnl") or 0.0)
    has_pos = bool(prev and prev.get("strike") is not None)

    atm_iv, iv_note = current_atm_iv(today)
    rv5d = po.realized_vol_5d(refresh=refresh)
    vrp = atm_iv - rv5d if (atm_iv is not None and rv5d is not None) else None

    window = [r["vrp"] for r in rows if r.get("vrp") is not None][-VRP_WINDOW:]
    gate_on, median, n_hist = _gate(vrp, window)

    fields, note = _act(prev, today, cum_prev, has_pos, gate_on)
    if vrp is None and iv_note:
        note = f"vrp_unavailable: {iv_note}" if note == "ok" else f"{note}; vrp_unavailable: {iv_note}"

    row = {
        "hypothesis_ref": "RL-2026-07-26-06", "book": BOOK,
        "kind": "paper_options_snapshot",
        "asof_ist": datetime.now(po.IST).strftime("%Y-%m-%d %H:%M:%S"),
        "run_date": today.isoformat(), "underlying": po.NIFTY, "mark_basis": "ltp",
        "realized_vol_5d": rv5d, "atm_iv_gate": po._r(atm_iv), "vrp": po._r(vrp),
        "vrp_median": po._r(median), "n_vrp_hist": n_hist, "gate_on": gate_on,
        "warmup": n_hist < VRP_WINDOW, **fields, "note": note,
    }
    if write:
        log_run(row, path=path)
    if verbose:
        _print_summary(row, path, write)
    return row


def _print_summary(row: dict, path: str, write: bool) -> None:
    print(f"[VRP-gated options] {row['run_date']}  action={row['action']}  "
          f"gate_on={row['gate_on']}  underlying={row['underlying']}")
    print(f"  VRP={row['vrp']} (ATM_IV {row['atm_iv_gate']} - RV5d {row['realized_vol_5d']})  "
          f"median={row['vrp_median']}  n_hist={row['n_vrp_hist']}  warmup={row['warmup']}")
    if row["strike"] is not None:
        print(f"  SHORT straddle  expiry={row['expiry']}  strike={row['strike']}  "
              f"lot={row['lot_size']}  dte={row['dte']}  (entered {row['entry_date']})")
        print(f"  entry credit={row['entry_credit']}  spot={row['spot']}  "
              f"mark={row['mark_value']} (CE {row['ce_ltp']} + PE {row['pe_ltp']})")
    else:
        print("  NO open position (flat)")
    if row.get("settled"):
        s = row["settled"]
        print(f"  closed {s['expiry']} @ {s['strike']}: close({s['basis']})={s['close_value']} "
              f"vs mark {s['prev_mark']} -> realized {s['realized_pnl']}")
    print(f"  daily P&L={row['daily_pnl']}  cumulative P&L={row['cumulative_pnl']}  "
          f"mark_basis={row['mark_basis']}")
    print(f"  note: {row['note']}")
    print(f"  snapshot {'appended to ' + path if write else 'NOT written (dry run)'}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Forward-only VRP-gated PAPER NIFTY short-straddle harness")
    p.add_argument("--dry-run", action="store_true", help="print but do not append a row")
    p.add_argument("--refresh", action="store_true",
                   help="refresh the ^NSEI Yahoo cache before computing realized vol")
    p.add_argument("--path", default=SNAPSHOT_PATH)
    a = p.parse_args()
    snapshot(path=a.path, write=not a.dry_run, refresh=a.refresh)


if __name__ == "__main__":
    main()
