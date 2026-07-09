"""Daily read-only F&O forward-collector (RL-2026-07-15).

FORWARD-ONLY. Live F&O contracts serve data only over their ~3-month life and
expired contracts are not resolvable, so basis / PCR / IV cannot be reconstructed
historically - this collector starts the dataset now and appends ONE row per
run-day to `experiments/fno_daily.jsonl`. The pre-registered forward hypotheses
(H1 basis cross-section, H2 NIFTY PCR extremes, H3 IV skew) get their first read
only after >=126 collection days; nothing here backtests or reads a test window.

READ-ONLY: every Groww call goes through `groww_client.call`, which rate-limits to
<=7 req/s and refuses order methods. A missing quote or a down API degrades to a
partial row with a `note`; a run never crashes on absent data.

Collected daily:
  - Basis: per F&O single-stock underlying (~210) - cash LTP + current- and
    next-month futures LTP -> annualized basis b = (fut/cash - 1) * (365/dte).
  - NIFTY chain (nearest expiry): OI put-call ratio, ATM IV, and a fixed-moneyness
    skew = IV(put ~0.95*spot) - IV(call ~1.05*spot).
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

from . import groww_client as gc
from .india import fno_shortable, groww_instruments
from .tracking import log_run

IST = timezone(timedelta(hours=5, minutes=30))
SNAPSHOT_PATH = "experiments/fno_daily.jsonl"
LTP_BATCH = 50                     # Groww caps get_ltp at 50 symbols/call
NIFTY = "NIFTY"                    # index underlying for the option-chain block
# read-only Groww methods this collector routes through gc.call (the method-spy
# test proves only these are ever dispatched; gc.call itself refuses order methods).
READ_METHODS = ("get_ltp", "get_expiries", "get_option_chain", "get_all_instruments")


def _num(x) -> float | None:
    """Coerce a Groww payload value to float, or None if it isn't numeric."""
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


def _parse_date(s) -> date | None:
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def annualized_basis(fut: float | None, cash: float | None, dte: int | None) -> float | None:
    """(fut/cash - 1) * 365/dte. None when any input is missing or dte<=0/cash<=0."""
    if fut is None or cash is None or cash <= 0 or dte is None or dte <= 0:
        return None
    return (fut / cash - 1.0) * (365.0 / dte)


def _ltp(symbols: list[str], segment: str) -> tuple[dict[str, float], str | None]:
    """Batched read-only Groww LTP (<=50/call). Returns (prices, first_error_or_None);
    never raises so the caller records a partial row instead of crashing. A failing
    batch (e.g. a transient Bad Request) is skipped, NOT fatal - later batches still
    run, so one blip never wipes the whole segment's coverage."""
    out: dict[str, float] = {}
    err: str | None = None
    for i in range(0, len(symbols), LTP_BATCH):
        chunk = symbols[i:i + LTP_BATCH]
        try:
            payload = gc.call("get_ltp", exchange_trading_symbols=tuple(chunk), segment=segment)
        except Exception as e:                # auth/entitlement/network/rate - all non-fatal
            err = err or f"{type(e).__name__}: {e}"
            continue
        for s in chunk:
            if (v := _num(payload.get(s))) is not None:
                out[s] = v
    return out, err


def futures_by_underlying(instruments, today: date) -> dict[str, list[tuple[date, str]]]:
    """For each single-stock underlying, the two nearest not-yet-expired NSE futures
    as (expiry_date, trading_symbol), sorted by expiry."""
    fut = instruments[(instruments["instrument_type"] == "FUT")
                      & (instruments["segment"] == "FNO")
                      & (instruments["exchange"] == "NSE")]
    out: dict[str, list[tuple[date, str]]] = {}
    for u, grp in fut.groupby("underlying_symbol"):
        rows = [(d, str(ts)) for ts, ex in zip(grp["trading_symbol"], grp["expiry_date"])
                if (d := _parse_date(ex)) is not None and d >= today]
        rows.sort()
        out[str(u)] = rows[:2]
    return out


def basis_block(underlyings: list[str], fut_by_u: dict[str, list[tuple[date, str]]],
                today: date) -> tuple[dict, dict, str | None]:
    """Cash + fut1/fut2 LTP and annualized basis per underlying. Returns
    (basis_dict, coverage_counters, note)."""
    cash_syms = [f"NSE_{u}" for u in underlyings]
    fut_syms = [f"NSE_{ts}" for u in underlyings for _, ts in fut_by_u.get(u, [])]
    cash_px, cash_err = _ltp(cash_syms, "CASH")
    fut_px, fut_err = _ltp(fut_syms, "FNO")

    basis: dict[str, dict] = {}
    n_cash = n_f1 = n_f2 = 0
    for u in underlyings:
        cash = cash_px.get(f"NSE_{u}")
        e = {"cash": cash, "fut1": None, "fut2": None,
             "dte1": None, "dte2": None, "b1": None, "b2": None}
        n_cash += cash is not None
        futs = fut_by_u.get(u, [])
        if futs:
            d1, ts1 = futs[0]
            e["fut1"] = fut_px.get(f"NSE_{ts1}")
            e["dte1"] = (d1 - today).days
            e["b1"] = annualized_basis(e["fut1"], cash, e["dte1"])
            n_f1 += e["fut1"] is not None
        if len(futs) > 1:
            d2, ts2 = futs[1]
            e["fut2"] = fut_px.get(f"NSE_{ts2}")
            e["dte2"] = (d2 - today).days
            e["b2"] = annualized_basis(e["fut2"], cash, e["dte2"])
            n_f2 += e["fut2"] is not None
        basis[u] = e

    counters = {"n_underlyings": len(underlyings), "n_cash_ok": n_cash,
                "n_fut1_ok": n_f1, "n_fut2_ok": n_f2}
    note = "; ".join(m for m in (cash_err and f"cash:{cash_err}",
                                 fut_err and f"fut:{fut_err}") if m) or None
    return basis, counters, note


def _iv(leg: dict) -> float | None:
    """Implied vol (percent) from a chain leg's greeks; 0/absent -> None."""
    iv = _num((leg.get("greeks") or {}).get("iv")) if isinstance(leg, dict) else None
    return iv if iv and iv > 0 else None


def chain_metrics(oc: dict) -> tuple[dict, str | None]:
    """Extract PCR (sum PE OI / sum CE OI), ATM IV (strike nearest spot, CE/PE mean),
    and fixed-moneyness skew = IV(put~0.95*spot) - IV(call~1.05*spot) from a NIFTY
    option-chain payload. Null-safe throughout."""
    spot = _num(oc.get("underlying_ltp"))
    strikes = {k: v for k, v in ((_num(k), v) for k, v in (oc.get("strikes") or {}).items())
               if k is not None and isinstance(v, dict)}
    out = {"spot": spot, "pcr": None, "atm_iv": None, "skew": None,
           "atm_strike": None, "n_strikes": len(strikes)}
    if not strikes or spot is None:
        return out, "chain_empty"

    ce_oi = sum(_num((strikes[k].get("CE") or {}).get("open_interest")) or 0.0 for k in strikes)
    pe_oi = sum(_num((strikes[k].get("PE") or {}).get("open_interest")) or 0.0 for k in strikes)
    out["pcr"] = pe_oi / ce_oi if ce_oi > 0 else None

    atm = min(strikes, key=lambda k: abs(k - spot))
    out["atm_strike"] = atm
    ivs = [iv for iv in (_iv(strikes[atm].get("CE", {})), _iv(strikes[atm].get("PE", {})))
           if iv is not None]
    out["atm_iv"] = sum(ivs) / len(ivs) if ivs else None

    put_iv = _iv(strikes[min(strikes, key=lambda k: abs(k - 0.95 * spot))].get("PE", {}))
    call_iv = _iv(strikes[min(strikes, key=lambda k: abs(k - 1.05 * spot))].get("CE", {}))
    out["skew"] = put_iv - call_iv if put_iv is not None and call_iv is not None else None
    return out, None


def nifty_chain_block(today: date) -> tuple[dict, bool, str | None]:
    """Nearest not-yet-expired NIFTY expiry -> option-chain metrics. Never raises."""
    try:
        exp = gc.call("get_expiries", exchange="NSE", underlying_symbol=NIFTY).get("expiries", [])
        future = sorted(e for e in exp if str(e) >= today.isoformat())
        if not future:
            return {"expiry": None, "spot": None, "pcr": None, "atm_iv": None,
                    "skew": None, "atm_strike": None, "n_strikes": 0}, False, "no_future_expiry"
        expiry = str(future[0])
        oc = gc.call("get_option_chain", exchange="NSE", underlying=NIFTY, expiry_date=expiry)
    except Exception as e:
        return {"expiry": None, "spot": None, "pcr": None, "atm_iv": None,
                "skew": None, "atm_strike": None, "n_strikes": 0}, False, f"{type(e).__name__}: {e}"
    m, note = chain_metrics(oc)
    m["expiry"] = expiry
    ok = note is None and m["pcr"] is not None
    return m, ok, note


def collect(today: date | None = None, refresh: bool = False, write: bool = True,
            path: str = SNAPSHOT_PATH, verbose: bool = True) -> dict:
    today = today or datetime.now(IST).date()
    notes: list[str] = []

    try:
        instruments = groww_instruments(refresh=refresh)
        underlyings = sorted(s.removesuffix(".NS") for s in fno_shortable(instruments))
        fut_by_u = futures_by_underlying(instruments, today)
        basis, counters, bnote = basis_block(underlyings, fut_by_u, today)
    except Exception as e:
        basis, bnote = {}, f"basis_error: {type(e).__name__}: {e}"
        counters = {"n_underlyings": 0, "n_cash_ok": 0, "n_fut1_ok": 0, "n_fut2_ok": 0}
    if bnote:
        notes.append(bnote)

    chain, chain_ok, cnote = nifty_chain_block(today)
    if cnote:
        notes.append(f"chain:{cnote}")

    row = {
        "hypothesis_ref": "RL-2026-07-15", "kind": "fno_daily_snapshot",
        "asof_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "collect_date": today.isoformat(),
        **counters, "chain_ok": chain_ok,
        "nifty_expiry": chain["expiry"], "nifty_spot": chain["spot"],
        "pcr": _round(chain["pcr"], 4), "atm_iv": _round(chain["atm_iv"], 4),
        "skew": _round(chain["skew"], 4), "atm_strike": chain["atm_strike"],
        "note": "; ".join(notes) or "ok",
        "basis": {u: _round_entry(e) for u, e in basis.items()},
    }
    if write:
        log_run(row, path=path)
    if verbose:
        _print_summary(row, path, write)
    return row


def _round(x: float | None, n: int) -> float | None:
    return None if x is None else round(x, n)


def _round_entry(e: dict) -> dict:
    return {k: (_round(v, 4) if isinstance(v, float) else v) for k, v in e.items()}


def _print_summary(row: dict, path: str, write: bool) -> None:
    print(f"[F&O collect] {row['collect_date']}  underlyings={row['n_underlyings']}  "
          f"cash_ok={row['n_cash_ok']}  fut1_ok={row['n_fut1_ok']}  fut2_ok={row['n_fut2_ok']}")
    print(f"NIFTY chain: expiry={row['nifty_expiry']}  spot={row['nifty_spot']}  "
          f"PCR={row['pcr']}  ATM_IV={row['atm_iv']}  skew={row['skew']}  ok={row['chain_ok']}")
    sample = list(row["basis"].items())[:3]
    if sample:
        print("basis sample (underlying: cash fut1 fut2 dte1 dte2 b1 b2):")
        for u, e in sample:
            print(f"  {u:14s} {e['cash']!s:>9} {e['fut1']!s:>9} {e['fut2']!s:>9} "
                  f"{e['dte1']!s:>4} {e['dte2']!s:>4} {e['b1']!s:>9} {e['b2']!s:>9}")
    print(f"note: {row['note']}")
    print(f"snapshot {'appended to ' + path if write else 'NOT written (dry run)'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Daily read-only F&O basis/PCR/IV collector")
    p.add_argument("--refresh", action="store_true",
                   help="refresh the Groww instrument master before collecting")
    p.add_argument("--dry-run", action="store_true", help="print but do not append a row")
    p.add_argument("--path", default=SNAPSHOT_PATH)
    a = p.parse_args()
    collect(refresh=a.refresh, write=not a.dry_run, path=a.path)


if __name__ == "__main__":
    main()
