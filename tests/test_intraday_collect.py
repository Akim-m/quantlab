"""Tests for the intraday 5-minute bar archive. No network, no real auth.

Load-bearing safety property: this collector fetches market data only and can NEVER
reach an order method - proven by spying on the groww_client dispatcher and asserting
every routed method is read-only. The rest pins the incremental-window paging, the
dedupe-and-sort merge, and graceful degradation (one bad symbol never stops the run).
"""

import csv
import json
from datetime import datetime, timedelta

import pytest

from quantlab import groww_client as gc
from quantlab import intraday_collect as ic

IST = ic.IST
NOW = datetime(2026, 7, 9, 16, 0, tzinfo=IST)


# ---- incremental window paging (last archived session -> <=14d windows to now) ----

def test_paged_windows_cover_and_cap():
    start = datetime(2026, 4, 1, tzinfo=IST)
    end = datetime(2026, 7, 1, tzinfo=IST)          # 91 days
    wins = ic.paged_windows(start, end)
    assert wins[0][0] == start and wins[-1][1] == end      # exact coverage
    assert all(b == c for (_, b), (c, _) in zip(wins, wins[1:]))   # contiguous
    assert all((b - a) <= timedelta(days=14) for a, b in wins)     # under the 15d cap
    assert len(wins) == 7                                   # ceil(91/14)
    assert ic.paged_windows(end, end) == []                 # already current
    assert ic.paged_windows(end, start) == []               # start after end


def test_last_archived_drives_incremental_start(tmp_path):
    fp = tmp_path / "X.csv"
    assert ic.last_archived(fp) is None                     # absent -> caller uses lookback
    ic.merge_bars(fp, [_bar("2026-07-07 09:15:00"), _bar("2026-07-07 15:25:00")])
    dt = ic.last_archived(fp)
    assert dt == datetime(2026, 7, 7, 15, 25, tzinfo=IST)   # final (max) row, IST-aware
    wins = ic.paged_windows(dt, datetime(2026, 7, 20, 16, 0, tzinfo=IST))
    assert wins[0][0] == dt                                 # resumes exactly where it stopped


# ---- dedupe-and-sort on overlapping appends ----

def _bar(ts, close=1.0, vol=10):
    return {"timestamp": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": close, "volume": vol}


def test_merge_dedupes_and_sorts(tmp_path):
    fp = tmp_path / "X.csv"
    n1 = ic.merge_bars(fp, [_bar("2026-07-07 09:20:00"), _bar("2026-07-07 09:15:00")])
    assert n1 == 2
    # overlapping append: one duplicate timestamp (new value) + one net-new, out of order
    n2 = ic.merge_bars(fp, [_bar("2026-07-07 09:20:00", close=9.0),
                            _bar("2026-07-07 09:25:00")])
    assert n2 == 1                                          # only 09:25 is net-new
    rows = list(csv.DictReader(fp.open(encoding="utf-8")))
    assert [r["timestamp"] for r in rows] == [
        "2026-07-07 09:15:00", "2026-07-07 09:20:00", "2026-07-07 09:25:00"]   # sorted
    assert rows[1]["close"] == "9.0"                        # duplicate overwritten, not doubled


def test_merge_write_false_counts_without_touching_disk(tmp_path):
    fp = tmp_path / "X.csv"
    assert ic.merge_bars(fp, [_bar("2026-07-07 09:15:00")], write=False) == 1
    assert not fp.exists()


# ---- read-only spy + graceful degradation end-to-end ----

class Spy:
    """Records every method routed through gc.call; serves canned 5-min candles.
    Raises for any symbol in `fail`, to prove one bad symbol never stops the run."""

    def __init__(self, fail=()):
        self.methods = []
        self.fail = set(fail)

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:
            raise PermissionError(method)
        if method != "get_historical_candle_data":
            raise AssertionError(f"unexpected method: {method}")
        sym = kwargs["trading_symbol"]
        if sym in self.fail:
            raise RuntimeError("no data for symbol")
        vol = None if sym == "NIFTY" else 1000            # index bars carry no volume
        return {"candles": [[1783395900, 1.0, 2.0, 0.5, 1.5, vol],
                            [1783396200, 1.5, 2.5, 1.0, 2.0, vol]],
                "interval_in_minutes": 5}


def test_collect_read_only_and_degrades(tmp_path, monkeypatch):
    spy = Spy(fail={"BADSYM"})
    monkeypatch.setattr(gc, "call", spy)
    universe = ("NIFTY", "RELIANCE", "BADSYM")
    out = tmp_path / "audit.jsonl"
    row = ic.collect(now=NOW, universe=universe, store_dir=tmp_path,
                     write=True, path=str(out), verbose=False)

    # SAFETY: only the declared read method is dispatched, never an order method.
    assert spy.methods and set(spy.methods) <= set(ic.READ_METHODS)
    assert not any(m in gc._ORDER_METHODS for m in spy.methods)

    # graceful degradation: BADSYM failed across all its windows, the rest archived.
    assert row["n_symbols"] == 3 and row["n_ok"] == 2 and row["n_failed"] == 1
    assert row["failed"] == ["BADSYM"] and row["note"].startswith("failed=BADSYM")

    # cross-window dedupe: each good symbol yields 2 net-new bars, not 2*n_windows.
    assert row["n_new_bars"] == 4
    assert row["first_session"] == "2026-07-07" and row["last_session"] == "2026-07-07"

    nifty = (tmp_path / "NIFTY.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(nifty) == 3                                  # header + 2 deduped bars
    assert nifty[1].endswith(",")                           # index volume blank
    rel = list(csv.DictReader((tmp_path / "RELIANCE.csv").open(encoding="utf-8")))
    assert rel[0]["volume"] == "1000"
    assert not (tmp_path / "BADSYM.csv").exists()           # nothing written for the failure

    rec = json.loads(out.read_text(encoding="utf-8").strip())
    assert rec["hypothesis_ref"] == "RL-2026-07-25" and rec["kind"] == "intraday_archive"
    for k in ("run_date", "n_symbols", "n_ok", "n_failed", "n_new_bars",
              "first_session", "last_session"):
        assert k in rec


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "call", Spy())
    out = tmp_path / "audit.jsonl"
    row = ic.collect(now=NOW, universe=("NIFTY",), store_dir=tmp_path,
                     write=False, path=str(out), verbose=False)
    assert not out.exists() and not (tmp_path / "NIFTY.csv").exists()
    assert row["n_new_bars"] == 2                           # counted without writing


def test_order_methods_refused_by_dispatcher():
    """The collector's only channel to Groww refuses order methods before any network."""
    with pytest.raises(PermissionError):
        gc.call("place_order")
