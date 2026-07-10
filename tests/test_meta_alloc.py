"""RL-2026-07-26-21 META-ALLOC: frozen maps, causal join, coverage, snapshot row."""
import json

import pandas as pd

from quantlab import meta_alloc as ma


def _days(n, start="2026-07-01"):
    return pd.bdate_range(start, periods=n)


# ---------------------------------------------------------------- frozen maps

def test_meta_maps_sum_to_one_with_cash():
    on, off = ma.alloc(True), ma.alloc(False)
    assert abs(sum(on.values()) - 1.0) < 1e-12
    assert abs(sum(off.values()) - 1.0) < 1e-12
    assert abs(off["CASH"] - 0.40) < 1e-12


def test_off_state_holds_no_directional():
    off = ma.alloc(False)
    assert all(s not in off for s in ma.DIRECTIONAL)


def test_frozen_map_values_pinned():
    # regression pin against tuning: these are the registered constants
    on = ma.alloc(True)
    assert on["regime"] == 0.50 and on["trend"] == 0.125 and on["dualrot"] == 0.075
    assert abs(on["ls"] - 0.30 / 7) < 1e-12
    st = ma.static_alloc()
    assert abs(sum(st.values()) - 1.0) < 1e-12 and st["CASH"] == 0.0
    assert abs(st["regime"] - 0.50 / 3) < 1e-12 and abs(st["deliv"] - 0.50 / 7) < 1e-12


# ---------------------------------------------------------------- causal join

def _row(day, meta, static=None):
    return {"panel_date": str(day.date()), "meta_alloc": meta,
            "static_alloc": static or ma.static_alloc()}


def test_combine_return_day_uses_strictly_prior_map():
    d = _days(3)
    # map flips 100% A -> 100% B on day 2; day-2's return must still be A's
    rows = [_row(d[0], {"A": 1.0, "CASH": 0.0}), _row(d[1], {"B": 1.0, "CASH": 0.0})]
    rets = {"A": pd.Series([0.10, 0.0], index=d[1:]), "B": pd.Series([0.0, 0.20], index=d[1:])}
    res = ma.combine(rows, rets)
    assert abs(res.loc[d[1], "meta"] - 0.10) < 1e-12   # day 2 earned A (map from day 1)
    assert abs(res.loc[d[2], "meta"] - 0.20) < 1e-12   # day 3 earned B (map from day 2)


def test_combine_no_return_before_first_map():
    d = _days(3)
    rows = [_row(d[1], {"A": 1.0, "CASH": 0.0})]
    rets = {"A": pd.Series([0.05, 0.05], index=[d[0], d[2]])}
    res = ma.combine(rows, rets)
    assert d[0] not in res.index and abs(res.loc[d[2], "meta"] - 0.05) < 1e-12


def test_combine_missing_sleeve_contributes_cash_and_coverage_reports_it():
    d = _days(2)
    rows = [_row(d[0], {"A": 0.6, "B": 0.4, "CASH": 0.0})]
    rets = {"A": pd.Series([0.10], index=[d[1]])}          # B unpriced that day
    res = ma.combine(rows, rets)
    assert abs(res.loc[d[1], "meta"] - 0.06) < 1e-12       # 0.6 * 10%, B -> cash
    assert abs(res.loc[d[1], "meta_coverage"] - 0.6) < 1e-12


def test_dedupe_keeps_last_row_per_panel_date():
    d = _days(1)[0]
    first = _row(d, {"A": 1.0, "CASH": 0.0})
    second = _row(d, {"B": 1.0, "CASH": 0.0})
    kept = ma._dedupe([first, second])
    assert len(kept) == 1 and kept[0]["meta_alloc"] == {"B": 1.0, "CASH": 0.0}


# ------------------------------------------------------------------- snapshot

def test_snapshot_row_schema_and_no_write(tmp_path):
    path = tmp_path / "meta.jsonl"
    row = ma.snapshot(write=False, path=str(path))
    assert not path.exists()
    assert row["hypothesis_ref"] == "RL-2026-07-26-21"
    assert row["regime_state"] in ("risk_on", "risk_off")
    assert abs(sum(row["meta_alloc"].values()) - 1.0) < 1e-12
    assert abs(sum(row["static_alloc"].values()) - 1.0) < 1e-12
    # the recorded map matches the recorded state (no hand-tweaked rows)
    assert row["meta_alloc"] == ma.alloc(row["regime_state"] == "risk_on")


def test_snapshot_writes_one_json_line(tmp_path):
    path = tmp_path / "meta.jsonl"
    ma.snapshot(write=True, path=str(path))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["kind"] == "meta_alloc_snapshot" and "git_commit" in rec
