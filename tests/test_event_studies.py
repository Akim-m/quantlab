"""Tests for the forward event collectors (JUMP-MOM + SSF-LIST).

No network: every test builds synthetic panels / ledgers. Coverage pins the jump
threshold at the 3-sigma boundary, proves sigma uses strictly pre-event data (no
look-ahead), the 21-day blackout dedupe, the volume-confirm flag, the F&O set-diff
(add/remove/baseline), the abnormal-return arithmetic against a hand-computed case,
that the peer basket excludes the event stock, and that maturity measurement is
idempotent.
"""

import json

import pandas as pd
import pytest

from quantlab import event_studies as es


# ---- synthetic-panel builder ----

def frame(cols_rets: dict, vols: dict | None = None, start="2019-01-01"):
    """(px, volume) from per-column daily-return arrays. Price base 100 at position 0;
    position t (t>=1) has return cols_rets[col][t-1]. Index = business days. `vols` gives
    a full-length (len+1) volume array per column; default all ones."""
    n = len(next(iter(cols_rets.values())))
    idx = pd.bdate_range(start, periods=n + 1)
    px = {}
    for c, rets in cols_rets.items():
        p = [100.0]
        for r in rets:
            p.append(p[-1] * (1.0 + r))
        px[c] = p
    px = pd.DataFrame(px, index=idx)
    if vols is None:
        vol = pd.DataFrame(1.0, index=idx, columns=list(cols_rets))
    else:
        vol = pd.DataFrame(vols, index=idx)
    return px, vol


IND = {"X": "IND1", "A.NS": "IND1", "B.NS": "IND1", "C.NS": "IND1", "D.NS": "IND2"}


# ---- jump detection: 3-sigma boundary + no look-ahead ----

def test_jump_fires_at_and_above_3sigma_not_below():
    pre = [0.01, -0.02, 0.015, -0.005, 0.02]          # 5 pre-event returns
    sigma = pd.Series(pre).std(ddof=1)
    # HIT event just above 3-sigma, MISS just below -> same sigma, boundary is `>=`.
    px, vol = frame({"HIT": pre + [3.0 * sigma * (1 + 1e-6)],
                     "MISS": pre + [3.0 * sigma * (1 - 1e-6)]})
    ev = es.detect_jumps(px, vol, {"HIT": "I", "MISS": "I"}, sigma_win=5)
    hit = {e["symbol"] for e in ev}
    assert "HIT" in hit and "MISS" not in hit
    row = next(e for e in ev if e["symbol"] == "HIT")
    assert row["z"] == pytest.approx(3.0, abs=1e-3) and row["direction"] == "up"
    assert row["sigma63"] == round(sigma, 6)


def test_sigma_excludes_event_day():
    """A calm pre-window then a large jump: the jump clears 3-sigma ONLY because sigma is
    computed pre-event. If the event day leaked into its own sigma, sigma would inflate
    and z would fall below 3 -> the detector must still fire."""
    pre = [0.004, -0.004, 0.004, -0.004, 0.004]
    sigma_pre = pd.Series(pre).std(ddof=1)
    jump = 3.05 * sigma_pre
    sigma_incl = pd.Series(pre + [jump]).std(ddof=1)
    assert jump / sigma_incl < 3.0                    # the leak WOULD flip the verdict
    px, vol = frame({"X": pre + [jump]})
    ev = es.detect_jumps(px, vol, IND, sigma_win=5)
    assert {e["symbol"] for e in ev} == {"X"}         # fired -> pre-event sigma only


def test_down_jump_direction():
    pre = [0.01, -0.01, 0.008, -0.006, 0.009]
    sigma = pd.Series(pre).std(ddof=1)
    px, vol = frame({"X": pre + [-3.2 * sigma]})
    ev = es.detect_jumps(px, vol, IND, sigma_win=5)
    assert ev[0]["direction"] == "down" and ev[0]["z"] < 0


# ---- blackout dedupe ----

def test_blackout_suppresses_within_window_allows_beyond():
    pre = [0.006, -0.006, 0.006, -0.006, 0.006]
    sigma = pd.Series(pre).std(ddof=1)
    noise = [0.001, -0.001] * 12                       # long calm tail before the event
    rets = noise + pre + [3.3 * sigma]
    px, vol = frame({"X": rets})
    ed = px.index[-1]
    within = {"X": px.index[-1 - 20]}                  # 20 trading days back -> blocked
    beyond = {"X": px.index[-1 - 21]}                  # 21 trading days back -> allowed
    assert es.detect_jumps(px, vol, IND, last_event=within, sigma_win=5) == []
    assert {e["symbol"] for e in es.detect_jumps(px, vol, IND, last_event=beyond,
                                                 sigma_win=5)} == {"X"}
    assert es._in_blackout(px, ed, ed, blackout=21) is True   # same-day re-detect blocked


# ---- volume-confirm flag ----

def test_volume_confirm_flag():
    pre = [0.01, -0.01, 0.01, -0.01, 0.01]
    sigma = pd.Series(pre).std(ddof=1)
    rets = pre + [3.5 * sigma]                          # 6 returns -> index length 7
    window = [10.0] * 6                                 # positions 0..5 (window is 1..5)
    px, vol_hi = frame({"X": rets}, vols={"X": window + [1000.0]})  # event vol >> p95
    px, vol_lo = frame({"X": rets}, vols={"X": window + [1.0]})     # event vol << p95
    assert es.detect_jumps(px, vol_hi, IND, sigma_win=5, vol_win=5)[0]["vol_confirmed"] is True
    assert es.detect_jumps(px, vol_lo, IND, sigma_win=5, vol_win=5)[0]["vol_confirmed"] is False


# ---- SSF set-diff: add / remove / baseline ----

def test_diff_universe_cases():
    assert es.diff_universe(None, {"A", "B"}) == ([], [])                 # baseline
    assert es.diff_universe({"A", "B"}, {"A", "B"}) == ([], [])           # no change
    assert es.diff_universe({"A", "B"}, {"A", "B", "C"}) == (["C"], [])   # add
    assert es.diff_universe({"A", "B"}, {"A"}) == ([], ["B"])             # remove
    assert es.diff_universe({"A", "B"}, {"A", "C"}) == (["C"], ["B"])     # add + remove


def _fno_row(collect_date, keys):
    return {"kind": "fno_daily_snapshot", "collect_date": collect_date,
            "basis": {k: {"cash": 1.0} for k in keys}}


def test_ssf_baseline_then_change(tmp_path):
    fno = tmp_path / "fno.jsonl"
    out = tmp_path / "ssf.jsonl"

    # one snapshot only -> baseline, no events
    fno.write_text(json.dumps(_fno_row("2026-07-10", ["ABB", "TCS"])) + "\n", encoding="utf-8")
    a = es.collect_ssf_changes(fno_path=str(fno), path=str(out), write=True, verbose=False)
    assert a["note"] == "baseline" and a["n_added"] == 0 and a["n_removed"] == 0
    assert es._read_ledger(out) and es._read_ledger(out)[-1]["kind"] == "audit"

    # second snapshot: TCS out, ZOMATO in
    with fno.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_fno_row("2026-07-11", ["ABB", "ZOMATO"])) + "\n")
    out2 = tmp_path / "ssf2.jsonl"
    a2 = es.collect_ssf_changes(fno_path=str(fno), path=str(out2), write=True, verbose=False)
    rows = es._read_ledger(out2)
    changes = {(r["symbol"], r["change"]) for r in rows if r["kind"] == "ssf_change"}
    assert changes == {("ZOMATO", "added"), ("TCS", "removed")}
    assert all(r["event_date"] == "2026-07-11" for r in rows if r["kind"] == "ssf_change")
    assert a2["n_added"] == 1 and a2["n_removed"] == 1 and a2["n_universe"] == 2
    for r in rows:
        if r["kind"] == "ssf_change":
            assert r["hypothesis_ref"] == es.SSF_REF


# ---- abnormal return: hand-computed + peer basket excludes self ----

def _mature_px():
    idx = pd.bdate_range("2026-01-01", periods=6)
    # only positions i=1 and j=3 carry the load-bearing prices
    return pd.DataFrame({
        "A.NS": [50, 100, 60, 110, 70, 80],    # i->j: 100 -> 110  = +0.10
        "B.NS": [90, 100, 95, 105, 96, 97],    #        100 -> 105  = +0.05
        "C.NS": [80, 100, 90, 103, 91, 92],    #        100 -> 103  = +0.03
        "D.NS": [10, 100, 20, 999, 30, 40],    # IND2 peer, must be excluded
    }, index=idx)


def test_abnormal_return_hand_computed_excludes_self_and_other_industry():
    px = _mature_px()
    abn, n = es.abnormal_return(px, IND, "A.NS", i=1, horizon=2)
    # peers = B,C (same IND1, self excluded, D in IND2 excluded); mean(0.05,0.03)=0.04
    assert n == 2 and abn == pytest.approx(0.10 - 0.04, rel=1e-9)


def test_abnormal_return_no_peers_returns_none():
    px = _mature_px()
    abn, n = es.abnormal_return(px, IND, "D.NS", i=1, horizon=2)   # only D in IND2
    assert abn is None and n == 0


# ---- maturity measurement: idempotent, respects horizon ----

def _seed_event(path, symbol, event_date, kind="jump"):
    es.log_run({"hypothesis_ref": es.JUMP_REF, "kind": kind,
                "event_date": event_date, "symbol": symbol}, path=str(path))


def test_measure_matured_idempotent_and_horizon(tmp_path):
    px = _mature_px()
    led = tmp_path / "led.jsonl"
    matured_date = px.index[1 + 2].date().isoformat()
    _seed_event(led, "A.NS", px.index[1].date().isoformat())    # matured (i=1, i+2=3 <= 5)
    _seed_event(led, "B.NS", px.index[-1].date().isoformat())   # too recent -> not matured

    r1 = es.measure_matured(str(led), px=px, industries=IND, horizon=2, write=True, verbose=False)
    assert len(r1) == 1 and r1[0]["symbol"] == "A.NS"
    assert r1[0]["abn_21d"] == pytest.approx(0.06, rel=1e-9)
    assert r1[0]["n_peers"] == 2 and r1[0]["matured_date"] == matured_date

    r2 = es.measure_matured(str(led), px=px, industries=IND, horizon=2, write=True, verbose=False)
    assert r2 == []                                             # nothing re-measured
    matured = [r for r in es._read_ledger(led) if r["kind"] == "matured"]
    assert len(matured) == 1                                    # only one matured row on disk


def test_measure_matured_reads_ssf_events(tmp_path):
    px = _mature_px()
    led = tmp_path / "ssf_led.jsonl"
    _seed_event(led, "A", px.index[1].date().isoformat(), kind="ssf_change")  # bare symbol
    r = es.measure_matured(str(led), px=px, industries=IND, ref=es.SSF_REF,
                           horizon=2, write=True, verbose=False)
    assert len(r) == 1 and r[0]["symbol"] == "A"               # 'A' -> 'A.NS' resolved
    assert r[0]["abn_21d"] == pytest.approx(0.06, rel=1e-9)


# ---- never-crash discipline ----

def test_collect_jumps_bad_panel_writes_error_audit(tmp_path):
    out = tmp_path / "j.jsonl"
    bad = pd.DataFrame({"X": [1.0, 2.0]})                       # no usable history
    audit = es.collect_jumps(px=bad, volume=bad, industries={}, path=str(out),
                             write=True, verbose=False)
    assert audit["kind"] == "audit" and audit["n_events"] == 0
    assert es._read_ledger(out)[-1]["kind"] == "audit"
