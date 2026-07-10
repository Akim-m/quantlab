"""Tests for the RL-2026-07-26-20 NSE MTO delivery-data layer.

Fully synthetic - no network. The parser is exercised on an inline MTO snippet built
from real record-type-20 lines (including a BE-series row that must drop, a wrong-series
bond row, and two malformed rows), the bhavcopy reference parser on a compact real-shape
CSV, and the backfill/collector on a fake fetch that proves resumability, 404 handling,
and manifest bookkeeping without touching the network.
"""

import json

import pandas as pd
import pytest

from quantlab import nse_mto as m

# Real MTO record-type-20 lines. RELIANCE 60/100=0.60, TCS 50/200=0.25 are the only two
# EQ rows; SOMEBOND (GS) and SOMEBE (BE) are non-EQ; BADLINE has a non-numeric quantity
# and SHORTROW has too few fields - both malformed.
MTO_TEXT = """Security Wise Delivery Position - Compulsory Rolling Settlement
10,MTO,01072013,222652471,0001413
Trade Date <01-JUL-2013>,Settlement Type <N>,Settlement No <2013125>,Settlement Date <03-JUL-2013>
Record Type,Sr No,Name of Security,Quantity Traded,Deliverable Quantity(gross across client level),% of Deliverable Quantity to Traded Quantity
20,1,RELIANCE,EQ,100,60,60.00
20,2,TCS,EQ,200,50,25.00
20,3,SOMEBOND,GS,20255,19244,95.01
20,4,SOMEBE,BE,1000,900,90.00
20,5,BADLINE,EQ,notanum,10,50.00
20,6,SHORTROW,EQ,100
"""

BHAV_TEXT = (
    " SYMBOL, SERIES, TTL_TRD_QNTY, DELIV_QTY, DELIV_PER\n"
    "RELIANCE, EQ, 100, 60, 60.00\n"
    "INFY, EQ, 200, 150, 75.00\n"
    "SGB, GS, 50, -, -\n"
)

_MTO_OK = "Security Wise Delivery Position\n10,MTO,01072013,1,1\n20,1,RELIANCE,EQ,100,60,60.00\n"


def test_parse_delivery_series_filter_and_malformed():
    ratios, stats = m.parse_delivery(MTO_TEXT)
    assert set(ratios) == {"RELIANCE", "TCS"}          # only EQ kept
    assert ratios["RELIANCE"] == pytest.approx(0.60)   # 60/100 from raw qty, not the %-field
    assert ratios["TCS"] == pytest.approx(0.25)
    assert stats["eq_rows"] == 2
    assert stats["dropped_series"] == 2                # GS + BE
    assert stats["malformed"] == 2                     # non-numeric qty + short row
    assert stats["dup_eq"] == 0


def test_ratios_in_unit_interval():
    ratios, _ = m.parse_delivery(MTO_TEXT)
    assert all(0.0 < v <= 1.0 for v in ratios.values())


def test_duplicate_eq_rows_are_summed():
    # A symbol appearing twice in EQ (multi-settlement-type file) aggregates by summing
    # traded and deliverable: (60+40)/(100+300) = 0.25, one symbol, one duplicate folded.
    text = ("20,1,DUP,EQ,100,60,60.00\n"
            "20,2,DUP,EQ,300,40,13.33\n")
    ratios, stats = m.parse_delivery(text)
    assert ratios == {"DUP": pytest.approx(0.25)}
    assert stats["eq_rows"] == 2 and stats["symbols"] == 1 and stats["dup_eq"] == 1


def test_parse_bhavcopy_delivery():
    bh = m.parse_bhavcopy_delivery(BHAV_TEXT)
    assert bh["RELIANCE"] == pytest.approx(0.60)
    assert bh["INFY"] == pytest.approx(0.75)
    assert "SGB" not in bh                              # non-EQ / non-numeric dropped


def test_date_from_name():
    assert m._date_from_name("MTO_09072026.DAT").isoformat() == "2026-07-09"
    assert m._date_from_name("MTO_01072013.DAT").isoformat() == "2013-07-01"
    assert m._date_from_name("not_a_file.txt") is None


def test_load_ratios_builds_symbol_panel(tmp_path):
    (tmp_path / "MTO_01072013.DAT").write_text(MTO_TEXT, encoding="utf-8")
    (tmp_path / "MTO_02072013.DAT").write_text(
        "20,1,RELIANCE,EQ,100,80,80.00\n20,2,TCS,EQ,200,100,50.00\n", encoding="utf-8")
    panel = m.load_ratios(str(tmp_path))
    assert list(panel.columns) == ["RELIANCE", "TCS"] or set(panel.columns) == {"RELIANCE", "TCS"}
    assert panel.shape == (2, 2)
    assert panel.loc[pd.Timestamp("2013-07-02"), "RELIANCE"] == pytest.approx(0.80)
    # a restricted window keeps only the requested date
    one = m.load_ratios(str(tmp_path), start="2013-07-02", end="2013-07-02")
    assert one.shape[0] == 1


class FakeFetch:
    """Canned (text, status, err) per URL; records every URL requested."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, url, *a, **k):
        self.calls.append(url)
        return self.table.get(url, (None, 404, "HTTP 404"))


def test_backfill_writes_manifest_and_is_resumable(tmp_path):
    data_dir = tmp_path / "mto"
    manifest = tmp_path / "manifest.json"
    ok_url = m.MTO_URL.format(d="01072013")             # 2013-07-01 Monday -> ok
    fetch = FakeFetch({ok_url: (_MTO_OK, 200, None)})   # 2013-07-02 Tuesday -> default 404

    s1 = m.backfill(spans=[("2013-07-01", "2013-07-02")], data_dir=str(data_dir),
                    manifest_path=str(manifest), sleep=0.0, fetch=fetch, verbose=False)
    assert s1["counts"]["ok"] == 1 and s1["counts"]["404"] == 1
    assert (data_dir / "MTO_01072013.DAT").exists()
    assert not (data_dir / "MTO_02072013.DAT").exists()   # holiday -> no file
    man = json.loads(manifest.read_text())
    assert man["2013-07-01"] == "ok" and man["2013-07-02"] == "404"

    # resume: the already-downloaded file is NOT refetched (only the 404 day is retried)
    fetch2 = FakeFetch({ok_url: (_MTO_OK, 200, None)})
    m.backfill(spans=[("2013-07-01", "2013-07-02")], data_dir=str(data_dir),
               manifest_path=str(manifest), sleep=0.0, fetch=fetch2, verbose=False)
    assert ok_url not in fetch2.calls                    # cached, skipped


def test_backfill_rejects_non_mto_200(tmp_path):
    # A 200 that is not an MTO file (anti-bot HTML) must not be saved as data.
    url = m.MTO_URL.format(d="01072013")
    fetch = FakeFetch({url: ("<html>blocked</html>", 200, None)})
    s = m.backfill(spans=[("2013-07-01", "2013-07-01")], data_dir=str(tmp_path / "d"),
                   manifest_path=str(tmp_path / "mf.json"), sleep=0.0, fetch=fetch, verbose=False)
    assert s["counts"]["error"] == 1 and s["counts"]["ok"] == 0


def test_collect_mto_today_idempotent(tmp_path):
    from datetime import date
    data_dir = tmp_path / "mto"
    manifest = tmp_path / "manifest.json"
    url = m.MTO_URL.format(d="01072013")
    fetch = FakeFetch({url: (_MTO_OK, 200, None)})
    r1 = m.collect_mto_today(today=date(2013, 7, 1), data_dir=str(data_dir),
                             manifest_path=str(manifest), fetch=fetch, verbose=False)
    assert r1["status"] == "ok"
    r2 = m.collect_mto_today(today=date(2013, 7, 1), data_dir=str(data_dir),
                             manifest_path=str(manifest), fetch=fetch, verbose=False)
    assert r2["status"] == "cached"                      # second run does not refetch
    assert fetch.calls.count(url) == 1
