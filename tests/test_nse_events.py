"""Tests for the NSE event-data archivers (ban list + index reconstitution).

No live HTTP: every collector takes an injectable `fetch`, so the network layer is
replaced by a router serving canned responses. Fixtures below are trimmed real shapes
(ban CSV probed 2026-07-10; constituent CSV is the repo's known format). Coverage:
symbol/date parsing, the set-diff baseline/add/remove/no-change logic, error paths that
return a row instead of crashing, the ledger row schema, and that the diff baseline is
never corrupted by an intervening failed fetch.
"""

import json

import pytest

from quantlab import nse_events as ne

# ---- fixtures (trimmed real shapes) ----

BAN_CSV_ONE = "Securities in Ban For Trade Date 10-JUL-2026:\n1,KAYNES\n"

BAN_CSV_MANY = (
    "Securities in Ban For Trade Date 09-JUL-2026:\n"
    "1,GNFC\n2,BANDHANBNK\n3,M&M\n4,BAJAJ-AUTO\n"      # ampersand + hyphen must survive
)

BAN_CSV_NIL = "Securities in Ban For Trade Date 11-JUL-2026:\n"   # zero-ban day

BAN_CSV_BARE = "Securities in Ban For Trade Date 12-JUL-2026:\nGNFC\nIDEA\n"  # no serials

BAN_JSON_DICT = json.dumps({"data": [{"symbol": "GNFC"}, {"symbol": "IDEA"}],
                            "timestamp": "09-Jul-2026 12:00"})
BAN_JSON_LIST = json.dumps(["GNFC", "IDEA"])

NIFTY50_CSV = (
    "Company Name,Industry,Symbol,Series,ISIN Code\n"
    "Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018\n"
    "Tata Consultancy Services Ltd.,Information Technology,TCS,EQ,INE467B01029\n"
    "Infosys Ltd.,Information Technology,INFY,EQ,INE009A01021\n"
)
# same index after a reconstitution: INFY out, HDFCBANK in
NIFTY50_CSV_V2 = (
    "Company Name,Industry,Symbol,Series,ISIN Code\n"
    "Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018\n"
    "Tata Consultancy Services Ltd.,Information Technology,TCS,EQ,INE467B01029\n"
    "HDFC Bank Ltd.,Financial Services,HDFCBANK,EQ,INE040A01034\n"
)


class Router:
    """Fake `fetch`: maps url -> (text, status, err) and records each call. Any url not
    in the map returns a 404 tuple (nothing crashes)."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, url, warmup=False, timeout=25):
        self.calls.append((url, warmup))
        return self.responses.get(url, (None, 404, "HTTP 404"))


def read_jsonl(path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ---- ban-list CSV parsing ----

def test_parse_ban_csv_one():
    syms, td = ne.parse_ban_csv(BAN_CSV_ONE)
    assert syms == ["KAYNES"] and td == "2026-07-10"


def test_parse_ban_csv_many_keeps_special_chars():
    syms, td = ne.parse_ban_csv(BAN_CSV_MANY)
    assert syms == ["GNFC", "BANDHANBNK", "M&M", "BAJAJ-AUTO"]      # & and - preserved
    assert td == "2026-07-09"


def test_parse_ban_csv_nil_day():
    syms, td = ne.parse_ban_csv(BAN_CSV_NIL)
    assert syms == [] and td == "2026-07-11"                       # header parsed, no names


def test_parse_ban_csv_bare_symbol_fallback():
    syms, td = ne.parse_ban_csv(BAN_CSV_BARE)
    assert syms == ["GNFC", "IDEA"] and td == "2026-07-12"


def test_parse_ban_csv_empty_text():
    assert ne.parse_ban_csv("") == ([], None)


# ---- ban-list JSON parsing (shape not live-verified; tolerant) ----

def test_symbols_from_json_dict_and_list():
    assert ne._symbols_from_json(json.loads(BAN_JSON_DICT)) == ["GNFC", "IDEA"]
    assert ne._symbols_from_json(json.loads(BAN_JSON_LIST)) == ["GNFC", "IDEA"]


def test_symbols_from_json_filters_non_symbols():
    obj = {"data": [{"symbol": "GNFC"}], "note": "these are not symbols at all"}
    assert ne._symbols_from_json(obj) == ["GNFC"]                  # prose value ignored


# ---- collect_ban_list: fallback, success, total failure ----

def test_ban_json_blocked_falls_back_to_csv(tmp_path):
    r = Router({ne.BAN_JSON_URL: (None, 404, "HTTP 404"),
                ne.BAN_CSV_URL: (BAN_CSV_ONE, 200, None)})
    out = tmp_path / "ban.jsonl"
    row = ne.collect_ban_list(today=ne.date(2026, 7, 10), write=True, path=str(out),
                              fetch=r, verbose=False)
    assert row["symbols"] == ["KAYNES"] and row["n_banned"] == 1
    assert row["source"] == "csv_archive" and row["trade_date"] == "2026-07-10"
    assert "json:HTTP 404" in row["note"]                          # fallback recorded honestly
    assert (ne.BAN_JSON_URL, True) in r.calls                      # warm-up requested
    rec = read_jsonl(out)[0]
    for k in ("hypothesis_ref", "kind", "date", "symbols", "n_banned", "source", "note"):
        assert k in rec
    assert rec["hypothesis_ref"] == "RL-2026-07-26-11" and rec["kind"] == "nse_ban_list"


def test_ban_json_success_skips_csv(tmp_path):
    r = Router({ne.BAN_JSON_URL: (BAN_JSON_DICT, 200, None)})
    row = ne.collect_ban_list(today=ne.date(2026, 7, 9), write=False, path=str(tmp_path / "b.jsonl"),
                              fetch=r, verbose=False)
    assert row["symbols"] == ["GNFC", "IDEA"] and row["source"] == "json_api"
    assert all(url != ne.BAN_CSV_URL for url, _ in r.calls)        # CSV never fetched
    assert not (tmp_path / "b.jsonl").exists()                     # dry run writes nothing


def test_ban_both_sources_fail_returns_error_row(tmp_path):
    r = Router({ne.BAN_JSON_URL: (None, 403, "HTTP 403"),
                ne.BAN_CSV_URL: (None, 403, "HTTP 403")})
    out = tmp_path / "ban.jsonl"
    row = ne.collect_ban_list(today=ne.date(2026, 7, 10), write=True, path=str(out),
                              fetch=r, verbose=False)
    assert row["symbols"] == [] and row["n_banned"] == 0 and row["source"] is None
    assert "json:HTTP 403" in row["note"] and "csv:HTTP 403" in row["note"]
    assert read_jsonl(out)[0]["n_banned"] == 0                     # error row still written


# ---- constituent parsing + set-diff ----

def test_parse_constituents():
    assert ne.parse_constituents(NIFTY50_CSV) == ["RELIANCE", "TCS", "INFY"]
    assert ne.parse_constituents("") == []


def test_diff_membership_cases():
    assert ne.diff_membership(None, ["A", "B"]) == ([], [])                 # baseline
    assert ne.diff_membership(["A", "B"], ["A", "B"]) == ([], [])           # no change
    assert ne.diff_membership(["A", "B"], ["A", "B", "C"]) == (["C"], [])   # add
    assert ne.diff_membership(["A", "B"], ["A"]) == ([], ["B"])             # remove
    assert ne.diff_membership(["A", "B"], ["A", "C"]) == (["C"], ["B"])     # add + remove


# ---- collect_index_changes: baseline -> diff, error isolation ----

def test_index_baseline_then_diff(tmp_path):
    out = tmp_path / "changes.jsonl"
    idx = ("nifty50",)

    r1 = Router({ne.INDEX_CSV_URL.format(index="nifty50"): (NIFTY50_CSV, 200, None)})
    rows1 = ne.collect_index_changes(today=ne.date(2026, 3, 1), indices=idx, write=True,
                                     path=str(out), fetch=r1, verbose=False)
    assert rows1[0]["note"] == "baseline"
    assert rows1[0]["added"] == [] and rows1[0]["removed"] == []
    assert rows1[0]["members"] == ["INFY", "RELIANCE", "TCS"]              # stored sorted
    assert rows1[0]["n_members"] == 3

    r2 = Router({ne.INDEX_CSV_URL.format(index="nifty50"): (NIFTY50_CSV_V2, 200, None)})
    rows2 = ne.collect_index_changes(today=ne.date(2026, 9, 1), indices=idx, write=True,
                                     path=str(out), fetch=r2, verbose=False)
    assert rows2[0]["added"] == ["HDFCBANK"] and rows2[0]["removed"] == ["INFY"]
    assert rows2[0]["note"] == "added=1 removed=1"

    rec = read_jsonl(out)[0]
    for k in ("hypothesis_ref", "kind", "date", "index", "added", "removed",
              "n_members", "note"):
        assert k in rec
    assert rec["hypothesis_ref"] == "RL-2026-07-26-17"


def test_index_no_change_note(tmp_path):
    out = tmp_path / "changes.jsonl"
    idx, url = ("nifty50",), ne.INDEX_CSV_URL.format(index="nifty50")
    ne.collect_index_changes(indices=idx, write=True, path=str(out),
                             fetch=Router({url: (NIFTY50_CSV, 200, None)}), verbose=False)
    rows = ne.collect_index_changes(indices=idx, write=True, path=str(out),
                                    fetch=Router({url: (NIFTY50_CSV, 200, None)}), verbose=False)
    assert rows[0]["note"] == "no_change"


def test_index_error_row_does_not_corrupt_baseline(tmp_path):
    """success -> failed fetch -> success: the third run must diff against the FIRST
    run's membership, not re-baseline off the error row (which carries members=null)."""
    out = tmp_path / "changes.jsonl"
    idx, url = ("nifty50",), ne.INDEX_CSV_URL.format(index="nifty50")

    ne.collect_index_changes(indices=idx, write=True, path=str(out),
                             fetch=Router({url: (NIFTY50_CSV, 200, None)}), verbose=False)
    err_rows = ne.collect_index_changes(indices=idx, write=True, path=str(out),
                                        fetch=Router({url: (None, 403, "HTTP 403")}), verbose=False)
    assert err_rows[0]["members"] is None and err_rows[0]["note"].startswith("error")

    final = ne.collect_index_changes(indices=idx, write=True, path=str(out),
                                     fetch=Router({url: (NIFTY50_CSV_V2, 200, None)}), verbose=False)
    assert final[0]["added"] == ["HDFCBANK"] and final[0]["removed"] == ["INFY"]  # vs run 1


def test_last_members_skips_error_rows(tmp_path):
    out = tmp_path / "changes.jsonl"
    ne.log_run({"kind": "nse_index_changes", "index": "nifty50",
                "members": ["A", "B"], "n_members": 2}, path=str(out))
    ne.log_run({"kind": "nse_index_changes", "index": "nifty50",
                "members": None, "n_members": 0}, path=str(out))     # error row
    assert ne._last_members(str(out), "nifty50") == ["A", "B"]       # error row skipped
    assert ne._last_members(str(out), "niftynext50") is None         # different index


# ---- network layer degrades gracefully (offline) ----

def test_http_get_never_raises_offline():
    text, status, err = ne.http_get("htp://bad-scheme", timeout=1)   # invalid scheme, no network
    assert text is None and status is None and err is not None
