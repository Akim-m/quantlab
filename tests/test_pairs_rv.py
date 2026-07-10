"""Tests for the RL-2026-07-26-09 same-sector cointegration pairs sleeve (PAIRS-RV).

FORWARD-ONLY: the formation set is frozen at registration, so no historical P&L is
exercised. What matters is that the cointegration/mean-reversion primitives behave on
synthetic series with a KNOWN answer (a constructed cointegrated pair passes, two
independent walks fail; an AR(1) half-life is recovered), that the entry/exit/time-stop
state machine and the per-pair dollar-neutral book are correct and deterministic, that
the frozen constants are valid members of the F&O-cap-N500 universe, and that the live
snapshot leg mirrors run_ls (read-only, signed, forward_track-compatible, ledger-driven
state). Mostly synthetic - one test rebuilds the offline universe to validate the frozen
set; none touch the network.
"""

import json

import numpy as np
import pandas as pd
import pytest

from quantlab import groww_client as gc
from quantlab import live_paper as lp
from quantlab import pairs_rv as pr


# ---- synthetic generators with a known relationship --------------------------------

def _rw(n, sigma, seed):
    return np.cumsum(np.random.default_rng(seed).normal(0.0, sigma, n))


def _ar1(n, phi, sigma, seed):
    """Stationary AR(1): x_t = phi*x_{t-1} + eps. Mean-reverting for |phi| < 1."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0.0, sigma)
    return x


# ---- Engle-Granger / ADF: cointegrated passes, independent walks fail --------------

def test_engle_granger_recovers_hedge_and_flags_cointegration():
    # log P_b is a random walk; log P_a = 1 + 1.5*log P_b + stationary noise -> cointegrated.
    lb = 3.0 + _rw(1500, 0.01, seed=1)
    la = 1.0 + 1.5 * lb + _ar1(1500, 0.90, 0.02, seed=2)

    beta, alpha = pr.engle_granger(la, lb)
    assert beta == pytest.approx(1.5, abs=0.1)          # hedge ratio recovered
    spread = la - beta * lb
    assert pr.adf_tstat(spread) < pr.ADF_CRIT           # residual is stationary -> passes


def test_independent_random_walks_are_not_cointegrated():
    la = 3.0 + _rw(1500, 0.01, seed=11)
    lb = 3.0 + _rw(1500, 0.01, seed=22)                 # unrelated second walk
    beta, _ = pr.engle_granger(la, lb)
    spread = la - beta * lb
    assert pr.adf_tstat(spread) > pr.ADF_CRIT           # unit root not rejected -> fails


def test_adf_strongly_rejects_pure_ar1_and_not_random_walk():
    assert pr.adf_tstat(_ar1(2000, 0.7, 0.01, seed=3)) < pr.ADF_CRIT   # stationary
    assert pr.adf_tstat(_rw(2000, 0.01, seed=4)) > pr.ADF_CRIT         # unit root


# ---- half-life from an AR(1) with a known phi --------------------------------------

def test_half_life_recovers_known_phi():
    for phi in (0.80, 0.90, 0.95):
        hl = pr.half_life(_ar1(6000, phi, 0.01, seed=int(phi * 100)))
        assert hl == pytest.approx(-np.log(2) / np.log(phi), rel=0.15)


def test_half_life_infinite_for_random_walk():
    # A random walk does not mean-revert -> half-life fails the < 60d filter (huge or inf).
    assert pr.half_life(_rw(4000, 0.01, seed=7)) > pr.HALF_LIFE_MAX


def test_spread_vol_is_daily_diff_std():
    x = _rw(500, 0.02, seed=8)
    assert pr.spread_vol(x) == pytest.approx(np.std(np.diff(x), ddof=1))


# ---- entry / exit / time-stop state machine (enter |z|>=2, exit z=0, stop 30d) -----

def test_entry_direction_thresholds():
    assert pr.entry_direction(2.5) == "short_a"      # a rich -> short a, long b
    assert pr.entry_direction(-2.5) == "long_a"      # a cheap -> long a, short b
    assert pr.entry_direction(2.0) == "short_a"      # boundary is inclusive
    assert pr.entry_direction(-2.0) == "long_a"
    assert pr.entry_direction(1.9) is None           # inside the band -> no trade
    assert pr.entry_direction(0.0) is None


def test_should_exit_zero_cross_and_time_stop():
    # long_a (entered z<=-2): exit when z crosses up to 0
    assert pr.should_exit("long_a", -1.0, 5) is False
    assert pr.should_exit("long_a", 0.0, 5) is True
    assert pr.should_exit("long_a", 0.2, 5) is True
    # short_a (entered z>=2): exit when z crosses down to 0
    assert pr.should_exit("short_a", 1.0, 5) is False
    assert pr.should_exit("short_a", -0.1, 5) is True
    # time stop fires regardless of z
    assert pr.should_exit("short_a", 3.0, pr.TIME_STOP) is True
    assert pr.should_exit("long_a", -3.0, pr.TIME_STOP) is True
    assert pr.should_exit("short_a", 3.0, pr.TIME_STOP - 1) is False


def test_state_machine_open_carry_exit_timestop(monkeypatch):
    monkeypatch.setattr(pr, "FROZEN_PAIRS", (("X.NS", "Y.NS", 1.0, 0.0, 0.1),))
    dates = pd.bdate_range("2026-01-01", periods=60)
    key = ("X.NS", "Y.NS")

    o1 = pr.update_open_pairs([], {key: 2.5}, dates, dates[0])          # flat -> open short_a
    assert len(o1) == 1 and o1[0]["direction"] == "short_a"
    assert o1[0]["z_entry"] == 2.5 and o1[0]["entry_date"] == str(dates[0].date())

    assert pr.update_open_pairs(o1, {key: 1.2}, dates, dates[3]) == o1   # inside band -> carry
    assert pr.update_open_pairs(o1, {key: -0.1}, dates, dates[3]) == []  # z<=0 -> exit
    assert len(pr.update_open_pairs(o1, {key: 2.5}, dates, dates[29])) == 1   # 29d -> still on
    assert pr.update_open_pairs(o1, {key: 2.5}, dates, dates[30]) == []       # 30d -> time stop

    oL = pr.update_open_pairs([], {key: -2.5}, dates, dates[0])         # flat -> open long_a
    assert oL[0]["direction"] == "long_a"
    assert pr.update_open_pairs(oL, {key: 0.05}, dates, dates[2]) == []  # z>=0 -> exit

    assert pr.update_open_pairs([], {key: 1.0}, dates, dates[0]) == []   # inside band -> no entry
    assert pr.update_open_pairs(o1, {key: float("nan")}, dates, dates[5]) == o1  # data gap -> carry


# ---- per-pair dollar-neutral book --------------------------------------------------

def test_target_weights_dollar_neutral_per_pair():
    short = pr.target_weights([{"a": "X.NS", "b": "Y.NS", "direction": "short_a",
                                "z_entry": 2.5, "entry_date": "2026-01-01"}])
    assert short["X.NS"] == pytest.approx(-pr.PER_LEG)   # short the rich leg a
    assert short["Y.NS"] == pytest.approx(pr.PER_LEG)    # long the cheap leg b
    assert short.sum() == pytest.approx(0.0)             # dollar-neutral
    assert short.abs().sum() == pytest.approx(2 * pr.PER_LEG)   # gross 0.1 per pair

    lng = pr.target_weights([{"a": "X.NS", "b": "Y.NS", "direction": "long_a",
                              "z_entry": -2.5, "entry_date": "2026-01-01"}])
    assert lng["X.NS"] == pytest.approx(pr.PER_LEG) and lng["Y.NS"] == pytest.approx(-pr.PER_LEG)

    assert pr.target_weights([]).empty                   # no open pairs -> all cash


def test_target_weights_book_nets_flat_across_pairs():
    book = pr.target_weights([
        {"a": "A.NS", "b": "B.NS", "direction": "short_a", "z_entry": 2.5, "entry_date": "d"},
        {"a": "C.NS", "b": "D.NS", "direction": "long_a", "z_entry": -2.5, "entry_date": "d"},
    ])
    assert book.sum() == pytest.approx(0.0)              # whole book net ~0
    assert book.abs().sum() == pytest.approx(4 * pr.PER_LEG)   # gross = 0.1 * n_open


# ---- frozen constants: shape + membership in the F&O-cap-N500 universe --------------

def test_frozen_pairs_structural():
    from quantlab.india import sector_map
    assert len(pr.FROZEN_PAIRS) == pr.TOP_N
    ind = {k.upper(): v for k, v in sector_map("nifty500").items()}
    seen = set()
    for a, b, beta, mu, sigma in pr.FROZEN_PAIRS:
        assert a.endswith(".NS") and b.endswith(".NS") and a != b
        assert np.isfinite(beta) and beta != 0.0
        assert np.isfinite(mu) and sigma > 0.0
        assert (a, b) not in seen                        # no duplicate pair
        seen.add((a, b))
        assert ind.get(a) is not None and ind[a] == ind[b]   # both legs same NSE industry


def test_frozen_pairs_are_in_the_fno_n500_universe():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, _, universe = pr.build_panel(refresh=False)
    uset = set(universe)
    for a, b, *_ in pr.FROZEN_PAIRS:
        assert a in uset and b in uset                   # both legs are F&O-shortable N500 names


# ---- determinism -------------------------------------------------------------------

def test_current_z_and_weights_deterministic():
    syms = sorted({s for a, b, *_ in pr.FROZEN_PAIRS for s in (a, b)})
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    log_px = pd.DataFrame(np.cumsum(rng.normal(0, 0.01, (400, len(syms))), axis=0) + 5.0,
                          index=idx, columns=syms)
    z1, z2 = pr.current_z(log_px), pr.current_z(log_px)
    assert z1 == z2 and set(z1) == {(a, b) for a, b, *_ in pr.FROZEN_PAIRS}
    assert all(np.isfinite(v) for v in z1.values())
    op = [{"a": "A.NS", "b": "B.NS", "direction": "short_a", "z_entry": 2.5, "entry_date": "d"}]
    pd.testing.assert_series_equal(pr.target_weights(op), pr.target_weights(op))


# ---- ledger-state round-trip: an open position survives write -> read -> recover ----

def test_ledger_state_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "FROZEN_PAIRS", (("X.NS", "Y.NS", 1.0, 0.0, 0.1),))
    dates = pd.bdate_range("2026-01-01", periods=40)
    op = [{"a": "X.NS", "b": "Y.NS", "z_entry": 2.5,
           "entry_date": str(dates[0].date()), "direction": "short_a"}]

    out = tmp_path / "paper_trades_pairs.jsonl"
    with open(out, "w") as f:
        f.write(json.dumps({"panel_date": str(dates[0].date()), "open_pairs": op,
                            "weights": {"X.NS": -0.05, "Y.NS": 0.05}}) + "\n")

    prev = lp._last_row(str(out))
    assert prev["open_pairs"] == op                       # persisted state read back intact
    # the recovered open position carries under a still-dislocated spread
    recovered = pr.update_open_pairs(prev["open_pairs"], {("X.NS", "Y.NS"): 1.5}, dates, dates[3])
    assert recovered == op


def test_last_row_missing_file_is_none(tmp_path):
    assert lp._last_row(str(tmp_path / "absent.jsonl")) is None


# ---- live snapshot leg: read-only, signed, ledger-driven, forward_track-compatible --

class Spy:
    """Records every method routed through groww_client.call; serves canned LTP."""

    PRICES = {"NSE_AAA": 110.0, "NSE_BBB": 50.0, "NSE_NIFTY": 20200.0, "NSE_NIFTYBEES": 110.0}

    def __init__(self):
        self.methods = []

    def __call__(self, method, *args, **kwargs):
        self.methods.append(method)
        if method in gc._ORDER_METHODS:                   # the spy honors the real guard
            raise PermissionError(method)
        syms = kwargs.get("exchange_trading_symbols", ())
        return {s: self.PRICES[s] for s in syms if s in self.PRICES}


def _synthetic_pairs_book():
    book = lp.Book(
        weights=pd.Series({"AAA.NS": 0.05, "BBB.NS": -0.05}),   # long_a: long AAA, short BBB
        regime_on=True, cash_frac=0.9, latest_date=pd.Timestamp("2026-07-09"),
        prev_close=pd.Series({"AAA.NS": 100.0, "BBB.NS": 40.0}),
        nsei_prev_close=20000.0,
    )
    open_pairs = [{"a": "AAA.NS", "b": "BBB.NS", "z_entry": -2.5,
                   "entry_date": "2026-07-09", "direction": "long_a"}]
    z_now = {("AAA.NS", "BBB.NS"): -2.5}
    return book, open_pairs, z_now


def test_run_pairs_read_only_signed_schema_and_pnl(tmp_path, monkeypatch):
    spy = Spy()
    monkeypatch.setattr(gc, "call", spy)
    captured = {}

    def fake_book(prev_open, refresh=False):
        captured["prev_open"] = prev_open                 # prove run_pairs feeds ledger state in
        return _synthetic_pairs_book()
    monkeypatch.setattr(lp, "current_pairs_book", fake_book)

    out = tmp_path / "paper_trades_pairs.jsonl"
    prior = [{"a": "AAA.NS", "b": "BBB.NS", "z_entry": -2.5,
              "entry_date": "2026-07-08", "direction": "long_a"}]
    with open(out, "w") as f:                             # a pre-existing open position
        f.write(json.dumps({"panel_date": "2026-07-08", "open_pairs": prior,
                            "weights": {"AAA.NS": 0.05, "BBB.NS": -0.05}}) + "\n")

    row = lp.run_pairs(path=str(out), write=True)

    assert captured["prev_open"] == prior                 # ledger last-row state was read + passed

    # SAFETY: only read-only methods dispatched, none of them order methods.
    assert spy.methods and set(spy.methods) <= set(lp.READ_METHODS)
    assert not any(m in gc._ORDER_METHODS for m in spy.methods)

    assert row["kind"] == "live_paper_pairs_snapshot"
    assert row["hypothesis_ref"] == "RL-2026-07-26-09"
    assert row["gross"] == pytest.approx(0.1) and row["net"] == pytest.approx(0.0)
    assert row["n_open"] == 1
    assert set(row["open_pairs"][0]) == {"a", "b", "z_entry", "entry_date", "direction"}
    # book_ret = 0.05*(110/100-1) + (-0.05)*(50/40-1) = 0.005 - 0.0125 = -0.0075
    assert row["book_intraday_ret"] == pytest.approx(-0.0075)
    assert row["nifty_intraday_ret"] == pytest.approx(0.01)
    assert row["quotes_ok"] is True

    rec = json.loads(out.read_text().strip().splitlines()[-1])
    assert "panel_date" in rec and "weights" in rec       # forward_track-compatible
    assert rec["weights"]["BBB.NS"] == pytest.approx(-0.05)   # short leg stays negative


def test_run_pairs_records_book_only_when_quotes_unavailable(tmp_path, monkeypatch):
    def boom(method, *a, **k):
        raise RuntimeError("no entitlement")
    monkeypatch.setattr(gc, "call", boom)
    monkeypatch.setattr(lp, "current_pairs_book", lambda prev_open, refresh=False: _synthetic_pairs_book())

    out = tmp_path / "p.jsonl"
    row = lp.run_pairs(path=str(out), write=True)
    assert row["book_intraday_ret"] is None               # no live P&L invented
    assert row["n_quotes_ok"] == 0 and row["quotes_ok"] is False
    assert row["weights"] and row["open_pairs"]           # book + open pairs still recorded
    assert out.read_text().strip()                        # snapshot still written


def test_run_pairs_empty_book_when_no_open_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "call", Spy())
    empty = lp.Book(weights=pd.Series(dtype=float), regime_on=False, cash_frac=1.0,
                    latest_date=pd.Timestamp("2026-07-09"),
                    prev_close=pd.Series(dtype=float), nsei_prev_close=20000.0)
    monkeypatch.setattr(lp, "current_pairs_book",
                        lambda prev_open, refresh=False: (empty, [], {}))
    out = tmp_path / "p.jsonl"
    row = lp.run_pairs(path=str(out), write=True)
    assert row["n_open"] == 0 and row["weights"] == {}    # all cash, no crash
    assert row["gross"] == pytest.approx(0.0)
    assert out.read_text().strip()


def test_order_methods_refused_by_dispatcher():
    """The harness's only channel to Groww refuses order methods before any network."""
    with pytest.raises(PermissionError):
        gc.call("place_order")
