"""PERFORMANCE.md generation: metrics are computed (not hardcoded), the md carries
the required section headers, and the forward section degrades cleanly on missing or
thin ledgers. All fast and network-free (synthetic series + tmp ledgers)."""

import json

import numpy as np
import pandas as pd

from quantlab import report
from quantlab.evaluation import sharpe_tstat


def _synth_data() -> dict:
    """Deterministic daily return series for the two books + benchmarks, spanning the
    test window so year/regime slicing has data. No backtests, no I/O."""
    idx = pd.bdate_range("2017-01-01", "2026-07-09")
    rng = np.random.default_rng(0)
    regime = pd.Series(0.0006 + 0.010 * rng.standard_normal(len(idx)), index=idx)
    ls = pd.Series(0.0002 + 0.004 * rng.standard_normal(len(idx)), index=idx)
    nsei = pd.Series(0.0003 + 0.009 * rng.standard_normal(len(idx)), index=idx)
    ew = pd.Series(0.0004 + 0.011 * rng.standard_normal(len(idx)), index=idx)
    mkt = pd.Series(20000.0 * (1.0 + nsei).cumprod().values, index=idx)  # ^NSEI level
    return {"regime": regime, "ls": ls, "nsei": nsei, "ew": ew, "mkt": mkt,
            "last_date": idx[-1], "n_stocks": 277, "n_shortable": 210, "n_overlap": 130}


def test_book_metrics_is_computed():
    idx = pd.bdate_range("2020-01-01", periods=300)
    ret = pd.Series(0.001 + 0.01 * np.random.default_rng(1).standard_normal(len(idx)), index=idx)
    m = report.book_metrics(ret, ret)
    assert m["sharpe"] == round(sharpe_tstat(ret)[0], 3)   # wired to the real Sharpe
    assert m["beta"] == 1.0                                 # regressed on itself


def test_render_has_headers_and_wired_numbers(tmp_path):
    d = _synth_data()
    missing = str(tmp_path / "none.jsonl")
    md = report.render(d, forward_paths=(missing, missing, missing))

    for header in ("## 1. Deployed strategies", "## 2. Scenario breakdown",
                   "## 3. Forward paper-track", "## 4. The graveyard"):
        assert header in md

    # the REGIME row's numbers must be exactly what book_metrics computes from the
    # SAME synthetic input -> proves the table is wired to the computation, not typed.
    m = report.book_metrics(d["regime"].loc[report.SPLIT:], d["nsei"].loc[report.SPLIT:])
    assert f"| REGIME (long-only) | {m['sharpe']:.3f} |" in md
    assert f"{m['ann'] * 100:+.1f}%" in md
    assert "No forward snapshots recorded yet." in md


def test_forward_section_missing_ledgers(tmp_path):
    missing = str(tmp_path / "absent.jsonl")
    md = report.forward_section(missing, missing, missing)
    assert "No forward snapshots recorded yet." in md


def test_forward_section_thin_ledger_no_network(tmp_path):
    reg = tmp_path / "reg.jsonl"
    reg.write_text(json.dumps({
        "panel_date": "2026-07-09", "asof_ist": "2026-07-09 14:00:00",
        "regime_state": "risk_off", "cash_frac": 0.5, "n_names": 2, "n_quotes_ok": 2,
        "groww_ok": True, "book_intraday_ret": 0.001, "nifty_intraday_ret": 0.002,
        "weights": {"A.NS": 0.5, "B.NS": 0.5},
    }) + "\n")
    absent = str(tmp_path / "absent.jsonl")
    # a single snapshot day cannot yield a forward return, and forward_track must not
    # reach Yahoo (it returns None before loading prices when < 2 books exist).
    md = report.forward_section(str(reg), absent, absent)
    assert "1 snapshot day" in md
    assert "risk_off" in md
    assert "needs" in md  # insufficient-days note for the cumulative line
