"""Manual forward paper-track runner for the deployable Indian books.

One run snapshots ALL FOUR live sleeves, runs the RL-2026-07-15 F&O daily collector
(basis/PCR/IV -> experiments/fno_daily.jsonl), marks the RL-2026-07-18 paper
NIFTY short-straddle (experiments/paper_options.jsonl), and prints all four forward
records: the RL-2026-07-10 long-only REGIME book (experiments/paper_trades.jsonl),
the RL-2026-07-12 F&O-shortable market-neutral L/S sleeve
(experiments/paper_trades_ls.jsonl), the RL-2026-07-17 multi-asset trend sleeve
(experiments/paper_trades_trend.jsonl), and the RL-2026-07-16 gold_lowbeta risk-off
variant (experiments/paper_trades_gl.jsonl). READ-ONLY w.r.t. Groww exactly as the
underlying quantlab modules: only read-only data methods, ever.

Yahoo is pulled once for the first book build; the second reuses that warm
cache (never refresh twice). --no-refresh skips the pull entirely. Each leg is
guarded: a failure is printed and the run continues, so one book's hiccup never
costs the other book its snapshot.

    uv run python scripts/snapshot.py               # refresh Yahoo first, then snapshot both
    uv run python scripts/snapshot.py --no-refresh  # skip the slow Yahoo pull
"""

import os
import subprocess
import sys
from datetime import date

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from quantlab import (event_studies, fno_collect, intraday_collect, live_paper,
                      nse_events, paper_options, paper_options_putw, paper_options_vrp)

refresh = "--no-refresh" not in sys.argv[1:]


def step(label, fn):
    print(f"\n===== {label} =====")
    try:
        fn()
    except Exception as e:
        print(f"[{label}] FAILED, continuing: {type(e).__name__}: {e}")


def archive_push():
    git = ["git", "-C", os.path.join(ROOT, "data", "raw", "intraday")]
    subprocess.run(git + ["add", "-A"], check=True)
    if subprocess.run(git + ["diff", "--cached", "--quiet"]).returncode == 0:
        print("no new bars")
        return
    subprocess.run(git + ["commit", "-m", f"archive {date.today().isoformat()}"], check=True)
    subprocess.run(git + ["push", "origin", "main"], check=True)


step("REGIME snapshot", lambda: live_paper.run(refresh=refresh))
step("F&O L/S sleeve snapshot", lambda: live_paper.run_ls(refresh=False))
step("TREND sleeve snapshot", lambda: live_paper.run_trend(refresh=False))
step("gold_lowbeta variant snapshot", lambda: live_paper.run_gl(refresh=False))
step("DUAL-ROT sleeve snapshot", lambda: live_paper.run_dualrot(refresh=False))
step("DIV-CARRY sleeve snapshot", lambda: live_paper.run_divcarry(refresh=False))
step("PAIRS sleeve snapshot", lambda: live_paper.run_pairs(refresh=False))
step("VOL-SHOCK sleeve snapshot", lambda: live_paper.run_volshock(refresh=False))
step("F&O daily collect", fno_collect.collect)
step("NSE ban-list collect", nse_events.collect_ban_list)
step("NSE index-changes collect", nse_events.collect_index_changes)
step("Event studies (jumps + SSF-list + maturity)", event_studies.run_all)
step("Intraday 5m archive", intraday_collect.collect)
step("Archive commit + push", archive_push)
step("PAPER options mark", paper_options.snapshot)
step("VRP-gated options mark", paper_options_vrp.snapshot)
step("PUT-WRITE options mark", paper_options_putw.snapshot)
step("REGIME forward record", lambda: live_paper.forward_track())
step("F&O L/S forward record",
     lambda: live_paper.forward_track(path=live_paper.LS_SNAPSHOT_PATH))
step("TREND forward record",
     lambda: live_paper.forward_track(path=live_paper.TREND_SNAPSHOT_PATH))
step("gold_lowbeta forward record",
     lambda: live_paper.forward_track(path=live_paper.GL_SNAPSHOT_PATH))
step("DUAL-ROT forward record",
     lambda: live_paper.forward_track(path=live_paper.DUALROT_SNAPSHOT_PATH))
step("DIV-CARRY forward record",
     lambda: live_paper.forward_track(path=live_paper.DIVCARRY_SNAPSHOT_PATH))
step("PAIRS forward record",
     lambda: live_paper.forward_track(path=live_paper.PAIRS_SNAPSHOT_PATH))
step("VOL-SHOCK forward record",
     lambda: live_paper.forward_track(path=live_paper.VOLSHOCK_SNAPSHOT_PATH))
