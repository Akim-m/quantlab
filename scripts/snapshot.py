"""Manual forward paper-track runner for the RL-2026-07-10 REGIME strategy.

Run any time after the 15:30 IST cash close (or intraday for a live mark). It
snapshots the current REGIME book, appends one row to experiments/paper_trades.jsonl,
then prints the accumulated forward record. READ-ONLY w.r.t. Groww exactly as
quantlab.live_paper: the only Groww method ever called is get_ltp.

    uv run python scripts/snapshot.py               # refresh Yahoo first, then snapshot
    uv run python scripts/snapshot.py --no-refresh  # skip the slow Yahoo pull
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from quantlab import live_paper

live_paper.run(refresh="--no-refresh" not in sys.argv[1:])
live_paper.forward_track()
