@echo off
REM RL-2026-07-10 forward paper-track: daily READ-ONLY REGIME snapshot from Groww.
REM Registered as a Windows Scheduled Task (see scripts/README-schedule.md). No trades.
cd /d "C:\Users\aydhi\OneDrive\Documents\ay\quant\quantlab"
set PYTHONIOENCODING=utf-8
set PYTHONPATH=src
"C:\Users\aydhi\.local\bin\uv.exe" run python -m quantlab.live_paper --refresh 1>> "experiments\paper_snapshot.log" 2>&1
