#!/usr/bin/env bash
# Wait for the parallel sweep to exit, then run the serial timing pass with the box to
# itself. Timings measured under contention are not compute costs, so this must not
# overlap the sweep.
set -uo pipefail
cd /home/wei/Desktop/AMR-DFJSP
while pgrep -f "sweep_all.py --workers" > /dev/null; do sleep 20; done
echo "=== sweep finished, starting serial timing pass $(date -Is) ==="
.venv/bin/python experiments/full_benchmark/bench_serial.py \
  --n_rules 3 --n_greedy 5 --n_best8 3 --n_ga 3 \
  --out experiments/full_benchmark/raw/timing.jsonl
echo "=== ALL COMPLETE $(date -Is) ==="
