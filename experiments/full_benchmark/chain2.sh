#!/usr/bin/env bash
# 1) serial timing (one process, clean compute numbers)
# 2) D best-of-16 (parallel) so the budget comparison is like-for-like
set -uo pipefail
cd /home/wei/Desktop/AMR-DFJSP
echo "=== SERIAL TIMING start $(date -Is) ==="
.venv/bin/python experiments/full_benchmark/bench_k.py
echo "=== D best-16 start $(date -Is) ==="
date -Is > experiments/full_benchmark/d16_start.txt
.venv/bin/python experiments/full_benchmark/sweep_all.py --workers 18 --what models \
  --models v9_only60_s42,v9_only60_s43,v9_only60_s44 --modes best16 \
  --out experiments/full_benchmark/raw/rows_d16.jsonl
date -Is > experiments/full_benchmark/d16_end.txt
echo "=== ALL DONE $(date -Is) ==="
