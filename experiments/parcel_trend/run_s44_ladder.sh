#!/usr/bin/env bash
# Best-of-8 and best-of-16 on ppo_gc1_s44_best -- the seed reported by
# experiments/policy_vs_rules/policy_in_grid.csv, and the strongest of the three
# (~11-13 makespan units ahead of s42 at n=60).
#
# No wait-gate. The previous version guarded with `pgrep -f "eval_extend_gnn.py"`, which
# deadlocked: the launching shell carries this script's whole text in its own command
# line, so the pattern matched the launcher itself and the loop never exited. Any -f
# pattern naming a string that also appears in this file has that problem. The caller is
# responsible for confirming the box is idle -- check with `pgrep -x python`, whose match
# is on the executable name and therefore cannot match a shell wrapper.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
RAW="$REPO/experiments/parcel_trend/raw"

cd "$REPO/Static_alogorithm"
echo "=== s44 LADDER START $(date +%H:%M:%S) ==="
for k in 8 16; do
  for n in 20 40 60 80 100; do
    OUT="$RAW/full_k${k}_s44_n${n}.jsonl"; rm -f "$OUT"
    "$PY" eval_extend_gnn.py --weights "$REPO/checkpoints_v8/ppo_gc1_s44_best.pth" \
      --run_key "ppo_gc1_s44_best" --samples "$k" \
      --inbox "$REPO/test_case/v3/trend/full_${n}.jsonl" --num_amrs 16 --out "$OUT"
    echo "S44 k=${k} n=${n} COMPLETE $(date +%H:%M:%S)"
  done
done
echo "S44 LADDER ALL COMPLETE $(date +%H:%M:%S)"
