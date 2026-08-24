#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
RAW="$REPO/experiments/parcel_trend/raw"
W="$REPO/checkpoints_v8/ppo_gc1_s42_best.pth,$REPO/checkpoints_v8/ppo_gc1_s43_best.pth,$REPO/checkpoints_v8/ppo_gc1_s44_best.pth"
K="ppo_gc1_s42_best,ppo_gc1_s43_best,ppo_gc1_s44_best"

cd "$REPO/Static_alogorithm"
for n in 20 40 60 80 100; do
  OUT="$RAW/full_policy_greedy_n${n}.jsonl"
  rm -f "$OUT"
  "$PY" eval_extend_gnn.py --weights "$W" --run_key "$K" \
    --inbox "$REPO/test_case/v3/trend/full_${n}.jsonl" --num_amrs 16 \
    --out "$OUT" --manifest "$RAW/manifest_full_n${n}.json"
  echo "GREEDY n=${n} COMPLETE"
done
echo "GREEDY ALL COMPLETE"
