#!/usr/bin/env bash
# Full run: 100 instances at n = 20/40/60/80/100.
#
# STRICTLY SERIAL, one process at a time. Runtime is a reported quantity here, and two
# schedulers sharing the box inflate each other's solve_s -- the rule side is CPU-bound
# Python and the policy side contends for the same cores feeding the GPU.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
RAW="$REPO/experiments/parcel_trend/raw"
SIZES="20 40 60 80 100"
COMBOS="milk_run+earliest_completion,milk_run+earliest_available,material_match+earliest_completion,milk_run+least_loaded,earliest_completion_job+earliest_completion,earliest_completion_job+material_match"

W3="$REPO/checkpoints_v8/ppo_gc1_s42_best.pth,$REPO/checkpoints_v8/ppo_gc1_s43_best.pth,$REPO/checkpoints_v8/ppo_gc1_s44_best.pth"
K3="ppo_gc1_s42_best,ppo_gc1_s43_best,ppo_gc1_s44_best"
W1="$REPO/checkpoints_v8/ppo_gc1_s42_best.pth"
K1="ppo_gc1_s42_best"

cd "$REPO/Static_alogorithm"

echo "=== STAGE 1/4: rules, 6 combos x 100 instances ==="
for n in $SIZES; do
  OUT="$RAW/full_rules_n${n}.jsonl"; rm -f "$OUT"
  "$PY" run_selected_rules.py --inbox "$REPO/test_case/v3/trend/full_${n}.jsonl" \
    --combos "$COMBOS" --amrs 16 --out "$OUT"
  echo "STAGE1 n=${n} COMPLETE"
done

echo "=== STAGE 2/4: policy greedy, 3 seeds x 100 instances ==="
for n in $SIZES; do
  OUT="$RAW/full_greedy_n${n}.jsonl"; rm -f "$OUT"
  "$PY" eval_extend_gnn.py --weights "$W3" --run_key "$K3" \
    --inbox "$REPO/test_case/v3/trend/full_${n}.jsonl" --num_amrs 16 \
    --out "$OUT" --manifest "$RAW/manifest_full_n${n}.json"
  echo "STAGE2 n=${n} COMPLETE"
done

echo "=== STAGE 3/4: policy best-of-8, seed 42 x 100 instances ==="
for n in $SIZES; do
  OUT="$RAW/full_k8_n${n}.jsonl"; rm -f "$OUT"
  "$PY" eval_extend_gnn.py --weights "$W1" --run_key "$K1" --samples 8 \
    --inbox "$REPO/test_case/v3/trend/full_${n}.jsonl" --num_amrs 16 --out "$OUT"
  echo "STAGE3 n=${n} COMPLETE"
done

echo "=== STAGE 4/4: policy best-of-16, seed 42 x 100 instances ==="
for n in $SIZES; do
  OUT="$RAW/full_k16_n${n}.jsonl"; rm -f "$OUT"
  "$PY" eval_extend_gnn.py --weights "$W1" --run_key "$K1" --samples 16 \
    --inbox "$REPO/test_case/v3/trend/full_${n}.jsonl" --num_amrs 16 --out "$OUT"
  echo "STAGE4 n=${n} COMPLETE"
done

echo "ALL STAGES COMPLETE"
