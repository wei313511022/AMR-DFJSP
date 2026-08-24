#!/usr/bin/env bash
# Full job-rule x AMR-rule grid at each parcel count, m=16, 10 instances.
# sweep_fleet.py covers all 6 AMR rules per call, so we loop only over job rules.
# Rows carry (amrs, instance, job_rule, rule, family="rule"), the join keys the
# policy rows from eval_extend_gnn.py use.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
RAW="$REPO/experiments/parcel_trend/raw"

JOB_RULES="fifo spt lpt nearest_station most_congested_station least_congested_station earliest_completion_job material_match milk_run random"

cd "$REPO/Static_alogorithm"
for n in 20 40 60 80 100; do
  OUT="$RAW/rules_n${n}.jsonl"
  rm -f "$OUT"
  for jr in $JOB_RULES; do
    "$PY" sweep_fleet.py --inbox "$REPO/test_case/v3/trend/trend_${n}.jsonl" \
      --amrs 16 --count 10 --job_rule "$jr" --out "$OUT" >/dev/null
  done
  echo "n=${n} done -> $(wc -l < "$OUT") rows"
done
