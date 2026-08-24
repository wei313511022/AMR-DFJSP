#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
RAW="$REPO/experiments/parcel_trend/raw"
COMBOS="milk_run+earliest_completion,milk_run+earliest_available,material_match+earliest_completion,milk_run+least_loaded,earliest_completion_job+earliest_completion,earliest_completion_job+material_match"

cd "$REPO/Static_alogorithm"
for n in 20 40 60 80 100; do
  OUT="$RAW/full_rules_n${n}.jsonl"
  rm -f "$OUT"
  "$PY" run_selected_rules.py --inbox "$REPO/test_case/v3/trend/full_${n}.jsonl" \
    --combos "$COMBOS" --amrs 16 --out "$OUT"
  echo "SIZE n=${n} COMPLETE -> $(wc -l < "$OUT") rows"
done
echo "RULES ALL COMPLETE"
