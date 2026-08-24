#!/usr/bin/env bash
# Final run of Experiment 1, on the CLEAN test set.
#
# Supersedes run_sweep.sh, which used instances_60.jsonl -- all 30 of whose
# instances sit inside train_60.jsonl. That is harmless for a rules-only sweep
# (rules do not train) but unusable once a learned policy shares the table.
# run_sweep.sh is kept as-is so it still reproduces raw/sweep.jsonl.
#
# Two products:
#   1. the fleet curve Lambda(eta), full sweep, 100 clean instances of 60 parcels
#   2. the parcel-count check at the headline fleet m=16, n = 60 / 120 / 240
#
# The n=60 point of (2) is the m=16 slice of (1) -- not re-run, so the two
# products cannot disagree.
#
# Cost: the executor is roughly quadratic in parcel count (measured 0.28 / 1.31 /
# 5.6 s per run at n = 60 / 120 / 240). A FULL fleet sweep at n=240 would be
# ~9.3 h, which is why (2) is m=16 only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
RAW="$HERE/raw"
mkdir -p "$RAW"

FLEETS=(8 12 16 20 24)
JOB_RULES=(milk_run lpt)
COUNT=100
HEADLINE_M=16

cd "$REPO/Static_alogorithm"

# --- 1. fleet curve, n=60 -------------------------------------------------
OUT_FLEET="$RAW/final_fleet_test60.jsonl"
rm -f "$OUT_FLEET"
for jr in "${JOB_RULES[@]}"; do
  for m in "${FLEETS[@]}"; do
    "$PY" sweep_fleet.py --amrs "$m" --count "$COUNT" --job_rule "$jr" \
      --inbox "$REPO/test_case/v3/test_60.jsonl" --out "$OUT_FLEET"
  done
done

# --- 2. parcel-count check at m=16 ---------------------------------------
for n in 120 240; do
  OUT_N="$RAW/parcels_${n}_m${HEADLINE_M}.jsonl"
  rm -f "$OUT_N"
  for jr in "${JOB_RULES[@]}"; do
    "$PY" sweep_fleet.py --amrs "$HEADLINE_M" --count "$COUNT" --job_rule "$jr" \
      --inbox "$REPO/test_case/v3/test_${n}.jsonl" --out "$OUT_N"
  done
done

echo
echo "fleet curve  -> $OUT_FLEET"
echo "parcel check -> $RAW/parcels_120_m${HEADLINE_M}.jsonl, $RAW/parcels_240_m${HEADLINE_M}.jsonl"
