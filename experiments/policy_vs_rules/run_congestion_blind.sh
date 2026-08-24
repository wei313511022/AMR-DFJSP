#!/usr/bin/env bash
# "Plan as though the facility were uncongested, then execute for real."
#
# The same intervention applied to both families, so they are directly comparable:
#
#   RULES   --congestion_blind makes estimated_dock_wait() return 0, so every planner that
#           consults it believes service points are never busy. Travel and service times are
#           unchanged. This is the fixed-travel-time assumption of the FJSP-T literature.
#
#   POLICY  --mask zeroes congestion feature slices at inference. Zero is the LEAST congested
#           value of each, so this installs the same false belief rather than hiding data.
#           The ladder is nested so it localises where congestion enters the policy:
#             L1  the four features proposed (local_density + the three dock queue terms)
#             L2  + service_remaining, committed_workload
#             L3  + estimated_dock_wait  <- the surrogate the policy actually plans against
#             control  four NON-congestion slices, to separate the effect from the
#                      distribution shift that any masking causes
#
# In both cases the EXECUTOR is untouched: full dock exclusivity, waiting lines, clearance
# and collisions. The planner is blind; the world is not.
#
# Caveat that belongs in the writeup: for the policy this is NOT an observability ablation.
# The network was trained WITH these inputs, so any degradation mixes loss of congestion
# signal with distribution shift. The `control` arm bounds the second term.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
RAW="$HERE/raw"
mkdir -p "$RAW"
cd "$REPO/Static_alogorithm"

ARM=gc1

# --- policy: mask ladder, all three seeds --------------------------------
OUT_MASK="$RAW/policy_masked_test60.jsonl"
rm -f "$OUT_MASK"
for lvl in L1 L2 L3 control; do
  for s in 42 43 44; do
    "$PY" eval_extend_gnn.py --model extend_gnn --mask "$lvl" \
      --weights "$REPO/checkpoints_v8/ppo_${ARM}_s${s}_best.pth" \
      --run_key "ppo_${ARM}_s${s}_best_mask${lvl}" \
      --inbox "$REPO/test_case/v3/test_60.jsonl" --num_amrs 16 \
      --out "$OUT_MASK"
  done
done

# --- rules: the full 60-combination grid, planned congestion-blind --------
OUT_RULES="$RAW/rule_grid_blind_test60_m16.jsonl"
rm -f "$OUT_RULES"
for jr in fifo spt lpt nearest_station most_congested_station least_congested_station \
          earliest_completion_job material_match milk_run random; do
  "$PY" sweep_fleet.py --amrs 16 --count 100 --job_rule "$jr" --congestion_blind \
    --inbox "$REPO/test_case/v3/test_60.jsonl" --out "$OUT_RULES"
done

echo
echo "policy masked -> $OUT_MASK  ($(wc -l < "$OUT_MASK") rows)"
echo "blind rules   -> $OUT_RULES ($(wc -l < "$OUT_RULES") rows)"
