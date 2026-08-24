#!/usr/bin/env bash
# Experiment 1 -- the congestion-penalty curve Lambda(eta).
#
# Sweeps the fleet size m over a FIXED layout, workload, rack and executor, so
# the only quantity that varies is the contention ratio eta = m / |D_in| = m/5.
# For every (m, instance, job rule, AMR rule) it records the executed makespan
# under the collision-aware executor Phi, the idealised makespan C~ under the
# fixed-travel-time abstraction, and the queueing / routing delay ratios.
#
# Reproduce:  bash run_sweep.sh
# Analyse:    python analyze.py
#
# Runtime is roughly 0.6 s per run; the grid below is 1800 runs (~20 min).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
OUT="$HERE/raw/sweep.jsonl"

# Fleet sizes: eta = 1.6, 2.4, 3.2 (headline), 4.0, 4.8.
# 24 is the depot's hard capacity -- aisle_bays() offers 6 columns x 4 rows.
FLEETS=(8 12 16 20 24)

# Two job rules, deliberately chosen to stress the doors differently:
#   milk_run -- batching; fills the rack, holds a door across consecutive
#               services, and is the strongest baseline in the rule grid.
#   lpt      -- single-trip; one parcel per tour, so door holds are short and
#               the fleet spends proportionally more time in transit.
# SCENARIO_SPEC_v3 known-gap #2 asks whether the penalty curve is rule
# dependent. Running both is what answers it.
JOB_RULES=(milk_run lpt)

# 30 instances of 60 parcels. All rules and all fleet sizes see the SAME 30,
# which is what makes the per-instance pairing in analyze.py legitimate.
COUNT=30

# sweep_fleet.py APPENDS, so a stale file would silently double-count.
rm -f "$OUT"
mkdir -p "$(dirname "$OUT")"

cd "$REPO/Static_alogorithm"
for jr in "${JOB_RULES[@]}"; do
  for m in "${FLEETS[@]}"; do
    "$PY" sweep_fleet.py \
      --amrs "$m" \
      --count "$COUNT" \
      --job_rule "$jr" \
      --inbox "$REPO/test_case/v3/instances_60.jsonl" \
      --out "$OUT"
  done
done

echo
echo "raw rows -> $OUT"
"$PY" sweep_fleet.py --summarise --job_rule milk_run --out "$OUT"
"$PY" sweep_fleet.py --summarise --job_rule lpt      --out "$OUT"
