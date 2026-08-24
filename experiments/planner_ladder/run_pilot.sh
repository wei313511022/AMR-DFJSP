#!/usr/bin/env bash
# Planner-fidelity ladder, 10-instance pilot.
#
# "Plan under each literature's transport assumptions, then execute for real."
# The EXECUTOR is identical in all three arms: full dock exclusivity, waiting
# lines, clearance and collisions. Only what the PLANNER believes changes.
#
#   ideal   raw Manhattan travel + zero dock wait
#           -> the fixed travel-time matrix of the FJSP-T / PDPTW literature
#   qblind  calibrated travel + zero dock wait
#           -> integrated conflict-free routing: knows legs detour, assumes
#              service begins on arrival
#   full    calibrated travel + calibrated queue-depth penalty
#           -> the surrogate the rules currently plan against
#
# Pilot scale: 10 instances, m=16, 2 job rules x 6 AMR rules = 120 runs/arm.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
RAW="$HERE/raw"
mkdir -p "$RAW"
cd "$REPO/Static_alogorithm"

COUNT=10
M=16
INBOX="$REPO/test_case/v3/test_60.jsonl"

run_arm () {
  local name="$1"; shift
  local out="$RAW/ladder_${name}.jsonl"
  rm -f "$out"
  for jr in lpt milk_run; do
    "$PY" sweep_fleet.py --amrs "$M" --count "$COUNT" --job_rule "$jr" \
      --inbox "$INBOX" --out "$out" "$@"
  done
  echo "  $name -> $out ($(wc -l < "$out") rows)"
}

run_arm full
run_arm qblind --congestion_blind
run_arm ideal  --congestion_blind --travel_blind
