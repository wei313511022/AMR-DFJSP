#!/usr/bin/env bash
# The complete dispatching-rule grid on test_60: 10 job rules x 6 AMR rules x 100 instances.
#
# The congestion sweep only ever ran milk_run and lpt, which is enough to characterise the
# facility but not enough to say what the strongest rule baseline actually is. A learned
# policy compared against a hand-picked subset of rules is comparing against a straw man,
# so this measures the whole grid on the same instances, with the same executor, and now
# with per-instance compute time.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
OUT="$HERE/raw/rule_grid_test60_m16.jsonl"
mkdir -p "$HERE/raw"
rm -f "$OUT"

cd "$REPO/Static_alogorithm"

# All 10 job rules. `edd`/`atc` are absent from JOB_RULES upstream: they died with the
# deadlines, and atc additionally underflowed (exp(-slack/(kappa*p_bar)) -> 0 for every
# action) and collapsed to "never batch", scoring worst of all 70 combinations.
for jr in fifo spt lpt nearest_station most_congested_station least_congested_station \
          earliest_completion_job material_match milk_run random; do
  "$PY" sweep_fleet.py --amrs 16 --count 100 --job_rule "$jr" \
    --inbox "$REPO/test_case/v3/test_60.jsonl" --out "$OUT"
done

echo
echo "rows -> $OUT  ($(wc -l < "$OUT") of an expected 6000)"
