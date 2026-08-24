#!/usr/bin/env bash
# Runs 7-9 -- ablations. Each isolates one confound that would otherwise make the headline
# number incomparable to something the reader already has in hand.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
RAW="$HERE/raw"
mkdir -p "$RAW"
cd "$REPO/Static_alogorithm"

ARM=gc1
HEADLINE_SEED=42

# --- Run 7: zero-shot fleet transfer -------------------------------------
# The actor builds its pairwise tensor at (B, 2, num_amrs, num_jobs, 4H+3) with num_amrs read
# from the input (extend_GNN.py:598), so the architecture is fleet-agnostic. Whether the
# LEARNED policy transfers is a different question. Rule baselines for all five fleet sizes
# already exist in congestion_penalty/raw/final_fleet_test60.jsonl.
OUT_FLEET="$RAW/policy_fleet_test60.jsonl"
rm -f "$OUT_FLEET"
for m in 8 12 20 24; do
  for s in 42 43 44; do
    "$PY" eval_extend_gnn.py \
      --weights "$REPO/checkpoints_v8/ppo_${ARM}_s${s}_best.pth" \
      --run_key "ppo_${ARM}_s${s}_best" \
      --inbox "$REPO/test_case/v3/test_60.jsonl" --num_amrs "$m" \
      --out "$OUT_FLEET"
  done
done

# --- Run 8: search budget, best-of-K at K = 8 and 16 ----------------------
# K rollouts is a budget the single-shot rules do not have. Analysed beside a matched-budget
# "oracle rule" column -- per-instance min over the 12 rule combos already on disk -- which
# is the honest counterpart: both generate a set and keep the best by executed makespan.
#
# Run over all three seeds rather than one. The headline showed between-seed sd (7.30)
# comparable to the treatment effect, so a single-seed answer to "what does search buy"
# would not be separable from seed noise. K=8 and K=16 together give the shape of the
# diminishing return, which one K alone cannot.
for K in 8 16; do
  OUT_K="$RAW/policy_test60_k${K}.jsonl"
  rm -f "$OUT_K"
  for s in 42 43 44; do
    "$PY" eval_extend_gnn.py \
      --weights "$REPO/checkpoints_v8/ppo_${ARM}_s${s}_best.pth" \
      --run_key "ppo_${ARM}_s${s}_best_k${K}" \
      --inbox "$REPO/test_case/v3/test_60.jsonl" --num_amrs 16 \
      --samples "$K" --sample_seed 1000 --out "$OUT_K"
  done
done

# --- Run 9: local improvement ---------------------------------------------
# benchmark_static_algorithms.py applies this; validation never did. Probes show it can make
# executed makespan WORSE, because its simplified stage optimises the collision-free decode
# rather than the executed one. The point of the run is the paired effect and the fraction of
# instances it worsens.
"$PY" eval_extend_gnn.py \
  --weights "$REPO/checkpoints_v8/ppo_${ARM}_s${HEADLINE_SEED}_best.pth" \
  --run_key "ppo_${ARM}_s${HEADLINE_SEED}_best_li" \
  --inbox "$REPO/test_case/v3/test_60.jsonl" --num_amrs 16 \
  --local_improve --li_simplified 1000 --li_collision 100 \
  --out "$RAW/policy_test60_li.jsonl"

echo
echo "rows -> $RAW"
