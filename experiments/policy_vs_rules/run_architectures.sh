#!/usr/bin/env bash
# Station-feature ablation: does giving the network dock/station state actually help?
#
# checkpoints_v6 is the matched comparison and the only one available. All three
# architectures were trained by one launcher (train_all_models_parallel.py) under an
# identical recipe -- REINFORCE/multisample, 2000 epochs, seeds 42/43/44, train_60/val_60 --
# so architecture is the only thing that varies:
#
#   gnn         SchedulerGNN,       job_in_dim=16, amr_in_dim=8. NO dock encoder, no
#               dock/AMR attention. This is the "ignores station features" arm.
#   extend_gnn  ExtendSchedulerGNN, adds dock_in_dim=9 and a dock<->AMR cross-attention
#               block on top of the same GIN job encoder.
#   attention   SchedulerAttention, a third architecture, included for context.
#
# Do NOT compare these against the v8 PPO checkpoints to answer the station-feature
# question: those differ in BOTH architecture and RL algorithm (PPO, 4000 epochs), which
# confounds the two. The v8 numbers belong in the table only as a separate reference row.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
OUT="$HERE/raw/architectures_test60_m16.jsonl"
mkdir -p "$HERE/raw"
rm -f "$OUT"
cd "$REPO/Static_alogorithm"

for arch in gnn extend_gnn attention; do
  for s in 42 43 44; do
    "$PY" eval_extend_gnn.py --model "$arch" \
      --weights "$REPO/checkpoints_v6/${arch}_s${s}_best.pth" \
      --run_key "v6_${arch}_s${s}_best" \
      --inbox "$REPO/test_case/v3/test_60.jsonl" --num_amrs 16 \
      --out "$OUT" --manifest "$HERE/raw/manifest_architectures.json"
  done
done

echo
echo "rows -> $OUT ($(wc -l < "$OUT") of an expected 900)"
