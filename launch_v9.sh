#!/usr/bin/env bash
# v9: does training across n=20..140 beat training at n=60 only?
#
# 6 runs = 2 arms x seeds {42,43,44}. Everything is held fixed except the training pool:
# same recipe as checkpoints_v8 (PPO, 4000 epochs, grad_clip 1.0), same seeds, and --
# importantly -- the SAME validation set for both arms. Selecting each arm on its own
# distribution would confound "which arm is better" with "which arm had the easier
# selection target", so both are selected on val_mix (20..140).
#
# Also new since v8, and applied to BOTH arms so it cannot confound the comparison:
#   --val_window 5          select on the mean of the last 5 validations, not argmin
#   --validation_interval   100 -> 40 validations instead of v8's 201
#   validation set          280 instances instead of 50
# Together these cut checkpoint-selection bias from roughly -7.8 makespan units to under 1.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
TRAINER="Static_alogorithm/extend_GNN/train_extend_gnn.py"
MIX="test_case/v3/mix"

TRAIN_MIX="$MIX/train_mix_20.jsonl,$MIX/train_mix_40.jsonl,$MIX/train_mix_60.jsonl,$MIX/train_mix_80.jsonl,$MIX/train_mix_100.jsonl,$MIX/train_mix_120.jsonl,$MIX/train_mix_140.jsonl"
VAL_MIX="$MIX/val_mix_20.jsonl,$MIX/val_mix_40.jsonl,$MIX/val_mix_60.jsonl,$MIX/val_mix_80.jsonl,$MIX/val_mix_100.jsonl,$MIX/val_mix_120.jsonl,$MIX/val_mix_140.jsonl"

# PREFLIGHT. load_training_events() only WARNS on a missing path and carries on with a
# partial pool, so a single typo would silently train one arm on 6 sizes instead of 7 and
# nothing in the logs would say so. Load the pools for real and assert the counts first.
"$PY" - "$TRAIN_MIX" "$VAL_MIX" <<'PYEOF'
import sys
sys.path.insert(0, "Static_alogorithm")
from reinforce_baseline import load_training_events
for label, spec, want in (("train_mix", sys.argv[1], 5005), ("val_mix", sys.argv[2], 280)):
    ev = load_training_events("", spec)
    sizes = sorted({len(e["jobs"]) for e in ev})
    if len(ev) != want:
        sys.exit(f"REFUSING: {label} loaded {len(ev)} events, expected {want}")
    if sizes != [20, 40, 60, 80, 100, 120, 140]:
        sys.exit(f"REFUSING: {label} covers sizes {sizes}, expected 20..140")
    print(f"  OK {label}: {len(ev)} events, sizes {sizes}")
ev = load_training_events("test_case/v3/train_60.jsonl", "")
if len(ev) != 5000:
    sys.exit(f"REFUSING: train_60 loaded {len(ev)} events, expected 5000")
print(f"  OK train_60: {len(ev)} events")
PYEOF

COMMON="--rl_method ppo --epochs 4000 --batch_size 8 --samples_per_instance 1 \
  --lr_actor 3e-4 --lr_critic 3e-4 --entropy_coef 0.01 --value_loss_coef 0.5 \
  --load_balance_coef 0.1 --grad_clip 1.0 --clip_eps 0.2 --ppo_epochs 4 \
  --baseline_mode stepwise --baseline_rule milk_run+earliest_completion \
  --validation_inboxes $VAL_MIX --validation_interval 100 --val_window 5"

launch () {  # launch <gpu> <run_name> <seed> <pool-args...>
  local gpu="$1" name="$2" seed="$3"; shift 3
  mkdir -p logs_v9
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" "$TRAINER" $COMMON "$@" \
    --seed "$seed" --run_name "$name" \
    --best_model_path "checkpoints_v9/${name}_best.pth" \
    --latest_checkpoint_path "checkpoints_v9/${name}_latest.pth" \
    > "logs_v9/${name}.log" 2>&1 &
  echo "  GPU$gpu  $name  (seed $seed)  pid $!"
  sleep 2
}

echo "launching 6 runs across 2 GPUs"
# Both arms appear on both GPUs, so any device-level difference cannot align with an arm.
# The mixed arm costs ~2.2x per epoch (its mean instance is larger), so the split is
# 2 mixed + 1 only60 / 1 mixed + 2 only60 to keep the two GPUs roughly level.
launch 0 mix_s42    42 --inboxes "$TRAIN_MIX"
launch 0 mix_s43    43 --inboxes "$TRAIN_MIX"
launch 0 only60_s42 42 --inbox test_case/v3/train_60.jsonl
launch 1 mix_s44    44 --inboxes "$TRAIN_MIX"
launch 1 only60_s43 43 --inbox test_case/v3/train_60.jsonl
launch 1 only60_s44 44 --inbox test_case/v3/train_60.jsonl

echo
echo "logs:   tail -f logs_v9/*.log"
echo "weights: checkpoints_v9/"
