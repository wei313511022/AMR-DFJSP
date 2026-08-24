#!/usr/bin/env bash
# Relaunch cells A and B of the v10 factorial with the dual-selection code.
#
# The first attempt (runs/superseded/*_v10_{A,B}_s44) started 14:45 on 2026-08-19,
# before the idealised-validation change landed at 16:16. Those runs recorded neither the
# val_ideal column nor the *_bestideal.pth checkpoint, and neither can be recovered after
# the fact -- the C~ selection has to be tracked while training. They were stopped at
# epoch 100/4000 and archived.
#
# Cells C and D are NOT touched: C is executor-trained and D is v9's only60_s44, already
# past epoch 1400. Selecting either under C~ would be meaningless, so they lose nothing.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
TRAINER="Static_alogorithm/extend_GNN/train_extend_gnn.py"
MIX="test_case/v3/mix"
VAL_MIX="$MIX/val_mix_20.jsonl,$MIX/val_mix_40.jsonl,$MIX/val_mix_60.jsonl,$MIX/val_mix_80.jsonl,$MIX/val_mix_100.jsonl,$MIX/val_mix_120.jsonl,$MIX/val_mix_140.jsonl"

# Same preflight as launch_v10.sh: A and B must still match cell D on every shared setting.
"$PY" - "runs/20260818_204340_only60_s44/args.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
want = {"rl_method":"ppo","epochs":4000,"batch_size":8,"samples_per_instance":1,
        "lr_actor":3e-4,"lr_critic":3e-4,"entropy_coef":0.01,"value_loss_coef":0.5,
        "load_balance_coef":0.1,"grad_clip":1.0,"clip_eps":0.2,"ppo_epochs":4,
        "baseline_mode":"stepwise","baseline_rule":"milk_run+earliest_completion",
        "validation_interval":100,"val_window":5,
        "inbox":"test_case/v3/train_60.jsonl","seed":44}
bad = [f"{k}: D has {d.get(k)!r}, script uses {v!r}" for k,v in want.items() if d.get(k)!=v]
if bad: sys.exit("REFUSING: A/B would not match cell D:\n  " + "\n  ".join(bad))
print("  preflight OK: shared config matches cell D")
PYEOF

# Refuse to start if the dual-selection code is not actually present, which is the whole
# reason for the restart.
grep -q "selector_ideal" "$TRAINER" || { echo "REFUSING: $TRAINER lacks selector_ideal" >&2; exit 1; }
grep -q "val_ideal" "$TRAINER"     || { echo "REFUSING: $TRAINER lacks val_ideal" >&2; exit 1; }
echo "  dual-selection code present"

COMMON="--rl_method ppo --epochs 4000 --batch_size 8 --samples_per_instance 1 \
  --lr_actor 3e-4 --lr_critic 3e-4 --entropy_coef 0.01 --value_loss_coef 0.5 \
  --load_balance_coef 0.1 --grad_clip 1.0 --clip_eps 0.2 --ppo_epochs 4 \
  --baseline_mode stepwise --baseline_rule milk_run+earliest_completion \
  --inbox test_case/v3/train_60.jsonl --seed 44 \
  --validation_inboxes $VAL_MIX --validation_interval 100 --val_window 5"

mkdir -p logs_v10 checkpoints_v10
launch () {  # launch <gpu> <cell> <mask> <evaluator>
  local gpu="$1" cell="$2" mask="$3" ev="$4" name="v10_${2}_s44"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" "$TRAINER" $COMMON \
    --train_mask "$mask" --train_evaluator "$ev" --run_name "$name" \
    --best_model_path "checkpoints_v10/${name}_best.pth" \
    --latest_checkpoint_path "checkpoints_v10/${name}_latest.pth" \
    > "logs_v10/${name}.log" 2>&1 &
  echo "  GPU$gpu  cell $cell  mask=$mask  return=$ev  -> $name  pid $!"
  sleep 3
}
# Both on GPU0, as before: keeps the A-vs-B mask contrast on one device.
launch 0 A L3   ideal
launch 0 B none ideal
