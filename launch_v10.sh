#!/usr/bin/env bash
# v10: the A/B/C/D factorial at seed 44.
#
#   cell  congestion channels        training return
#   A     masked (L3)                idealised C~   (eq. 5)
#   B     full                       idealised C~
#   C     masked (L3)                executor Phi   (eq. 8)
#   D     full                       executor Phi   <- NOT run here
#
# D is v9's `only60_s44`, already training. Every argument below is copied from
# runs/20260818_204340_only60_s44/args.json so that A, B and C differ from D in the two
# factors and NOTHING else -- same seed, same 4000 epochs, same PPO hyperparameters, and
# critically the same val_mix validation set and --val_window 5 selection. Selection
# stays executor-relative in all four cells, so every cell is chosen and reported on the
# deployment objective even when trained against C~.
#
# VALIDATION. Every cell validates on the same val_mix set, scored by the EXECUTOR, so
# the factorial changes one thing per axis. The idealised arms are additionally scored
# under C~ on the same decoded schedules, and a second checkpoint `*_bestideal.pth` is
# selected on C~ alone -- with no invalid penalty, because a practitioner inside the
# fixed-travel-time paradigm cannot see a routing failure either. One run, two selections:
#   *_best.pth       selection held fixed across all four cells (primary factorial)
#   *_bestideal.pth  idealised paradigm end to end (train AND select on C~)
#
# SINGLE SEED. This is a pilot: it can show the direction and rough size of the
# evaluator effect, which we expect to be large. It cannot resolve the observability
# axis or the interaction, because the between-seed spread measured on this recipe
# (~11 makespan units) is larger than either is likely to be.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
TRAINER="Static_alogorithm/extend_GNN/train_extend_gnn.py"
MIX="test_case/v3/mix"
VAL_MIX="$MIX/val_mix_20.jsonl,$MIX/val_mix_40.jsonl,$MIX/val_mix_60.jsonl,$MIX/val_mix_80.jsonl,$MIX/val_mix_100.jsonl,$MIX/val_mix_120.jsonl,$MIX/val_mix_140.jsonl"
D_ARGS="runs/20260818_204340_only60_s44/args.json"

# PREFLIGHT: assert the shared configuration really matches cell D. A silent drift here
# turns the factorial into a comparison of two unrelated recipes.
"$PY" - "$D_ARGS" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
want = {"rl_method":"ppo","epochs":4000,"batch_size":8,"samples_per_instance":1,
        "lr_actor":3e-4,"lr_critic":3e-4,"entropy_coef":0.01,"value_loss_coef":0.5,
        "load_balance_coef":0.1,"grad_clip":1.0,"clip_eps":0.2,"ppo_epochs":4,
        "baseline_mode":"stepwise","baseline_rule":"milk_run+earliest_completion",
        "validation_interval":100,"val_window":5,
        "inbox":"test_case/v3/train_60.jsonl","seed":44}
bad = [f"{k}: D has {d.get(k)!r}, this script uses {v!r}"
       for k, v in want.items() if d.get(k) != v]
if bad:
    sys.exit("REFUSING: A/B/C would not match cell D:\n  " + "\n  ".join(bad))
print(f"  preflight OK: shared config matches cell D ({sys.argv[1]})")
PYEOF

for f in $(echo "$VAL_MIX" | tr ',' ' '); do
  [ -f "$f" ] || { echo "REFUSING: missing validation file $f" >&2; exit 1; }
done
[ -f test_case/v3/train_60.jsonl ] || { echo "REFUSING: missing train_60.jsonl" >&2; exit 1; }

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
    --train_mask "$mask" --train_evaluator "$ev" \
    --run_name "$name" \
    --best_model_path "checkpoints_v10/${name}_best.pth" \
    --latest_checkpoint_path "checkpoints_v10/${name}_latest.pth" \
    > "logs_v10/${name}.log" 2>&1 &
  echo "  GPU$gpu  cell $cell  mask=$mask  return=$ev  -> $name  pid $!"
  sleep 3
}

# GPU0 is at ~39% utilisation with 6.6 GB free; GPU1 at ~73% and holds cell D. Two here,
# one there, keeping the C-vs-D mask contrast on a single device.
echo "launching A, B, C (seed 44). D = v9 only60_s44, already running on GPU1."
launch 0 A L3   ideal
launch 0 B none ideal
launch 1 C L3   executor

echo
echo "logs:    tail -f logs_v10/*.log"
echo "weights: checkpoints_v10/  (*_best.pth = executor-selected, *_bestideal.pth = C~-selected)"
echo "D cell:  checkpoints_v9/only60_s44_{best,latest}.pth"
