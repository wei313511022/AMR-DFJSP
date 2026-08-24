#!/usr/bin/env bash
# v10, extended to the 2x3 factorial: 12 runs on 12 CPUs and 2 GPUs.
#
#   cell  congestion channels   training return              seeds
#   A     masked (L3)           idealised C~                 42 43     (44 done)
#   B     full                  idealised C~                 42 43     (44 done)
#   C     masked (L3)           executor Phi                 42 43     (44 done)
#   D     full                  executor Phi                 --        (42 43 44 done = v9 only60)
#   E     masked (L3)           calibrated surrogate Psi^    42 43 44  (NEW)
#   F     full                  calibrated surrogate Psi^    42 43 44  (NEW)
#
# WHY E AND F. A and B price the training return with C~, which experiments/surrogate_fidelity
# measures as a weak ranker: at n=60, 15.3% value error and Kendall tau 0.454 against the
# executor, versus Psi^'s 7.0% and 0.736. An arm beaten by the executor-trained arm may only
# be losing to that strawman. Psi^ is the cheap pipeline a deployment would actually build --
# 0.3 ms against the executor's 85 ms per schedule -- so E/F is what turns "the idealised
# model is bad" into "here is what a practitioner should use instead".
#
# WHY THREE SEEDS. Between-seed sd on this recipe is ~6 makespan units (v9 only60 across
# 42/43/44: 444.98, 433.38, 437.53). Three seeds per cell resolves a gap of ~20 units and
# nothing finer. The A/B/C-vs-D gaps are 28 to 251 units, so three seeds is ample there.
# E-vs-C and F-vs-D may be much smaller, and that is a deliberate EQUIVALENCE question:
# three seeds supports a claim of "within +-16 units (~4%)". If the gap lands between 5 and
# 20 units, extend E, C, F and D to six seeds -- decide that BEFORE looking, not after.
#
# SCHEDULE. 6 concurrent runs, 3 per GPU, each pinned to 2 dedicated cores (cores 0-11) with
# 2 BLAS threads -- 12 CPUs total, leaving 12-23 free. Each slot runs its two jobs in series,
# so the whole set is two waves of ~60-65 h: about 5.5 days. One run at this exact config
# was measured at 1.1 GB of GPU memory, so three per 16 GB card is bounded by compute, not
# memory -- three per card is nonetheless untested here (v10 ran two on GPU0), so watch the
# first hour and drop a slot if the per-epoch time inflates.
#
# WAVE 1 IS THE WHOLE SURROGATE ARM (E and F at all three seeds). Cell D already has three
# seeds, so the decisive F-vs-D comparison is complete and fully powered after wave 1 alone,
# in ~3 days, before the A/B/C seed top-ups in wave 2 have finished. Each cell's seeds are
# also split across both GPUs, so no cell is confounded with a device.
#
#   ./launch_v10_2x3.sh --dry-run   print the plan and exit
#   ./launch_v10_2x3.sh             launch
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
TRAINER="Static_alogorithm/extend_GNN/train_extend_gnn.py"
MIX="test_case/v3/mix"
VAL_MIX="$MIX/val_mix_20.jsonl,$MIX/val_mix_40.jsonl,$MIX/val_mix_60.jsonl,$MIX/val_mix_80.jsonl,$MIX/val_mix_100.jsonl,$MIX/val_mix_120.jsonl,$MIX/val_mix_140.jsonl"
D_ARGS="runs/20260818_204340_only60_s44/args.json"
CORES_PER_RUN=2

COMMON="--rl_method ppo --epochs 4000 --batch_size 8 --samples_per_instance 1 \
  --lr_actor 3e-4 --lr_critic 3e-4 --entropy_coef 0.01 --value_loss_coef 0.5 \
  --load_balance_coef 0.1 --grad_clip 1.0 --clip_eps 0.2 --ppo_epochs 4 \
  --baseline_mode stepwise --baseline_rule milk_run+earliest_completion \
  --inbox test_case/v3/train_60.jsonl \
  --validation_inboxes $VAL_MIX --validation_interval 100 --val_window 5"

cell_mask () { case "$1" in A|C|E) echo L3 ;; B|F) echo none ;; *) echo "bad cell $1" >&2; exit 1 ;; esac; }
cell_eval () { case "$1" in A|B) echo ideal ;; C) echo executor ;; E|F) echo surrogate ;; *) exit 1 ;; esac; }

# slot -> its two jobs, in order. Slot s runs on GPU (s % 2) and cores 2s,2s+1.
SLOT_JOBS=(
  "E:42 A:43"   # slot 0  GPU0  cores 0,1
  "E:43 A:42"   # slot 1  GPU1  cores 2,3
  "F:42 B:43"   # slot 2  GPU0  cores 4,5
  "F:43 B:42"   # slot 3  GPU1  cores 6,7
  "E:44 C:43"   # slot 4  GPU0  cores 8,9
  "F:44 C:42"   # slot 5  GPU1  cores 10,11
)

# ---------------------------------------------------------------- worker mode
if [ "${1:-}" = "--worker" ]; then
  slot="$2"; shift 2
  gpu=$(( slot % 2 ))
  cores="$(( slot * CORES_PER_RUN ))-$(( slot * CORES_PER_RUN + CORES_PER_RUN - 1 ))"
  for job in "$@"; do
    cell="${job%%:*}"; seed="${job##*:}"
    name="v10_${cell}_s${seed}"
    echo "[slot $slot] $(date '+%F %T') starting $name on GPU$gpu cores $cores"
    # Pinning matters for more than tidiness: six unpinned PyTorch processes each spawn a
    # thread pool sized to all 24 cores and spend the run fighting over them.
    taskset -c "$cores" env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS="$CORES_PER_RUN" MKL_NUM_THREADS="$CORES_PER_RUN" \
      OPENBLAS_NUM_THREADS="$CORES_PER_RUN" NUMEXPR_NUM_THREADS="$CORES_PER_RUN" \
      "$PY" "$TRAINER" $COMMON \
        --seed "$seed" \
        --train_mask "$(cell_mask "$cell")" \
        --train_evaluator "$(cell_eval "$cell")" \
        --run_name "$name" \
        --best_model_path "checkpoints_v10/${name}_best.pth" \
        --latest_checkpoint_path "checkpoints_v10/${name}_latest.pth" \
        > "logs_v10/${name}.log" 2>&1 && status=0 || status=$?
    echo "[slot $slot] $(date '+%F %T') finished $name (exit $status)"
    # A crashed run must not take its slot-mate down with it: 60 h of the other cell is
    # still worth having, and `set -e` would otherwise end the slot here.
    if [ "$status" -ne 0 ]; then
      echo "[slot $slot] WARNING: $name FAILED -- see logs_v10/${name}.log"
    fi
  done
  echo "[slot $slot] $(date '+%F %T') slot done"
  exit 0
fi

# ------------------------------------------------------------------ preflight
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# 1. every shared setting still matches cell D, or A/B/C/E/F are not a factorial with it.
"$PY" - "$D_ARGS" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
want = {"rl_method":"ppo","epochs":4000,"batch_size":8,"samples_per_instance":1,
        "lr_actor":3e-4,"lr_critic":3e-4,"entropy_coef":0.01,"value_loss_coef":0.5,
        "load_balance_coef":0.1,"grad_clip":1.0,"clip_eps":0.2,"ppo_epochs":4,
        "baseline_mode":"stepwise","baseline_rule":"milk_run+earliest_completion",
        "validation_interval":100,"val_window":5,
        "inbox":"test_case/v3/train_60.jsonl"}
bad = [f"{k}: D has {d.get(k)!r}, this script uses {v!r}"
       for k, v in want.items() if d.get(k) != v]
if bad:
    sys.exit("REFUSING: the new cells would not match cell D:\n  " + "\n  ".join(bad))
print("  preflight: shared config matches cell D")
PYEOF

# 2. the surrogate path must actually be in the trainer -- the whole point of E and F.
grep -q '"surrogate"' "$TRAINER"           || { echo "REFUSING: $TRAINER has no surrogate arm" >&2; exit 1; }
grep -q "NATIVE_SELECTION" "$TRAINER"      || { echo "REFUSING: $TRAINER lacks NATIVE_SELECTION" >&2; exit 1; }
grep -q "val_surrogate" "$TRAINER"         || { echo "REFUSING: $TRAINER lacks the val_surrogate column" >&2; exit 1; }
"$PY" -c "
import sys; sys.path.insert(0, 'Static_alogorithm')
from surrogate_evaluator import surrogate_makespan" \
  || { echo "REFUSING: surrogate_evaluator does not import" >&2; exit 1; }
echo "  preflight: surrogate arm present and importable"

# 3. data.
for f in $(echo "$VAL_MIX" | tr ',' ' ') test_case/v3/train_60.jsonl; do
  [ -f "$f" ] || { echo "REFUSING: missing $f" >&2; exit 1; }
done
echo "  preflight: training and validation sets present"

# 4. never clobber a finished run. checkpoints_v10 already holds the s44 A/B/C weights.
mkdir -p logs_v10 checkpoints_v10
clash=""
for spec in "${SLOT_JOBS[@]}"; do
  for job in $spec; do
    name="v10_${job%%:*}_s${job##*:}"
    [ -e "checkpoints_v10/${name}_best.pth" ] && clash="$clash $name"
  done
done
[ -n "$clash" ] && { echo "REFUSING: checkpoints already exist for:$clash" >&2
                     echo "  move them aside first; this script never overwrites weights." >&2; exit 1; }
echo "  preflight: no checkpoint would be overwritten"

# 5. the GPUs must be idle. Six runs plus a forgotten job is how you OOM on day three.
"$PY" - <<'PYEOF'
import subprocess, sys
out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True)
if out.returncode != 0:
    sys.exit("REFUSING: nvidia-smi failed; cannot confirm the GPUs are free")
busy = []
for line in out.stdout.strip().splitlines():
    idx, used, total = (int(x) for x in line.split(","))
    print(f"  preflight: GPU{idx} {used} MiB / {total} MiB used")
    if used > 2000:
        busy.append(f"GPU{idx} has {used} MiB in use")
if busy:
    sys.exit("REFUSING: " + "; ".join(busy) + " -- 3 runs per GPU needs the card to itself")
PYEOF

command -v taskset >/dev/null || { echo "REFUSING: taskset not found (util-linux)" >&2; exit 1; }
ncpu=$(nproc)
need=$(( ${#SLOT_JOBS[@]} * CORES_PER_RUN ))
[ "$ncpu" -ge "$need" ] || { echo "REFUSING: need $need cores, host has $ncpu" >&2; exit 1; }
echo "  preflight: pinning $need of $ncpu cores (cores 0-$(( need - 1 )))"

# --------------------------------------------------------------------- launch
echo
echo "PLAN -- 12 runs, ${#SLOT_JOBS[@]} concurrent, $CORES_PER_RUN cores each, 2 GPUs"
printf '  %-6s %-5s %-7s  %-14s %-14s\n' slot GPU cores wave1 wave2
for slot in "${!SLOT_JOBS[@]}"; do
  read -r j1 j2 <<< "${SLOT_JOBS[$slot]}"
  printf '  %-6s %-5s %-7s  %-14s %-14s\n' "$slot" "$(( slot % 2 ))" \
    "$(( slot * CORES_PER_RUN ))-$(( slot * CORES_PER_RUN + CORES_PER_RUN - 1 ))" \
    "v10_${j1%%:*}_s${j1##*:}" "v10_${j2%%:*}_s${j2##*:}"
done
echo "  wave 1 = the complete E/F surrogate arm; wave 2 = A/B/C seed top-ups"
echo "  git HEAD $(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo ' (WORKING TREE DIRTY)')"

if [ "$DRY" = 1 ]; then echo; echo "dry run -- nothing launched"; exit 0; fi

echo
for slot in "${!SLOT_JOBS[@]}"; do
  # setsid so the slots outlive this shell and its terminal.
  setsid nohup "$0" --worker "$slot" ${SLOT_JOBS[$slot]} \
    >> "logs_v10/slot${slot}.log" 2>&1 &
  echo "  launched slot $slot (pid $!) -> logs_v10/slot${slot}.log"
  sleep 3
done

cat <<'EOF'

  progress:  tail -f logs_v10/slot*.log
  a run:     tail -f logs_v10/v10_E_s42.log
  curves:    runs/*_v10_*/metrics.csv  (val_score = executor, val_ideal = C~, val_surrogate = Psi^)
  weights:   checkpoints_v10/
               *_best.pth       executor-selected -- the primary factorial, every cell
               *_bestideal.pth  C~-selected       -- cells A and B only
               *_bestsurr.pth   Psi^-selected     -- cells E and F only
  stop all:  pkill -f 'train_extend_gnn.py.*v10_'
EOF
