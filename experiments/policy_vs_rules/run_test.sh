#!/usr/bin/env bash
# Runs 2, 5, 6 -- the reported numbers. Everything here is on test_60/120/240.
#
# Refuses to start unless README.md is committed and clean: section 3 of that file is the
# pre-registration, and it has to be frozen before any test row exists or it is not a
# pre-registration at all.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
RAW="$HERE/raw"
mkdir -p "$RAW"

cd "$REPO"
if ! git diff --quiet -- experiments/policy_vs_rules/README.md; then
  echo "REFUSING: README.md has uncommitted changes." >&2
  echo "Section 3 is the pre-registration and must be frozen before test rows exist." >&2
  exit 1
fi
if ! git ls-files --error-unmatch experiments/policy_vs_rules/README.md >/dev/null 2>&1; then
  echo "REFUSING: README.md is not committed." >&2
  exit 1
fi
PREREG=$(git log -1 --format=%H -- experiments/policy_vs_rules/README.md)
echo "pre-registration commit: $PREREG"

# The zero-shot baselines in experiments/congestion_penalty/raw/parcels_*_m16.jsonl were
# computed against specific test_120/test_240 files. Those two datasets are uncommitted
# working-tree files, so a git checkout would revert them to their 50-instance versions and
# the join would silently pair different instances under the same ids.
"$PY" - <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
repo = Path(__file__).resolve().parent if False else Path.cwd()
for name, want_n in (("test_60", 100), ("test_120", 100), ("test_240", 100)):
    p = repo / "test_case" / "v3" / f"{name}.jsonl"
    n = sum(1 for _ in p.open())
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    if n != want_n:
        sys.exit(f"REFUSING: {name}.jsonl has {n} instances, expected {want_n}. "
                 f"Was it reverted by a git checkout?")
    print(f"  {name}.jsonl  {n} instances  sha256 {sha[:16]}")
PYEOF

cd "$REPO/Static_alogorithm"
ARM_HEADLINE=gc1
ARM_OTHER=gc1.5

# --- Run 2: HEADLINE, m=16, n=60 -----------------------------------------
# Both arms: the val selection tied, so gc1 is the headline by pre-declared tie-break and
# gc1.5 is reported alongside, labelled exploratory.
OUT="$RAW/policy_test60_m16.jsonl"
rm -f "$OUT"
for arm in "$ARM_HEADLINE" "$ARM_OTHER"; do
  for s in 42 43 44; do
    "$PY" eval_extend_gnn.py \
      --weights "$REPO/checkpoints_v8/ppo_${arm}_s${s}_best.pth" \
      --run_key "ppo_${arm}_s${s}_best" \
      --inbox "$REPO/test_case/v3/test_60.jsonl" --num_amrs 16 \
      --out "$OUT" --manifest "$RAW/manifest_test60.json"
  done
done

# _latest is val-independent, so test(_best) - test(_latest) measures what the
# 201-evaluation selection actually bought. Ablation.
for s in 42 43 44; do
  "$PY" eval_extend_gnn.py \
    --weights "$REPO/checkpoints_v8/ppo_${ARM_HEADLINE}_s${s}_latest.pth" \
    --run_key "ppo_${ARM_HEADLINE}_s${s}_latest" \
    --inbox "$REPO/test_case/v3/test_60.jsonl" --num_amrs 16 --out "$OUT"
done

# --- Run 6: v7 -- PPO vs REINFORCE at matched 2000 epochs ------------------
# stepwise_* are excluded: those runs were killed at epoch 528/536 of 2000, so they are a
# lower bound on REINFORCE rather than a measurement of it.
OUT_V7="$RAW/policy_test60_v7.jsonl"
rm -f "$OUT_V7"
for ck in ppo_s42 episode_s42 clip50_s42; do
  "$PY" eval_extend_gnn.py \
    --weights "$REPO/checkpoints_v7/${ck}_best.pth" --run_key "v7_${ck}_best" \
    --inbox "$REPO/test_case/v3/test_60.jsonl" --num_amrs 16 --out "$OUT_V7"
done

# --- Run 5: zero-shot parcel count ---------------------------------------
for n in 120 240; do
  OUT_N="$RAW/policy_test${n}_m16.jsonl"
  rm -f "$OUT_N"
  for s in 42 43 44; do
    "$PY" eval_extend_gnn.py \
      --weights "$REPO/checkpoints_v8/ppo_${ARM_HEADLINE}_s${s}_best.pth" \
      --run_key "ppo_${ARM_HEADLINE}_s${s}_best" \
      --inbox "$REPO/test_case/v3/test_${n}.jsonl" --num_amrs 16 \
      --out "$OUT_N" --manifest "$RAW/manifest_test${n}.json"
  done
done

echo
echo "pre-registration commit: $PREREG"
echo "rows -> $RAW"
