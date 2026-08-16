#!/usr/bin/env bash
# Runs 0-1 -- SELECTION ONLY, on val_60. Nothing here may be reported as a result.
#
# `_best.pth` is argmin over 201 validation evaluations on these same 50 instances, so a
# val score is an optimistic order statistic, not a generalisation estimate. val_60 is used
# here for exactly two decisions -- which grad_clip arm, and which dispatching rule the
# headline compares against -- and for the harness self-test. Every reported number comes
# from test_60.
#
# Safe to re-run: both output files are deleted first.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
RAW="$HERE/raw"
mkdir -p "$RAW"
cd "$REPO/Static_alogorithm"

# --- Run 0: rule baseline on val_60 --------------------------------------
# Missing until now. Without it we cannot tell whether val_60 is easier or harder than
# test_60, so the policy's val score (335-347) cannot be placed against the test rule
# field (347.2-350.6).
OUT_RULES="$RAW/rules_val60_m16.jsonl"
rm -f "$OUT_RULES"
for jr in milk_run lpt; do
  "$PY" sweep_fleet.py --amrs 16 --count 50 --job_rule "$jr" \
    --inbox "$REPO/test_case/v3/val_60.jsonl" --out "$OUT_RULES"
done

# --- Run 1: every candidate checkpoint on val_60 --------------------------
OUT_POLICY="$RAW/policy_val60_m16.jsonl"
rm -f "$OUT_POLICY"

# v8: both grad_clip arms x 3 seeds x {best, latest}. `latest` is val-independent, so it
# is the control for how much the 201-evaluation selection actually bought.
for ck in ppo_gc1_s42 ppo_gc1.5_s42 ppo_gc1_s43 ppo_gc1.5_s43 ppo_gc1_s44 ppo_gc1.5_s44; do
  for kind in best latest; do
    "$PY" eval_extend_gnn.py \
      --weights "$REPO/checkpoints_v8/${ck}_${kind}.pth" --run_key "${ck}_${kind}" \
      --inbox "$REPO/test_case/v3/val_60.jsonl" --num_amrs 16 \
      --out "$OUT_POLICY" --manifest "$RAW/manifest_val60.json"
  done
done

# v7: the wave-1 family. stepwise_* were killed at epoch 528/536 of 2000 and are a lower
# bound on REINFORCE, not a measurement of it -- labelled at analysis time.
for ck in ppo_s42 episode_s42 clip50_s42 stepwise_s42 stepwise_clip50_s42; do
  "$PY" eval_extend_gnn.py \
    --weights "$REPO/checkpoints_v7/${ck}_best.pth" --run_key "v7_${ck}_best" \
    --inbox "$REPO/test_case/v3/val_60.jsonl" --num_amrs 16 \
    --out "$OUT_POLICY"
done

echo
echo "rules  -> $OUT_RULES"
echo "policy -> $OUT_POLICY"
