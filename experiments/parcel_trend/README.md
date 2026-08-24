# Parcel-count trend — extend_GNN + PPO vs dispatching rules, n = 20…100

**Question.** `experiments/policy_vs_rules` stage `scaling` measured the policy at
n = 60 / 120 / 240 and found a sign flip: −3.2% at n=60, +6.1% at n=120, +15.6% at
n=240. That leaves a factor-of-two gap with no resolution inside it. This experiment
fills in n = 20, 40, 60, 80, 100 to see the shape of the curve on the near side of the
crossover.

**This is a trend probe, not a powered comparison.** 10 instances per size. Every CI
here is roughly 3× wider than the 100-instance headline. Read the sign and the slope of
delta%, not the individual p-values.

## Instances

`test_case/v3/trend/trend_{20,40,60,80,100}.jsonl` — 10 instances each, generated with

```
python generate_instances_v3.py --jobs N --count 10 --seed 20260818 --amrs 16 \
    --out ../test_case/v3/trend/trend_N.jsonl
```

One generator, one seed, one call per size, so the five points differ only in n.

`trend_60.jsonl` was checked against all 5000 instances of `train_60.jsonl` by parcel
tuple `(type, inbound_dock, station)` per instance: **0 overlap**. The other four sizes
cannot collide with the training set, which is 60-parcel instances only. These are fresh
instances rather than a slice of `test_60.jsonl` so that all five points come from the
same generator call pattern; the n=60 point is therefore *not* a subset of the
pre-registered test set and will not reproduce the headline number exactly.

## What is being compared

| | |
|---|---|
| policy | `checkpoints_v8/ppo_gc1_s{42,43,44}_best.pth` — the pre-registered `gc1` arm |
| decoding | deterministic greedy, 1 rollout, no local improvement (the primary configuration) |
| fleet | m = 16 for every size, so n/robot sweeps 1.25 → 6.25 |
| rules | the full grid — 10 job rules × 6 AMR rules — re-ranked at each size |
| baseline | `milk_run+earliest_completion`, carried over unchanged from `policy_vs_rules/selection.json` |

The baseline rule is **not** re-selected here. It was chosen on `val_60` before any test
row existed and is also the rule the policy was trained against; re-picking it per size
would turn this into a rule search. The strongest rule *at each n* is reported in a
second table and is post-hoc by construction — its purpose is to show whether the rule
ranking itself moves with workload, not to serve as the comparison.

## Harness

- policy rows — `Static_alogorithm/eval_extend_gnn.py`, all four of its guard rails active
- rule rows — `Static_alogorithm/sweep_fleet.py`, one call per job rule (each covers all
  6 AMR rules), which emits the `(amrs, instance, job_rule, rule, family)` join keys
- analysis — `analyze_trend.py`, which imports `analyze_policy.py` rather than
  re-implementing pairing, seed averaging, bootstrap CI, or the permutation test, so the
  two analyses cannot drift apart

Seeds are averaged **within** instance, then across instances. Instances where either
side failed to route (`nu > 0`) are dropped from a pair.

## Reproduce

```
cd Static_alogorithm
for n in 20 40 60 80 100; do
  python eval_extend_gnn.py \
    --weights ../checkpoints_v8/ppo_gc1_s42_best.pth,../checkpoints_v8/ppo_gc1_s43_best.pth,../checkpoints_v8/ppo_gc1_s44_best.pth \
    --run_key ppo_gc1_s42_best,ppo_gc1_s43_best,ppo_gc1_s44_best \
    --inbox ../test_case/v3/trend/trend_${n}.jsonl --num_amrs 16 \
    --out ../experiments/parcel_trend/raw/policy_n${n}.jsonl \
    --manifest ../experiments/parcel_trend/raw/manifest_n${n}.json
done
../experiments/parcel_trend/run_rules.sh
python ../experiments/parcel_trend/analyze_trend.py
```

## Outputs

| File | Contents |
|---|---|
| `trend.csv` | one row per size: seed-averaged policy vs baseline rule and vs the best rule at that size |
| `trend_by_seed.csv` | the same paired stats per seed, unaveraged |
| `rule_ranking.csv` | all 60 rule combinations ranked by executed makespan, per size |
| `trend_summary.txt` | the printed tables |
| `fig_parcel_trend.pdf/.png` | absolute makespan, and delta% vs n |
