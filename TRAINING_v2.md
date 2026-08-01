# Training the scenario-v2 model

All paths relative to `Static_alogorithm/`. Every checkpoint from v1 is invalid:
the job feature vector changed from 16 to 18 dimensions, so `--init_checkpoint`
against an old `.pth` will fail on a shape mismatch. Train from scratch.

---

## Step 0 — calibrate the scenario first (do not skip)

Health checks currently **fail two of four targets**, and training before fixing
that wastes the GPU time. See "Outstanding issues" below. Decide the size mix,
regenerate, and re-run until `health_check_v2.py` passes.

```bash
python health_check_v2.py --inbox ../test_case/v2/train_60_g100.jsonl --events 20
```

## Step 1 — generate instances

Train on one gamma, evaluate across all three. Keep validation disjoint from training.

```bash
# training set
python generate_instances_v2.py --jobs 60 --count 200 --gamma 1.0 --seed 20260731 \
    --out ../test_case/v2/train_60_g100.jsonl

# validation (different seed)
python generate_instances_v2.py --jobs 60 --count 50  --gamma 1.0 --seed 99991 \
    --out ../test_case/v2/val_60_g100.jsonl

# evaluation: deadline-tightness sweep + zero-shot size generalisation
for g in 0.8 1.0 1.2; do
  python generate_instances_v2.py --jobs 60 --count 100 --gamma $g --seed 4242 \
      --out ../test_case/v2/test_60_g${g/./}.jsonl
done
for n in 120 240; do
  python generate_instances_v2.py --jobs $n --count 50 --gamma 1.0 --seed 4242 \
      --out ../test_case/v2/test_${n}_g100.jsonl
done
```

## Step 2 — train

```bash
python GNN/train_gnn.py \
    --rl_method reinforce \
    --inbox ../test_case/v2/train_60_g100.jsonl \
    --validation_inbox ../test_case/v2/val_60_g100.jsonl \
    --validation_interval 10 \
    --validation_invalid_penalty 100 \
    --baseline_rule "edd+material_match" \
    --baseline_mode stepwise \
    --epochs 400 --batch_size 8 \
    --lr_actor 1e-4 --entropy_coef 0.01 --load_balance_coef 0.0 \
    --normalize_advantage \
    --best_model_path ../checkpoints_v2/gnn_v2_best.pth \
    --latest_checkpoint_path ../checkpoints_v2/gnn_v2_latest.pth \
    --seed 42
```

Three choices worth understanding rather than copying:

**`--baseline_rule edd+material_match`.** Chosen from the full 70-combination grid
(6 instances, n=60, uniform mix). It is the best rule on the combined objective
F = makespan + tardiness + 100*nu, at F = 483 against 595 for the runner-up.

The counterfactual baseline must be able to express the behaviour you want the
policy to beat. A deadline-blind reference such as `earliest_completion_job`
produces a deadline-blind advantage signal, and the policy inherits that blindness
— it would be rewarded for finishing early rather than finishing on time.

**`--load_balance_coef 0.0`.** The load-balance shaping term was tuned for the pure
makespan objective. With tardiness in the signal it pulls against deadline
adherence. Turn it on later as an ablation, not by default.

**`--baseline_mode stepwise`.** This is the expensive one: two executor rollouts per
decision, so O(n²) collision-aware executions per episode. At n=60 that is roughly
7,200 executor calls per episode. Budget accordingly, and run the `episode` mode
ablation to confirm the cost is justified — if it is not, the honest result is that
the episode baseline suffices.

## Step 3 — the ablation that matters most

Train an identical policy on a fixed travel-time model and evaluate it under the
real executor. This is the direct causal evidence for the paper's premise and it
costs one extra training run.

```bash
# 1. neutralise the calibration so the surrogate is a pure distance matrix
cp calibration/fast_model_calibration.json calibration/_backup.json
python -c "import json; json.dump({'travel':{'scale':1.0,'offset':0.0}, \
    'dock_wait_penalty':[0,0,0,0]}, open('calibration/fast_model_calibration.json','w'))"

# 2. train with collision-aware evaluation disabled in the advantage computation
#    (set check_collision=False in compute_dispatch_baseline_comparison)
python GNN/train_gnn.py ... --best_model_path ../checkpoints_v2/gnn_v2_matrix.pth

# 3. restore, then evaluate BOTH policies under the real executor
mv calibration/_backup.json calibration/fast_model_calibration.json
```

Report both policies' executed performance as a function of the contention ratio
eta. If the matrix-trained policy degrades as eta rises while the executor-trained
one does not, the paper's central claim is established.

## Step 4 — evaluate

```bash
python benchmark_static_algorithms.py \
    --algorithms gnn,ga,milk_run+earliest_completion,edd+earliest_completion,\
atc+earliest_completion,material_match+earliest_completion \
    --inbox ../test_case/v2/test_60_g100.jsonl
```

Report makespan, total tardiness, late-shipment count, and unroutable parcels
separately. Do not report the scalarised objective as a headline number — it hides
the trade-off behind an arbitrary `w_T`.

---

## The headline finding: a strategy crossover at eta ~ 2.6

Tardiness is dropped; the objective is pure makespan (`use_objective=False`). The
paper's mechanism is now the consolidation/contention interaction:

| m | eta | milk_run | best single-trip rule | milk_run edge | tax | unroutable |
|---|---|---|---|---|---|---|
| 8 | 1.6 | 438.0 | 594.0 | **-26.3%** | 1.9% | 0.00 |
| 10 | 2.0 | 395.0 | 499.2 | -20.9% | 3.4% | 0.04 |
| 12 | 2.4 | 358.2 | 417.0 | -14.1% | 3.4% | 0.00 |
| 14 | 2.8 | 387.4 | 377.8 | **+2.5%** | 9.4% | 0.56 |
| 16 | 3.2 | 420.0 | 392.7 | +7.0% | 23.4% | 1.87 |
| 18 | 3.6 | 410.0 | 368.0 | +11.4% | 23.5% | 2.00 |

Below eta ~ 2.6 consolidation wins by up to 26%; above it, consolidation *loses* by
up to 11%. The optimal strategy inverts. A dispatching rule commits to one side of
the crossover for the entire episode; a policy conditioned on live dock congestion
does not have to. That is the argument for learning here, and it needs no deadlines.

Mechanism: a batching robot holds an inbound door across consecutive services.
That is free when doors are idle and expensive when eleven other robots are queued.

## Outstanding issues, in priority order

**1. Congestion tax is 5.8%, target 13–18%.** The cause is the size mix. Measured on
identical layouts:

| size mix | mean service time | congestion tax | AMR-rule spread |
|---|---|---|---|
| uniform A/B/C | 10.0 | **15.9%** | 16.3% |
| 60/30/10 (v2 default) | 7.5 | 5.8% | 10.2% |

The realistic skew I recommended costs two thirds of the effect the paper depends
on, because shorter services mean less dock occupancy and therefore less queueing.
This is a real trade-off between fidelity and signal strength, and it needs a
decision before training. Options: revert to a uniform mix; use an intermediate
mix such as 40/35/25; or keep 60/30/10 and raise the fleet to restore eta. I would
test 40/35/25 first — it keeps some skew while retaining the larger parcels that
drive dock occupancy.

**2. The capacity-mask diagnostic is measuring the wrong thing.** It reports 0.0%
because `material_match` and friends never batch — they pick one parcel and deliver
it immediately, so they never approach the rack limit no matter how small it is.
That is also why `Q=3/3/3` and `Q=2/2/2` produce byte-identical results. Rerun the
diagnostic against `milk_run`, which is the only rule that deliberately fills the
rack, and size the slots against that.

**3. Unroutable parcels 0.19/episode, target < 0.1.** Concentrated in the `random`
AMR rule (1.00/episode). Prioritised planning without replanning is the cause. This
is the most likely reviewer attack and it is a planner problem, not a scenario one.

**4. Door rows are load-bearing.** `(1, 6, 10, 14, 19)` gives 15.9% tax; the tidier
`(1, 6, 11, 16, 19)` gives 10.1% on identical instances. A door on the centre row
forces cross-traffic through the depot column. Do not regularise the spacing for
readability without re-running the health check.

---

## What changed in the code

| File | Change |
|---|---|
| `scenario_v2.py` | new — layout, nested rack, size mix, shipments, deadlines |
| `generate_instances_v2.py` | new — instance generator with shipments and deadlines |
| `health_check_v2.py` | new — the four scenario health checks |
| `GA/GA.py` | `Job` gains `shipment_id`/`deadline`; `rack_can_load` nested capacity; `decode_schedule` gains `completion_out`; `evaluate_solution` reports makespan/tardiness/late/unroutable |
| `operation_policy.py` | `legal_actions` uses the nested rack constraint |
| `dispatching_rules.py` | nested capacity; `milk_run` now obeys the same rack as the policy; `home_material` removed; `edd` and `atc` added |
| `GNN/GNN.py` | job features 16 -> 18 (deadline slack, shipment progress); default `job_in_dim=18` |
| `reinforce_baseline.py` | `evaluate_objective` and `score_solution`; counterfactual advantage now sees tardiness |

`USE_NESTED_CAPACITY = False` in `GA/GA.py` recovers the v1 strict per-class rule
for comparison.
