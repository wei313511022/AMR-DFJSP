# Training the scenario-v3 model

All paths relative to `Static_alogorithm/`. Every checkpoint from v1 and v2 is
invalid — not because of a shape change (the job feature vector is 16 in both,
v2's "18 dims" was a docstring that was never implemented) but because the
facility changed. Old checkpoints were trained on the 10x10 / 3-AMR floor with
coordinates normalised by 10.0. Train from scratch.

---

## Step 0 — health check first (do not skip)

```bash
python health_check_v3.py --inbox ../test_case/v3/instances_60.jsonl --events 20
```

Targets:

| metric | target | why |
|---|---|---|
| Lambda (execution penalty) | >= 13% | below ~10% the execution-aware premise is not credible |
| queueing share of Lambda | > 50% | the part an assignment policy actually controls |
| AMR-rule spread | > 5% | confirms the assignment decision is worth making |
| capacity mask rate | > 5% | confirms the rack binds (measure under `milk_run` only) |
| unroutable / episode | < 0.1 | executor-relative feasibility holds |

**Lambda is measured against the idealised evaluator**, not against a
collision-free decode. The v2 "congestion tax" compared the executor to
`decode_schedule(check_collision=False)`, which still enforces dock exclusivity,
waiting-line reservations and upstream holds — it already contained the
queueing, so it subtracted out exactly the term prior work omits. The 13–18%
band in SCENARIO_SPEC_v3 was calibrated on that old reference and will need
re-deriving from a 30-instance paired run; expect the true Lambda to read
higher.

## Step 1 — generate instances

```bash
# training set
python generate_instances_v3.py --jobs 60 --count 200 --seed 20260803 \
    --out ../test_case/v3/train_60.jsonl

# validation (different seed, disjoint from training)
python generate_instances_v3.py --jobs 60 --count 50 --seed 99991 \
    --out ../test_case/v3/val_60.jsonl

# zero-shot size generalisation
for n in 60 120 240; do
  python generate_instances_v3.py --jobs $n --count 30 --seed 4242 \
      --out ../test_case/v3/test_${n}.jsonl
done
```

No `--gamma`, no `--reference`: there are no deadlines. All time structure comes
from contention for the doors, so the instance data is just parcels.

## Step 2 — train

```bash
python GNN/train_gnn.py \
    --rl_method reinforce \
    --inbox ../test_case/v3/train_60.jsonl \
    --validation_inbox ../test_case/v3/val_60.jsonl \
    --validation_interval 10 \
    --validation_invalid_penalty 100 \
    --baseline_rule "milk_run+earliest_completion" \
    --baseline_mode stepwise \
    --epochs 400 --batch_size 8 \
    --lr_actor 1e-4 --entropy_coef 0.01 --load_balance_coef 0.0 \
    --normalize_advantage \
    --best_model_path ../checkpoints_v3/gnn_v3_best.pth \
    --latest_checkpoint_path ../checkpoints_v3/gnn_v3_latest.pth \
    --seed 42
```

Three choices worth understanding rather than copying:

**`--baseline_rule milk_run+earliest_completion`.** `milk_run` is the batching
baseline and the real target; `earliest_completion` is the dominant AMR rule,
best or tied-best for nearly every job rule. The v2 default was
`edd+material_match`, which no longer exists — `edd` was a deadline rule.

The counterfactual baseline must be able to express the behaviour you want the
policy to beat. `milk_run` is the rule that commits to consolidation for the
whole episode, so the advantage signal measures exactly what a
congestion-conditioned policy can add over that commitment.

**`--load_balance_coef 0.0`.** Shaping term; turn it on as an ablation, not by
default.

**`--baseline_mode stepwise`.** The expensive one: two executor rollouts per
decision, so O(n^2) collision-aware executions per episode. At n=60 that is
roughly 7,200 executor calls per episode. Run the `episode` mode ablation to
confirm the cost is justified — if it is not, the honest result is that the
episode baseline suffices.

## Step 3 — the ablation that matters most

Train an identical policy on the idealised model and evaluate it under the real
executor. This is the direct causal evidence for the paper's premise and costs
one extra training run.

```bash
# 1. neutralise the calibration so the surrogate is a pure distance matrix
cp calibration/fast_model_calibration.json calibration/_backup.json
python -c "import json; json.dump({'travel':{'scale':1.0,'offset':0.0}, \
    'dock_wait_penalty':[0,0,0,0]}, open('calibration/fast_model_calibration.json','w'))"

# 2. train with collision-aware evaluation disabled in the advantage computation
#    (set check_collision=False in compute_dispatch_baseline_comparison)
python GNN/train_gnn.py ... --best_model_path ../checkpoints_v3/gnn_v3_matrix.pth

# 3. restore, then evaluate BOTH policies under the real executor
mv calibration/_backup.json calibration/fast_model_calibration.json
```

Report both policies as a function of eta. If the matrix-trained policy degrades
as eta rises while the executor-trained one does not, the central claim holds.

## Step 4 — evaluate

```bash
python benchmark_static_algorithms.py --generate \
    --algorithms gnn,ga,milk_run+earliest_completion,lpt+earliest_completion,\
most_congested_station+earliest_completion,material_match+earliest_completion \
    --job_counts 60 --samples 30
```

`--generate` regenerates the case files against the live facility. Reusing case
files written before the v3 layout landed gives you the same dock and station
*names* at the old *coordinates*.

The benchmark now writes `ideal_makespan`, `penalty`, `omega_q`, `omega_r`
alongside `makespan`, and every duration-valued summary column is computed over
**cleanly-routed runs only**. Report makespan, Lambda, and unroutable parcels
separately; there is no scalarised objective to report.

## Step 5 — the contention sweep

```bash
for m in 8 12 16 20; do
  python eval_fleet_size.py --num_amrs $m --events 30 \
      --inbox ../test_case/v3/test_60.jsonl
done
```

Every fleet size goes through `scenario_v3.apply_layout`, so the sweep varies
only eta = m / 5. Fleet capacity is 4 x (number of pods) = 20; going past that
raises rather than cycling, because cycling once gave two AMRs the same bay,
which blocks it from t=0.

---

## Measured at m=16, 30 instances x 60 parcels, `milk_run` job rule

| AMR rule | idealised | executed | Lambda | Omega_q | Omega_r | q-share | nu | clean |
|---|---|---|---|---|---|---|---|---|
| earliest_available | 283.1 | 329.4 | 16.6% | 33.0% | 4.2% | 89% | 0.00 | 30/30 |
| earliest_completion | 284.4 | 330.0 | 16.1% | 33.9% | 4.5% | 88% | 0.00 | 30/30 |
| least_loaded | 277.3 | 332.1 | 20.3% | 32.2% | 4.3% | 88% | 0.00 | 30/30 |
| material_match | 288.5 | 330.2 | 14.6% | 26.9% | 4.1% | 87% | 0.00 | 30/30 |
| nearest_amr | 284.4 | 330.0 | 16.1% | 33.9% | 4.5% | 88% | 0.00 | 30/30 |
| random | 283.6 | 329.0 | 16.3% | 31.8% | 4.4% | 88% | 0.00 | 30/30 |

**Lambda 16.7%, queueing share 88%, unroutable 0.00 across all 180 runs.** The
penalty lands inside the spec's 13–18% band even though it is now measured
against the stricter idealised reference, and the queueing share is well above
the 50% the motivation needs. Reproduce with `run_health_chunk.py`, which
appends per-run rows so the sweep can be built across several short sessions.

### The AMR-rule spread does not survive 30 instances

At 4 instances the spread across AMR rules read 11.8%. At 30 it is **0.93%**
under `milk_run` — the earlier figure was noise. This matters, because "the
assignment decision is worth making" is one of the four health checks.

Under a single-trip job rule the picture inverts (6 instances, `lpt`):

| AMR rule | executed | Lambda | Omega_q |
|---|---|---|---|
| earliest_available | 378.4 | 14.6% | 24.2% |
| earliest_completion | 387.2 | 15.2% | 27.9% |
| least_loaded | 382.2 | 20.0% | 24.3% |
| material_match | 385.2 | 15.4% | 24.1% |
| nearest_amr | 544.8 | 28.2% | 56.1% |
| random | 938.7 | 86.4% | 154.1% |

Spread over sensible rules 53.9%, over all rules 145%. But almost all of that
is `nearest_amr` and `random` being bad; among the four sensible rules that are
actually competitive the range is ~2%.

So the honest reading is: **the choice among reasonable AMR rules is worth
little at m=16; the choice of job rule and the choice to batch at all is worth
a great deal.** `health_check_v3.py` now runs both `milk_run` and `lpt` and
reports spread with `random` excluded, so this cannot be papered over by rule
selection. Decide before training whether the paper's claim rests on the
assignment decision (in which case this needs addressing — a higher eta, or a
tighter rack) or on the batching decision (in which case it is already made).

Note that Omega_q + Omega_r (~37%) exceeds Lambda (~17%). That is expected, not
a bug: Lambda is a max over robots, the Omegas are fleet-summed ratios, and
delay on a non-critical robot moves the Omegas without moving Lambda. They
diagnose where robot-time is lost; they are not an additive decomposition of
makespan.

---

## What changed from v2

| File | Change |
|---|---|
| `GA/GA.py` | v3 facility is now the module DEFAULT (20x19, 5 doors at rows 2/6/10/14/18, 16 AMRs in 2x2 pods at x=4,5). `build_pod_depot` added. `Job` loses `shipment_id`/`deadline`. `evaluate_solution` returns makespan/nu/routable, no tardiness. Grid bounds set explicitly, not derived from occupied cells. `SUPPLY_LOCATIONS` no longer aliases size classes to doors. |
| `scenario_v3.py` | replaces `scenario_v2.py`. No longer owns the layout — re-exports GA's and rebuilds it in place for fleet sweeps. Its contradictory second `SLOT_CAPACITY` (3/3/3) is gone; GA's 1/1/1 is the only definition. No shipments, no deadlines. |
| `ideal_evaluator.py` | **new** — eq. (5) idealised makespan, eq. (6) Lambda, eq. (7) Omega_q/Omega_r, and `aggregate()` which averages over routing-clean runs only. |
| `generate_instances_v3.py` | replaces the v2 generator; no shipments, no deadlines, no reference-rule circularity. |
| `health_check_v3.py` | replaces the v2 check; measures Lambda against the idealised model, not the collision-free decode. |
| `benchmark_static_algorithms.py` | reports ideal/Lambda/Omega per run; `mean_all_makespan` (which pooled MAX_DEPTH penalties into a "makespan") removed. |
| `eval_fleet_size.py` | sweeps through `apply_layout` instead of the v1 bay geometry; reports Lambda. |
| `decompose_time_budget.py` | reports Omega alongside fleet-time shares, and skips failed episodes. |
| `dispatching_rules.py` | `edd` and `atc` removed with deadlines. |
| `operation_policy.py` | `position_scale` / `rack_scale` / `fleet_scale` — shared feature normalisers, replacing the hard-coded 10.0 and 3.0 in five separate encoders. |
| `GNN`, `Attention`, `extend_GNN` | coordinates normalised by floor extent, rack counts by suffix cap, per-AMR counts by fleet size. `extend_GNN` now uses the nested rack instead of the flat `AMR_LOAD_CAPACITY`. |
| `reinforce_baseline.py` | `evaluate_objective` and the `use_objective` switch removed; the credit signal is pure makespan. |

`USE_NESTED_CAPACITY = False` in `GA/GA.py` still recovers the strict per-class
rule for comparison.
