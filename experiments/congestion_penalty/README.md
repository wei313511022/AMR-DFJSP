# Experiment 1 — the congestion-penalty curve Λ(η)

**Question.** How much makespan does the fixed-travel-time abstraction used by the
FJSP-T and learned-dispatcher literature *lose* when the same schedule is executed
by a collision-aware router against exclusive service points — and how does that
loss grow with contention?

This is the measurement the paper's motivation rests on. If Λ is small, modelling
congestion is not worth a paper; if Λ is large and dominated by *queueing* rather
than *routing*, then the error is concentrated in exactly the quantity an
assignment policy controls, which is the argument for executor-relative training.

---

## 1. Design

One controlled factor. Layout, workload, rack, size mix, executor and instance set
are all held fixed; the only thing that moves is fleet size `m`, and therefore the
contention ratio

```
eta = m / |D_in| = m / 5
```

| factor | setting |
|---|---|
| fleet sizes `m` | 8, 12, 16, 20, 24 → η = 1.6, 2.4, 3.2, 4.0, 4.8 |
| instances | 30 × 60 parcels, `test_case/v3/instances_60.jsonl`, indices 0–29 |
| grid | 20 × 19, open floor, four-connected, unit speed |
| service points | 5 inbound doors (x=0) and 5 outbound stations (x=19), rows 2/6/10/14/18 |
| waiting line | depth 3, single file running inward along each service row |
| depot | aisle rows y = 4/8/12/16, filled column-major from x=0; 24 bays total |
| rack | one slot per class, nested: `n_C ≤ 1`, `n_B+n_C ≤ 2`, `n_A+n_B+n_C ≤ 3` |
| service times | p(A)=5, p(B)=10, p(C)=15, equal at both ends |
| size mix | uniform over A/B/C |
| releases | all parcels at t = 0 |
| door clearance | θ = 1 tick |

`m = 24` is the depot's hard capacity — `aisle_bays()` offers 6 columns × 4 aisle
rows. Beyond that `build_pod_depot` raises rather than cycling, because cycling
would give two AMRs the same bay and block it from t=0.

`scenario_v3.apply_layout` rebuilds and **validates** the depot at every sweep
point, so no point can quietly run on a geometry that walls off the floor or parks
robots inside a dock queue.

## 2. Which rules, and why

A schedule has to come from *somewhere* before it can be executed, and Λ is a
property of the (schedule, executor) pair. So the sweep generates schedules with
dispatching rules and measures Λ on what they produce. Every rule is a pair
`job_rule + amr_rule`: the job rule picks which parcel to act on next, the AMR
rule picks which robot does it.

### Job rules — two, chosen to stress the doors differently

| rule | behaviour | why it is here |
|---|---|---|
| `milk_run` | keeps collecting while a nearby pickup exists and the rack has room, then delivers the batch | the **batching** regime: fills the rack, holds a door across consecutive services, and is the strongest baseline in the full rule grid |
| `lpt` | longest processing time first, one parcel per tour | the **single-trip** regime: short door holds, proportionally more time in transit |

#### `lpt` in detail

The score is `(-job.duration, job.idx)`, minimised
(`dispatching_rules.py:219`). Service time is exactly the size class — A=5,
B=10, C=15 — so on a uniform mix LPT is precisely "all ~20 C parcels first,
then the ~20 B, then the ~20 A", with parcel index as a deterministic
tiebreak. The score ignores which robot is acting, so the AMR rule
(`earliest_completion` in the headline column) makes every robot choice.

**LPT is strictly single-trip, and provably so** — this is not a tuning
accident. Suppose `pickup(j)` was just chosen, i.e. it was the minimum-scoring
legal action. Then every other available pickup `k` satisfies either
`p_k < p_j`, or `p_k = p_j` with `k.idx > j.idx`. The newly legal `unload(j)`
scores exactly `(-p_j, j.idx)`, which is therefore `<=` every remaining pickup
score. So `unload(j)` is chosen immediately next, every time. Measured over 5
instances:

| job rule | mean parcels onboard after a pickup | max |
|---|---|---|
| `milk_run` | 1.67 | 3 |
| `lpt` | 1.00 | 1 |
| `spt`, `fifo` | 1.00 | 1 |

Consequences that matter for this experiment: LPT never exercises the rack
constraint, occupies a door for exactly one service per visit, and moves its
robots in dock→station→dock round trips. `milk_run` can chain several services
at one door and holds it across them. That is the door-pressure contrast the
two rules are here to provide.

One honest caveat: LPT's reputation comes from `P||C_max`, where it carries
Graham's 4/3 − 1/(3m) bound. Here the parallel resources that bind are the
five doors, not the sixteen robots, so that guarantee does **not** transfer.
LPT is used as a representative single-trip heuristic, not as an approximation
algorithm.

Running both answers `SCENARIO_SPEC_v3.md` known-gap #2, which asks whether the
penalty curve is rule-dependent. If the two curves have the same shape, Λ is a
property of the facility rather than of one heuristic — which is what the paper
needs to claim.

`milk_run` uses `MILK_RUN_RADIUS = 3.0`. Note that doors sit 4–5 cells apart and
the calibrated metric inflates a 4-cell gap to 4.97, so **every radius in [0,4]
means the same thing**: consolidate only at the door you are already at. That is
the measured optimum; radius 20 is catastrophic (2378 makespan).

### AMR rules — all six

`earliest_available`, `earliest_completion`, `least_loaded`, `material_match`,
`nearest_amr`, `random`.

The **headline column is `earliest_completion`**, which is the dominant AMR rule —
best or tied-best for nearly every job rule in the full grid. The other five are
recorded so the curve can be shown to survive the choice of robot rule
(`by_rule.csv`); `random` is kept as a spread reference but excluded from any
"sensible rules" aggregate.

Two rules that used to be in this grid are gone and should not be added back:
`edd`/`atc` died with the deadlines (`atc` also underflowed and collapsed to
"never batch", scoring worst of all 70 combinations), and `home_material` is
incoherent once every AMR carries all three slot classes.

**This is a rule-generated sweep, not a policy evaluation.** It measures the
abstraction's error on schedules of a realistic shape. It does not tell you how a
learned policy will do — that is the matrix-trained ablation, a separate
experiment.

## 3. What is measured

Each of the 1800 runs (2 job rules × 5 fleets × 30 instances × 6 AMR rules) emits
one JSON row to `raw/sweep.jsonl` via `ideal_evaluator.evaluate`:

- **`executed`** — `C_max` under the collision-aware executor Φ: space–time A* on a
  shared reservation table, one-cell following gap, no head-on swaps, exclusive
  service points, depth-3 waiting lines, θ=1 door clearance, and each robot's
  return to its bay.
- **`ideal`** — `C̃`, the idealised makespan: for each robot, the sum of free-space
  Manhattan distances plus service times along its committed sequence, plus the
  return leg; then the max over robots. This is the fixed-travel-time model prior
  work actually uses — service starts on arrival, no queueing, no collisions.
- **`penalty`** — `Λ = (executed − ideal) / ideal`. The headline quantity.
- **`omega_q`, `omega_r`** — fleet queueing and routing delay ratios, both over
  `Σ_k C̃_k`. Queueing is the executor's `wait_inbound_line`, `wait_outbound_line`
  and `hold_upstream` spans; routing is executed travel beyond free-space travel.
  Overlapping time is counted once. These diagnose robot-time inflation; they are
  **not** an additive decomposition of makespan and not independent causal effects.
- **`nu`** — routing failures, reported separately and never folded into makespan.

The comparator is deliberately the *idealised* model and **not** a collision-free
decode with dock exclusivity. That older reference already contains the queueing,
so it subtracts out precisely the term prior work omits.

### On the infeasibility rate

`nu` is reported for every sweep point (`nu_per_episode` and `clean/instances` in
`summary.txt`, raw per-run values in `sweep.jsonl`). It must be, for two reasons:
it is the precondition that licenses every makespan mean in the table, and a
sweep point where it drifted upward would mean Λ was measuring routing failure
rather than congestion.

**But it is identically zero here, and that is itself the finding.** Every
rule-generated run in this sweep routes cleanly, at every fleet size, under both
job rules. So the infeasibility rate cannot be used as evidence against
dispatching rules — on this axis the rules are flawless.

It is worth being precise about where the non-zero numbers in this project come
from, because the two are easy to conflate. The 21–26% → 0–1% figures recorded in
`GA/GA.py` are the *share of **sampled** schedules with at least one un-executable
job*, measured on val_60 with seeds 43/44. Those are schedules drawn from the
policy-rollout distribution, not produced by dispatching rules. The distinction is
real:

- **Dispatching rules occupy a benign corner of schedule space.** They are
  conservative and locally greedy, they keep each robot on simple dock→station
  cycles, and they never construct the pathological interleavings that strand a
  robot on a door with all three neighbours busy.
- **A learning policy explores the whole space**, including that pathology — which
  is exactly why executor-in-the-loop training is needed, and why the idealised
  model is dangerous there. `C̃` assigns a perfectly good score to a schedule the
  router cannot execute at all.

So infeasibility belongs in the paper as an argument about **schedule
distributions and why a policy needs the executor in the loop**, not as an
argument against dispatching rules. The case against the rules here is Λ: they are
systematically optimistic under the abstraction, by a margin that grows with
contention. Reported as a column of zeros, `nu` plays the role of a control — it
certifies that the geometry is clean, so that Λ is congestion and nothing else,
and that no filtering bias enters from dropping failed runs.

## 4. Two analysis rules that are not optional

1. **Never pool failed runs into a makespan mean.** On failure the executor charges
   `availability[amr] += MAX_DEPTH` (=100) — a penalty constant, not elapsed time.
   Several early "congestion" spikes in this project were that arithmetic: one
   apparent 33.9% collapsed to 16.3% once restricted to cleanly-routed runs.
   `analyze.py` excludes `nu > 0` from every timing column and reports the clean
   count alongside.

2. **Pair by instance.** All fleet sizes are scored on the same 30 instances, so
   the fleet-to-fleet difference is a within-instance quantity. Its confidence
   interval is computed from per-instance deltas, not from the pooled spread of
   absolute makespans — that spread is dominated by instance difficulty and would
   hide a real effect. Λ itself is averaged per-instance rather than computed as
   (mean executed)/(mean ideal), which would silently weight easy and hard
   instances differently.

## 5. Reproduce

```bash
bash run_sweep.sh     # ~20 min, 1800 runs at ~0.6 s each
python analyze.py     # writes curve.csv, by_rule.csv, summary.txt, fig_penalty_curve.pdf
```

`run_sweep.sh` deletes `raw/sweep.jsonl` first, because `sweep_fleet.py` appends
and a stale file would silently double-count.

**Requires the working-tree `aisle_bays(width=6)`.** On the committed `width=4` the
depot offers only 16 bays and `m=20` raises immediately.

## 6. Files

| file | contents |
|---|---|
| `run_sweep.sh` | the exact sweep, reproducible |
| `analyze.py` | pairing, CIs, CSV + figure |
| `raw/sweep.jsonl` | one row per run — the primary record |
| `curve.csv` | headline curve, `earliest_completion`, both job rules |
| `by_rule.csv` | the same curve under all six AMR rules |
| `summary.txt` | printed tables incl. paired within-instance deltas |
| `fig_penalty_curve.pdf` | Λ vs η, and the queueing/routing split |

## 7. Results

See [RESULTS.md](RESULTS.md). Final run: 100 clean test instances. Λ = 20.9% ± 1.9 at
m=16 with 88% of it queueing; Λ keeps growing to η=4.8 (no plateau); Λ is flat
across n = 60/120/240; 7 routing failures in 8400 runs, none in a headline column.
