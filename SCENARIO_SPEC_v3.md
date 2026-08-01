# Scenario specification v3

Supersedes SCENARIO_SPEC_v2.md. Every number marked *(measured)* comes from runs on
this repository's executor during the v3 design session; everything else is a choice
still to be validated.

**What changed from v2, and why**

| v2 | v3 | reason |
|---|---|---|
| outbound deadlines, bi-objective | dropped; pure makespan | tardiness gradient is sparse and flat-then-spiky; deadline generator was degenerate (2 distinct values/instance) |
| cross-docking framing | capacitated multi-robot pickup and delivery | no deadlines left to earn the label |
| 3x3 slot rack (9 slots) | 1 per class (3 slots, nested) | at 9 slots consolidation is harmful and `milk_run` looks artificially weak |
| 12 AMRs, single-column depot | 16 AMRs, 2x2 pods | the column was a wall; see §3 |
| congestion tax vs collision-free decode | penalty vs **idealised** model | the old reference already contained queueing, i.e. it subtracted out the thing prior work omits |

---

## 1. Problem

Capacitated multi-robot pickup and delivery with **exclusive service points**.
Parcels sit at inbound doors at t=0 and must be carried to outbound stations by a
fleet of AMRs with compartmented racks. Objective: minimise makespan, evaluated
under collision-aware execution.

No deadlines, no release times, no storage. All time structure comes from
contention for the doors.

## 2. Floor

Open floor, no obstacles.

| element | value |
|---|---|
| grid | x in [0,19], y in [1,19] — 380 cells, four-connected, unit speed |
| inbound doors | x=0, rows y = 2, 6, 10, 14, 18 |
| outbound stations | x=19, same five rows |
| waiting cells | the two rows adjacent to each service point, 4 cells along the wall (`WAIT_LINE_DEPTH = 3`) |
| **contention ratio** | **eta = m / 5 = 3.2** at the headline fleet |

Doors are **inset one cell from each boundary**. A door on row 1 or 19 loses one of
its two waiting rows to the grid edge and gets half the queue capacity of the others,
which makes the five servers non-exchangeable for reasons unrelated to scheduling.
Insetting also raised the execution penalty *(measured, m=16, 18 clean runs)*:

| door rows | queueing | routing | penalty | queueing share |
|---|---|---|---|---|
| 1, 6, 10, 14, 19 | +8.3% | +6.6% | 14.9% | 56% |
| **2, 6, 10, 14, 18** | **+11.3%** | **+5.5%** | **16.8%** | **67%** |

An earlier test appeared to show that even spacing hurts (`1,6,11,16,19` gave 10.1%
against 15.9%); that comparison used the walled depot and the superseded comparator,
and does not survive re-measurement.

## 3. Fleet and depot

**16 homogeneous AMRs**, unit speed, no kinematics or battery model.

**Depot: 2x2 charging pods at x = 4,5**, one pod beside each door row, extending
away from the mid-row so the central lane stays clear:

```
y=19 ....................
y=18 D...RR.............S
y=17 ....RR..............
y=14 D...RR.............S
y=13 ....RR..............
y=10 D..................S     <- centre lane deliberately clear
y= 7 ....RR..............
y= 6 D...RR.............S
y= 3 ....RR..............
y= 2 D...RR.............S
y= 1 ....................
```

Depot geometry turned out to be the single most consequential parameter in the whole
design, and three of four candidate layouts were broken *(measured, m=16 / m=24)*:

| layout | unroutable m=16 | unroutable m=24 | verdict |
|---|---|---|---|
| single column x=2 | 0.89 | — | column is a **wall**; 3 free rows at m=16, 0 at m=20 |
| block 4x4 at x=2 | 0.06 | 0.56 | bays sat **inside dock queues** (queues reach x=3) |
| block 4x4 at x=5 | 0.00 | 0.17 | clean but low congestion (4.1%) |
| bottom rows y=0,-1 | 0.44 | 3.72 | 12-bay row is a **horizontal wall** |
| **2x2 pods at x=4,5** | **0.00** | n/a | clean *and* highest congestion of the valid layouts |

Two hard constraints for any depot design:

1. **Never a contiguous run** across a full row or column. Idle AMRs hold their cells
   in the reservation table permanently (`next_t >= free_t`), so a packed line is an
   impassable barrier, not a temporary obstacle.
2. **Clear the service-point queues** by at least one cell. `dock_waiting_slots` does
   *not* queue inward along the door's own row: it fills the two **adjacent rows**,
   spanning x=0..3 (verified against the implementation). Bays start at x=4. A robot
   assigned a waiting slot that is another robot's parked bay can never arrive.

Fleet capacity is `4 x (number of pods)` = 20. `apply_layout` raises rather than
cycling — cycling silently gave two AMRs the same bay, which blocks it from t=0.

## 4. Parcels and rack

| property | value |
|---|---|
| count | n = 60 for training; evaluate zero-shot at 60 / 120 / 240 |
| size class | A / B / C, **uniform** |
| service duration | p(A)=5, p(B)=10, p(C)=15, equal at both ends |
| release | r_j = 0 for all j |
| origin / destination | drawn independently of size class |

Uniform, not the realistic 60/30/10 skew: the skew drops mean service time from 10.0
to 7.5, which cut the congestion tax from 15.9% to 5.8% *(measured)*. Signal strength
beat that particular realism argument; record it as a limitation.

**Rack: one slot per size class, with downward substitution.** A class-s parcel takes
a slot of class s or larger, so the binding constraints are suffix sums:

    n_C <= 1,   n_B + n_C <= 2,   n_A + n_B + n_C <= 3

A 3x3 rack (9 slots) was measured well past the useful batch size: `milk_run` scored
445.7 makespan at 9 slots versus **365.3 at 3** *(measured)*. Over-consolidation delays
the first-picked parcel through every later pickup and holds a door across consecutive
services — and the door is the binding resource.

## 5. Objective and evaluation

Minimise makespan under the collision-aware executor. Report separately:
makespan, unroutable parcels `nu`, and the execution penalty of §6.

The executor is a **partial function** — prioritised planning is incomplete. Define

    Phi(a, sigma) -> (C_max, nu),   minimise C_max + kappa * nu

and state that feasibility is **executor-relative**: a schedule this planner cannot
route may be routable by CBS. Frame it as optimising against the router that will
actually be deployed.

**Never pool runs that had failures into a makespan mean.** On failure the code does
`availability[amr] += MAX_DEPTH` (=100), a penalty constant, not elapsed time. Several
early "congestion" spikes in this project were that arithmetic: 33.9% collapsed to
16.3% once restricted to cleanly-routed runs *(measured)*.

## 6. The execution-penalty metric — the paper's central quantity

Compare the executed makespan against the model the FJSP-T / learned-dispatcher
literature actually uses: **fixed travel-time matrix, service starts on arrival, no
queueing, no collisions.**

    ideal    = max over AMRs of sum(dist(prev, next) + p_j) along its committed sequence
    executed = C_max under the collision-aware executor
    penalty  = (executed - ideal) / ideal

Do **not** use the old reference (collision-free decode with dock exclusivity). It
already contains the queueing, so it subtracts out exactly what prior work omits.

Measured, pod depot, cleanly-routed runs only, 3 rules x 6 instances:

| m | eta | idealised | executed | queueing | routing | **penalty** |
|---|---|---|---|---|---|---|
| 8 | 1.6 | 529.4 | 568.8 | +3.9% | +3.5% | 7.4% $^{\dagger}$ |
| 12 | 2.4 | 371.3 | 414.9 | +7.7% | +4.0% | 11.7% $^{\dagger}$ |
| **16** | **3.2** | **301.8** | **352.3** | **+11.3%** | **+5.5%** | **16.8%** |
| 20 | 4.0 | 290.2 | 339.8 | +11.5% | +5.6% | 17.1% $^{\dagger}$ |

$^{\dagger}$ **Measured with the superseded door rows (1,6,10,14,19).** Only m=16 has
been re-measured since the doors were inset. Re-run the whole sweep before the curve
goes in a figure; the m=16 point moved 14.9% -> 16.8% under the change, so the others
will move too.

The queueing share of the error rises from 53% to 67% as the fleet grows (67% at the headline setting). **The
dominant and fastest-growing component of the abstraction's error is the one an
assignment policy controls.** That is the paper's motivation, and it needs neither
deadlines nor the consolidation crossover.

Decompose using the executor's own timeline kinds: `wait_inbound_line`,
`wait_outbound_line`, `hold_upstream` are queueing; `travel`, `return` are routing.

## 7. Baselines

Full grid measured at m=16 *(measured, 10 instances)*: best executed makespan was
`milk_run+earliest_completion` at 385.8, then `most_congested_station+earliest_completion`
387.2 and `lpt+earliest_completion` 389.0. **`earliest_completion` is the dominant AMR
rule**, best or tied-best for nearly every job rule.

- `milk_run` — the batching baseline, and the real target. Tune it: radius must be
  reported as a ratio to inter-door spacing. Doors are 4–5 apart and the calibrated
  metric inflates a 4-cell gap to 4.97, so **every radius in [0,4] means "consolidate
  only at the door you are already at"**, which is optimal. Radius 20 is catastrophic
  (2378 makespan) *(measured)*.
- All job x AMR rule combinations except `home_material` (removed: incoherent once
  every AMR carries all three slot classes, and it was the worst performer everywhere).
- `atc` is **currently broken** — 930.7 makespan, worst of all 70 combinations. The
  urgency term `exp(-slack/(kappa*p_bar))` underflows, every action scores 0, and the
  tiebreak collapses to "never batch". Fix or drop; do not publish as-is.
- GA at matched wall-clock, with an anytime curve.
- CP-SAT relaxation for a lower bound.
- **Matrix-trained ablation** — an identical policy trained on the idealised model,
  evaluated under the executor. This is the direct causal evidence for §6 and the
  single most valuable experiment in the plan.

## 8. Health checks — run before spending GPU time

| check | target | status at m=16 |
|---|---|---|
| execution penalty vs idealised (§6) | 13–18% | **16.8% PASS** |
| unroutable / episode | < 0.1 | **0.00 PASS** |
| AMR-rule spread, sensible rules | > 5% | ~10% within `milk_run` PASS |
| capacity mask rate under `milk_run` | > 5% | 24.6% PASS |

The capacity check **must** use `milk_run`. Every other rule carries exactly 1.00
parcel at a time and never approaches the rack limit, which is why `Q=3/3/3` and
`Q=2/2/2` once produced byte-identical results.

## 9. Known gaps

1. All §6 numbers are 6 instances x 3 rules. Enough to choose the design, not to
   publish. Re-run at 30+ with paired statistics before writing.
2. The penalty curve is rule-dependent; confirm it holds separately for `milk_run`
   and the single-trip rules, which stress the doors differently.
3. m=20 had one run with routing failures (17/18 clean), so 17.1% is slightly
   optimistic.
4. The 72-combination grid was measured with the *old* depot and needs re-running.
5. `atc` needs fixing or removing.
6. Aisle/rack layouts were never re-measured with a correct depot. If §6's 14.9% is
   judged too small, that is the remaining lever — and racks are permissible again
   now that cross-docking is dropped.
