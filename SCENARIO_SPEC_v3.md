# Scenario specification v3

Supersedes SCENARIO_SPEC_v2.md. Every number marked *(measured)* comes from runs on
this repository's executor, during either the v3 design session or the geometry revision
after it; everything else is a choice still to be validated. Where the two disagree about
the layout, the geometry revision wins — see the second table below.

**What changed from v2, and why**

| v2 | v3 | reason |
|---|---|---|
| outbound deadlines, bi-objective | dropped; pure makespan | tardiness gradient is sparse and flat-then-spiky; deadline generator was degenerate (2 distinct values/instance) |
| cross-docking framing | capacitated multi-robot pickup and delivery | no deadlines left to earn the label |
| 3x3 slot rack (9 slots) | 1 per class (3 slots, nested) | at 9 slots consolidation is harmful and `milk_run` looks artificially weak |
| 12 AMRs, single-column depot | 16 AMRs, 2x2 pods (since superseded by aisle bays) | the column was a wall; see §3 |
| congestion tax vs collision-free decode | penalty vs **idealised** model | the old reference already contained queueing, i.e. it subtracted out the thing prior work omits |

**What changed after v3 was written** (commit `d2f014f`, plus the `aisle_bays` widening
in the working tree). All three are geometry fixes for the same failure: a service point
sits ON a wall, so an AMR being served there has **three** neighbours, not four, and what
occupies those three decides whether it can leave.

| v3 as written | current | reason |
|---|---|---|
| waiting cells on the two rows adjacent to each door | `QUEUE_GEOMETRY = "inward"` — single file of 3 cells along the door's own row (§2) | the lateral line took two of the three neighbours; the served AMR had one way out |
| 2x2 charging pods at x = 4,5 (§3) | `DEPOT_LAYOUT = "aisle"` — bays on the rows between doors, x = 0..5 (§3) | pods parked AMRs on the door rows themselves, so four of five dock lanes dead-ended into two parked robots |
| door free the instant service ends | `DOCK_CLEARANCE_TICKS = 1` (§5.1) | the next AMR reserved the cell while the incumbent was still standing on it |

The three are complementary and **none is sufficient alone**. Share of sampled schedules
on `val_60` with >=1 un-executable job, seeds 43 / 44 *(measured)*:

| queue | depot | clearance | infeasible |
|---|---|---|---|
| lateral | pods | 0 | 26% / 21%  <- v3–v5 |
| lateral | pods | 1 | 20% / 6% |
| inward | pods | 0 | 6% / 9% |
| inward | aisle | 0 | 6% / 9% |
| **inward** | **aisle** | **1** | **0% / 1%**  <- current default |

Set `QUEUE_GEOMETRY="lateral"`, `DEPOT_LAYOUT="pods"`, `DOCK_CLEARANCE_TICKS=0` to
reproduce the v3–v5 results exactly. Every *(measured)* number in §2, §6, §7 and §8 was
taken under those settings and predates this table; the design conclusions stand, the
magnitudes need re-running.

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
| waiting cells | a **single file of 3 cells running inward from each service point along its own row** (`QUEUE_GEOMETRY = "inward"`, `WAIT_LINE_DEPTH = 3`): dock (0,y) -> (1,y),(2,y),(3,y); station (19,y) -> (18,y),(17,y),(16,y) |
| **contention ratio** | **eta = m / 5 = 3.2** at the headline fleet |

The queue is a single file precisely so that **the two cells flanking a service point are
never queue cells** — they are the served AMR's only exits. `QUEUE_GEOMETRY = "lateral"`
restores the old geometry (the two adjacent rows, x=0..3 at a dock), which cost 26%/21%
feasibility; see the table at the top.

Doors are **inset one cell from each boundary**. The original reason is now moot: under
the lateral queue a door on row 1 or 19 lost one of its two waiting rows to the grid edge
and got half the queue capacity of the others, making the five servers non-exchangeable
for reasons unrelated to scheduling. The inward line does not touch the boundary rows, so
every door row would now carry a full 3-cell queue either way. The rows stay at
2,6,10,14,18 because insetting also raised the execution penalty *(measured, m=16,
18 clean runs, lateral queue)*:

| door rows | queueing | routing | penalty | queueing share |
|---|---|---|---|---|
| 1, 6, 10, 14, 19 | +8.3% | +6.6% | 14.9% | 56% |
| **2, 6, 10, 14, 18** | **+11.3%** | **+5.5%** | **16.8%** | **67%** |

An earlier test appeared to show that even spacing hurts (`1,6,11,16,19` gave 10.1%
against 15.9%); that comparison used the walled depot and the superseded comparator,
and does not survive re-measurement.

## 3. Fleet and depot

**16 homogeneous AMRs**, unit speed, no kinematics or battery model.

**Depot: aisle bays** (`DEPOT_LAYOUT = "aisle"`, `aisle_rows` / `aisle_bays`). Bays sit
on the rows strictly **between** consecutive doors — y = 4, 8, 12, 16 for
`DOOR_ROWS = (2,6,10,14,18)` — spanning columns x = 0..5 (`width = 6`), so the depot
offers **24 bays** and every door row stays clear from wall to wall. Bays are filled
column-major, so a fleet smaller than 24 spreads across the four rows instead of piling
onto one; at m=16 the fleet occupies x = 0..3 of all four aisle rows:

```
y=19 ....................
y=18 Dwww............wwwS
y=17 ....................
y=16 RRRR................    <- aisle bays; x=4,5 spare at m=16
y=15 ....................
y=14 Dwww............wwwS
y=13 ....................
y=12 RRRR................
y=11 ....................
y=10 Dwww............wwwS
y= 9 ....................
y= 8 RRRR................
y= 7 ....................
y= 6 Dwww............wwwS
y= 5 ....................
y= 4 RRRR................
y= 3 ....................
y= 2 Dwww............wwwS
y= 1 ....................

D door (x=0)   S station (x=19)   w waiting cell (§2)   R parked AMR
```

`width` was 4 (16 bays) when the aisle depot landed, which silently capped contention
sweeps at the headline fleet; it is 6 in the working tree. The column-major fill order
means the widening moves no bay at m <= 16. m=24 exactly fills the depot — raise `width`
again before sweeping past eta = 4.8, and re-check constraint 1 below when you do.

The **pod depot this replaces** (`DEPOT_LAYOUT = "pods"`) parked AMRs on the door rows
themselves — x=4,5 beside rows 2/3, 6/7, 13/14, 17/18 — so four of the five dock lanes
dead-ended into two parked robots. It is retained only to reproduce the v3–v5 numbers.

Depot geometry turned out to be the single most consequential parameter in the whole
design, and three of four early candidates were broken *(measured, m=16 / m=24, unroutable
parcels per episode, lateral queue)*:

| layout | unroutable m=16 | unroutable m=24 | verdict |
|---|---|---|---|
| single column x=2 | 0.89 | — | column is a **wall**; 3 free rows at m=16, 0 at m=20 |
| block 4x4 at x=2 | 0.06 | 0.56 | bays sat **inside dock queues** (lateral queues reach x=3) |
| block 4x4 at x=5 | 0.00 | 0.17 | clean but low congestion (4.1%) |
| bottom rows y=0,-1 | 0.44 | 3.72 | 12-bay row is a **horizontal wall** |
| 2x2 pods at x=4,5 | 0.00 | n/a | clean on this metric *and* highest congestion of the valid layouts — but blocks the dock rows; superseded |

That sweep says nothing about the aisle depot, which was chosen on the different
feasibility metric in the table at the top (share of schedules with any un-executable job,
not unroutable parcels per episode).

Two hard constraints for any depot design:

1. **Never a contiguous run** across a full row or column. Idle AMRs hold their cells
   in the reservation table permanently (`next_t >= free_t`), so a packed line is an
   impassable barrier, not a temporary obstacle.
2. **Clear the service-point queues** entirely. A robot assigned a waiting slot that is
   another robot's parked bay can never arrive. The inward queue puts the waiting cells
   *on* the door row (x=1..3 at a dock, x=16..18 at a station), so the aisle depot
   satisfies this by construction — no bay is ever on a door row. Under the pod layout
   the queues occupied the two adjacent rows out to x=3, which is what forced bays to
   x >= 4.

Fleet capacity is `width x |aisle rows|` = 6 x 4 = **24**. `build_pod_depot` raises rather
than cycling — cycling silently gave two AMRs the same bay, which blocks it from t=0 —
and `apply_layout` then runs `validate_depot`, which checks duplicate bays, bays inside
waiting sets, and solid rows/columns. None of it is left to inspection.

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

### 5.1 Door clearance (`DOCK_CLEARANCE_TICKS = 1`)

The service cell stays **reserved by the incumbent for 1 tick past service end**, so it
has room to pull away before the next AMR takes the cell. Two things are deliberately
*not* done:

- `availability[amr]` is **not** advanced — the incumbent is free to leave at service
  end, it simply keeps the right to stand there. The clearance is standing room, not
  extra service time.
- `dock_available` is **not** extended. Delaying it pushes the next AMR down the
  wait-slot / `hold_upstream` branch, and `hold_upstream` drives any AMR standing on a
  door all the way back to its bay: on one instance that produced 51 pointless depot
  round trips and took the makespan from 309 to 578 *(measured)*. The cell reservation
  alone is enough — the arriving AMR waits a tick on approach, as a real queue would.

The failure it fixes: an AMR finishes service and, on the tick its successor books the
cell, all three of its neighbours are taken (a head-on swap, a robot still inside the
one-cell following gap, a queued robot). It must move and cannot, so the leg fails. One
extra tick of standing room and it departs with full headway — no safety rule relaxed.

Measured on `val_60` under the inward queue, paired (identical schedules scored at each
setting; makespan over schedules clean at *every* setting):

| ticks | infeasible s43 / s44 | makespan cost s43 / s44 | heuristic cost |
|---|---|---|---|
| 0 | 6% / 9% | +0.0 | +0.0 |
| **1** | **0% / 1%** | **+9.7 / +8.2** | **+8.7** |
| 2 | 1% / 0% | +22.4 / +18.9 | +17.2 |
| 3 | 0% / 0% | +37.9 / +32.1 | +25.3 |

1 buys essentially all of the feasibility for the least time; larger values cost roughly
linearly and add nothing. The model and the dispatch heuristic pay almost the same price,
so relative comparisons are unaffected. Set `DOCK_CLEARANCE_TICKS = 0` to reproduce the
v3–v5 results exactly.

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

Every row also predates the inward queue, the aisle depot and the clearance tick. None of
the three touches `ideal`, which has no queueing to alter, and the clearance alone adds
about +9 to `executed`, so expect the re-measured penalty to read **higher** than the
table — the direction of the argument is safe, the levels are not.

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
6. Floor **obstacles** (storage aisles / racks as blocked cells — not the aisle depot
   of §3) were never re-measured with a correct depot. If §6's 14.9% is
   judged too small, that is the remaining lever — and racks are permissible again
   now that cross-docking is dropped.
