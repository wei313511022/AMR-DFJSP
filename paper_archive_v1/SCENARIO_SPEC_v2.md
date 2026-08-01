# Scenario specification v2 — AMR cross-dock transfer with outbound deadlines

Decision: **A + D** — all parcels released at t=0, outbound shipment deadlines added.
Supersedes the 10x10 / 3-AMR setting. All numbers below marked *(measured)* come from
runs on this repository's executor; everything else is a design choice to be verified.

---

## 1. Floor

Open floor, **no obstacles**. Cross-docking has no storage, so racks are excluded by
definition — and measurement showed they were not needed anyway (§8).

| Element | Value |
|---|---|
| Grid | x in [0, 19], y in [1, 19] — 380 cells, four-connected, unit speed |
| Obstacles | none (`OBSTACLES = set()`) |
| Inbound doors | x = 0, rows y = 1, 6, 10, 14, 19 |
| Outbound stations | x = 19, rows y = 1, 6, 10, 14, 19 |
| AMR depot | bank along x = 2, 12 robots spread over y |
| Dock waiting line | `WAIT_LINE_DEPTH = 3` cells adjacent to each service point |

**Design parameter to report explicitly: 2.4 robots per service point** (12 robots,
5 service points per side). This — not floor size, not obstacles — is what produces
contention. State it in the paper as the density parameter.

Depot geometry is the second-largest driver and must be fixed and documented: scattering
robots across the floor drops the congestion tax to 3.8%, a wall bank raises it to 15.6%
*(measured)*. The wall bank is both more realistic and more contended.

## 2. Fleet

- 12 homogeneous AMRs, unit speed, four-connected motion, no kinematics or battery model.
- Each AMR carries a **3x3 slot rack**: 3 slots each for size classes A (small),
  B (medium), C (large).
- **Downward substitution is allowed** — a smaller parcel may occupy a larger slot.
  This replaces the strict per-class constraint and yields a nested capacity structure:

  n_C <= 3,  n_B + n_C <= 6,  n_A + n_B + n_C <= 9

  Rationale: physically correct for compartment carriers, and it creates an opportunity
  cost (burning a large slot on a small parcel) that dispatching rules cannot evaluate
  without lookahead. This is a decision class the learned policy can exploit.

Put a small figure of the slot rack in the paper. The 3x3 structure currently reads as
arbitrary because it is nowhere motivated in the text.

## 3. Parcels

| Property | Value |
|---|---|
| Count | n = 60 for training; evaluate zero-shot at n = 60, 120, 240 |
| Size class | A / B / C — **skewed distribution, suggest 60 / 30 / 10** |
| Service duration | p(A)=5, p(B)=10, p(C)=15, identical at pickup and delivery |
| Release time | **r_j = 0 for all j** |
| Origin | inbound door, drawn independently of size class |
| Destination | outbound station, drawn independently |

Terminology: stop calling these "material types" — they are **parcel size classes**.
"Material" implies a fungible commodity and invites a misreading of the capacity constraint.

The skew from uniform to 60/30/10 is deliberate: it matches real parcel mixes *and* makes
the small-slot constraint bind more often. Verify with the mask-rate check in §7.

Since r_j = 0 everywhere, either delete the release constraint from the formulation or
state r_j = 0 explicitly. Leaving a constraint that never binds is a credibility cost.

## 4. Outbound shipments and deadlines

This is what earns the cross-docking label. Deadlines — not release times — are what
distinguish cross-docking from generic pickup-and-delivery, and they are absent from the
capacitated-MAPD literature, so this is a genuine differentiator.

- Jobs destined for the same outbound station are partitioned into **shipments** of 6–10
  parcels. Shipment g departs on one outbound truck.
- Shipment completion: `C_g = max_{j in g} c_delta_j` (all parcels loaded).
- Tardiness: `T_g = max(0, C_g - D_g)`.

**Deadline generation.** Let `C_ref` be the collision-free makespan of one fixed reference
rule on that instance. For shipment i of K at a station:

    D_g = gamma * C_ref * (i + 1) / K

Staggered departures through the horizon are realistic (trucks leave across a shift) and
create a genuine "which shipment first" sequencing decision. Sweep gamma in {0.8, 1.0, 1.2}
— tight, nominal, loose.

Caveat to handle in the paper: deriving deadlines from a reference heuristic is mildly
circular. Fix the reference rule, publish it, and confirm conclusions are stable across two
different reference rules.

## 5. Objective

Report **makespan and total tardiness separately**. Do not fold them into a weighted scalar
— reviewers distrust an arbitrary weight, and you already carry one undisclosed weighted
objective (`fitness()` returns `makespan + 0.001 * total_active_time`, which must either be
disclosed or aligned with the stated objective).

Also report: number of late shipments, and unroutable jobs per episode.

The executor is a **partial function** — prioritized planning is incomplete and can fail.
Define the objective on failure:

    Phi(a, sigma) -> (C_max, nu),  minimize C_max + kappa * nu

and state plainly that feasibility here is **executor-relative**: a schedule this planner
cannot route might be routable by CBS. Frame that as a deployment-realistic commitment
(the scheduler optimizes against the router it will actually run with), not as an oversight.

## 6. Why deadlines fix the weakest result

`milk_run` batches by proximity. Under deadlines that is the wrong heuristic — you need
batching by shipment. This is the axis where the learned policy should separate from the
best hand-crafted rule by much more than the current ~4%, and it gives a *mechanism*
explanation rather than a bare number.

Prerequisite: `MILK_RUN_MAX_LOAD = 6` is a **total** onboard check while the policy uses a
per-class limit. The baseline is currently playing by different physics than the policy.
Unify both on the nested constraint in §2 before publishing any milk_run comparison.

## 7. Health checks — run these before generating final instances

| Check | Target | Why |
|---|---|---|
| Congestion tax | 13–18% | Below ~10% the execution-aware claim is not credible |
| AMR-rule spread | > 10% | Confirms the assignment decision is worth making |
| Capacity mask hit rate | > 5% of pickup decisions | Below this, the consolidation claim is unsupported |
| Unroutable jobs | < 0.1 / episode | Currently 0.70 — planner work required |
| Kendall tau (free vs executed rank) | near 0 | The ranking-inversion evidence |

## 8. Measured evidence for this configuration

20x19 open floor, 12 AMRs, wall depot, n = 60, 5 rules, 4–10 instances:

| configuration | collision-free | executed | tax | rule spread | inv/ep |
|---|---|---|---|---|---|
| 5 doors spread (chosen) | 444.6 | 514.0 | **15.6%** | **16.6%** | 0.70 |
| 4 doors | 415.8 | 509.3 | 22.5% | 20.5% | 1.45 |
| 5 doors clustered | 400.6 | 455.1 | 13.6% | 10.3% | 0.55 |
| racked variant (rejected) | 450.7 | 519.6 | 15.3% | 9.1% | 0.76 |

The open floor matches the racked layout on congestion and nearly doubles the rule spread,
because door contention is a **scheduling** bottleneck (which robot queues behind which)
while corridor contention is a **routing** bottleneck. The contribution is scheduling.

Fleet-density sweep on the original 10x10 floor, for the regime-map figure:

| m | 3 | 5 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|
| congestion tax | 1.5% | 3.5% | 7.3% | 15.6% | 31.8% | 29.8% |
| AMR-rule spread | 0.04% | 0.49% | 0.91% | 6.96% | 9.58% | 7.57% |
| Kendall tau | 1.00 | 0.33 | -0.23 | 0.23 | 0.14 | -0.29 |

At m=3 the travel-time model predicts the executed ranking perfectly. From m=8 it is
uncorrelated or anti-correlated — optimizing the surrogate stops predicting what runs well.
From m=8 to m=16, **random robot assignment beats every "smart" rule under execution**,
because rules like `earliest_available` cluster robots in space and clustering causes
interference.

## 9. Code changes required

| File | Change |
|---|---|
| `GA/GA.py` | New layout constants; grid bounds; `AMR_STARTS` (12, wall bank) |
| `GA/GA.py` `Job` | Add `shipment_id`, `deadline` |
| `GA/GA.py` ~1381 | Nested capacity in `repair_operation_order` |
| `GA/GA.py` `fitness()` | Disclose or remove the `0.001 * total_active_time` term |
| instance generator | Shipment partition, deadline assignment, 60/30/10 size skew |
| `operation_policy.py:229` | Nested capacity in `legal_actions` |
| `dispatching_rules.py:130` | Nested capacity |
| `dispatching_rules.py:63` | `MILK_RUN_MAX_LOAD` -> nested constraint |
| `dispatching_rules.py:76` | Delete `HOME_MATERIAL_AMR` / `home_material` rule |
| `dispatching_rules.py` | Add EDD and ATC job rules |
| `GNN/GNN.py` | job_in_dim 16 -> 18 (slack to deadline, shipment progress) |
| training | Add tardiness to the objective; retrain |

`home_material` should go: it maps size class to a dedicated robot, which is incoherent
once every AMR carries all three slot classes, and it performs catastrophically
(`lpt+home_material` at n=100: 3281 vs 1176 for `lpt+material_match`). Leaving obviously
broken rules in the baseline table makes the rule family look weaker than it is and feeds
the weak-baseline suspicion.

## 10. Baselines

- All sensible job-rule x AMR-rule combinations (drop `home_material`)
- `milk_run` with corrected capacity — the one that matters
- **EDD / ATC** deadline-aware rules — without these the deadline comparison is unfair
- GA at matched wall-clock budget, with an anytime curve
- CP-SAT relaxation for a lower bound
- **Matrix-trained ablation**: an identical policy trained on fixed travel times, evaluated
  under the executor. This is the direct causal evidence for the paper's premise and the
  single most valuable experiment in the plan.

## 11. Open items

1. Fix prioritized planning failures (0.70 -> <0.1 per episode). This is the most likely
   reviewer attack and it is a planner problem, not a scenario problem.
2. Re-run the chosen configuration with ~50 instances and paired statistics; current
   numbers use 4–10 and are directional only.
3. Decide the training scale. The stepwise counterfactual baseline costs O(n^2) executor
   calls per episode — train at n=60, evaluate zero-shot to n=240.
4. Regularize the door rows to even spacing (y = 1, 6, 11, 16, 19 or similar) so the
   layout is describable in one sentence.
