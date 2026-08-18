# CD-GNN / RA-L Positioning Review

Based on: `Static_alogorithm/` (GA, GNN, Attention, dispatching_rules, operation_policy,
reinforce_baseline, calibration), `Static_alogorithm/benchmark_results/`,
`paper/*.tex`. Attention branch read only for cross-checking numbers.

---

## 0. What the code actually does (so we agree on the object under review)

| Element | Implementation |
|---|---|
| Floor | 10×10 grid, `OBSTACLES = set()` (empty), 5 inbound docks at x=0, 5 stations at x=9 |
| Fleet | `AMR_STARTS` = 3 AMRs in `GA.GA`; checkpoints and `eval_fleet_size.py` refer to a 5-AMR model |
| Job | (pickup at dock, unload at station) pair, type ∈ {A,B,C}, service 5/10/15 s |
| Coupling | `AMR_LOAD_CAPACITY = 3` → batching; docks/stations single-server |
| Executor | `decode_schedule` — prioritized space-time A* with cell/edge reservations, `WAIT_LINE_DEPTH=3` dock queues, deadlock bail-out at t>10000 → `invalid_jobs` |
| Surrogate | affine travel calibration (scale 0.963, offset 1.119, R²=0.906) + 4-bin dock-wait penalty |
| Policy | GIN (3 layers, adding-arc adjacency over jobs) + multi-pointer job/AMR heads, ~2.8e5 params |
| Training | REINFORCE with a **stepwise counterfactual dispatch-rule rollout baseline**, each step evaluated under the collision-aware executor |

### Measured results (`benchmark_results/static_benchmark_summary.csv`, 100 samples/size)

| n | best rule combo | GA | Attention | **GNN** | GNN vs GA | GNN speedup vs GA |
|---|---|---|---|---|---|---|
| 20 | 238.87 | 144.70 (7.2 s) | 141.74 | **140.38** (0.67 s) | −3.0 % | 10.8× |
| 40 | 456.49 | 295.32 (19.0 s) | **280.05** | 282.32 (1.42 s) | −4.4 % | 13.4× |
| 60 | 668.77 | 464.75 (36.0 s) | **423.28** | 433.43 (2.38 s) | −6.7 % | 15.2× |
| 80 | 885.57 | 647.19 (58.5 s) | **575.70** | 594.72 (3.77 s) | −8.1 % | 15.5× |
| 100 | 1104.12 | 837.39 (86.2 s) | **735.87** | 758.17 (5.25 s) | −9.5 % | 16.4× |

Three facts fall straight out of this table and they drive everything below:

1. **The "two orders of magnitude faster" claim in the abstract is not supported.**
   It is 11–16×. One order of magnitude, not two. `gnn_precise` is 79× faster but
   *loses* to GA (503.5 vs 464.8 at n=60), so it cannot carry the claim either.
2. **The Attention model beats the GNN at every size ≥ 40** (0.8 %, 2.3 %, 3.2 %, 2.9 %).
   You are preparing to publish the weaker of your own two models.
3. **The margin over hand-crafted rules (≈35 %) is too large to be believed.**
   `milk_run` — the only rule that expresses batching — is implemented in
   `dispatching_rules.py` but is *absent from the benchmark CSV*. Your own claim guard in
   `abstract_intro.tex` says milk_run reaches ≈450 at m=5 vs GNN 433. That is a ≈4 % margin,
   not 35 %. Reviewers of learned-scheduling papers are trained to distrust exactly this gap.

---

## 1. Is the topic good?

**Yes, but not for the reason you stated.** Grade the three candidate framings separately:

| Framing | Verdict |
|---|---|
| "GNN for scheduling" | **Dead.** L2D (2020), Song et al. (2023), Moon et al. (2024) already own this. A GIN over a disjunctive graph is 2020 technology; in 2026 it is a component, not a contribution. Never put "GNN" in the title. |
| "Cross-docking is hard" | **Over-claims, and the code does not back it.** See §2. |
| "The makespan of a schedule is only defined through its collision-aware execution, so the *learning signal* must come from the executor" | **This is publishable and, as far as I can find, unoccupied.** |

The intersection is real. The FJSP-T learning line (Moon et al. and successors) reduces transport
to a fixed travel-time matrix. The MAPF/MAPD line routes rigorously but carries a deliberately
thin task model — one agent, one task, no single-server service resources, no capacitated
consolidation. The integrated-scheduling-and-conflict-free-routing line (container terminals)
re-solves per instance with metaheuristics and learns nothing reusable. Nobody trains a
constructive policy whose *advantage estimate* is computed by running a collision-aware executor.
Your `reinforce_baseline.compute_dispatch_baseline_comparison` does exactly that. That is the paper.

**But there is a trap inside your own motivation.** You want to argue execution-awareness is
necessary. Your own planned number says the congestion tax is **2.6 % of makespan** at the base
fleet, rising to 12 % as the fleet doubles. A reviewer will read "2.6 %" and write: *the authors'
own data show the effect they build their method around is negligible in the operating regime they
evaluate.* On a 10×10 grid with no obstacles and 3–5 robots, congestion essentially cannot happen.
**You must either move into a regime where congestion is large (narrow aisles, m=10–20, obstacle
racks) or restructure the claim as a regime map — "here is where it pays and here is where it does
not."** The second option is cheaper, more honest, and more interesting to RA-L than a 6 % win.

### RA-L fit: the real risk

RA-L is a robotics venue. Its reviewers expect either hardware or a high-fidelity simulator.
Your evaluation is a 10×10 abstract grid with instantaneous unit-cost moves, no kinematics, no
localization error, no sensing. Two of three reviewers will ask "where are the robots?"
Mitigations, in order of cost:

- **Cheapest credible fix:** re-run the top 2–3 methods on a standard MAPF benchmark map
  (`warehouse-10-20-10-2-1` or a Kiva-style map) with 10–20 AMRs. This simultaneously fixes
  the scale objection *and* the "congestion is only 2.6 %" objection.
- **Better:** ROS 2 / Gazebo or a Nav2-based executor for a subset of instances, showing the
  schedule ranking is preserved under a physical controller.
- **Best:** 2–3 real AMRs executing a schedule, even at toy scale, as a feasibility figure.

Without at least the first, I would rate acceptance at RA-L as unlikely on scope grounds alone,
independent of the method's merit. Honest alternatives with a better content fit:
**IEEE T-ASE** (automation science — scheduling + logistics is squarely in scope, no hardware
expectation, longer page budget), **RCIM**, or **ICRA** with the benchmark-map experiments added.

---

## 2. Should you rephrase your perspective?

Yes — three specific rewrites.

### 2.1 Drop or earn the "cross-docking" label

Real cross-docking is defined by: inbound *and* outbound doors, truck arrival/departure **time
windows**, and the absence of storage. Your model has inbound docks and processing stations, no
outbound doors, no truck schedule, no deadlines, and a pure makespan objective. A logistics-literate
reviewer will call this a dock-to-station transfer bay, not a cross-dock, and will be right.

Two ways out:

- **(a) Earn it (recommended if you have a week):** attach outbound release deadlines to job
  groups and add a tardiness term. Then "cross-docking" is justified *and* you get a second
  objective that hand-crafted rules handle badly — which strengthens the learning argument.
- **(b) Rename it:** "capacitated pickup-and-delivery flexible job shop with exclusive service
  points and collision-aware execution." Ugly, accurate, unattackable.

Do not keep the cross-docking framing without (a). It is a free target.

### 2.2 Rewrite the "RL observes AMR state" claim — it is currently trivially true and old

> "using an RL to observe the AMR state for assigning jobs to AMRs is necessary"

Every FJSP-T DRL paper since 2023 puts vehicle features in the state. As written this is not a
contribution; it is table stakes. The non-obvious version of the same idea:

> Travel and wait times are **endogenous**: they are produced by the assignments already made.
> A fixed travel-time matrix — the representation used by every learned FJSP-T dispatcher — is
> therefore not an approximation of the transport model but a different problem. The consequence
> is not merely a modelling inaccuracy: it changes the *ranking* of schedules, so a policy trained
> on the matrix model optimizes the wrong objective. We quantify this ranking inversion and show
> that grounding the *training signal* — not just the state features — in a collision-aware
> executor is what recovers it.

Note what this buys you: **"ranking inversion" is a measurable claim you can turn into a figure.**
Take N schedules, score them under (i) the fixed-travel-time model and (ii) the collision-aware
executor, and report Kendall's τ. If τ is low, your entire paper is justified in one plot, and
the 2.6 %-congestion objection dies — because the argument becomes about *ranking*, not magnitude.
**If I could add one experiment to this paper, it would be this one.** It is cheap: you already
have both simulators and `calibrate_fast_model.py` already samples the pairs.

### 2.3 Reorder the contributions

Current order leads with formulation and architecture. Both are the weakest items. Proposed:

1. **Execution-grounded credit assignment** (stepwise counterfactual rollout under the executor)
   — the method contribution.
2. **The schedule–execution ranking gap**, measured, and the calibration that halves it
   — the empirical justification (this is where §2.2's τ figure goes).
3. **A regime map**: congestion tax and the value of execution-awareness as a function of fleet
   density, with the time-budget decomposition — *when this matters and when it does not*.
4. Architecture and the sparse-vs-dense over-smoothing result — demoted to a subsection + ablation.

Item 3 is the most RA-L-friendly thing in the repo and it is currently listed last as "VII-F".

---

## 3. What is your weakness?

Ordered by how likely each is to cause rejection.

### Fatal-if-unaddressed

**W1. Scale and physical fidelity vs venue.** 10×10 empty grid, 3–5 robots, ≤100 jobs, unit-cost
moves. MAPF reviewers work on 33×46+ maps with 100+ agents. See §1 for fixes.

**W2. Self-defeating motivation number.** Congestion = 2.6 % of makespan at base fleet.
You cannot argue necessity from 2.6 %. Fix by changing regime, or by reframing around ranking
inversion (§2.2) rather than magnitude.

**W3. No lower bound, no optimality gap.** "Beats GA by 6 %" is only meaningful if GA is strong.
The CP-SAT bound is a `TODO` in `main.tex`. At n=100 GA runs 86 s — a reviewer will ask whether it
converged, and will ask for an anytime curve (GA quality vs wall-clock) with the GNN plotted as a
point. You have the machinery (`benchmark_neural_local_improvement.py` already produces Pareto
plots); this is the single highest-value missing experiment after the τ figure.

### Serious

**W4. Your Attention model beats your GNN.** Publishing the GNN alone and holding Attention back
means (i) you report your second-best result, and (ii) the follow-up paper obsoletes this one, which
reviewers of the second paper will notice. **Recommended: keep it one paper and convert Attention
into an encoder ablation.** "The contribution is execution-grounded training; we instantiate it with
a graph encoder and a set-attention encoder and find the encoder is worth ≤3 % while execution
grounding is worth X %." That is a *stronger* paper than either alone, and it neutralizes the "why
a GNN?" question that a graph-skeptical reviewer will otherwise ask.

**W5. Weak-baseline signature.** 35 % over dispatching rules, with the one batching-capable rule
(`milk_run`) missing from the benchmark. Run it. If the honest margin is 4 %, report 4 % — the
credibility gain exceeds the headline loss, provided contributions 2 and 3 carry the paper.

**W6. Executor is incomplete prioritized planning.** Space-time A* with reservations, no
replanning, no priority ordering search; it deadlocks (`t > 10000` bail-out, nonzero `invalid_jobs`
across the CSV). Two attacks follow. (a) *Completeness:* why not PBS / CBS / LNS? (b) **The
premise attack, which is worse:** if the congestion your method learns to avoid is an artifact of a
weak router, a better router dissolves the contribution. **Defense: report results under two
routers** (your prioritized planner and one stronger, e.g. PBS) and show the schedule ranking is
preserved. Without this, W6 is the reviewer question I would least like to answer.

**W7. Training-cost scalability of the stepwise baseline.** For each sampled trajectory of length
2n you call `complete_with_dispatch_rule` + `evaluate_makespan` **twice per step** → O(n) collision-aware
executions per step, O(n²) per episode. At n=100 that is ~4×10⁴ executor calls per episode. State
this cost explicitly with wall-clock numbers and show the stepwise-vs-episode ablation justifies it.
If it does not, the episode baseline is the honest choice and the "counterfactual" contribution
shrinks to a negative result.

### Moderate

**W8. Novelty of the training scheme is narrower than the draft implies.** A greedy-rollout
baseline is standard (self-critical / Kool et al.); yours is the stepwise, executor-grounded variant.
Cite the rollout-baseline lineage explicitly and claim the *executor grounding*, not the rollout.
Claiming more invites a reviewer to supply the citation for you.

**W9. Reproducibility / a real bug in the fleet study.** Fleet size is a module-level global that
`eval_fleet_size.py` mutates in place after import, and its own docstring says
`DOCK_QUEUE_SCALE` is *deliberately left at its import-time value*. So in the zero-shot
fleet-generalization experiment the queue features are normalized for the training fleet, not the
evaluation fleet. Whatever the intent, as written this is a feature-scale mismatch inside the
headline generalization claim, and it is discoverable from the released code. Fix it or document
it as an intentional invariance and show it does not matter. Separately: no results CSV records
`m`, so the m=3 vs m=5 provenance of the benchmark table cannot be reconstructed.

**W10. Over-smoothing result is a strawman risk.** "Dense dock-sharing edges collapse embeddings"
is a nice negative result, but a reviewer may answer "of course — that is why nobody does that,"
or "use edge types / attention and it goes away." Frame it as *why the sparse dynamic graph is the
right inductive bias*, with the dense variant as evidence, rather than as a discovery.

**W11. Page budget.** Six planned experiment subsections (VII-A…VII-F) will not fit in RA-L's
6+2 pages alongside four method sections. Cut to three: main comparison + Pareto, the ranking-gap
/ calibration result, and the regime map. Ablations go in a single compact table.

**W12. Single instance family, deterministic durations, static release times.** State as
limitations; the dynamic-arrival branch exists in `Random_Job_Arrivals/` and is the natural
follow-up paper — which is a cleaner second paper than "the same thing with attention."

---

## 4. Shortest path to a submittable paper

| # | Action | Cost | Buys |
|---|---|---|---|
| 1 | Ranking-inversion figure (Kendall's τ: matrix model vs executor) | ~1 day | Kills W2, justifies the whole premise |
| 2 | Add `milk_run` to the benchmark; restate margins honestly | hours | Kills W5 |
| 3 | Fix the speed claim (11–16×, not 100×) | minutes | Kills an integrity flag |
| 4 | GA anytime curve + CP-SAT / combinatorial lower bound | 2–3 days | Kills W3 |
| 5 | Fold Attention in as an encoder ablation | ~1 day | Kills W4, pre-empts "why a GNN?" |
| 6 | Re-run on one standard MAPF warehouse map, m=10–20 | 3–5 days | Kills W1 + W2 together |
| 7 | Second router (PBS) robustness check | 2–3 days | Kills W6 |
| 8 | Retitle/reframe: execution grounding first, cross-docking dropped or earned | ~1 day | Kills the framing objections |

Items 1–5 are the minimum I would submit on. Items 6–7 are what turn "borderline" into "accept."

**Working title in the reframed direction:**
*"Scheduling Robots That Actually Drive: Execution-Grounded Policy Learning for
Capacitated Multi-Robot Pickup and Delivery"*
