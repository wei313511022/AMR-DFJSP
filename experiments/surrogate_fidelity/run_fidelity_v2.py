"""Fidelity and cost of the three evaluators, over the full candidate field.

Supersedes run_fidelity.py, which measured 3 sizes x 30 instances x 12 rule-generated
schedules. This measures the SAME quantities at the same coverage as the main benchmark --
n = 20/40/60/80/100, 100 instances each, from test_case/v3/trend/full_*.jsonl -- and over a
candidate field that is no longer rules-only:

    12  dispatching-rule combinations   6 job rules x 2 AMR rules (unchanged from v1)
     6  learned policies                cells A-F of the v10 factorial, seed 44, greedy
     1  GA                              random.seed(1000 + instance), as in the main sweep
    --
    19  candidate schedules per instance, 9500 schedules in total

WHY THE POLICIES AND THE GA MATTER HERE. run_fidelity.py's own README records the flaw
they fix: its 12 candidates were produced by rules that themselves plan under Psi-hat, so
Psi-hat was ranking schedules it helped construct. Cells A and B were TRAINED to minimise
C~ and cells E and F to minimise Psi-hat, so the field now contains schedules optimised
against each evaluator under test -- the adversarial case, not the friendly one. The GA
optimises the collision-free decode and is a fourth, independent generator.

WHY THESE 12 RULES AND NOT ALL 60. The 60-combination grid is reported in
../full_benchmark/results.csv; these 12 are its top half (mean rank 3-30 of 60) and they
span the batching spectrum, from one parcel per trip to a full milk-run rack. The bottom
half is `random`, `nearest_station` and `least_congested_station`, which execute 2-4x worse
than the field and would be ranked correctly by any evaluator, inflating tau for all three.
`rule_selection.py` regenerates the evidence for both halves of that claim.

Every schedule carries its measured batching depth (pickups per trip) so the spread is a
number in the table, not an assertion in the text.

    python run_fidelity_v2.py --workers 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STATIC = REPO / "Static_alogorithm"
for p in (STATIC, STATIC / "extend_GNN"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SIZES = (20, 40, 60, 80, 100)
AMRS = 16
N_INSTANCES = 100


def dataset(n: int) -> Path:
    return REPO / f"test_case/v3/trend/full_{n}.jsonl"


# --- candidate field -------------------------------------------------------------------
# 6 job rules x 2 AMR rules. Identical to run_fidelity.py so the two tables are comparable.
RULE_COMBOS = [f"{j}+{a}" for j in ("fifo", "spt", "lpt", "milk_run",
                                    "material_match", "earliest_completion_job")
               for a in ("earliest_available", "earliest_completion")]

# The v10 factorial: mask x training return. Seed 44 is the only seed with all six cells
# finished at 4000 epochs (../full_benchmark/RESULTS_A-F.md), so one seed keeps the six
# candidates on a common footing. D is v9's only60, this project's prior main method.
# The mask must match training: a checkpoint fed channels it never saw is not that model.
POLICIES = {
    "A": ("checkpoints_v10/v10_A_s44_best.pth", "L3"),    # masked, C~ return
    "B": ("checkpoints_v10/v10_B_s44_best.pth", "none"),  # full,   C~ return
    "C": ("checkpoints_v10/v10_C_s44_best.pth", "L3"),    # masked, Phi return
    "D": ("checkpoints_v9/only60_s44_best.pth", "none"),  # full,   Phi return
    "E": ("checkpoints_v10/v10_E_s44_best.pth", "L3"),    # masked, Psi-hat return
    "F": ("checkpoints_v10/v10_F_s44_best.pth", "none"),  # full,   Psi-hat return
}
MASKS = {
    "none": {},
    "L3": {"amr": (9,), "dock": (2, 3, 4, 5, 6), "action": (2,)},
}

_G = {}


def worker_init():
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import scenario_v3 as sc
    sc.apply_layout(num_amrs=AMRS)
    import GA.GA as GA
    import operation_policy as _op
    if len(GA.AMR_KEYS) != AMRS:
        raise SystemExit(f"apply_layout gave {len(GA.AMR_KEYS)} AMRs")
    if _op.DOCK_QUEUE_SCALE != float(AMRS):
        raise SystemExit("apply_layout did not patch operation_policy")
    import ideal_evaluator as ie
    from surrogate_evaluator import surrogate_makespan
    from GA.GA import load_dispatch_events
    import extend_GNN
    _G["ie"], _G["GA"], _G["psi"] = ie, GA, surrogate_makespan
    _G["events"] = {n: load_dispatch_events(dataset(n)) for n in SIZES}
    _G["xg"] = sys.modules.get("extend_GNN.extend_GNN", extend_GNN)
    _G["pristine"] = _G["xg"].extract_state_extend_gnn
    _G["models"] = {}


def set_mask(level):
    """Rebind extract_state_extend_gnn to the pristine function wrapped by `level`."""
    xg, original = _G["xg"], _G["pristine"]
    spec = MASKS[level]
    if not spec:
        fn = original
    else:
        def fn(*a, **kw):
            t = list(original(*a, **kw))
            for i in spec.get("amr", ()):    t[0][..., i] = 0.0
            for i in spec.get("dock", ()):   t[2][..., i] = 0.0
            for i in spec.get("action", ()): t[3][..., i] = 0.0
            return tuple(t)
    xg.extract_state_extend_gnn = fn
    xg.solve_with_extend_gnn.__globals__["extract_state_extend_gnn"] = fn


def get_model(cell):
    if cell not in _G["models"]:
        import torch
        from operation_policy import load_required_operation_checkpoint
        path, _ = POLICIES[cell]
        m = _G["xg"].ExtendSchedulerGNN()
        load_required_operation_checkpoint(
            m, REPO / path, torch, required_keys=("op_emb.weight", "operation_actor.0.weight"))
        m.eval()
        _G["models"][cell] = m
    return _G["models"][cell]


def batch_profile(ind, jobs):
    """Batching depth of a schedule: parcels picked up, and trips taken to move them.

    A trip is a maximal run of one robot's committed operations from an empty rack back to
    empty. pickups/trips is therefore 1.0 for a rule that fetches one parcel and returns,
    and grows with the size of the batch a rule accumulates before delivering. Purely
    combinatorial -- read off the order, no simulation -- so it costs nothing to record.
    """
    GA = _G["GA"]
    ops = GA.repair_operation_order(ind.order, list(jobs))
    per = defaultdict(list)
    for o in ops:
        per[ind.amr_assignment[o.job_idx]].append(o)
    pickups = trips = 0
    for seq in per.values():
        onboard = 0
        for o in seq:
            if o.kind == GA.PICKUP:
                if onboard == 0:
                    trips += 1
                onboard += 1
                pickups += 1
            else:
                onboard -= 1
    return pickups, trips


def score(ind, jobs, n, inst, family, method, gen_s):
    """Score one schedule with all three evaluators and time each.

    phi_s times ie.evaluate, which recomputes C~ internally. That is the convention
    run_fidelity.py used and the contamination is under 0.5% (C~ is ~0.1 ms against Phi's
    30-140 ms), so the two tables' cost columns stay comparable.
    """
    ie = _G["ie"]
    t0 = time.perf_counter()
    ideal = ie.per_robot_ideal(ind, jobs)
    c_tilde = max(ideal.values()) if ideal else 0.0
    ideal_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    psi = _G["psi"](ind, jobs)
    psi_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    m = ie.evaluate(ind, jobs)
    phi_s = time.perf_counter() - t0

    pickups, trips = batch_profile(ind, jobs)
    return {
        "n_jobs": n, "instance": inst, "family": family, "method": method,
        "c_tilde": c_tilde, "psi_hat": psi,
        "executed": m["executed"], "nu": m["nu"], "routable": m["routable"],
        "pickups": pickups, "trips": trips,
        "load_per_trip": round(pickups / trips, 4) if trips else float("nan"),
        "gen_s": round(gen_s, 6), "ideal_s": round(ideal_s, 6),
        "psi_s": round(psi_s, 6), "phi_s": round(phi_s, 6),
    }


def run_task(task):
    kind, arg, n, i0, i1 = task
    rows = []
    for ev in _G["events"][n][i0:i1]:
        jobs = list(ev["jobs"])
        inst = ev["index"]
        if kind == "rule":
            from reinforce_baseline import complete_with_dispatch_rule
            t0 = time.perf_counter()
            ind = complete_with_dispatch_rule(jobs, [], {}, baseline_rule=arg, seed=42)
            gen_s = time.perf_counter() - t0
            rows.append(score(ind, jobs, n, inst, "rule", arg, gen_s))
        elif kind == "ga":
            random.seed(1000 + inst)
            t0 = time.perf_counter()
            ind, _ = _G["GA"].evolve(jobs)
            gen_s = time.perf_counter() - t0
            rows.append(score(ind, jobs, n, inst, "ga", "GA", gen_s))
        else:
            import torch
            set_mask(POLICIES[arg][1])
            model = get_model(arg)
            t0 = time.perf_counter()
            with torch.no_grad():
                ind, _, _ = _G["xg"].solve_with_extend_gnn(jobs, model, deterministic=True)
            gen_s = time.perf_counter() - t0
            model.eval()
            rows.append(score(ind, jobs, n, inst, "policy", arg, gen_s))
    return rows


def build_tasks(sizes, n_inst, what):
    """Chunk by cost: the GA is ~200 s an instance at n=100, a rule is under a second."""
    tasks = []
    for n in sizes:
        if "ga" in what:
            tasks += [("ga", None, n, i, i + 1) for i in range(n_inst)]
        if "policy" in what:
            for cell in POLICIES:
                tasks += [("policy", cell, n, i, min(i + 5, n_inst))
                          for i in range(0, n_inst, 5)]
        if "rule" in what:
            for combo in RULE_COMBOS:
                tasks += [("rule", combo, n, i, min(i + 25, n_inst))
                          for i in range(0, n_inst, 25)]
    # Longest first: the GA tasks must start early or they set the wall clock.
    cost = {"ga": 3, "policy": 2, "rule": 1}
    tasks.sort(key=lambda t: (-cost[t[0]], -t[2]))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--instances", type=int, default=N_INSTANCES)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--what", default="rule,policy,ga")
    ap.add_argument("--out", default=str(HERE / "fidelity_v2.jsonl"))
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    what = set(args.what.split(","))
    tasks = build_tasks(sizes, args.instances, what)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    done = 0
    t_start = time.perf_counter()
    with out.open("w", encoding="utf-8") as fh:
        with ctx.Pool(args.workers, initializer=worker_init) as pool:
            for rows in pool.imap_unordered(run_task, tasks):
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                done += 1
                if done % 25 == 0 or done == len(tasks):
                    el = time.perf_counter() - t_start
                    print(f"  {done}/{len(tasks)} tasks  {el/60:.1f} min elapsed  "
                          f"eta {el/done*(len(tasks)-done)/60:.1f} min", flush=True)
    print(f"rows -> {out}")


if __name__ == "__main__":
    main()
