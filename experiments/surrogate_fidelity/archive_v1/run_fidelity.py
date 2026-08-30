"""Fidelity and cost of the three evaluators, on a common set of schedules.

Section IV asserts that the calibrated surrogate Psi-hat is needed to build decision-state
features, and Section III measures how far the idealised decode C~ sits from the executor.
Neither is quantified for the SURROGATE, which is what the policy actually conditions on.
This measures all three on identical schedules:

    C~      per_robot_ideal        free-space travel, service on arrival, no interference
    Psi-hat apply_fast_action      calibrated travel + queue-depth penalty, no routing
    Phi     decode_schedule        space-time A*, exclusivity, waiting lines, collisions

Two kinds of error are reported, and they are not interchangeable:

  VALUE ERROR      |predicted - executed| / executed.  What Lambda measures for C~.
  DECISION ERROR   Kendall tau between the predicted and executed ORDERING of candidate
                   schedules FOR ONE INSTANCE, plus the regret of picking the predicted
                   best. This is the quantity a scheduler actually suffers, and it must be
                   computed within an instance: ranking across instances mostly recovers
                   "more parcels take longer" and inflates tau for every evaluator.

Candidates are 12 dispatching-rule combinations spanning the rule families, which gives a
spread of genuinely different schedules per instance at CPU-only cost.

    python run_fidelity.py --sizes 20,60,100 --instances 30
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO / "Static_alogorithm", REPO / "Static_alogorithm" / "extend_GNN"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ideal_evaluator as ie          # noqa: E402
import operation_policy as op         # noqa: E402
import scenario_v3 as sc              # noqa: E402
from surrogate_evaluator import surrogate_makespan  # noqa: E402
from GA.GA import load_dispatch_events                      # noqa: E402
from reinforce_baseline import complete_with_dispatch_rule  # noqa: E402

COMBOS = [f"{j}+{a}" for j in ("fifo", "spt", "lpt", "milk_run",
                               "material_match", "earliest_completion_job")
          for a in ("earliest_available", "earliest_completion")]


def kendall_tau(pairs) -> float:
    conc = disc = 0
    for (a1, b1), (a2, b2) in itertools.combinations(pairs, 2):
        s = (a1 - a2) * (b1 - b2)
        if s > 0:
            conc += 1
        elif s < 0:
            disc += 1
    return (conc - disc) / (conc + disc) if conc + disc else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="20,60,100")
    ap.add_argument("--instances", type=int, default=30)
    ap.add_argument("--amrs", type=int, default=16)
    ap.add_argument("--out", default=str(HERE / "fidelity.jsonl"))
    args = ap.parse_args()

    sc.apply_layout(num_amrs=args.amrs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for n in [int(s) for s in args.sizes.split(",")]:
        inbox = REPO / "test_case" / "v3" / "trend" / f"full_{n}.jsonl"
        events = load_dispatch_events(inbox)[: args.instances]
        for ev in events:
            jobs = list(ev["jobs"])
            for combo in COMBOS:
                t0 = time.perf_counter()
                ind = complete_with_dispatch_rule(jobs, [], {}, baseline_rule=combo, seed=42)
                gen_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                ideal = ie.per_robot_ideal(ind, jobs)
                c_tilde = max(ideal.values()) if ideal else 0.0
                ideal_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                psi = surrogate_makespan(ind, jobs)
                psi_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                metrics = ie.evaluate(ind, jobs)
                phi_s = time.perf_counter() - t0

                rows.append({
                    "n_jobs": n, "instance": ev["index"], "combo": combo,
                    "c_tilde": c_tilde, "psi_hat": psi,
                    "executed": metrics["executed"], "nu": metrics["nu"],
                    "gen_s": round(gen_s, 6), "ideal_s": round(ideal_s, 6),
                    "psi_s": round(psi_s, 6), "phi_s": round(phi_s, 6),
                })
        print(f"  n={n}: {len(events)} instances x {len(COMBOS)} schedules")

    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"rows -> {out}")


if __name__ == "__main__":
    main()
