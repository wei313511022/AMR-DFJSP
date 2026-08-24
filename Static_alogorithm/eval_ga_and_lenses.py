"""GA arms, and the three reporting lenses applied to any schedule.

Two things this produces.

1. THREE LENSES. The same schedule scored three ways:

     ideal        C~  free-space travel + service, no queueing, no collisions. The
                      fixed-travel-time model the FJSP-T / learned-dispatcher literature
                      actually uses.
     collision_free   the executor with check_collision=False: exclusive service points and
                      waiting lines are still enforced, but robots pass through each other.
     executed     Phi  the full collision-aware executor.

   Reporting `ideal` or `collision_free` is what a paper that ignores congestion would
   publish; `executed` is what the robots would actually do. Note that `collision_free`
   already contains the queueing, so it is NOT the right comparator for the paper's Lambda
   -- it subtracts out precisely the term prior work omits. It is reported here because it
   separates the queueing share of the gap from the collision share.

2. GA ARMS. `evolve(search_check_collision=False|True)` -- the search optimises the
   collision-free surrogate or the real executor. Both are REPORTED under Phi, so the pair
   isolates "optimised against the wrong model" from "searched badly".

    python eval_ga_and_lenses.py --mode lenses --rules milk_run+earliest_available,... \
        --count 100 --out raw/lenses.jsonl
    python eval_ga_and_lenses.py --mode ga --ga_collision 0 --count 25 --out raw/ga_cf.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

STATIC_DIR = os.path.abspath(os.path.dirname(__file__))
if STATIC_DIR not in sys.path:
    sys.path.insert(0, STATIC_DIR)

import GA.GA as GA  # noqa: E402
import ideal_evaluator as ie  # noqa: E402
import scenario_v3 as sc  # noqa: E402
from GA.GA import Individual, decode_schedule, load_dispatch_events  # noqa: E402
from reinforce_baseline import complete_with_dispatch_rule  # noqa: E402


def collision_free_makespan(individual: Individual, jobs) -> tuple:
    """Makespan under the executor with collisions disabled but docks still exclusive."""
    availability, _, _, _, invalid = decode_schedule(
        individual, jobs, need_log=False, check_collision=False)
    return float(max(availability.values())), float(invalid)


def three_lenses(individual: Individual, jobs) -> dict:
    row = ie.evaluate(individual, jobs)          # gives ideal (C~) and executed (Phi)
    cf, cf_invalid = collision_free_makespan(individual, jobs)
    row["collision_free"] = cf
    row["collision_free_nu"] = cf_invalid
    # Split the gap: how much of executed-minus-ideal is queueing (already present in the
    # collision-free decode) and how much is collision avoidance on top of it.
    if row["ideal"] > 0:
        row["gap_queueing_pct"] = 100.0 * (cf - row["ideal"]) / row["ideal"]
        row["gap_collision_pct"] = 100.0 * (row["executed"] - cf) / row["ideal"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("lenses", "ga"), required=True)
    ap.add_argument("--rules", type=str, default="milk_run+earliest_available")
    ap.add_argument("--ga_collision", type=int, default=0,
                    help="1 = GA search optimises the collision-aware executor")
    ap.add_argument("--ga_seed", type=int, default=42)
    ap.add_argument("--inbox", type=str,
                    default=os.path.join(STATIC_DIR, "..", "test_case", "v3", "test_60.jsonl"))
    ap.add_argument("--num_amrs", type=int, default=16)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    sc.apply_layout(num_amrs=args.num_amrs)
    inbox = Path(args.inbox).resolve()
    if inbox.name == "instances_60.jsonl":
        raise SystemExit("instances_60.jsonl is inside train_60.jsonl; use test_60.jsonl")
    events = load_dispatch_events(inbox)[args.start: args.start + args.count]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset = inbox.stem
    n_jobs = len(events[0]["jobs"])

    with out.open("a", encoding="utf-8") as fh:
        if args.mode == "lenses":
            for combo in [c for c in args.rules.split(",") if c.strip()]:
                job_rule, amr_rule = combo.split("+", 1)
                execs = []
                for ev in events:
                    jobs = list(ev["jobs"])
                    t0 = time.perf_counter()
                    ind = complete_with_dispatch_rule(
                        jobs, [], {}, baseline_rule=combo, seed=42)
                    solve_s = time.perf_counter() - t0
                    row = three_lenses(ind, jobs)
                    row.update({"amrs": args.num_amrs, "instance": ev["index"],
                                "job_rule": job_rule, "rule": amr_rule,
                                "family": "rule", "combo": combo, "dataset": dataset,
                                "n_jobs": n_jobs, "solve_s": round(solve_s, 4)})
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    if row["nu"] == 0:
                        execs.append(row["executed"])
                print(f"  {combo:>42}: executed {statistics.mean(execs):8.2f} "
                      f"| clean {len(execs)}/{len(events)}")
        else:
            search_ca = bool(args.ga_collision)
            label = "ga_collision_aware" if search_ca else "ga_collision_free"
            import random as _random
            execs = []
            for ev in events:
                jobs = list(ev["jobs"])
                # Seed per instance so the run is reproducible and two arms share a stream.
                _random.seed(args.ga_seed + 7919 * ev["index"])
                t0 = time.perf_counter()
                ind, _ = GA.evolve(jobs, search_check_collision=search_ca)
                solve_s = time.perf_counter() - t0
                row = three_lenses(ind, jobs)
                row.update({"amrs": args.num_amrs, "instance": ev["index"],
                            "job_rule": label, "rule": label, "family": "ga",
                            "combo": label, "dataset": dataset, "n_jobs": n_jobs,
                            "ga_search_collision": search_ca,
                            "ga_population": GA.POPULATION_SIZE,
                            "ga_generations": GA.GENERATIONS,
                            "solve_s": round(solve_s, 4)})
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if row["nu"] == 0:
                    execs.append(row["executed"])
                print(f"    inst {ev['index']:>3}: executed {row['executed']:8.1f} "
                      f"| cf {row['collision_free']:8.1f} | ideal {row['ideal']:8.1f} "
                      f"| {solve_s:6.1f}s", flush=True)
            print(f"  {label}: executed {statistics.mean(execs):8.2f} "
                  f"| clean {len(execs)}/{len(events)}")
    print(f"rows -> {out}")


if __name__ == "__main__":
    main()
