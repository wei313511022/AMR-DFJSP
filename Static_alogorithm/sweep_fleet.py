"""Contention sweep over eta = m / |D_in| -- chunked so it survives short sessions.

Runs every AMR rule at one fleet size and APPENDS one JSON row per
(fleet, instance, job rule, AMR rule) to --out, so the sweep can be built up
across several invocations and summarised once at the end.

    python sweep_fleet.py --amrs 4  --count 5 --out ../sweep.jsonl
    python sweep_fleet.py --amrs 8  --count 5 --out ../sweep.jsonl
    ...
    python sweep_fleet.py --summarise --out ../sweep.jsonl

Everything except the fleet is held fixed: same instances, same layout, same
rack, same executor. `scenario_v3.apply_layout` rebuilds the depot at each
size and validates it, so a sweep point can never quietly run on a geometry
that walls off the floor or parks robots inside dock queues.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

STATIC_DIR = os.path.abspath(os.path.dirname(__file__))
if STATIC_DIR not in sys.path:
    sys.path.insert(0, STATIC_DIR)

import GA.GA as GA  # noqa: E402
import scenario_v3 as sc  # noqa: E402
import ideal_evaluator as ie  # noqa: E402
from GA.GA import load_dispatch_events  # noqa: E402
from reinforce_baseline import complete_with_dispatch_rule  # noqa: E402

AMR_RULES = ["earliest_available", "earliest_completion", "least_loaded",
             "material_match", "nearest_amr", "random"]
SENSIBLE = [r for r in AMR_RULES if r != "random"]
REFERENCE = "earliest_completion"   # dominant AMR rule; the headline column


def summarise(path: Path, job_rule: str) -> None:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["job_rule"] == job_rule]
    if not rows:
        print(f"no rows for job rule {job_rule!r}")
        return

    by_fleet_rule = defaultdict(list)
    for r in rows:
        by_fleet_rule[(r["amrs"], r["rule"])].append(r)
    fleets = sorted({r["amrs"] for r in rows})
    n_inst = len({r["instance"] for r in rows})

    print(f"\njob rule: {job_rule}   |   {n_inst} instances x 60 parcels   "
          f"|   AMR rule for the headline columns: {REFERENCE}")
    print(f"{'m':>3} {'eta':>5} {'ideal':>8} {'exec':>8} {'Lambda':>8} {'Om_q':>7} "
          f"{'Om_r':>7} {'q-shr':>6} {'spread*':>8} {'nu/ep':>6} {'clean':>7}")
    for m in fleets:
        ref = ie.aggregate(by_fleet_rule[(m, REFERENCE)])
        execs, nus = {}, []
        for rule in AMR_RULES:
            agg = ie.aggregate(by_fleet_rule[(m, rule)])
            execs[rule] = agg["executed"]
            nus.append(agg["nu_per_episode"])
        good = [execs[r] for r in SENSIBLE if execs[r] == execs[r]]
        spread = 100 * (max(good) - min(good)) / min(good) if good else float("nan")
        print(f"{m:>3} {m / 5:>5.1f} {ref['ideal']:>8.1f} {ref['executed']:>8.1f} "
              f"{100 * ref['penalty']:>7.1f}% {100 * ref['omega_q']:>6.1f}% "
              f"{100 * ref['omega_r']:>6.1f}% {100 * ref['queue_share']:>5.0f}% "
              f"{spread:>7.2f}% {statistics.mean(nus):>6.2f} "
              f"{int(ref['clean_instances'])}/{int(ref['instances']):<5}")
    print("* spread = best-to-worst executed makespan over SENSIBLE AMR rules "
          "(random excluded).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", type=str, default="../test_case/v3/instances_60.jsonl")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--amrs", type=int, default=16)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--job_rule", type=str, default="milk_run")
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.summarise:
        summarise(out, args.job_rule)
        return

    sc.apply_layout(num_amrs=args.amrs)
    events = load_dispatch_events(Path(args.inbox))[args.start: args.start + args.count]

    with out.open("a", encoding="utf-8") as fh:
        for ev in events:
            jobs = ev["jobs"]
            for amr_rule in AMR_RULES:
                ind = complete_with_dispatch_rule(
                    jobs, [], {}, baseline_rule=f"{args.job_rule}+{amr_rule}", seed=42)
                row = ie.evaluate(ind, jobs)
                row.update({"amrs": args.amrs, "instance": ev["index"],
                            "job_rule": args.job_rule, "rule": amr_rule})
                fh.write(json.dumps(row) + "\n")
    print(f"m={args.amrs} (eta={sc.contention_ratio(args.amrs):.1f}): "
          f"{len(events)} instances x {len(AMR_RULES)} AMR rules -> {out}")


if __name__ == "__main__":
    main()
