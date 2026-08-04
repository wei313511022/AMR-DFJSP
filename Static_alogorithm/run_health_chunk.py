"""Chunked driver for health_check_v3 -- evaluate a slice of instances, append rows.

The executor is slow enough that a 30-instance x 6-rule sweep does not fit in a
single short-lived shell. This runs instances [--start, --start+--count) and
APPENDS one JSON row per (instance, rule) to --out, so the sweep can be built up
across several invocations and aggregated once at the end with --summarise.

    python run_health_chunk.py --start 0  --count 5 --out ../rows.jsonl
    python run_health_chunk.py --start 5  --count 5 --out ../rows.jsonl
    ...
    python run_health_chunk.py --summarise --out ../rows.jsonl
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


def summarise(path: Path) -> None:
    by_rule = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_rule[row["rule"]].append(row)

    print(f"{'AMR rule':<20} {'ideal':>8} {'exec':>8} {'Lambda':>8} "
          f"{'Om_q':>7} {'Om_r':>7} {'q-share':>8} {'nu':>6}  clean")
    aggs = {}
    for rule in AMR_RULES:
        rows = by_rule.get(rule, [])
        if not rows:
            continue
        agg = ie.aggregate(rows)
        aggs[rule] = agg
        print(f"{rule:<20} {agg['ideal']:>8.1f} {agg['executed']:>8.1f} "
              f"{100 * agg['penalty']:>7.1f}% {100 * agg['omega_q']:>6.1f}% "
              f"{100 * agg['omega_r']:>6.1f}% {100 * agg['queue_share']:>7.0f}% "
              f"{agg['nu_per_episode']:>6.2f}  "
              f"{int(agg['clean_instances'])}/{int(agg['instances'])}")

    lam = 100 * statistics.mean([a["penalty"] for a in aggs.values()])
    qsh = 100 * statistics.mean([a["queue_share"] for a in aggs.values()])
    ex = [a["executed"] for a in aggs.values()]
    spread = 100 * (max(ex) - min(ex)) / min(ex)
    nu = statistics.mean([a["nu_per_episode"] for a in aggs.values()])
    n = int(max(a["instances"] for a in aggs.values()))

    print(f"\nover {n} instances x {len(aggs)} AMR rules")
    print(f"  Lambda                   {lam:6.1f}%   (floor 13%)      "
          f"{'PASS' if lam >= 13 else 'CHECK'}")
    print(f"  queueing share of error  {qsh:6.1f}%   (> 50%)          "
          f"{'PASS' if qsh > 50 else 'CHECK'}")
    print(f"  AMR-rule spread          {spread:6.2f}%   (> 5%)           "
          f"{'PASS' if spread > 5 else 'CHECK'}")
    print(f"  unroutable / episode     {nu:7.2f}   (< 0.1)          "
          f"{'PASS' if nu < 0.1 else 'CHECK'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", type=str, default="../test_case/v3/instances_60.jsonl")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--amrs", type=int, default=sc.NUM_AMRS)
    ap.add_argument("--job_rule", type=str, default="milk_run")
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.summarise:
        summarise(out)
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
                row.update({"instance": ev["index"], "rule": amr_rule})
                fh.write(json.dumps(row) + "\n")
    print(f"appended {len(events)} instances x {len(AMR_RULES)} rules -> {out}")


if __name__ == "__main__":
    main()
