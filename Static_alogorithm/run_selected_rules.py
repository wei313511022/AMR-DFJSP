"""Run a named list of (job_rule, AMR_rule) combinations and record makespan + runtime.

`sweep_fleet.py` always runs all six AMR rules for one job rule, which wastes ~10x the
compute when only a handful of combinations are wanted. This runs exactly the requested
pairs and emits the identical row schema -- same join keys (`amrs`, `instance`,
`job_rule`, `rule`, `family="rule"`), same `solve_s` / `eval_s` split -- so its rows drop
into the same analyses.

    python run_selected_rules.py --inbox ../test_case/v3/trend/full_60.jsonl \
        --combos milk_run+earliest_completion,milk_run+earliest_available \
        --amrs 16 --out rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

STATIC_DIR = os.path.abspath(os.path.dirname(__file__))
if STATIC_DIR not in sys.path:
    sys.path.insert(0, STATIC_DIR)

import GA.GA as GA  # noqa: E402
import ideal_evaluator as ie  # noqa: E402
import operation_policy  # noqa: E402
import scenario_v3 as sc  # noqa: E402
from GA.GA import load_dispatch_events  # noqa: E402
from dispatching_rules.dispatching_rules import AMR_RULES, JOB_RULES  # noqa: E402
from reinforce_baseline import complete_with_dispatch_rule  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", required=True)
    ap.add_argument("--combos", required=True, help="comma-separated job_rule+amr_rule")
    ap.add_argument("--amrs", type=int, default=16)
    ap.add_argument("--count", type=int, default=0, help="0 = all")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    combos = []
    for spec in args.combos.split(","):
        spec = spec.strip()
        if not spec:
            continue
        jr, ar = spec.split("+", 1)
        # Typo-proofing: an unknown rule name would otherwise surface as a KeyError deep
        # inside the scorer, after an arbitrary amount of the sweep had already run.
        if jr not in JOB_RULES:
            raise SystemExit(f"unknown job rule {jr!r}; valid: {', '.join(JOB_RULES)}")
        if ar not in AMR_RULES:
            raise SystemExit(f"unknown AMR rule {ar!r}; valid: {', '.join(AMR_RULES)}")
        combos.append((jr, ar))

    sc.apply_layout(num_amrs=args.amrs)
    # Same assertion eval_extend_gnn.install_fleet makes: apply_layout patches
    # operation_policy inside a bare `except Exception: pass`, and a swallowed failure
    # leaves the dock-queue features in the previous fleet's units.
    if len(GA.AMR_KEYS) != args.amrs:
        raise SystemExit(f"apply_layout produced {len(GA.AMR_KEYS)} AMRs, expected {args.amrs}")
    if operation_policy.DOCK_QUEUE_SCALE != float(args.amrs):
        raise SystemExit("apply_layout's operation_policy patch did not run")

    events = load_dispatch_events(Path(args.inbox))[args.start:]
    if args.count:
        events = events[: args.count]
    if not events:
        raise SystemExit("no events selected")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_jobs = len(events[0]["jobs"])

    with out.open("a", encoding="utf-8") as fh:
        for jr, ar in combos:
            t_combo = time.perf_counter()
            execs = []
            for ev in events:
                jobs = ev["jobs"]
                t0 = time.perf_counter()
                ind = complete_with_dispatch_rule(
                    jobs, [], {}, baseline_rule=f"{jr}+{ar}", seed=args.seed)
                solve_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                row = ie.evaluate(ind, jobs)
                eval_s = time.perf_counter() - t0
                row.update({"amrs": args.amrs, "instance": ev["index"],
                            "job_rule": jr, "rule": ar, "family": "rule",
                            "n_jobs": len(jobs), "dataset": Path(args.inbox).stem,
                            "solve_s": round(solve_s, 4), "eval_s": round(eval_s, 4)})
                fh.write(json.dumps(row) + "\n")
                if row["nu"] == 0:
                    execs.append(row["executed"])
            mean = sum(execs) / len(execs) if execs else float("nan")
            print(f"  n={n_jobs:>3} {jr}+{ar:<20} executed {mean:>8.2f} | "
                  f"clean {len(execs)}/{len(events)} | {time.perf_counter() - t_combo:>6.1f}s")


if __name__ == "__main__":
    main()
