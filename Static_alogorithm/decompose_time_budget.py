"""Time-budget decomposition of executed schedules.

Decodes schedules with the collision-aware executor and attributes every
AMR-second to a category from the logged timeline:

  travel     loaded/empty movement, incl. waiting-line approach legs
  service    dock loading (load_inbound) and station processing (process_*)
  dock_wait  queueing at inbound/outbound docks (wait_*_line entries)
  hold       upstream holds when a waiting line is saturated
  return     final trip back to the depot
  idle       fleet-time not covered above (parked at base / finished early)

Fleet-time per episode is num_amrs x makespan, so category percentages sum
to 100% including idle. Reports the dispatch-rule baseline and the trained
GNN on the same instances; supports zero-shot fleet scaling via --num_amrs
(reuses set_fleet from eval_fleet_size).

Usage:
    python decompose_time_budget.py --num_amrs 5 --events 25
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

STATIC_DIR = os.path.abspath(os.path.dirname(__file__))
GNN_DIR = os.path.join(STATIC_DIR, "GNN")
for path in (STATIC_DIR, GNN_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import GA.GA as GA  # noqa: E402
from GA.GA import decode_schedule_tick_by_tick, load_dispatch_events  # noqa: E402
from GNN import SchedulerGNN, solve_with_gnn  # noqa: E402
from eval_fleet_size import set_fleet  # noqa: E402
from operation_policy import load_required_operation_checkpoint  # noqa: E402
from reinforce_baseline import (  # noqa: E402
    DEFAULT_BASELINE_RULE,
    complete_with_dispatch_rule,
)

CATEGORY_OF_KIND = {
    "travel": "travel",
    "return": "return",
    "hold_upstream": "hold",
    "wait_inbound_line": "dock_wait",
    "wait_outbound_line": "dock_wait",
    "load_inbound": "service",
}
CATEGORIES = ("travel", "service", "dock_wait", "hold", "return", "idle")


def categorize(kind: str) -> str:
    if kind.startswith("process_"):
        return "service"
    try:
        return CATEGORY_OF_KIND[kind]
    except KeyError:
        raise ValueError(f"Unrecognized timeline kind: {kind!r}") from None


def decompose(individual, jobs):
    availability, timelines, _, _, invalid_count = decode_schedule_tick_by_tick(
        individual, jobs, need_log=True, check_collision=True
    )
    makespan = max(availability.values())
    budget = defaultdict(float)
    for _amr, start, end, kind, _label in timelines:
        budget[categorize(kind)] += max(0.0, end - start)
    fleet_time = makespan * len(GA.AMR_KEYS)
    budget["idle"] = fleet_time - sum(budget.values())
    return makespan, invalid_count, budget, fleet_time


def report(name: str, results) -> None:
    n = len(results)
    avg_makespan = sum(r[0] for r in results) / n
    avg_invalid = sum(r[1] for r in results) / n
    totals = {c: sum(r[2][c] for r in results) / n for c in CATEGORIES}
    fleet_time = sum(r[3] for r in results) / n
    print(f"\n{name}: makespan {avg_makespan:.1f} | invalid/ep {avg_invalid:.2f} | fleet-time {fleet_time:.0f}")
    for cat in CATEGORIES:
        print(f"  {cat:<10s} {totals[cat]:8.1f}  ({100 * totals[cat] / fleet_time:5.1f}% of fleet-time)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_amrs", type=int, default=5)
    parser.add_argument("--events", type=int, default=25)
    parser.add_argument("--inbox", type=str, default=os.path.join(STATIC_DIR, "..", "test_case", "static", "dispatch_validation_60.jsonl"))
    parser.add_argument("--weights", type=str, default=os.path.join(STATIC_DIR, "..", "gnn_mpn_scheduler_best.pth"))
    parser.add_argument("--baseline_rule", type=str, default=DEFAULT_BASELINE_RULE)
    parser.add_argument("--skip_gnn", action="store_true", help="Decompose only the dispatch-rule schedules")
    args = parser.parse_args()

    set_fleet(args.num_amrs)
    print(f"Fleet: {len(GA.AMR_KEYS)} AMRs | events: {args.events} | rule: {args.baseline_rule}")

    model = None
    if not args.skip_gnn:
        torch.manual_seed(42)
        model = SchedulerGNN(job_in_dim=16, amr_in_dim=8, hidden_dim=128, gin_layers=3)
        status = load_required_operation_checkpoint(
            model,
            Path(args.weights),
            torch,
            required_keys=("op_emb.weight", "operation_actor.0.weight"),
        )
        print(f"Checkpoint: {status}")

    events = load_dispatch_events(Path(args.inbox))[: args.events]
    rule_results, gnn_results = [], []
    for event in events:
        jobs = event["jobs"]
        rule_ind = complete_with_dispatch_rule(
            jobs, prefix_operations=[], prefix_assignment={},
            baseline_rule=args.baseline_rule, seed=42,
        )
        rule_results.append(decompose(rule_ind, jobs))
        if model is not None:
            gnn_ind, _, _ = solve_with_gnn(jobs, model, deterministic=True)
            gnn_results.append(decompose(gnn_ind, jobs))

    report(f"rule ({args.baseline_rule}), {args.num_amrs} AMRs", rule_results)
    if gnn_results:
        report(f"GNN (5-AMR weights), {args.num_amrs} AMRs", gnn_results)


if __name__ == "__main__":
    main()
