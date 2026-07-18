"""Zero-shot fleet-size evaluation for the trained GNN scheduler.

Runs the best 5-AMR GNN checkpoint on a fleet of --num_amrs AMRs without
retraining, against the dispatch-rule baseline on the same instances, and
reports collision-aware makespans plus the congestion tax (collision-aware
minus collision-free decode of the same schedule).

The fleet is defined by module-level constants in GA.GA; this script mutates
AMR_STARTS / AMR_KEYS / BASES *in place* after imports, so every module that
from-imported them sees the new fleet. DOCK_QUEUE_SCALE in operation_policy
is intentionally left at its import-time value so queue features keep the
units the checkpoint was trained on.

Usage:
    python eval_fleet_size.py --num_amrs 10 --events 25
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

STATIC_DIR = os.path.abspath(os.path.dirname(__file__))
GNN_DIR = os.path.join(STATIC_DIR, "GNN")
for path in (STATIC_DIR, GNN_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import GA.GA as GA  # noqa: E402
from GA.GA import load_dispatch_events  # noqa: E402
from GNN import SchedulerGNN, solve_with_gnn  # noqa: E402
from operation_policy import load_required_operation_checkpoint  # noqa: E402
from reinforce_baseline import (  # noqa: E402
    DEFAULT_BASELINE_RULE,
    complete_with_dispatch_rule,
    evaluate_makespan,
)

BASE_YS = (9, 7, 5, 3, 1)


def set_fleet(num_amrs: int) -> None:
    """Mutate the global fleet in place. First 5 AMRs match the original layout."""
    if num_amrs < 1:
        raise ValueError("--num_amrs must be at least 1")
    max_cols = 5  # columns x=2..6 keep bases clear of docks (x=0) and stations (x=9)
    if num_amrs > max_cols * len(BASE_YS):
        raise ValueError(f"--num_amrs must be at most {max_cols * len(BASE_YS)}")
    starts = {}
    for i in range(num_amrs):
        col = 2 + i // len(BASE_YS)
        starts[f"AMR{i + 1}"] = (col, BASE_YS[i % len(BASE_YS)])
    GA.AMR_STARTS.clear()
    GA.AMR_STARTS.update(starts)
    GA.AMR_KEYS.clear()
    GA.AMR_KEYS.extend(starts.keys())
    GA.BASES.clear()
    GA.BASES.extend(starts.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_amrs", type=int, default=10)
    parser.add_argument("--events", type=int, default=25, help="Number of validation events to evaluate")
    parser.add_argument("--inbox", type=str, default=os.path.join(STATIC_DIR, "..", "test_case", "static", "dispatch_validation_60.jsonl"))
    parser.add_argument("--weights", type=str, default=os.path.join(STATIC_DIR, "..", "gnn_mpn_scheduler_best.pth"))
    parser.add_argument("--baseline_rule", type=str, default=DEFAULT_BASELINE_RULE)
    args = parser.parse_args()

    set_fleet(args.num_amrs)
    print(f"Fleet: {len(GA.AMR_KEYS)} AMRs at {list(GA.AMR_STARTS.values())}")

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
    rows = []
    for event in events:
        jobs = event["jobs"]

        rule_ind = complete_with_dispatch_rule(
            jobs, prefix_operations=[], prefix_assignment={},
            baseline_rule=args.baseline_rule, seed=42,
        )
        rule_free, _ = evaluate_makespan(rule_ind, jobs, check_collision=False)
        rule_coll, rule_inv = evaluate_makespan(rule_ind, jobs, check_collision=True)

        t0 = time.perf_counter()
        gnn_ind, _, _ = solve_with_gnn(jobs, model, deterministic=True)
        gnn_time = time.perf_counter() - t0
        gnn_free, _ = evaluate_makespan(gnn_ind, jobs, check_collision=False)
        gnn_coll, gnn_inv = evaluate_makespan(gnn_ind, jobs, check_collision=True)

        rows.append((rule_free, rule_coll, rule_inv, gnn_free, gnn_coll, gnn_inv, gnn_time))
        print(
            f"event {event['index']:>3}: rule {rule_coll:7.1f} (tax {rule_coll - rule_free:5.1f}, inv {rule_inv}) "
            f"| GNN {gnn_coll:7.1f} (tax {gnn_coll - gnn_free:5.1f}, inv {gnn_inv}) [{gnn_time:.1f}s]",
            flush=True,
        )

    n = len(rows)
    avg = lambda idx: sum(r[idx] for r in rows) / n
    rule_tax = avg(1) - avg(0)
    gnn_tax = avg(4) - avg(3)
    print(f"\n=== {args.num_amrs} AMRs | {n} events x {len(events[0]['jobs'])} jobs ===")
    print(f"rule ({args.baseline_rule}):")
    print(f"  makespan {avg(1):7.1f} | congestion tax {rule_tax:5.1f} ({100 * rule_tax / avg(0):4.1f}%) | invalid/ep {avg(2):.2f}")
    print("GNN (zero-shot 5-AMR weights):")
    print(f"  makespan {avg(4):7.1f} | congestion tax {gnn_tax:5.1f} ({100 * gnn_tax / avg(3):4.1f}%) | invalid/ep {avg(5):.2f} | {avg(6):.1f}s/ep")


if __name__ == "__main__":
    main()
