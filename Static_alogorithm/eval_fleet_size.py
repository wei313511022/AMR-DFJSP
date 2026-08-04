"""Zero-shot fleet-size evaluation -- the contention sweep over eta = m / |D_in|.

Runs the trained GNN checkpoint on a fleet of --num_amrs AMRs without
retraining, against the dispatch-rule baseline on the same instances, and
reports executed makespan, the idealised makespan, and Lambda / Omega_q /
Omega_r from ideal_evaluator.

Two things changed from the v2 version of this script:

  * The fleet is installed through scenario_v3.apply_layout, so a sweep varies
    ONLY eta. It used to lay bays out on the v1 geometry, which on the v3 floor
    sits inside the door queues and walls off columns.
  * The reported error is Lambda against the IDEALISED model, not the old
    "congestion tax" against a collision-free decode. That decode still
    enforced dock exclusivity and waiting lines, so it already contained the
    queueing it was supposed to expose.

apply_layout also resets DOCK_QUEUE_SCALE in operation_policy to the new fleet
size. That is deliberate: the queue features are ratios to fleet size, so a
checkpoint trained at m=16 sees the same units at m=24.

Usage:
    python eval_fleet_size.py --num_amrs 20 --events 25
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
import scenario_v3 as sc  # noqa: E402
import ideal_evaluator as ie  # noqa: E402
from GA.GA import load_dispatch_events  # noqa: E402
from GNN import SchedulerGNN, solve_with_gnn  # noqa: E402
from operation_policy import load_required_operation_checkpoint  # noqa: E402
from reinforce_baseline import (  # noqa: E402
    DEFAULT_BASELINE_RULE,
    complete_with_dispatch_rule,
    evaluate_makespan,
)

def set_fleet(num_amrs: int) -> None:
    """Install a fleet of `num_amrs` robots in the v3 pod depot.

    This used to lay bays out in columns x=2..6 on rows (9,7,5,3,1) -- the v1
    geometry, which on the v3 floor puts bays INSIDE the door waiting areas
    (dock_waiting_slots reaches x=3) and packs columns into walls that idle
    robots hold permanently in the reservation table. Both faults show up as
    unroutable parcels rather than as congestion. The sweep now goes through
    scenario_v3.apply_layout, so every fleet size gets the same pod geometry
    the headline m=16 setting uses and eta is the only thing that changes.
    """
    if num_amrs < 1:
        raise ValueError("--num_amrs must be at least 1")
    sc.apply_layout(num_amrs=num_amrs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_amrs", type=int, default=16)
    parser.add_argument("--events", type=int, default=25, help="Number of validation events to evaluate")
    parser.add_argument("--inbox", type=str, default=os.path.join(STATIC_DIR, "..", "test_case", "v3", "instances_60.jsonl"))
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
        rule_m = ie.evaluate(rule_ind, jobs)

        t0 = time.perf_counter()
        gnn_ind, _, _ = solve_with_gnn(jobs, model, deterministic=True)
        gnn_time = time.perf_counter() - t0
        gnn_m = ie.evaluate(gnn_ind, jobs)

        rows.append((rule_m, gnn_m, gnn_time))
        print(
            f"event {event['index']:>3}: rule {rule_m['executed']:7.1f} "
            f"(Lambda {100 * rule_m['penalty']:4.1f}%, nu {int(rule_m['nu'])}) "
            f"| GNN {gnn_m['executed']:7.1f} "
            f"(Lambda {100 * gnn_m['penalty']:4.1f}%, nu {int(gnn_m['nu'])}) [{gnn_time:.1f}s]",
            flush=True,
        )

    n = len(rows)
    rule_agg = ie.aggregate([r[0] for r in rows])
    gnn_agg = ie.aggregate([r[1] for r in rows])
    gnn_secs = sum(r[2] for r in rows) / n

    print(f"\n=== {args.num_amrs} AMRs | eta = {sc.contention_ratio(args.num_amrs):.2f} "
          f"| {n} events x {len(events[0]['jobs'])} jobs ===")
    for name, agg, extra in (
        (f"rule ({args.baseline_rule})", rule_agg, ""),
        ("GNN (zero-shot)", gnn_agg, f" | {gnn_secs:.1f}s/ep"),
    ):
        print(f"{name}:")
        print(f"  executed {agg['executed']:7.1f} | idealised {agg['ideal']:7.1f} "
              f"| Lambda {100 * agg['penalty']:5.1f}% "
              f"(Om_q {100 * agg['omega_q']:4.1f}%, Om_r {100 * agg['omega_r']:4.1f}%) "
              f"| nu/ep {agg['nu_per_episode']:.2f} "
              f"| clean {int(agg['clean_instances'])}/{int(agg['instances'])}{extra}")
    print("Durations are means over cleanly-routed episodes only; nu/ep is over all.")


if __name__ == "__main__":
    main()
