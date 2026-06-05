"""Collision-heavy compatibility entrypoint for the corrected GA model.

This module used to contain a full copy of the old GA implementation. It now
delegates to GA.py so it cannot drift back to the obsolete type-as-dock model.
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path

try:
    from . import GA as _ga
except ImportError:
    import GA as _ga

_ga.routing_iters = 0
_ga.collision_routing_iters = 2000

globals().update({name: value for name, value in vars(_ga).items() if not name.startswith("_")})
routing_iters = _ga.routing_iters
collision_routing_iters = _ga.collision_routing_iters


def evolve(jobs, init_state=None):
    _ga.routing_iters = routing_iters
    _ga.collision_routing_iters = collision_routing_iters
    return _ga.evolve(jobs, init_state=init_state)


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    dispatch_events = _ga.load_dispatch_events(Path(args.inbox)) if args.inbox else _ga.load_dispatch_events()
    target_index = os.environ.get(_ga.DISPATCH_EVENT_INDEX_ENV)
    if dispatch_events and target_index is not None:
        dispatch_events = [event for event in dispatch_events if str(event["index"]) == str(target_index)]

    rows = []
    if dispatch_events:
        for event in dispatch_events:
            print(f"\n=== Processing Dispatch Event {event['index']} (Jobs: {len(event['jobs'])}) ===")
            start = time.perf_counter()
            best_ind, _ = evolve(event["jobs"])
            solve_time = time.perf_counter() - start
            img_path = f"{args.save_img.split('.')[0]}_{event['index']}.png" if args.save_img else None
            makespan, computation_time = _ga.describe_solution(
                best_ind,
                event["jobs"],
                solve_time=solve_time,
                show_gantt=args.gantt,
                save_img=img_path,
            )
            rows.append([event["index"], f"{makespan:.2f}", f"{computation_time:.4f}"])
    else:
        print("No dispatch file found. Generating random jobs...")
        jobs = _ga.make_jobs()
        start = time.perf_counter()
        best_ind, _ = evolve(jobs)
        solve_time = time.perf_counter() - start
        makespan, computation_time = _ga.describe_solution(
            best_ind,
            jobs,
            solve_time=solve_time,
            show_gantt=args.gantt,
            save_img=args.save_img,
        )
        rows.append(["random", f"{makespan:.2f}", f"{computation_time:.4f}"])

    if rows:
        with open(args.output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Event_Index", "Makespan", "Computation_Time"])
            writer.writerows(rows)
        print(f"\nSummary results saved to {args.output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gantt", action="store_true", help="Plot Gantt Chart")
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file")
    parser.add_argument("--save_img", type=str, default="", help="Save the schedule Gantt chart to this file")
    parser.add_argument("--output_csv", type=str, default="GA_with_collisions_summary_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
