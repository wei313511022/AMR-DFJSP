"""Routing/schedule-output entrypoint for the corrected GA model.

The old copy in this file used the obsolete type-as-dock refill model. This
wrapper delegates scheduling to GA.py and writes operation-aware JSONL outputs.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

try:
    from . import GA as _ga
except ImportError:
    import GA as _ga

globals().update({name: value for name, value in vars(_ga).items() if not name.startswith("_")})


def _write_routing_output(path_logs, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    max_len = max((len(points) for points in path_logs.values()), default=0)
    with output_file.open("w", encoding="utf-8") as f:
        for step in range(max_len):
            record = {"step": step}
            for amr in _ga.AMR_KEYS:
                points = path_logs.get(amr, [_ga.AMR_STARTS[amr]])
                point = points[step] if step < len(points) else points[-1]
                record[amr] = {"x": point[0], "y": point[1]}
            f.write(json.dumps(record) + "\n")
    print(f"AMR routing output saved to {output_path} ({max_len} steps)")


def _write_schedule_output(individual, jobs, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    job_map = {job.idx: job for job in jobs}
    amr_ops = {amr: [] for amr in _ga.AMR_KEYS}
    for op in _ga.repair_operation_order(list(individual.order), jobs):
        job = job_map[op.job_idx]
        amr = individual.amr_assignment[job.idx]
        pickup_location = _ga.job_pickup_location(job)
        amr_ops[amr].append(
            {
                "job": job.idx,
                "operation": op.kind,
                "type": job.type_,
                "duration": job.duration,
                "inbound_dock": job.inbound_dock,
                "pickup_location": list(pickup_location),
                "station": job.station,
                "arrival_time": job.arrival_time,
            }
        )

    with output_file.open("w", encoding="utf-8") as f:
        for amr in _ga.AMR_KEYS:
            f.write(json.dumps({"amr": amr, "operations": amr_ops[amr]}) + "\n")
    print(f"AMR schedule output saved to {output_path}")


def describe_solution(
    individual,
    jobs: List[_ga.Job],
    solve_time: float = None,
    show_gantt: bool = False,
    save_img: str = None,
    routing_output: str = "amr_routing.jsonl",
    schedule_output: str = "amr_schedule.jsonl",
) -> Tuple[float, float]:
    availability, timeline, queue_infos, path_logs, invalid_count = _ga.decode_schedule_tick_by_tick(
        individual,
        jobs,
        need_log=True,
        check_collision=True,
    )
    makespan = max(availability.values())
    print(f"Optimal Makespan Found: {makespan:.2f}s")
    print(f"Invalid Jobs Count: {invalid_count}")
    if solve_time is not None:
        print(f"Computation Time: {solve_time:.4f}s")
    if show_gantt or save_img:
        _ga.plot_gantt(
            timeline,
            queue_infos,
            jobs,
            solve_time=solve_time,
            invalid_count=invalid_count,
            show_gantt=show_gantt,
            save_img=save_img,
        )
    _write_routing_output(path_logs, routing_output)
    _write_schedule_output(individual, jobs, schedule_output)
    return makespan, solve_time


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
            best_ind, _ = _ga.evolve(event["jobs"])
            solve_time = time.perf_counter() - start
            img_path = f"{args.save_img.split('.')[0]}_{event['index']}.png" if args.save_img else None
            routing_path = f"amr_routing_{event['index']}.jsonl"
            schedule_path = f"amr_schedule_{event['index']}.jsonl"
            makespan, computation_time = describe_solution(
                best_ind,
                event["jobs"],
                solve_time=solve_time,
                show_gantt=args.gantt,
                save_img=img_path,
                routing_output=routing_path,
                schedule_output=schedule_path,
            )
            rows.append([event["index"], f"{makespan:.2f}", f"{computation_time:.4f}"])
    else:
        print("No dispatch file found. Generating random jobs...")
        jobs = _ga.make_jobs()
        start = time.perf_counter()
        best_ind, _ = _ga.evolve(jobs)
        solve_time = time.perf_counter() - start
        makespan, computation_time = describe_solution(
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
    parser.add_argument("--output_csv", type=str, default="GA_routing_output_summary_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
