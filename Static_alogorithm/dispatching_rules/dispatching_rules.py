import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from GA.GA import (  # noqa: E402
    AMR_KEYS,
    AMR_LOAD_CAPACITY,
    AMR_STARTS,
    DISPATCH_EVENT_INDEX_ENV,
    INBOUND_DOCK_LOCATIONS,
    Individual,
    Job,
    Operation,
    PICKUP,
    STATIONS,
    TYPE_DURATION,
    UNLOAD,
    decode_schedule_tick_by_tick,
    dock_key_from_value,
    heuristic,
    job_pickup_location,
    load_dispatch_events,
    make_jobs,
    plot_gantt,
)
from operation_policy import (  # noqa: E402
    OperationAction,
    OperationEstimate,
    apply_fast_action,
    empty_dock_service_events,
    estimate_action,
    estimated_dock_wait,
    estimated_travel_time,
    legal_actions,
    station_remaining_workload as active_station_workload,
)


JOB_RULES = (
    "fifo",
    "spt",
    "lpt",
    "nearest_station",
    "most_congested_station",
    "least_congested_station",
    "earliest_completion_job",
    "material_match",
    "random",
)

AMR_RULES = (
    "earliest_available",
    "nearest_amr",
    "material_match",
    "earliest_completion",
    "least_loaded",
    "home_material",
    "random",
)

HOME_MATERIAL_AMR = {
    "A": "AMR1",
    "B": "AMR2",
    "C": "AMR3",
}


@dataclass
class RuleState:
    amr_positions: Dict[str, Tuple[int, int]]
    amr_availabilities: Dict[str, float]
    station_availabilities: Dict[str, float]
    inventory: Dict[str, Dict[str, int]]
    assigned_count: Dict[str, int]
    picked_jobs: set
    completed_jobs: set
    carrier_map: Dict[int, str]
    dock_service_events: Dict[str, List[Tuple[float, float, int, str]]] = None


@dataclass(frozen=True)
class AssignmentEstimate:
    amr: str
    start_time: float
    supply_end: float
    travel_end: float
    process_start: float
    process_end: float
    available_end: float
    completion_time: float
    travel_time: float
    supply_needed: bool


def initial_state() -> RuleState:
    dock_availabilities = {station: 0.0 for station in STATIONS.keys()}
    dock_availabilities.update({dock: 0.0 for dock in INBOUND_DOCK_LOCATIONS.keys()})
    return RuleState(
        amr_positions={amr: AMR_STARTS[amr] for amr in AMR_KEYS},
        amr_availabilities={amr: 0.0 for amr in AMR_KEYS},
        station_availabilities=dock_availabilities,
        inventory={amr: {mat: 0 for mat in TYPE_DURATION.keys()} for amr in AMR_KEYS},
        assigned_count={amr: 0 for amr in AMR_KEYS},
        picked_jobs=set(),
        completed_jobs=set(),
        carrier_map={},
        dock_service_events=empty_dock_service_events(),
    )


def estimate_assignment(job: Job, amr: str, state: RuleState) -> AssignmentEstimate:
    material = job.type_
    curr_pos = state.amr_positions[amr]
    avail = state.amr_availabilities[amr]
    capacity_blocked = state.inventory[amr].get(material, 0) >= AMR_LOAD_CAPACITY
    pickup_location = job_pickup_location(job)
    to_pickup = estimated_travel_time(curr_pos, pickup_location)
    inbound_dock = dock_key_from_value(job.inbound_dock)
    ready_time = max(avail + to_pickup, float(job.arrival_time))
    pickup_start = ready_time + estimated_dock_wait(
        inbound_dock, ready_time, state.station_availabilities, state.dock_service_events
    )
    pickup_end = pickup_start + job.duration

    target_station = STATIONS[job.station]
    to_station = estimated_travel_time(pickup_location, target_station)
    travel_end = pickup_end + to_station
    process_start = travel_end + estimated_dock_wait(
        job.station, travel_end, state.station_availabilities, state.dock_service_events
    )
    process_end = process_start + job.duration
    travel_time = to_pickup + to_station

    return AssignmentEstimate(
        amr=amr,
        start_time=avail,
        supply_end=pickup_end,
        travel_end=travel_end,
        process_start=process_start,
        process_end=process_end,
        available_end=process_end,
        completion_time=process_end,
        travel_time=travel_time,
        supply_needed=capacity_blocked,
    )


def apply_assignment(job: Job, estimate: AssignmentEstimate, state: RuleState) -> None:
    amr = estimate.amr
    material = job.type_

    state.inventory[amr][material] = min(
        AMR_LOAD_CAPACITY,
        state.inventory[amr].get(material, 0) + 1,
    )
    state.inventory[amr][material] -= 1
    state.station_availabilities[dock_key_from_value(job.inbound_dock)] = estimate.supply_end
    state.station_availabilities[job.station] = estimate.process_end
    state.amr_availabilities[amr] = estimate.process_end
    state.amr_positions[amr] = STATIONS[job.station]
    state.assigned_count[amr] += 1


def station_remaining_workload(jobs: Sequence[Job]) -> Dict[str, float]:
    workload = {station: 0.0 for station in STATIONS.keys()}
    for job in jobs:
        workload[job.station] += job.duration
    return workload


def _job_rule_score(action: OperationAction, estimate: OperationEstimate, jobs: Sequence[Job], state: RuleState, job_rule: str, rng: random.Random):
    job = jobs[action.job_list_idx]
    if job_rule == "fifo":
        return (job.idx,)
    if job_rule == "spt":
        return (job.duration, job.idx)
    if job_rule == "lpt":
        return (-job.duration, job.idx)
    if job_rule == "nearest_station":
        distance = heuristic(state.amr_positions[action.amr], STATIONS[job.station])
        return (distance, estimate.travel_time, job.duration, job.idx)
    if job_rule in {"most_congested_station", "least_congested_station"}:
        workload = estimate.station_workload
        if job_rule == "most_congested_station":
            return (-workload, -job.duration, job.idx)
        return (workload, job.duration, job.idx)
    if job_rule == "earliest_completion_job":
        return (estimate.projected_completion, job.duration, job.idx)
    if job_rule == "material_match":
        return (estimate.material_match, estimate.projected_completion, job.idx)
    if job_rule == "random":
        return (rng.random(), job.idx)
    raise ValueError(f"Unknown job rule: {job_rule}")


def _amr_rule_score(action: OperationAction, estimate: OperationEstimate, jobs: Sequence[Job], state: RuleState, amr_rule: str, rng: random.Random):
    job = jobs[action.job_list_idx]
    if amr_rule == "earliest_available":
        return (state.amr_availabilities[action.amr], estimate.projected_completion, action.amr)
    if amr_rule == "nearest_amr":
        return (estimate.travel_time, estimate.projected_completion, action.amr)
    if amr_rule == "material_match":
        return (estimate.material_match, estimate.projected_completion, action.amr)
    if amr_rule == "earliest_completion":
        return (estimate.projected_completion, estimate.travel_time, action.amr)
    if amr_rule == "least_loaded":
        return (estimate.assigned_count, state.amr_availabilities[action.amr], estimate.projected_completion, action.amr)
    if amr_rule == "home_material":
        preferred = HOME_MATERIAL_AMR.get(job.type_)
        return (0 if action.amr == preferred else 1, estimate.projected_completion, action.amr)
    if amr_rule == "random":
        return (rng.random(), action.amr)
    raise ValueError(f"Unknown AMR rule: {amr_rule}")


def choose_operation(
    jobs: Sequence[Job],
    state: RuleState,
    job_rule: str,
    amr_rule: str,
    rng: random.Random,
) -> Tuple[OperationAction, OperationEstimate]:
    actions = legal_actions(
        jobs,
        state.picked_jobs,
        state.completed_jobs,
        state.carrier_map,
        state.inventory,
    )
    if not actions:
        raise RuntimeError("No legal dispatch operation is available before all jobs completed.")

    workloads = active_station_workload(jobs, state.completed_jobs)
    estimates = [
        estimate_action(
            action,
            jobs,
            state.amr_positions,
            state.amr_availabilities,
            state.station_availabilities,
            state.inventory,
            state.assigned_count,
            workloads,
            dock_service_events=state.dock_service_events,
        )
        for action in actions
    ]

    def score(estimate):
        action = estimate.action
        return (
            _job_rule_score(action, estimate, jobs, state, job_rule, rng),
            _amr_rule_score(action, estimate, jobs, state, amr_rule, rng),
            0 if action.kind == UNLOAD else 1,
            action.job_id,
            action.amr,
        )

    best = min(estimates, key=score)
    return best.action, best


def apply_operation(action: OperationAction, state: RuleState, jobs: Sequence[Job]) -> None:
    apply_fast_action(
        action,
        jobs,
        state.picked_jobs,
        state.completed_jobs,
        state.carrier_map,
        state.amr_positions,
        state.amr_availabilities,
        state.station_availabilities,
        state.inventory,
        state.assigned_count,
        dock_service_events=state.dock_service_events,
    )


def best_estimate_for_job(job: Job, state: RuleState) -> AssignmentEstimate:
    return min(
        (estimate_assignment(job, amr, state) for amr in AMR_KEYS),
        key=lambda est: (est.completion_time, est.travel_time, est.amr),
    )


def choose_job(
    unscheduled: Sequence[Job],
    state: RuleState,
    job_rule: str,
    rng: random.Random,
) -> Job:
    if job_rule == "fifo":
        return min(unscheduled, key=lambda job: job.idx)
    if job_rule == "spt":
        return min(unscheduled, key=lambda job: (job.duration, job.idx))
    if job_rule == "lpt":
        return max(unscheduled, key=lambda job: (job.duration, -job.idx))
    if job_rule == "nearest_station":
        return min(
            unscheduled,
            key=lambda job: (
                min(heuristic(state.amr_positions[amr], STATIONS[job.station]) for amr in AMR_KEYS),
                job.duration,
                job.idx,
            ),
        )
    if job_rule in {"most_congested_station", "least_congested_station"}:
        workload = station_remaining_workload(unscheduled)
        if job_rule == "most_congested_station":
            return max(unscheduled, key=lambda job: (workload[job.station], -job.duration, -job.idx))
        return min(unscheduled, key=lambda job: (workload[job.station], job.duration, job.idx))
    if job_rule == "earliest_completion_job":
        return min(
            unscheduled,
            key=lambda job: (
                best_estimate_for_job(job, state).completion_time,
                job.duration,
                job.idx,
            ),
        )
    if job_rule == "material_match":
        return min(
            unscheduled,
            key=lambda job: (
                0 if any(state.inventory[amr].get(job.type_, 0) > 0 for amr in AMR_KEYS) else 1,
                best_estimate_for_job(job, state).completion_time,
                job.idx,
            ),
        )
    if job_rule == "random":
        return rng.choice(list(unscheduled))

    raise ValueError(f"Unknown job rule: {job_rule}")


def choose_amr(job: Job, state: RuleState, amr_rule: str, rng: random.Random) -> AssignmentEstimate:
    estimates = {amr: estimate_assignment(job, amr, state) for amr in AMR_KEYS}

    if amr_rule == "earliest_available":
        chosen = min(
            AMR_KEYS,
            key=lambda amr: (
                state.amr_availabilities[amr],
                estimates[amr].completion_time,
                amr,
            ),
        )
    elif amr_rule == "nearest_amr":
        chosen = min(
            AMR_KEYS,
            key=lambda amr: (
                estimates[amr].travel_time,
                estimates[amr].completion_time,
                amr,
            ),
        )
    elif amr_rule == "material_match":
        chosen = min(
            AMR_KEYS,
            key=lambda amr: (
                0 if state.inventory[amr].get(job.type_, 0) > 0 else 1,
                estimates[amr].completion_time,
                amr,
            ),
        )
    elif amr_rule == "earliest_completion":
        chosen = min(
            AMR_KEYS,
            key=lambda amr: (
                estimates[amr].completion_time,
                estimates[amr].travel_time,
                amr,
            ),
        )
    elif amr_rule == "least_loaded":
        chosen = min(
            AMR_KEYS,
            key=lambda amr: (
                state.assigned_count[amr],
                state.amr_availabilities[amr],
                estimates[amr].completion_time,
                amr,
            ),
        )
    elif amr_rule == "home_material":
        preferred = HOME_MATERIAL_AMR.get(job.type_)
        chosen = preferred if preferred in estimates else min(AMR_KEYS)
    elif amr_rule == "random":
        chosen = rng.choice(list(AMR_KEYS))
    else:
        raise ValueError(f"Unknown AMR rule: {amr_rule}")

    return estimates[chosen]


def solve_with_dispatching_rules(
    jobs: Sequence[Job],
    job_rule: str,
    amr_rule: str,
    seed: int = 42,
) -> Tuple[Individual, float]:
    start_time = time.perf_counter()
    rng = random.Random(seed)
    state = initial_state()

    order: List[Operation] = []
    amr_assignment = [""] * len(jobs)

    while len(state.completed_jobs) < len(jobs):
        action, _ = choose_operation(jobs, state, job_rule, amr_rule, rng)
        order.append(Operation(action.job_id, action.kind))
        if action.kind == PICKUP:
            amr_assignment[action.job_id] = action.amr
        apply_operation(action, state, jobs)

    solve_time = time.perf_counter() - start_time
    return Individual(order=order, amr_assignment=amr_assignment), solve_time


def evaluate_individual(individual: Individual, jobs: Sequence[Job]) -> Tuple[float, int]:
    availability, _, _, _, invalid_count = decode_schedule_tick_by_tick(
        individual,
        list(jobs),
        need_log=False,
        check_collision=True,
    )
    return max(availability.values()), invalid_count


def make_rule_pairs(selected_job_rules: Sequence[str], selected_amr_rules: Sequence[str]) -> List[Tuple[str, str]]:
    return [(job_rule, amr_rule) for job_rule in selected_job_rules for amr_rule in selected_amr_rules]


def parse_rule_list(raw: str, valid: Sequence[str], label: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(valid)

    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in selected if item not in valid]
    if unknown:
        raise ValueError(f"Unknown {label} rule(s): {', '.join(unknown)}. Valid: {', '.join(valid)}")
    return selected


def maybe_save_gantt(
    individual: Individual,
    jobs: Sequence[Job],
    solve_time: float,
    save_path: str,
) -> None:
    availability, timeline, queue_infos, _, invalid_count = decode_schedule_tick_by_tick(
        individual,
        list(jobs),
        need_log=True,
        check_collision=True,
    )
    _ = availability
    plot_gantt(
        timeline,
        queue_infos,
        list(jobs),
        solve_time=solve_time,
        invalid_count=invalid_count,
        show_gantt=False,
        save_img=save_path,
    )


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    selected_job_rules = parse_rule_list(args.job_rules, JOB_RULES, "job")
    selected_amr_rules = parse_rule_list(args.amr_rules, AMR_RULES, "AMR")
    rule_pairs = make_rule_pairs(selected_job_rules, selected_amr_rules)

    if args.inbox:
        dispatch_events = load_dispatch_events(Path(args.inbox))
    else:
        dispatch_events = load_dispatch_events()

    target_index = os.environ.get(DISPATCH_EVENT_INDEX_ENV)
    if dispatch_events and target_index is not None:
        dispatch_events = [event for event in dispatch_events if str(event["index"]) == str(target_index)]

    if not dispatch_events:
        print("No dispatch file found. Generating random jobs...")
        dispatch_events = [{"index": "random", "jobs": make_jobs()}]

    rows = []
    best_by_event: Dict[object, Tuple[float, Individual, Sequence[Job], str, float]] = {}

    print(f"Testing {len(rule_pairs)} dispatching-rule combinations on {len(dispatch_events)} event(s).")

    for event in dispatch_events:
        jobs = event["jobs"]
        print(f"\n=== Event {event['index']} | Jobs: {len(jobs)} ===")

        for job_rule, amr_rule in rule_pairs:
            rule_name = f"{job_rule}+{amr_rule}"
            individual, solve_time = solve_with_dispatching_rules(
                jobs,
                job_rule,
                amr_rule,
                seed=args.seed,
            )
            makespan, invalid_count = evaluate_individual(individual, jobs)

            print(
                f"{rule_name:45s} | Makespan: {makespan:8.2f} | "
                f"Invalid: {invalid_count:3d} | Time: {solve_time:.4f}s"
            )

            rows.append(
                [
                    event["index"],
                    rule_name,
                    job_rule,
                    amr_rule,
                    f"{makespan:.2f}",
                    invalid_count,
                    f"{solve_time:.6f}",
                ]
            )

            best = best_by_event.get(event["index"])
            if best is None or makespan < best[0]:
                best_by_event[event["index"]] = (makespan, individual, jobs, rule_name, solve_time)

    output_path = Path(args.output_csv)
    with output_path.open(mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Event_Index",
                "Rule",
                "Job_Rule",
                "AMR_Rule",
                "Makespan",
                "Invalid_Jobs",
                "Computation_Time",
            ]
        )
        writer.writerows(rows)

    print(f"\nDispatching-rule summary saved to {output_path}")

    if args.save_best_gantt:
        gantt_root = Path(args.save_best_gantt)
        for event_index, (makespan, individual, jobs, rule_name, solve_time) in best_by_event.items():
            if gantt_root.suffix:
                save_path = gantt_root
            else:
                gantt_root.mkdir(parents=True, exist_ok=True)
                safe_rule = rule_name.replace("+", "_").replace("/", "_")
                save_path = gantt_root / f"event_{event_index}_{safe_rule}.png"

            maybe_save_gantt(individual, jobs, solve_time, str(save_path))
            print(f"Saved best Gantt for event {event_index} ({rule_name}, {makespan:.2f}s) to {save_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate dispatching-rule combinations for the static AMR-DFJSP problem."
    )
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file")
    parser.add_argument("--output_csv", type=str, default="dispatching_rules_summary_results.csv")
    parser.add_argument("--job_rules", type=str, default="all", help=f"Comma list or all. Valid: {', '.join(JOB_RULES)}")
    parser.add_argument("--amr_rules", type=str, default="all", help=f"Comma list or all. Valid: {', '.join(AMR_RULES)}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_best_gantt",
        type=str,
        default="",
        help="Optional PNG path or directory for the best rule per event.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
