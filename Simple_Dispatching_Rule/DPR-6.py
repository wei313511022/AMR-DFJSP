#!/usr/bin/env python3
"""DPR-6 for route-aware FJSP: max processing time."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import map as map_config
import visualization
from calcu_dist import make_calculate_distance

_EPS = 1e-12

calculate_distance = make_calculate_distance(
    map_config.GRID_SIZE, map_config.BARRIER_NODES
)


@dataclass(frozen=True)
class MachineOption:
    machine: int
    processing: float


@dataclass
class JobState:
    job_id: int
    operations: List[List[MachineOption]]
    material: str
    next_op: int = 0
    ready_time: float = 0.0
    current_node: int = 0

    def remaining_ops(self) -> int:
        return len(self.operations) - self.next_op

    def next_options(self) -> List[MachineOption]:
        return self.operations[self.next_op]


@dataclass
class AmrState:
    amr_id: int
    ready_time: float
    current_node: int


@dataclass(frozen=True)
class RoutePlan:
    option: MachineOption
    amr_id: int
    amr_ready: float
    amr_prev_node: int
    pickup_node: int
    delivery_node: int
    depart: float
    arrival: float
    start: float
    end: float
    to_pick_travel: float
    transport: float
    machine_wait: float

    @property
    def travel_time(self) -> float:
        return float(self.to_pick_travel) + float(self.transport)

    @property
    def route_nodes(self) -> List[int]:
        return [int(self.amr_prev_node), int(self.pickup_node), int(self.delivery_node)]

    @property
    def route_legs(self) -> List[float]:
        return [float(self.to_pick_travel), float(self.transport)]


SelectFunc = Callable[
    [List[JobState], Dict[int, float], Dict[int, AmrState], Optional[int], Optional[int]],
    Tuple[JobState, MachineOption],
]


def load_instance(jsonl_path: str, index: int) -> dict:
    if index < 0:
        raise ValueError("index must be >= 0")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        current = 0
        for line in f:
            if not line.strip():
                continue
            if current == index:
                return json.loads(line)
            current += 1
    raise ValueError(f"index {index} out of range for {jsonl_path}")


def parse_jobs(instance: dict) -> List[JobState]:
    jobs_raw = instance.get("jobs", [])
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError("instance has no jobs")

    jobs: List[JobState] = []
    for j_idx, job in enumerate(jobs_raw):
        ops_raw = job.get("operations", [])
        if not isinstance(ops_raw, list) or not ops_raw:
            raise ValueError(f"job {j_idx} has no operations")

        material = str(job.get("material", "A"))
        if material not in map_config.TYPE_TO_MATERIAL_NODE:
            raise ValueError(f"job {j_idx} has unknown material: {material}")

        operations: List[List[MachineOption]] = []
        for op_idx, op_raw in enumerate(ops_raw):
            if not isinstance(op_raw, list) or not op_raw:
                raise ValueError(f"job {j_idx} op {op_idx} has no machine options")
            options: List[MachineOption] = []
            for opt in op_raw:
                machine = int(opt.get("machine"))
                processing = float(opt.get("processing"))
                options.append(MachineOption(machine=machine, processing=processing))
            operations.append(options)

        jobs.append(
            JobState(
                job_id=j_idx,
                operations=operations,
                material=material,
                current_node=int(map_config.TYPE_TO_MATERIAL_NODE[material]),
            )
        )
    return jobs


def init_machine_times(instance: dict, jobs: List[JobState]) -> Dict[int, float]:
    machine_count = instance.get("machines")
    if isinstance(machine_count, int) and machine_count > 0:
        machine_ids = list(range(int(machine_count)))
    else:
        machine_ids = sorted(
            {opt.machine for job in jobs for op in job.operations for opt in op}
        )
    return {int(m): 0.0 for m in machine_ids}


def init_amrs() -> Dict[int, AmrState]:
    return {
        int(amr_id): AmrState(
            amr_id=int(amr_id),
            ready_time=0.0,
            current_node=int(start_node),
        )
        for amr_id, start_node in map_config.S_m.items()
    }


def machine_to_node(machine: int) -> int:
    station_id = int(machine) + 1
    try:
        return int(map_config.JSON_STATION_MAPPING[station_id])
    except KeyError as exc:
        raise ValueError(f"machine {machine} has no station node mapping") from exc


def route_plan_for_option(
    job: JobState,
    option: MachineOption,
    machine_times: Dict[int, float],
    amrs: Dict[int, AmrState],
) -> RoutePlan:
    pickup_node = int(job.current_node)
    delivery_node = machine_to_node(int(option.machine))
    machine_ready = float(machine_times.get(int(option.machine), 0.0))

    best: Optional[RoutePlan] = None
    tied: List[RoutePlan] = []
    for amr in amrs.values():
        to_pick = float(calculate_distance(int(amr.current_node), pickup_node))
        transport = float(calculate_distance(pickup_node, delivery_node))

        # The AMR can wait before leaving, so it does not need to sit at pickup
        # before the job/material is ready.
        depart = max(float(amr.ready_time), float(job.ready_time) - to_pick)
        arrival = depart + to_pick + transport
        start = max(machine_ready, arrival)
        end = start + float(option.processing)
        plan = RoutePlan(
            option=option,
            amr_id=int(amr.amr_id),
            amr_ready=float(amr.ready_time),
            amr_prev_node=int(amr.current_node),
            pickup_node=pickup_node,
            delivery_node=delivery_node,
            depart=float(depart),
            arrival=float(arrival),
            start=float(start),
            end=float(end),
            to_pick_travel=float(to_pick),
            transport=float(transport),
            machine_wait=float(start - arrival),
        )

        if best is None:
            best = plan
            tied = [plan]
            continue

        if (
            plan.start < best.start - _EPS
            or (
                abs(plan.start - best.start) <= _EPS
                and plan.end < best.end - _EPS
            )
            or (
                abs(plan.start - best.start) <= _EPS
                and abs(plan.end - best.end) <= _EPS
                and plan.travel_time < best.travel_time - _EPS
            )
        ):
            best = plan
            tied = [plan]
        elif (
            abs(plan.start - best.start) <= _EPS
            and abs(plan.end - best.end) <= _EPS
            and abs(plan.travel_time - best.travel_time) <= _EPS
        ):
            tied.append(plan)

    if not tied:
        raise ValueError("no AMR route plan could be built")
    return random.choice(tied)


def choose_machine_earliest_start(
    job: JobState,
    options: List[MachineOption],
    machine_times: Dict[int, float],
    amrs: Dict[int, AmrState],
) -> MachineOption:
    best_start: Optional[float] = None
    tied: List[MachineOption] = []
    for opt in options:
        start = route_plan_for_option(job, opt, machine_times, amrs).start
        if best_start is None or start < best_start - _EPS:
            best_start = start
            tied = [opt]
        elif abs(start - best_start) <= _EPS:
            tied.append(opt)
    return random.choice(tied)


def choose_machine_by_processing(
    options: List[MachineOption], *, prefer_min: bool
) -> MachineOption:
    if prefer_min:
        target = min(opt.processing for opt in options)
    else:
        target = max(opt.processing for opt in options)
    tied = [opt for opt in options if abs(opt.processing - target) <= _EPS]
    return random.choice(tied)


def find_option_for_machine(
    options: List[MachineOption], machine_id: int
) -> Optional[MachineOption]:
    matches = [opt for opt in options if int(opt.machine) == int(machine_id)]
    if not matches:
        return None
    min_proc = min(opt.processing for opt in matches)
    tied = [opt for opt in matches if abs(opt.processing - min_proc) <= _EPS]
    return random.choice(tied)


def _dispatch_instance(
    instance: dict,
    *,
    seed: Optional[int] = None,
    rule_tag: str,
    select_next: SelectFunc,
) -> Tuple[List[dict], float]:
    if seed is not None:
        random.seed(int(seed))

    jobs = parse_jobs(instance)
    machine_times = init_machine_times(instance, jobs)
    amrs = init_amrs()

    total_ops = sum(len(job.operations) for job in jobs)
    schedule: List[dict] = []
    last_job: Optional[int] = None
    last_machine: Optional[int] = None

    while len(schedule) < total_ops:
        available = [job for job in jobs if job.remaining_ops() > 0]
        if not available:
            break

        job, option = select_next(
            available, machine_times, amrs, last_job, last_machine
        )

        if option.machine not in machine_times:
            machine_times[int(option.machine)] = 0.0

        plan = route_plan_for_option(job, option, machine_times, amrs)
        rec = {
            "job": int(job.job_id),
            "op_index": int(job.next_op),
            "machine": int(option.machine),
            "amr": int(plan.amr_id),
            "material": job.material,
            "processing": float(option.processing),
            "start": float(plan.start),
            "end": float(plan.end),
            "rule": rule_tag,
            "pickup_node": int(plan.pickup_node),
            "delivery_node": int(plan.delivery_node),
            "amr_prev_node": int(plan.amr_prev_node),
            "amr_ready": float(plan.amr_ready),
            "depart": float(plan.depart),
            "arrival": float(plan.arrival),
            "to_pick_travel": float(plan.to_pick_travel),
            "transport": float(plan.transport),
            "travel_time": float(plan.travel_time),
            "machine_wait": float(plan.machine_wait),
            "route_nodes": plan.route_nodes,
            "route_legs": plan.route_legs,
        }
        schedule.append(rec)

        job.ready_time = float(plan.end)
        job.current_node = int(plan.delivery_node)
        job.next_op += 1
        machine_times[int(option.machine)] = float(plan.end)
        amrs[int(plan.amr_id)].ready_time = float(plan.end)
        amrs[int(plan.amr_id)].current_node = int(plan.delivery_node)
        last_job = int(job.job_id)
        last_machine = int(option.machine)

    makespan = max((float(job.ready_time) for job in jobs), default=0.0)
    return schedule, makespan


def write_schedule(
    out_path: str,
    schedule: List[dict],
    *,
    instance_index: int,
    makespan: float,
) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in schedule:
            rec_out = dict(rec)
            rec_out["instance_index"] = int(instance_index)
            rec_out["makespan"] = float(makespan)
            f.write(json.dumps(rec_out) + "\n")


def build_parser(rule_name: str, default_out: str, default_fig_out: str) -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parent
    default_data = str(base_dir.parent / "fjssp_training_dataset.jsonl")

    parser = argparse.ArgumentParser(
        description=f"Dispatch a FJSP instance using {rule_name}."
    )
    parser.add_argument(
        "--data",
        default=default_data,
        help="Path to fjssp_training_dataset.jsonl",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="0-based instance index in the JSONL file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for tie-breaking",
    )
    parser.add_argument(
        "--out",
        default=str(base_dir / default_out),
        help="Output JSONL schedule path",
    )
    parser.add_argument(
        "--fig-out",
        default=str(base_dir / default_fig_out),
        help="Output PNG visualization path",
    )
    parser.add_argument(
        "--no-fig",
        action="store_true",
        help="Skip PNG visualization output",
    )
    return parser


def run_cli(
    *,
    rule_name: str,
    rule_tag: str,
    default_out: str,
    default_fig_out: str,
    select_next: SelectFunc,
) -> None:
    parser = build_parser(rule_name, default_out, default_fig_out)
    args = parser.parse_args()

    instance = load_instance(args.data, args.index)
    schedule, makespan = _dispatch_instance(
        instance,
        seed=args.seed,
        rule_tag=rule_tag,
        select_next=select_next,
    )
    write_schedule(
        args.out,
        schedule,
        instance_index=args.index,
        makespan=makespan,
    )

    fig_msg = ""
    if not args.no_fig:
        visualization.save_dpr_schedule_image(
            schedule,
            args.fig_out,
            rule_name=rule_name,
            instance_index=args.index,
            makespan=makespan,
        )
        fig_msg = f"; figure={args.fig_out}"

    print(
        f"{rule_name} scheduled {len(schedule)} operations; makespan={makespan:.2f}; "
        f"output={args.out}{fig_msg}"
    )
RULE_NAME = "DPR-6"
RULE_TAG = "dpr-6"
DEFAULT_OUT = "dpr-6_schedule.jsonl"
DEFAULT_FIG_OUT = "dpr-6_schedule.png"

def _select_rule(jobs, machine_times, amrs, last_job, last_machine):
    best = None
    candidates = []
    for job in jobs:
        option = choose_machine_by_processing(job.next_options(), prefer_min=False)
        score = float(option.processing)
        if best is None or score > best + _EPS:
            best = score
            candidates = [(job, option)]
        elif abs(score - best) <= _EPS:
            candidates.append((job, option))
    return random.choice(candidates)


def dispatch_instance(instance, *, seed=None):
    return _dispatch_instance(
        instance,
        seed=seed,
        rule_tag=RULE_TAG,
        select_next=_select_rule,
    )


def main() -> None:
    run_cli(
        rule_name=RULE_NAME,
        rule_tag=RULE_TAG,
        default_out=DEFAULT_OUT,
        default_fig_out=DEFAULT_FIG_OUT,
        select_next=_select_rule,
    )


if __name__ == "__main__":
    main()
