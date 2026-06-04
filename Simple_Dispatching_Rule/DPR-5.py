#!/usr/bin/env python3
"""DPR-5 for route-aware FJSP: continue same job and machine when possible."""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import map as map_config
import visualization
from calcu_dist import make_calculate_distance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "../fjssp_training_dataset.jsonl"
INSTANCE_INDEX = 0
OUT_PATH = "../results/dpr-5_result.jsonl"
FIG_OUT_PATH = "../results/dpr-5_result.html"

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
    material: str = "A"
    next_op: int = 0
    ready_time: float = 0.0
    current_node: int = 10
    is_exited: bool = False

    def remaining_ops(self) -> int:
        return len(self.operations) - self.next_op

    def next_options(self) -> List[MachineOption]:
        return self.operations[self.next_op]


@dataclass
class AmrState:
    amr_id: int
    ready_time: float
    current_node: int
    inventory: Dict[str, int] = None

    def __post_init__(self):
        if self.inventory is None:
            self.inventory = {"A": 0, "B": 0, "C": 0}


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
    replenished_material: bool = False

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
                                current_node=10
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
    delivery_node = machine_to_node(int(option.machine))
    machine_ready = float(machine_times.get(int(option.machine), 0.0))

    best: Optional[RoutePlan] = None
    tied: List[RoutePlan] = []
    for amr in amrs.values():
        replenished = False
        if job.next_op == 0:
            mat = job.material
            if amr.inventory[mat] > 0:
                pickup_node = int(amr.current_node)
            else:
                pickup_node = int(map_config.TYPE_TO_MATERIAL_NODE[mat])
                replenished = True
        else:
            pickup_node = int(job.current_node)

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
            replenished_material=replenished,
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
    select_next: SelectFunc,
) -> Tuple[List[dict], float]:
    if seed is not None:
        random.seed(int(seed))

    jobs = parse_jobs(instance)
    machine_times = init_machine_times(instance, jobs)
    amrs = init_amrs()

    total_ops = sum(len(job.operations) for job in jobs) + len(jobs)
    schedule: List[dict] = []
    last_job: Optional[int] = None
    last_machine: Optional[int] = None

    while len(schedule) < total_ops:
        jobs_to_exit = [job for job in jobs if job.remaining_ops() == 0 and not getattr(job, "is_exited", False)]
        if jobs_to_exit:
            job = jobs_to_exit[0]
            option = MachineOption(machine=-1, processing=0.0)
            pickup_node = int(job.current_node)
            delivery_node = getattr(map_config, "EXIT_NODE", 138)
            best_plan = None
            for amr in amrs.values():
                to_pick = float(calculate_distance(int(amr.current_node), pickup_node))
                transport = float(calculate_distance(pickup_node, delivery_node))
                depart = max(float(amr.ready_time), float(job.ready_time) - to_pick)
                arrival = depart + to_pick + transport
                plan = RoutePlan(option=option, amr_id=int(amr.amr_id), amr_ready=float(amr.ready_time), amr_prev_node=int(amr.current_node), pickup_node=pickup_node, delivery_node=delivery_node, depart=float(depart), arrival=float(arrival), start=float(arrival), end=float(arrival), to_pick_travel=float(to_pick), transport=float(transport), machine_wait=0.0, replenished_material=False)
                if best_plan is None or plan.end < best_plan.end - _EPS:
                    best_plan = plan
            plan = best_plan
            rec = {"job": int(job.job_id), "op_index": int(job.next_op), "machine": -1, "amr": int(plan.amr_id), "start": float(plan.start), "end": float(plan.end)}
            schedule.append(rec)
            job.ready_time = float(plan.end)
            job.current_node = int(plan.delivery_node)
            job.is_exited = True
            amrs[int(plan.amr_id)].ready_time = float(plan.end)
            amrs[int(plan.amr_id)].current_node = int(plan.delivery_node)
            last_job = int(job.job_id)
            last_machine = None
            continue

        available = [job for job in jobs if job.remaining_ops() > 0]
        if not available:
            break

        job, option = select_next(available, machine_times, amrs, last_job, last_machine)

        if option.machine not in machine_times:
            machine_times[int(option.machine)] = 0.0

        plan = route_plan_for_option(job, option, machine_times, amrs)
        amr_obj = amrs[int(plan.amr_id)]
        if job.next_op == 0:
            mat = job.material
            if plan.replenished_material:
                amr_obj.inventory[mat] += map_config.MATERIAL_PICK_QTY
            amr_obj.inventory[mat] -= 1
        rec = {
            "job": int(job.job_id),
            "op_index": int(job.next_op),
            "machine": int(option.machine),
            "amr": int(plan.amr_id),
            "start": float(plan.start),
            "end": float(plan.end),
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
            f.write(json.dumps(rec_out) + "\n")


def _select_rule(jobs, machine_times, amrs, last_job, last_machine):
    if last_job is not None:
        for job in jobs:
            if int(job.job_id) != int(last_job):
                continue
            if last_machine is not None:
                option = find_option_for_machine(job.next_options(), last_machine)
                if option is not None:
                    return job, option
            option = choose_machine_earliest_start(
                job, job.next_options(), machine_times, amrs
            )
            return job, option

    job = random.choice(jobs)
    option = choose_machine_earliest_start(
        job, job.next_options(), machine_times, amrs
    )
    return job, option


def dispatch_instance(instance, *, seed=None):
    return _dispatch_instance(
        instance,
        seed=67,
        select_next=_select_rule,
    )


def main() -> None:
    instance = load_instance(DATA_PATH, INSTANCE_INDEX)
    schedule, makespan = _dispatch_instance(
        instance,
        select_next=_select_rule,
    )
    write_schedule(
        OUT_PATH,
        schedule,
        instance_index=INSTANCE_INDEX,
        makespan=makespan,
    )

    fig_msg = ""

    visualization.save_dpr_schedule_image(
        schedule,
        FIG_OUT_PATH,
        instance_index=INSTANCE_INDEX,
        makespan=makespan,
    )
    fig_msg = f"; figure={FIG_OUT_PATH}"

    print(
        f"DPR-5; makespan={makespan:.2f}; "
        f"output={OUT_PATH}{fig_msg}"
    )

if __name__ == "__main__":
    main()
