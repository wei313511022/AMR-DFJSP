from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from GA.GA import (
    AMR_KEYS,
    AMR_LOAD_CAPACITY,
    AMR_STARTS,
    rack_can_load,
    SIZE_ORDER,
    SUFFIX_CAP,
    TOTAL_SLOTS_PER_AMR,
    INBOUND_DOCK_LOCATIONS,
    PICKUP,
    STATIONS,
    TYPE_DURATION,
    UNLOAD,
    Job,
    Operation,
    dock_key_from_value,
    empty_count_inventory,
    heuristic,
    job_pickup_location,
    normalize_count_inventory,
)

OPERATION_KINDS = (PICKUP, UNLOAD)
PICKUP_OP = 0
UNLOAD_OP = 1
NUM_OPERATION_TYPES = 2
DOCK_DELAY_SCALE = 100.0
DOCK_QUEUE_SCALE = float(max(len(AMR_KEYS), 1))
LOWER_BOUND_SCALE = 500.0

# --- feature normalisers -----------------------------------------------------
# Shared by every encoder (GNN, GNN_precise, Attention, Attention_precise,
# extend_GNN) so the five of them cannot drift apart again.
#
# All of these used to be the literals 10.0 and 3.0, inherited from the v1
# 10x10 / 3-slot setting. On the v3 20x19 floor a hard-coded 10.0 let position
# features run to 1.9 while every other channel sat in [0, 1], making position
# the loudest input to the encoder for no principled reason; and dividing a
# rack count by 3.0 when the class cap is 1 meant a FULL slot read as 0.33.
#
# Resolved at call time against the live GA globals, because apply_layout
# rebinds them for the contention sweep.


def position_scale() -> Tuple[float, float]:
    """(x, y) divisors that map any floor cell into [0, 1]."""
    import GA.GA as _GA

    return (float(max(_GA.GRID_MAX_X, 1)), float(max(_GA.GRID_MAX_Y, 1)))


def rack_scale(size_class: str) -> float:
    """Divisor for an onboard count of `size_class`: its binding suffix cap."""
    import GA.GA as _GA

    return float(max(_GA.SUFFIX_CAP.get(size_class, 1), 1))


def fleet_scale() -> float:
    """Divisor for per-AMR job counts: the current fleet size."""
    import GA.GA as _GA

    return float(max(len(_GA.AMR_KEYS), 1))

DOCK_KEYS = list(INBOUND_DOCK_LOCATIONS.keys()) + list(STATIONS.keys())
DockEvent = Tuple[float, float, int, str]
DockServiceEvents = Dict[str, List[DockEvent]]

# Calibration corrects the fast rollout's optimistic estimates toward the
# collision-aware decoder: travel is an affine map of Manhattan distance and
# dock waits gain a penalty indexed by how many committed services are still
# unfinished when the AMR becomes ready. Identity values keep the legacy
# behavior when no artifact has been fitted yet.
CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration" / "fast_model_calibration.json"


def load_calibration(path: Optional[Path] = CALIBRATION_PATH) -> dict:
    calibration = {
        "travel": {"scale": 1.0, "offset": 0.0},
        "dock_wait_penalty": [0.0, 0.0, 0.0, 0.0],
    }
    if path is not None and Path(path).exists():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        travel = raw.get("travel", {})
        calibration["travel"]["scale"] = float(travel.get("scale", 1.0))
        calibration["travel"]["offset"] = float(travel.get("offset", 0.0))
        for i, value in enumerate(raw.get("dock_wait_penalty", [])[:4]):
            calibration["dock_wait_penalty"][i] = float(value)
    return calibration


CALIBRATION = load_calibration()


def set_calibration(calibration: Optional[dict]) -> None:
    """Override the active calibration; None resets to identity."""
    global CALIBRATION
    CALIBRATION = calibration if calibration is not None else load_calibration(None)


def estimated_travel_time(origin: Tuple[int, int], destination: Tuple[int, int]) -> float:
    base = heuristic(origin, destination)
    if base <= 0.0:
        return 0.0
    travel = CALIBRATION["travel"]
    # Real grid travel can never beat Manhattan distance.
    return max(float(base), travel["scale"] * base + travel["offset"])


def empty_dock_service_events() -> DockServiceEvents:
    return {dock: [] for dock in DOCK_KEYS}


def dock_queue_depth(
    dock_service_events: Optional[DockServiceEvents],
    dock_key: str,
    ready_time: float,
) -> int:
    if not dock_service_events:
        return 0
    return sum(1 for event in dock_service_events.get(dock_key, []) if event[1] > ready_time)


def estimated_dock_wait(
    dock_key: str,
    ready_time: float,
    station_availabilities: Dict[str, float],
    dock_service_events: Optional[DockServiceEvents] = None,
) -> float:
    base_wait = max(0.0, station_availabilities.get(dock_key, 0.0) - ready_time)
    penalties = CALIBRATION["dock_wait_penalty"]
    depth = dock_queue_depth(dock_service_events, dock_key, ready_time)
    return base_wait + penalties[min(depth, len(penalties) - 1)]


@dataclass(frozen=True)
class OperationAction:
    kind: str
    job_list_idx: int
    job_id: int
    amr: str
    amr_idx: int
    action_id: int


@dataclass(frozen=True)
class OperationEstimate:
    action: OperationAction
    start_time: float
    end_time: float
    projected_completion: float
    travel_time: float
    material_match: int
    assigned_count: int
    station_workload: float


def action_id(kind: str, amr_idx: int, job_list_idx: int, num_jobs: int) -> int:
    op_idx = PICKUP_OP if kind == PICKUP else UNLOAD_OP
    return op_idx * (len(AMR_KEYS) * num_jobs) + amr_idx * num_jobs + job_list_idx


def decode_action_id(raw_action_id: int, jobs: Sequence[Job]) -> OperationAction:
    num_jobs = len(jobs)
    per_kind = len(AMR_KEYS) * num_jobs
    op_idx = raw_action_id // per_kind
    remainder = raw_action_id % per_kind
    amr_idx = remainder // num_jobs
    job_list_idx = remainder % num_jobs
    kind = PICKUP if op_idx == PICKUP_OP else UNLOAD
    job = jobs[job_list_idx]
    return OperationAction(
        kind=kind,
        job_list_idx=job_list_idx,
        job_id=job.idx,
        amr=AMR_KEYS[amr_idx],
        amr_idx=amr_idx,
        action_id=raw_action_id,
    )


def initial_operation_state(init_state: Optional[dict] = None, precise: bool = False):
    if init_state:
        amr_positions = {amr: init_state["positions"].get(amr, AMR_STARTS[amr]) for amr in AMR_KEYS}
        amr_availabilities = {amr: float(init_state["availability"].get(amr, 0.0)) for amr in AMR_KEYS}
        station_availabilities = {s: float(init_state["time"]) for s in STATIONS.keys()}
        station_availabilities.update({dock: float(init_state["time"]) for dock in INBOUND_DOCK_LOCATIONS.keys()})
        amr_inventory = normalize_count_inventory(init_state.get("inventory", {}))
        current_time = float(init_state.get("time", 0.0))
    else:
        amr_positions = {amr: AMR_STARTS[amr] for amr in AMR_KEYS}
        amr_availabilities = {amr: 0.0 for amr in AMR_KEYS}
        station_availabilities = {s: 0.0 for s in STATIONS.keys()}
        station_availabilities.update({dock: 0.0 for dock in INBOUND_DOCK_LOCATIONS.keys()})
        amr_inventory = empty_count_inventory()
        current_time = 0.0

    if precise:
        amr_states = {amr: (amr_positions[amr], amr_availabilities[amr]) for amr in AMR_KEYS}
        return amr_positions, amr_availabilities, station_availabilities, amr_inventory, amr_states, {}
    return amr_positions, amr_availabilities, station_availabilities, amr_inventory


def carrier_for_job(carrier_map: Dict[int, str], job_id: int) -> Optional[str]:
    return carrier_map.get(job_id)


def job_status_value(job_id: int, picked_jobs_set: set, completed_jobs_set: set) -> float:
    if job_id in completed_jobs_set:
        return 1.0
    if job_id in picked_jobs_set:
        return 0.5
    return 0.0


def carrier_feature(job_id: int, carrier_map: Dict[int, str]) -> float:
    carrier = carrier_map.get(job_id)
    if carrier not in AMR_KEYS:
        return 0.0
    return float(AMR_KEYS.index(carrier) + 1) / float(len(AMR_KEYS))


def station_remaining_workload(jobs: Sequence[Job], completed_jobs_set: set) -> Dict[str, float]:
    workload = {station: 0.0 for station in STATIONS.keys()}
    for job in jobs:
        if job.idx not in completed_jobs_set:
            workload[job.station] += job.duration
    return workload


def legal_actions(
    jobs: Sequence[Job],
    picked_jobs_set: set,
    completed_jobs_set: set,
    carrier_map: Dict[int, str],
    amr_inventory: Dict[str, Dict[str, int]],
) -> List[OperationAction]:
    actions: List[OperationAction] = []
    num_jobs = len(jobs)
    for job_list_idx, job in enumerate(jobs):
        if job.idx in completed_jobs_set:
            continue
        if job.idx in picked_jobs_set:
            carrier = carrier_map.get(job.idx)
            if carrier in AMR_KEYS:
                amr_idx = AMR_KEYS.index(carrier)
                actions.append(
                    OperationAction(
                        kind=UNLOAD,
                        job_list_idx=job_list_idx,
                        job_id=job.idx,
                        amr=carrier,
                        amr_idx=amr_idx,
                        action_id=action_id(UNLOAD, amr_idx, job_list_idx, num_jobs),
                    )
                )
            continue

        for amr_idx, amr in enumerate(AMR_KEYS):
            # Nested slot rack: a class-s parcel needs a free slot of class s or
            # larger, so the binding test is over suffix sums, not a per-class cap.
            if not rack_can_load(amr_inventory[amr], job.type_):
                continue
            actions.append(
                OperationAction(
                    kind=PICKUP,
                    job_list_idx=job_list_idx,
                    job_id=job.idx,
                    amr=amr,
                    amr_idx=amr_idx,
                    action_id=action_id(PICKUP, amr_idx, job_list_idx, num_jobs),
                )
            )
    return actions


def action_mask(
    jobs: Sequence[Job],
    picked_jobs_set: set,
    completed_jobs_set: set,
    carrier_map: Dict[int, str],
    amr_inventory: Dict[str, Dict[str, int]],
) -> List[bool]:
    mask = [True] * (NUM_OPERATION_TYPES * len(AMR_KEYS) * len(jobs))
    for action in legal_actions(jobs, picked_jobs_set, completed_jobs_set, carrier_map, amr_inventory):
        mask[action.action_id] = False
    return mask


def completed_job_mask(jobs: Sequence[Job], completed_jobs_set: set) -> List[bool]:
    return [job.idx in completed_jobs_set for job in jobs]


def decision_time(amr_availabilities: Dict[str, float]) -> float:
    return min(amr_availabilities.values()) if amr_availabilities else 0.0


def completion_time_estimate(
    job: Job,
    amr_positions: Dict[str, Tuple[int, int]],
    amr_availabilities: Dict[str, float],
    station_availabilities: Dict[str, float],
) -> float:
    """Best-case completion over all AMRs, using calibrated travel times.

    With an identity calibration this is a true lower bound; once calibrated
    it becomes an estimate aligned with the collision-aware decoder.
    """
    if not amr_positions:
        return 0.0

    pickup_location = job_pickup_location(job)
    target_station = STATIONS[job.station]
    inbound_dock = dock_key_from_value(job.inbound_dock)
    estimates = []

    for amr in AMR_KEYS:
        if amr not in amr_positions or amr not in amr_availabilities:
            continue
        pickup_travel = estimated_travel_time(amr_positions[amr], pickup_location)
        pickup_start = max(
            amr_availabilities[amr] + pickup_travel,
            float(job.arrival_time),
            station_availabilities.get(inbound_dock, 0.0),
        )
        pickup_end = pickup_start + job.duration
        outbound_travel_end = pickup_end + estimated_travel_time(pickup_location, target_station)
        completion = max(
            outbound_travel_end,
            station_availabilities.get(job.station, 0.0),
        ) + job.duration
        estimates.append(completion)

    return min(estimates) if estimates else 0.0


# Backward-compatible alias (the quantity is no longer a strict bound once calibrated).
completion_time_lower_bound = completion_time_estimate


def dock_congestion_features(
    job: Job,
    jobs: Sequence[Job],
    picked_jobs_set: set,
    completed_jobs_set: set,
    station_availabilities: Dict[str, float],
    current_time: float,
) -> Tuple[float, float, float, float]:
    inbound_dock = dock_key_from_value(job.inbound_dock)
    inbound_delay = max(
        0.0,
        station_availabilities.get(inbound_dock, 0.0) - current_time,
    ) / DOCK_DELAY_SCALE
    outbound_delay = max(
        0.0,
        station_availabilities.get(job.station, 0.0) - current_time,
    ) / DOCK_DELAY_SCALE

    inbound_committed = sum(
        1
        for other in jobs
        if other.idx != job.idx
        and other.idx not in picked_jobs_set
        and other.idx not in completed_jobs_set
        and dock_key_from_value(other.inbound_dock) == inbound_dock
    )
    outbound_committed = sum(
        1
        for other in jobs
        if other.idx in picked_jobs_set
        and other.idx not in completed_jobs_set
        and other.station == job.station
    )

    return (
        inbound_delay,
        outbound_delay,
        inbound_committed / DOCK_QUEUE_SCALE,
        outbound_committed / DOCK_QUEUE_SCALE,
    )


def estimate_action(
    action: OperationAction,
    jobs: Sequence[Job],
    amr_positions: Dict[str, Tuple[int, int]],
    amr_availabilities: Dict[str, float],
    station_availabilities: Dict[str, float],
    amr_inventory: Dict[str, Dict[str, int]],
    assigned_count: Dict[str, int],
    station_workload: Dict[str, float],
    dock_service_events: Optional[DockServiceEvents] = None,
) -> OperationEstimate:
    job = jobs[action.job_list_idx]
    start_time = amr_availabilities[action.amr]
    material_match = 0 if amr_inventory[action.amr].get(job.type_, 0) > 0 else 1

    if action.kind == PICKUP:
        pickup_location = job_pickup_location(job)
        to_pickup = estimated_travel_time(amr_positions[action.amr], pickup_location)
        inbound_dock = dock_key_from_value(job.inbound_dock)
        ready_time = max(start_time + to_pickup, float(job.arrival_time))
        pickup_start = ready_time + estimated_dock_wait(
            inbound_dock, ready_time, station_availabilities, dock_service_events
        )
        pickup_end = pickup_start + job.duration
        target_station = STATIONS[job.station]
        to_station = estimated_travel_time(pickup_location, target_station)
        projected_travel_end = pickup_end + to_station
        projected_process_start = projected_travel_end + estimated_dock_wait(
            job.station, projected_travel_end, station_availabilities, dock_service_events
        )
        projected_completion = projected_process_start + job.duration
        return OperationEstimate(
            action=action,
            start_time=start_time,
            end_time=pickup_end,
            projected_completion=projected_completion,
            travel_time=to_pickup,
            material_match=material_match,
            assigned_count=assigned_count[action.amr],
            station_workload=station_workload[job.station],
        )

    target_station = STATIONS[job.station]
    to_station = estimated_travel_time(amr_positions[action.amr], target_station)
    travel_end = start_time + to_station
    process_start = travel_end + estimated_dock_wait(
        job.station, travel_end, station_availabilities, dock_service_events
    )
    process_end = process_start + job.duration
    return OperationEstimate(
        action=action,
        start_time=start_time,
        end_time=process_end,
        projected_completion=process_end,
        travel_time=to_station,
        material_match=0,
        assigned_count=assigned_count[action.amr],
        station_workload=station_workload[job.station],
    )


def apply_fast_action(
    action: OperationAction,
    jobs: Sequence[Job],
    picked_jobs_set: set,
    completed_jobs_set: set,
    carrier_map: Dict[int, str],
    amr_positions: Dict[str, Tuple[int, int]],
    amr_availabilities: Dict[str, float],
    station_availabilities: Dict[str, float],
    amr_inventory: Dict[str, Dict[str, int]],
    assigned_count: Optional[Dict[str, int]] = None,
    dock_service_events: Optional[DockServiceEvents] = None,
) -> None:
    job = jobs[action.job_list_idx]
    material = job.type_

    if action.kind == PICKUP:
        pickup_location = job_pickup_location(job)
        start_time = amr_availabilities[action.amr]
        inbound_dock = dock_key_from_value(job.inbound_dock)
        ready_time = max(
            start_time + estimated_travel_time(amr_positions[action.amr], pickup_location),
            float(job.arrival_time),
        )
        pickup_start = ready_time + estimated_dock_wait(
            inbound_dock, ready_time, station_availabilities, dock_service_events
        )
        pickup_end = pickup_start + job.duration
        amr_availabilities[action.amr] = pickup_end
        amr_positions[action.amr] = pickup_location
        station_availabilities[inbound_dock] = pickup_end
        amr_inventory[action.amr][material] += 1
        picked_jobs_set.add(job.idx)
        carrier_map[job.idx] = action.amr
        if assigned_count is not None:
            assigned_count[action.amr] += 1
        if dock_service_events is not None:
            dock_service_events.setdefault(inbound_dock, []).append(
                (pickup_start, pickup_end, job.idx, PICKUP)
            )
        return

    target_station = STATIONS[job.station]
    start_time = amr_availabilities[action.amr]
    travel_end = start_time + estimated_travel_time(amr_positions[action.amr], target_station)
    process_start = travel_end + estimated_dock_wait(
        job.station, travel_end, station_availabilities, dock_service_events
    )
    process_end = process_start + job.duration
    amr_availabilities[action.amr] = process_end
    amr_positions[action.amr] = target_station
    station_availabilities[job.station] = process_end
    amr_inventory[action.amr][material] -= 1
    completed_jobs_set.add(job.idx)
    if dock_service_events is not None:
        dock_service_events.setdefault(job.station, []).append(
            (process_start, process_end, job.idx, UNLOAD)
        )


def operation_sequence_from_individual(individual, jobs: Sequence[Job]) -> List[int]:
    from GA.GA import repair_operation_order

    job_id_to_list_idx = {job.idx: idx for idx, job in enumerate(jobs)}
    actions = []
    carrier_map: Dict[int, str] = {}
    for op in repair_operation_order(list(individual.order), list(jobs)):
        job_list_idx = job_id_to_list_idx[op.job_idx]
        if op.kind == PICKUP:
            amr = individual.amr_assignment[op.job_idx]
            carrier_map[op.job_idx] = amr
        else:
            amr = carrier_map.get(op.job_idx, individual.amr_assignment[op.job_idx])
        actions.append(action_id(op.kind, AMR_KEYS.index(amr), job_list_idx, len(jobs)))
    return actions


def load_balance_step_advantages_from_actions(action_seq: Iterable[int], jobs: Sequence[Job]) -> List[float]:
    counts = [0 for _ in AMR_KEYS]
    advantages = []
    num_jobs = len(jobs)
    per_kind = len(AMR_KEYS) * num_jobs
    for raw_action_id in action_seq:
        op_idx = raw_action_id // per_kind
        remainder = raw_action_id % per_kind
        amr_idx = remainder // num_jobs
        if op_idx == PICKUP_OP:
            advantages.append(float(min(counts) - counts[amr_idx]))
            counts[amr_idx] += 1
        else:
            advantages.append(0.0)
    return advantages


def load_required_operation_checkpoint(model, checkpoint: Path, torch_module, required_keys: Sequence[str]) -> str:
    if not checkpoint or not checkpoint.exists():
        raise RuntimeError("no operation-policy checkpoint found; retrain this model before inference")

    loaded = torch_module.load(checkpoint, map_location="cpu")
    state_dict = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and getattr(value, "shape", None) == model_state[key].shape
    }
    missing_required = [key for key in required_keys if key not in compatible]
    if missing_required:
        raise RuntimeError(
            "checkpoint is not an operation-policy checkpoint; retrain after the pickup/unload conversion. "
            f"Missing/incompatible tensors: {', '.join(missing_required)}"
        )
    model.load_state_dict(compatible, strict=False)
    return f"loaded {len(compatible)}/{len(model_state)} tensors from {checkpoint}"
