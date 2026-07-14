from __future__ import annotations

import csv
import os
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from GA.GA import (  # noqa: E402
    AMR_KEYS,
    AMR_LOAD_CAPACITY,
    AMR_STARTS,
    DISPATCH_EVENT_INDEX_ENV,
    INBOUND_DOCK_LOCATIONS,
    JOB_COUNT,
    PICKUP,
    STATIONS,
    TYPE_DURATION,
    UNLOAD,
    Individual,
    Job,
    Operation,
    _adjacent_points,
    decode_schedule_tick_by_tick,
    dock_key_from_value,
    dock_waiting_slots,
    empty_count_inventory,
    grid_distance,
    heuristic,
    load_dispatch_events,
    make_jobs,
    normalize_count_inventory,
    plot_gantt,
    routing_iters,
    collision_routing_iters,
)
from neural_local_improvement import apply_neural_local_improvement  # noqa: E402
from operation_policy import (  # noqa: E402
    LOWER_BOUND_SCALE,
    action_mask,
    apply_fast_action,
    carrier_feature,
    completion_time_lower_bound,
    decode_action_id,
    decision_time,
    empty_dock_service_events,
    estimated_dock_wait,
    estimated_travel_time,
    initial_operation_state,
    job_status_value,
    load_required_operation_checkpoint,
)


AMR_IN_DIM = 12
JOB_IN_DIM = 11
DOCK_IN_DIM = 9
ACTION_IN_DIM = 3
DISTANCE_SCALE = 20.0
TIME_SCALE = 100.0
WORKLOAD_SCALE = 500.0

INBOUND_DOCK_KEYS = list(INBOUND_DOCK_LOCATIONS.keys())
OUTBOUND_DOCK_KEYS = list(STATIONS.keys())
DOCK_KEYS = INBOUND_DOCK_KEYS + OUTBOUND_DOCK_KEYS
DOCK_INDEX = {dock: idx for idx, dock in enumerate(DOCK_KEYS)}
DockEvent = Tuple[float, float, int, str]
DockServiceEvents = Dict[str, List[DockEvent]]


def _dock_position(dock_key: str) -> Tuple[int, int]:
    if dock_key in INBOUND_DOCK_LOCATIONS:
        return INBOUND_DOCK_LOCATIONS[dock_key]
    return STATIONS[dock_key]


def _job_inbound_dock(job: Job) -> str:
    return dock_key_from_value(job.inbound_dock)


def _job_outbound_dock(job: Job) -> str:
    return job.station


def _dock_norm(dock_key: str, keys: Sequence[str]) -> float:
    return float(keys.index(dock_key) + 1) / float(max(len(keys), 1))


def _empty_dock_service_events() -> DockServiceEvents:
    # Shared with operation_policy so apply_fast_action records into the same structure.
    return empty_dock_service_events()


def normalize_dock_service_events(dock_service_events: Optional[DockServiceEvents]) -> DockServiceEvents:
    normalized = _empty_dock_service_events()
    if dock_service_events:
        for dock, events in dock_service_events.items():
            normalized.setdefault(dock, [])
            normalized[dock].extend(events)
    return normalized


@lru_cache(maxsize=1)
def _near_dock_cells() -> frozenset[Tuple[int, int]]:
    cells = set()
    for dock in DOCK_KEYS:
        dock_pos = _dock_position(dock)
        related = {dock_pos, *dock_waiting_slots(dock_pos)}
        for point in related:
            cells.add(point)
            cells.update(_adjacent_points(point))
    return frozenset(cells)


def compute_action_service_event(
    action,
    jobs: Sequence[Job],
    amr_positions: Dict[str, Tuple[int, int]],
    amr_availabilities: Dict[str, float],
    station_availabilities: Dict[str, float],
    dock_service_events: Optional[DockServiceEvents] = None,
) -> Tuple[str, DockEvent]:
    job = jobs[action.job_list_idx]
    start_time = amr_availabilities[action.amr]

    if action.kind == PICKUP:
        dock_key = _job_inbound_dock(job)
        dock_pos = INBOUND_DOCK_LOCATIONS[dock_key]
        ready_time = max(
            start_time + estimated_travel_time(amr_positions[action.amr], dock_pos),
            float(job.arrival_time),
        )
        service_start = ready_time + estimated_dock_wait(
            dock_key, ready_time, station_availabilities, dock_service_events
        )
        service_end = service_start + job.duration
        return dock_key, (service_start, service_end, job.idx, action.kind)

    dock_key = _job_outbound_dock(job)
    dock_pos = STATIONS[dock_key]
    ready_time = start_time + estimated_travel_time(amr_positions[action.amr], dock_pos)
    service_start = ready_time + estimated_dock_wait(
        dock_key, ready_time, station_availabilities, dock_service_events
    )
    service_end = service_start + job.duration
    return dock_key, (service_start, service_end, job.idx, action.kind)


def record_dock_service_event(
    dock_service_events: DockServiceEvents,
    action,
    jobs: Sequence[Job],
    amr_positions: Dict[str, Tuple[int, int]],
    amr_availabilities: Dict[str, float],
    station_availabilities: Dict[str, float],
) -> DockEvent:
    dock_key, event = compute_action_service_event(
        action,
        jobs,
        amr_positions,
        amr_availabilities,
        station_availabilities,
        dock_service_events,
    )
    dock_service_events.setdefault(dock_key, []).append(event)
    return event


def _valid_waiting_slot_count(dock_key: str) -> int:
    return max(1, len(dock_waiting_slots(_dock_position(dock_key))))


def _dock_feature_row(
    dock_key: str,
    station_availabilities: Dict[str, float],
    dock_service_events: DockServiceEvents,
    current_time: float,
) -> List[float]:
    events = dock_service_events.get(dock_key, [])
    unfinished = [event for event in events if event[1] > current_time]
    future = [event for event in unfinished if event[0] > current_time]
    current = [event for event in unfinished if event[0] <= current_time < event[1]]
    queue_count = len(future)
    service_remaining = max((event[1] - current_time for event in current), default=0.0)
    committed_workload = sum(event[1] - max(event[0], current_time) for event in unfinished)
    available_delay = max(0.0, station_availabilities.get(dock_key, 0.0) - current_time)
    dock_pos = _dock_position(dock_key)

    return [
        1.0 if dock_key in INBOUND_DOCK_LOCATIONS else 0.0,
        1.0 if dock_key in STATIONS else 0.0,
        available_delay / TIME_SCALE,
        queue_count / float(max(len(AMR_KEYS), 1)),
        min(queue_count / float(_valid_waiting_slot_count(dock_key)), 1.0),
        service_remaining / TIME_SCALE,
        committed_workload / WORKLOAD_SCALE,
        dock_pos[0] / 10.0,
        dock_pos[1] / 10.0,
    ]


def _action_feature_row(
    kind: str,
    amr: str,
    job: Job,
    amr_positions: Dict[str, Tuple[int, int]],
    amr_availabilities: Dict[str, float],
    station_availabilities: Dict[str, float],
    dock_service_events: Optional[DockServiceEvents] = None,
) -> List[float]:
    start_time = amr_availabilities[amr]

    if kind == PICKUP:
        inbound_dock = _job_inbound_dock(job)
        inbound_pos = INBOUND_DOCK_LOCATIONS[inbound_dock]
        outbound_pos = STATIONS[_job_outbound_dock(job)]
        distance_to_service = estimated_travel_time(amr_positions[amr], inbound_pos)
        amr_arrival = start_time + distance_to_service
        ready_time = max(amr_arrival, float(job.arrival_time))
        dock_wait = estimated_dock_wait(
            inbound_dock, ready_time, station_availabilities, dock_service_events
        )
        downstream = estimated_travel_time(inbound_pos, outbound_pos)
        return [
            distance_to_service / DISTANCE_SCALE,
            downstream / DISTANCE_SCALE,
            dock_wait / TIME_SCALE,
        ]

    outbound_dock = _job_outbound_dock(job)
    outbound_pos = STATIONS[outbound_dock]
    distance_to_service = estimated_travel_time(amr_positions[amr], outbound_pos)
    amr_arrival = start_time + distance_to_service
    dock_wait = estimated_dock_wait(
        outbound_dock, amr_arrival, station_availabilities, dock_service_events
    )
    return [
        distance_to_service / DISTANCE_SCALE,
        0.0,
        dock_wait / TIME_SCALE,
    ]


def _build_job_adjacency(
    jobs: Sequence[Job],
    picked_jobs_set: set,
    completed_jobs_set: set,
    carrier_map: Dict[int, str],
    order_seq: Sequence[Operation],
) -> List[List[float]]:
    num_jobs = len(jobs)
    adj = [[0.0] * num_jobs for _ in range(num_jobs)]
    job_idx_to_pos = {job.idx: pos for pos, job in enumerate(jobs)}

    def add_edge(left: int, right: int) -> None:
        if left == right:
            return
        adj[left][right] = 1.0
        adj[right][left] = 1.0

    for i, left in enumerate(jobs):
        if left.idx in completed_jobs_set:
            continue
        for j in range(i + 1, num_jobs):
            right = jobs[j]
            if right.idx in completed_jobs_set:
                continue
            if _job_inbound_dock(left) == _job_inbound_dock(right):
                add_edge(i, j)
            if _job_outbound_dock(left) == _job_outbound_dock(right):
                add_edge(i, j)

    last_job_per_amr: Dict[str, int] = {}
    last_job_per_station: Dict[str, int] = {}
    job_map = {job.idx: job for job in jobs}
    for scheduled_op in order_seq:
        job_idx = scheduled_op.job_idx if isinstance(scheduled_op, Operation) else int(scheduled_op)
        if job_idx not in job_idx_to_pos:
            continue
        pos = job_idx_to_pos[job_idx]
        amr = carrier_map.get(job_idx)
        job_obj = job_map[job_idx]
        station = _job_outbound_dock(job_obj)
        if amr is not None and amr in last_job_per_amr:
            add_edge(last_job_per_amr[amr], pos)
        if station in last_job_per_station:
            add_edge(last_job_per_station[station], pos)
        if amr is not None:
            last_job_per_amr[amr] = pos
        last_job_per_station[station] = pos

    return adj


def extract_state_extend_gnn(
    jobs: Sequence[Job],
    picked_jobs_set: set,
    completed_jobs_set: set,
    carrier_map: Dict[int, str],
    amr_positions: Dict[str, Tuple[int, int]],
    amr_availabilities: Dict[str, float],
    amr_inventory: Dict[str, Dict[str, int]],
    station_availabilities: Dict[str, float],
    order_seq: Sequence[Operation],
    dock_service_events: Optional[DockServiceEvents] = None,
):
    dock_service_events = normalize_dock_service_events(dock_service_events)
    current_time = decision_time(amr_availabilities)
    material_keys = list(TYPE_DURATION.keys())

    amr_feat = []
    for amr in AMR_KEYS:
        pos = amr_positions[amr]
        avail = amr_availabilities[amr]
        inv = amr_inventory[amr]
        total_remaining = sum(max(0, AMR_LOAD_CAPACITY - inv.get(mat, 0)) for mat in material_keys)
        active_carried = sum(
            1
            for job in jobs
            if job.idx in picked_jobs_set
            and job.idx not in completed_jobs_set
            and carrier_map.get(job.idx) == amr
        )
        local_density = sum(
            1
            for other in AMR_KEYS
            if other != amr and heuristic(pos, amr_positions[other]) <= 2
        ) / float(max(len(AMR_KEYS) - 1, 1))
        nearest_dock_distance = min(heuristic(pos, _dock_position(dock)) for dock in DOCK_KEYS)
        amr_feat.append(
            [
                1.0 if avail > current_time else 0.0,
                max(0.0, avail - current_time) / TIME_SCALE,
                inv.get("A", 0) / AMR_LOAD_CAPACITY,
                inv.get("B", 0) / AMR_LOAD_CAPACITY,
                inv.get("C", 0) / AMR_LOAD_CAPACITY,
                total_remaining / float(max(len(material_keys) * AMR_LOAD_CAPACITY, 1)),
                active_carried / 10.0,
                1.0 if pos in _near_dock_cells() else 0.0,
                nearest_dock_distance / DISTANCE_SCALE,
                local_density,
                pos[0] / 10.0,
                pos[1] / 10.0,
            ]
        )

    dock_feat = [
        _dock_feature_row(dock, station_availabilities, dock_service_events, current_time)
        for dock in DOCK_KEYS
    ]

    job_feat = []
    job_mask = []
    inbound_indices = []
    outbound_indices = []
    for job in jobs:
        inbound_dock = _job_inbound_dock(job)
        outbound_dock = _job_outbound_dock(job)
        is_completed = job.idx in completed_jobs_set
        job_status = job_status_value(job.idx, picked_jobs_set, completed_jobs_set)
        lb_val = 0.0 if is_completed else completion_time_lower_bound(
            job,
            amr_positions,
            amr_availabilities,
            station_availabilities,
        )
        job_feat.append(
            [
                1.0,
                job.duration / 25.0,
                max(0.0, current_time - float(job.arrival_time)) / TIME_SCALE,
                carrier_feature(job.idx, carrier_map),
                job_status,
                1.0 if job.type_ == "A" else 0.0,
                1.0 if job.type_ == "B" else 0.0,
                1.0 if job.type_ == "C" else 0.0,
                _dock_norm(inbound_dock, INBOUND_DOCK_KEYS),
                _dock_norm(outbound_dock, OUTBOUND_DOCK_KEYS),
                lb_val / LOWER_BOUND_SCALE,
            ]
        )
        job_mask.append(is_completed)
        inbound_indices.append(DOCK_INDEX[inbound_dock])
        outbound_indices.append(DOCK_INDEX[outbound_dock])

    action_features = []
    for kind in (PICKUP, UNLOAD):
        op_rows = []
        for amr in AMR_KEYS:
            op_rows.append(
                [
                    _action_feature_row(
                        kind,
                        amr,
                        job,
                        amr_positions,
                        amr_availabilities,
                        station_availabilities,
                        dock_service_events,
                    )
                    for job in jobs
                ]
            )
        action_features.append(op_rows)

    adj = _build_job_adjacency(
        jobs,
        picked_jobs_set,
        completed_jobs_set,
        carrier_map,
        order_seq,
    )

    return (
        torch.tensor([amr_feat], dtype=torch.float32),
        torch.tensor([job_feat], dtype=torch.float32),
        torch.tensor([dock_feat], dtype=torch.float32),
        torch.tensor([action_features], dtype=torch.float32),
        torch.tensor([job_mask], dtype=torch.bool),
        torch.tensor([adj], dtype=torch.float32),
        torch.tensor([inbound_indices], dtype=torch.long),
        torch.tensor([outbound_indices], dtype=torch.long),
    )


class GINConv(nn.Module):
    def __init__(self, hidden_dim: int, eps_init: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps_init))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj):
        # Shared-dock edges can form dense cliques; mean aggregation keeps
        # message scale stable as job count and dock contention grow.
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        agg = torch.bmm(adj, x) / degree
        return self.norm(self.mlp((1.0 + self.eps) * x + agg))


class ExtendSchedulerGNN(nn.Module):
    def __init__(
        self,
        amr_in_dim: int = AMR_IN_DIM,
        job_in_dim: int = JOB_IN_DIM,
        dock_in_dim: int = DOCK_IN_DIM,
        action_in_dim: int = ACTION_IN_DIM,
        hidden_dim: int = 128,
        gin_layers: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_in_dim = action_in_dim

        self.amr_emb = nn.Sequential(
            nn.Linear(amr_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.job_emb = nn.Linear(job_in_dim, hidden_dim)
        self.dock_emb = nn.Sequential(
            nn.Linear(dock_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.job_dock_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # AMR<->dock interaction round (pre-LN residual attention): docks learn
        # which AMRs are loaded/heading their way, AMRs learn about each other
        # and about dock congestion, before jobs fuse the dock embeddings.
        self.dock_from_amr_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.dock_from_amr_norm_q = nn.LayerNorm(hidden_dim)
        self.dock_from_amr_norm_kv = nn.LayerNorm(hidden_dim)
        self.amr_self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.amr_self_norm = nn.LayerNorm(hidden_dim)
        self.amr_from_dock_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.amr_from_dock_norm_q = nn.LayerNorm(hidden_dim)
        self.amr_from_dock_norm_kv = nn.LayerNorm(hidden_dim)
        # Zero-init the residual branches so the interaction block starts as an
        # exact identity; otherwise the randomly-initialized attention outputs
        # (at LayerNorm scale) swamp the much smaller embedding stream and the
        # policy trains against noise it cannot escape.
        for attn in (self.dock_from_amr_attn, self.amr_self_attn, self.amr_from_dock_attn):
            nn.init.zeros_(attn.out_proj.weight)
            nn.init.zeros_(attn.out_proj.bias)
        self.gin_layers = nn.ModuleList([GINConv(hidden_dim) for _ in range(gin_layers)])
        self.op_emb = nn.Embedding(2, hidden_dim)
        self.operation_actor = nn.Sequential(
            nn.Linear(hidden_dim * 4 + action_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        modules = (
            self.amr_emb,
            self.job_emb,
            self.dock_emb,
            self.job_dock_fusion,
            self.dock_from_amr_attn,
            self.dock_from_amr_norm_q,
            self.dock_from_amr_norm_kv,
            self.amr_self_attn,
            self.amr_self_norm,
            self.amr_from_dock_attn,
            self.amr_from_dock_norm_q,
            self.amr_from_dock_norm_kv,
            self.gin_layers,
            self.op_emb,
            self.operation_actor,
        )
        for module in modules:
            yield from module.parameters()

    def encode_amrs(self, amr_features):
        return self.amr_emb(amr_features)

    def encode_docks(self, dock_features):
        return self.dock_emb(dock_features)

    def interact_amrs_docks(self, amr_embeddings, dock_embeddings):
        """One pre-LN residual attention round between AMRs and docks."""
        dock_q = self.dock_from_amr_norm_q(dock_embeddings)
        amr_kv = self.dock_from_amr_norm_kv(amr_embeddings)
        dock_update, _ = self.dock_from_amr_attn(dock_q, amr_kv, amr_kv)
        dock_embeddings = dock_embeddings + dock_update

        amr_q = self.amr_self_norm(amr_embeddings)
        amr_update, _ = self.amr_self_attn(amr_q, amr_q, amr_q)
        amr_embeddings = amr_embeddings + amr_update

        amr_q = self.amr_from_dock_norm_q(amr_embeddings)
        dock_kv = self.amr_from_dock_norm_kv(dock_embeddings)
        amr_update, _ = self.amr_from_dock_attn(amr_q, dock_kv, dock_kv)
        amr_embeddings = amr_embeddings + amr_update

        return amr_embeddings, dock_embeddings

    @staticmethod
    def _gather_docks(dock_embeddings, dock_indices):
        hidden_dim = dock_embeddings.size(-1)
        gather_index = dock_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        return torch.gather(dock_embeddings, 1, gather_index)

    def encode_jobs(self, job_features, dock_embeddings, adj, inbound_dock_idx_by_job, outbound_dock_idx_by_job):
        job_base = self.job_emb(job_features)
        inbound_docks = self._gather_docks(dock_embeddings, inbound_dock_idx_by_job)
        outbound_docks = self._gather_docks(dock_embeddings, outbound_dock_idx_by_job)
        x = self.job_dock_fusion(torch.cat([job_base, inbound_docks, outbound_docks], dim=-1))
        for gin in self.gin_layers:
            x = x + gin(x, adj)
        return x

    def forward_operation_actor(
        self,
        job_embeddings,
        amr_embeddings,
        dock_embeddings,
        action_features,
        inbound_dock_idx_by_job,
        outbound_dock_idx_by_job,
        operation_mask=None,
    ):
        batch_size, num_jobs, hidden_dim = job_embeddings.shape
        num_amrs = amr_embeddings.size(1)

        inbound_docks = self._gather_docks(dock_embeddings, inbound_dock_idx_by_job)
        outbound_docks = self._gather_docks(dock_embeddings, outbound_dock_idx_by_job)
        dock_by_op = torch.stack([inbound_docks, outbound_docks], dim=1)
        dock_expand = dock_by_op.unsqueeze(2).expand(-1, -1, num_amrs, -1, -1)

        op_ids = torch.arange(2, device=job_embeddings.device)
        op_expand = self.op_emb(op_ids).view(1, 2, 1, 1, hidden_dim).expand(batch_size, -1, num_amrs, num_jobs, -1)
        amr_expand = amr_embeddings.unsqueeze(1).unsqueeze(3).expand(-1, 2, -1, num_jobs, -1)
        job_expand = job_embeddings.unsqueeze(1).unsqueeze(2).expand(-1, 2, num_amrs, -1, -1)

        pairs = torch.cat(
            [op_expand, amr_expand, job_expand, dock_expand, action_features],
            dim=-1,
        )
        logits = self.operation_actor(pairs).squeeze(-1)
        if operation_mask is not None:
            logits = logits.masked_fill(operation_mask.view_as(logits), float("-inf"))
        return logits

    def forward_critic(self, job_embeddings, amr_embeddings, dock_embeddings, job_mask):
        mask_float = (~job_mask).float().unsqueeze(-1)
        num_unmasked = mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        job_pool = (job_embeddings * mask_float).sum(dim=1) / num_unmasked.squeeze(-1)
        amr_pool = amr_embeddings.mean(dim=1)
        dock_pool = dock_embeddings.mean(dim=1)
        return self.critic(torch.cat([job_pool, amr_pool, dock_pool], dim=-1)).squeeze(-1)


def _encode_state_tensors(model: ExtendSchedulerGNN, tensors, device):
    (
        amr_feat,
        job_feat,
        dock_feat,
        action_feat,
        job_mask,
        adj,
        inbound_idx,
        outbound_idx,
    ) = tensors
    amr_feat = amr_feat.to(device)
    job_feat = job_feat.to(device)
    dock_feat = dock_feat.to(device)
    action_feat = action_feat.to(device)
    job_mask = job_mask.to(device)
    adj = adj.to(device)
    inbound_idx = inbound_idx.to(device)
    outbound_idx = outbound_idx.to(device)

    dock_embeddings = model.encode_docks(dock_feat)
    amr_embeddings = model.encode_amrs(amr_feat)
    amr_embeddings, dock_embeddings = model.interact_amrs_docks(amr_embeddings, dock_embeddings)
    job_embeddings = model.encode_jobs(job_feat, dock_embeddings, adj, inbound_idx, outbound_idx)
    return (
        job_embeddings,
        amr_embeddings,
        dock_embeddings,
        action_feat,
        job_mask,
        inbound_idx,
        outbound_idx,
    )


def describe_solution_extend_gnn(
    individual: Individual,
    jobs: List[Job],
    solve_time: float = None,
    show_gantt: bool = False,
    save_img: str = None,
) -> Tuple[float, float]:
    availability, decoded_timeline, queue_infos, path_logs, invalid_count = decode_schedule_tick_by_tick(
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
        plot_gantt(
            decoded_timeline,
            queue_infos,
            jobs,
            solve_time=solve_time,
            invalid_count=invalid_count,
            show_gantt=show_gantt,
            save_img=save_img,
        )
    return makespan, solve_time


def solve_with_extend_gnn(jobs, model, deterministic=True, init_state: dict = None):
    perf_start_time = time.perf_counter()
    amr_positions, amr_availabilities, station_availabilities, amr_inventory = initial_operation_state(init_state)
    picked_jobs_set = set()
    completed_jobs_set = set()
    carrier_map = {}
    order_seq = []
    amr_assignment_map = {}
    dock_service_events = _empty_dock_service_events()

    total_log_prob = 0.0
    if deterministic:
        model.eval()
    else:
        model.train()

    device = next(model.parameters()).device

    for _ in range(2 * len(jobs)):
        tensors = extract_state_extend_gnn(
            jobs,
            picked_jobs_set,
            completed_jobs_set,
            carrier_map,
            amr_positions,
            amr_availabilities,
            amr_inventory,
            station_availabilities,
            order_seq,
            dock_service_events,
        )
        (
            job_embeddings,
            amr_embeddings,
            dock_embeddings,
            action_feat,
            job_mask,
            inbound_idx,
            outbound_idx,
        ) = _encode_state_tensors(model, tensors, device)
        op_mask = torch.tensor(
            [action_mask(jobs, picked_jobs_set, completed_jobs_set, carrier_map, amr_inventory)],
            dtype=torch.bool,
            device=device,
        )
        logits = model.forward_operation_actor(
            job_embeddings,
            amr_embeddings,
            dock_embeddings,
            action_feat,
            inbound_idx,
            outbound_idx,
            op_mask,
        )
        flat_logits = logits.view(-1)

        if deterministic:
            best_action = torch.argmax(flat_logits).item()
        else:
            log_probs = F.log_softmax(flat_logits, dim=0)
            probs = torch.exp(log_probs)
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            best_action = action_tensor.item()
            total_log_prob += log_probs[best_action]

        action = decode_action_id(best_action, jobs)
        order_seq.append(Operation(action.job_id, action.kind))
        if action.kind == PICKUP:
            amr_assignment_map[action.job_id] = action.amr
        apply_fast_action(
            action,
            jobs,
            picked_jobs_set,
            completed_jobs_set,
            carrier_map,
            amr_positions,
            amr_availabilities,
            station_availabilities,
            amr_inventory,
            dock_service_events=dock_service_events,
        )

    final_assignment = [amr_assignment_map[i] for i in range(len(jobs))]
    ind = Individual(order=order_seq, amr_assignment=final_assignment)
    solve_dur = time.perf_counter() - perf_start_time
    return ind, total_log_prob, solve_dur


if __name__ == "__main__":
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--gantt", action="store_true", help="Plot Gantt Chart")
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file")
    parser.add_argument("--save_img", type=str, default="", help="Save the schedule Gantt chart to this file")
    parser.add_argument("--local_iters", type=int, default=routing_iters, help="Number of simplified local-improvement iterations")
    parser.add_argument("--collision_iters", type=int, default=collision_routing_iters, help="Number of collision routing iterations")
    parser.add_argument("--output_csv", type=str, default="extend_gnn_summary_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    gnn_model = ExtendSchedulerGNN()
    weights_path = Path("extend_gnn_scheduler_best.pth")
    print(f"Loading trained weights from {weights_path}...")
    status = load_required_operation_checkpoint(
        gnn_model,
        weights_path,
        torch,
        required_keys=("op_emb.weight", "operation_actor.0.weight"),
    )
    print(f"extend_GNN: {status}")

    dispatch_events = load_dispatch_events(Path(args.inbox)) if args.inbox else load_dispatch_events()
    target_index = os.environ.get(DISPATCH_EVENT_INDEX_ENV)
    results_data = []

    print("=== Using hybrid dock-aware extend_GNN static rollout ===")
    if dispatch_events:
        if target_index is not None:
            dispatch_events = [event for event in dispatch_events if str(event["index"]) == str(target_index)]
        for event in dispatch_events:
            print(f"\n=== Processing Dispatch Event {event['index']} (Jobs: {len(event['jobs'])}) ===")
            best_ind, _, solve_dur = solve_with_extend_gnn(event["jobs"], gnn_model)
            improvement = apply_neural_local_improvement(
                best_ind,
                event["jobs"],
                simplified_iters=args.local_iters,
                collision_iters=args.collision_iters,
            )
            best_ind = improvement.individual
            solve_dur += improvement.postprocess_time
            img_path = f"{args.save_img.split('.')[0]}_{event['index']}.png" if args.save_img else None
            makespan, computation_time = describe_solution_extend_gnn(
                best_ind,
                event["jobs"],
                solve_time=solve_dur,
                show_gantt=args.gantt,
                save_img=img_path,
            )
            results_data.append([event["index"], f"{makespan:.2f}", f"{computation_time:.4f}"])
    else:
        print("No dispatch file found. Generating random jobs...")
        jobs = make_jobs()
        best_ind, _, solve_dur = solve_with_extend_gnn(jobs, gnn_model)
        improvement = apply_neural_local_improvement(
            best_ind,
            jobs,
            simplified_iters=args.local_iters,
            collision_iters=args.collision_iters,
        )
        best_ind = improvement.individual
        solve_dur += improvement.postprocess_time
        makespan, computation_time = describe_solution_extend_gnn(
            best_ind,
            jobs,
            solve_time=solve_dur,
            show_gantt=args.gantt,
            save_img=args.save_img,
        )
        results_data.append(["random", f"{makespan:.2f}", f"{computation_time:.4f}"])

    if results_data:
        with open(args.output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Event_Index", "Makespan", "Computation_Time"])
            writer.writerows(results_data)
        print(f"\nSummary results saved to {args.output_csv}")
