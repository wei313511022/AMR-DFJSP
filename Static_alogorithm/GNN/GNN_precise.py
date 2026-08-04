import csv
import json
import os
import random
import time
from collections import deque
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import all constants and simulation logic from GA
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import GA.GA as _GA          # live module handle: apply_layout REBINDS scalars
                            # like GRID_MAX_X, so a from-import of those names
                            # would freeze the v3 defaults and silently ignore
                            # any fleet/layout sweep.
from GA.GA import (
    Job, Individual, Operation, AMR_STARTS, AMR_KEYS, STATIONS, OBSTACLES,
    GRID_MIN_X, GRID_MAX_X, GRID_MIN_Y, GRID_MAX_Y, BASES, TYPE_DURATION,
    SCHEDULE_OUTBOX, DISPATCH_INBOX, DISPATCH_EVENT_INDEX_ENV,
    JOB_COUNT, MAX_DEPTH,
    empty_count_inventory, job_pickup_location, normalize_count_inventory,
    _is_within_bounds, _DELTAS, _adjacent_points, _build_path, _manhattan_path,
    heuristic, shortest_path, find_dynamic_path, _extend_path_log, grid_distance,
    nearest_base_to_station, _diagnose_and_print_failure, decode_schedule, decode_schedule_tick_by_tick, fitness,
    plot_gantt, station_key_from_value, dock_key_from_value, load_dispatch_events, make_jobs
)
from operation_policy import (
    position_scale,
    rack_scale,
    fleet_scale,
    action_mask,
    carrier_feature,
    completion_time_lower_bound,
    decode_action_id,
    decision_time,
    dock_congestion_features,
    initial_operation_state,
    job_status_value,
    load_required_operation_checkpoint,
    LOWER_BOUND_SCALE,
)
from neural_local_improvement import apply_neural_local_improvement

NUM_AMRS = len(AMR_KEYS)
STATION_KEYS = list(STATIONS.keys())

# Overwrite describe_solution locally to pass save_img
def describe_solution_gnn(individual: Individual, jobs: List[Job], solve_time: float = None, show_gantt: bool = False, save_img: str = None) -> Tuple[float, float]:
    availability, decoded_timeline, queue_infos, path_logs, invalid_count = decode_schedule_tick_by_tick(individual, jobs, need_log=True, check_collision=True)
    makespan = max(availability.values())
    print(f"Optimal Makespan Found: {makespan:.2f}s")
    print(f"Invalid Jobs Count: {invalid_count}")
    if solve_time is not None:
        print(f"Computation Time: {solve_time:.4f}s")
    if show_gantt or save_img:
        plot_gantt(decoded_timeline, queue_infos, jobs, solve_time=solve_time, invalid_count=invalid_count, show_gantt=show_gantt, save_img=save_img)
    return makespan, solve_time


# ===== GIN Convolutional Layer =====
class GINConv(nn.Module):
    """
    Graph Isomorphism Network convolutional layer.
    h_v^(l+1) = MLP^(l)( (1 + eps) * h_v^(l) + sum_{u in N(v)} h_u^(l) )
    """
    def __init__(self, hidden_dim, eps_init=0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps_init))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj):
        """
        x:   (batch, num_nodes, hidden_dim)
        adj: (batch, num_nodes, num_nodes) adjacency matrix (float, 0/1)
        """
        # Aggregate neighbor features via adjacency matrix multiplication
        agg = torch.bmm(adj, x)  # (batch, num_nodes, hidden_dim)
        out = (1.0 + self.eps) * x + agg
        out = self.mlp(out)
        out = self.norm(out)
        return out


# ===== Multi-Pointer GNN Scheduler =====
class SchedulerGNN(nn.Module):
    """
    Multi-Pointer Network for FJSP scheduling:
    - Job Encoder: GIN operating on the disjunctive graph of jobs
    - Machine Encoder: MLP for AMR/machine features
    - Job Actor: MLP decoder selecting the next job operation
    - Machine Actor: MLP decoder selecting the AMR/machine for the chosen job
    - Joint Critic: Shared value network estimating V(s_t)
    """
    def __init__(self, job_in_dim=16, amr_in_dim=8, hidden_dim=128, gin_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Encoders ---
        self.job_emb = nn.Linear(job_in_dim, hidden_dim)
        self.amr_emb = nn.Linear(amr_in_dim, hidden_dim)

        # GIN layers for the job disjunctive graph
        self.gin_layers = nn.ModuleList([
            GINConv(hidden_dim) for _ in range(gin_layers)
        ])

        self.op_emb = nn.Embedding(2, hidden_dim)

        # --- Operation Actor Decoder ---
        # Takes concatenation of [operation_emb, job_emb, amr_emb] -> score
        self.operation_actor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # --- Joint Critic ---
        # Takes pooled job + pooled amr representations
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_jobs(self, job_features, adj):
        """
        job_features: (batch, num_jobs, job_in_dim)
        adj: (batch, num_jobs, num_jobs) adjacency matrix
        Returns: (batch, num_jobs, hidden_dim)
        """
        x = self.job_emb(job_features)
        for gin in self.gin_layers:
            x = x + gin(x, adj)  # Residual connections
        return x

    def encode_amrs(self, amr_features):
        """
        amr_features: (batch, num_amrs, amr_in_dim)
        Returns: (batch, num_amrs, hidden_dim)
        """
        return self.amr_emb(amr_features)

    def forward_job_actor(self, job_embeddings, job_mask):
        """
        Deprecated paired-policy path.
        """
        raise RuntimeError("forward_job_actor is deprecated; use forward_operation_actor.")

    def forward_machine_actor(self, selected_job_emb, amr_embeddings):
        """
        selected_job_emb: (batch, hidden_dim)
        amr_embeddings: (batch, num_amrs, hidden_dim)
        Returns: machine_logits (batch, num_amrs)
        """
        raise RuntimeError("forward_machine_actor is deprecated; use forward_operation_actor.")

    def forward_operation_actor(self, job_embeddings, amr_embeddings, operation_mask=None):
        batch_size, num_jobs, hidden_dim = job_embeddings.shape
        num_amrs = amr_embeddings.size(1)
        op_ids = torch.arange(2, device=job_embeddings.device)
        op_embeddings = self.op_emb(op_ids).view(1, 2, 1, 1, hidden_dim).expand(batch_size, -1, num_amrs, num_jobs, -1)
        amr_expand = amr_embeddings.unsqueeze(1).unsqueeze(3).expand(-1, 2, -1, num_jobs, -1)
        job_expand = job_embeddings.unsqueeze(1).unsqueeze(2).expand(-1, 2, num_amrs, -1, -1)
        pairs = torch.cat([op_embeddings, job_expand, amr_expand], dim=-1)
        logits = self.operation_actor(pairs).squeeze(-1)
        if operation_mask is not None:
            logits = logits.masked_fill(operation_mask.view_as(logits), float("-inf"))
        return logits

    def forward_critic(self, job_embeddings, amr_embeddings, job_mask):
        """
        Returns scalar state value V(s_t).
        job_embeddings: (batch, num_jobs, hidden_dim)
        amr_embeddings: (batch, num_amrs, hidden_dim)
        job_mask: (batch, num_jobs)
        """
        # Mean-pool job embeddings (only over unassigned jobs)
        mask_float = (~job_mask).float().unsqueeze(-1)  # (batch, num_jobs, 1)
        num_unmasked = mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        job_pool = (job_embeddings * mask_float).sum(dim=1) / num_unmasked.squeeze(-1)  # (batch, hidden_dim)

        # Mean-pool AMR embeddings
        amr_pool = amr_embeddings.mean(dim=1)  # (batch, hidden_dim)

        combined = torch.cat([job_pool, amr_pool], dim=-1)  # (batch, hidden_dim*2)
        value = self.critic(combined).squeeze(-1)  # (batch,)
        return value


# ===== State Extraction =====
def extract_state_gnn(jobs, picked_jobs_set, completed_jobs_set, carrier_map, amr_positions, amr_availabilities,
                      amr_inventory, station_availabilities,
                      order_seq):
    """
    Constructs the tensor inputs for the GNN model.
    Returns: amr_features, job_features, job_mask, adj_matrix

    Job Features (16):
     0:  exist_flag (1.0)
     1:  duration / 25.0
     2:  carrier AMR index proxy
     3:  dest_pos_x / grid extent
     4:  dest_pos_y / grid extent
     5:  supply_pos_x / grid extent
     6:  supply_pos_y / grid extent
     7:  type A flag
     8:  type B flag
     9:  type C flag
     10: job_status (0.0 unpicked, 0.5 onboard, 1.0 completed)
     11: completion_time_lower_bound / 500.0
     12: inbound_dock_available_delay / 100.0
     13: outbound_dock_available_delay / 100.0
     14: inbound_dock_queue_length / num_amrs
     15: outbound_dock_queue_length / num_amrs

    AMR Features (8):
     0: status_val
     1: (avail - min_avail) / 50.0
     2: inventory A ratio
     3: inventory B ratio
     4: inventory C ratio
     5: pos_x / grid extent
     6: pos_y / grid extent
     7: queue_depth / grid extent

    Adjacency Matrix (num_jobs x num_jobs):
     - "Adding arc scheme": directed edges added when jobs are scheduled
       on the same AMR or same workstation in sequence.
    """
    num_jobs = len(jobs)
    _sx, _sy = position_scale()

    # --- AMR Features ---
    amr_feat = []
    min_avail = min(amr_availabilities.values())

    for amr in AMR_KEYS:
        pos = amr_positions[amr]
        avail = amr_availabilities[amr]
        inv = amr_inventory[amr]

        status_val = 1.0 if avail > min_avail else 0.0
        rem = avail - min_avail

        queue_depth = sum(1 for a in carrier_map.values() if a == amr) / fleet_scale()

        feat = [
            status_val,
            rem / 50.0,
            inv.get("A", 0) / rack_scale("A"),
            inv.get("B", 0) / rack_scale("B"),
            inv.get("C", 0) / rack_scale("C"),
            pos[0] / _sx,
            pos[1] / _sy,
            queue_depth,
        ]
        amr_feat.append(feat)

    # --- Job Features ---
    job_feat = []
    job_mask = []
    current_time = decision_time(amr_availabilities)

    for job in jobs:
        pos = STATIONS[job.station]
        supply_pos = job_pickup_location(job)

        is_completed = job.idx in completed_jobs_set
        job_status = job_status_value(job.idx, picked_jobs_set, completed_jobs_set)
        lb_val = 0.0 if is_completed else completion_time_lower_bound(
            job,
            amr_positions,
            amr_availabilities,
            station_availabilities,
        )
        dock_features = dock_congestion_features(
            job,
            jobs,
            picked_jobs_set,
            completed_jobs_set,
            station_availabilities,
            current_time,
        )

        feat = [
            1.0,
            job.duration / 25.0,
            carrier_feature(job.idx, carrier_map),
            pos[0] / _sx,
            pos[1] / _sy,
            supply_pos[0] / _sx,
            supply_pos[1] / _sy,
            1.0 if job.type_ == "A" else 0.0,
            1.0 if job.type_ == "B" else 0.0,
            1.0 if job.type_ == "C" else 0.0,
            job_status,
            lb_val / LOWER_BOUND_SCALE,
            *dock_features,
        ]
        job_feat.append(feat)
        job_mask.append(is_completed)

    # --- Adjacency Matrix (Adding Arc Scheme) ---
    # Build from the currently committed order_seq and amr_assignment_map
    adj = [[0.0] * num_jobs for _ in range(num_jobs)]

    # Build index maps: job.idx -> position in jobs list
    job_idx_to_list_pos = {job.idx: i for i, job in enumerate(jobs)}

    # Track last job scheduled per AMR and per station
    last_job_per_amr = {}   # amr -> last job list index
    last_job_per_station = {}  # station -> last job list index

    job_map = {job.idx: job for job in jobs}

    for scheduled_op in order_seq:
        scheduled_job_idx = scheduled_op.job_idx if isinstance(scheduled_op, Operation) else scheduled_op
        list_pos = job_idx_to_list_pos[scheduled_job_idx]
        amr = carrier_map.get(scheduled_job_idx)
        job_obj = job_map[scheduled_job_idx]
        station = job_obj.station

        # AMR sequential arc: this job aggregates from the previous job on the same AMR
        # (GINConv row = receiver: agg[i] = sum_j adj[i][j] * x[j])
        if amr is not None and amr in last_job_per_amr:
            prev_pos = last_job_per_amr[amr]
            if prev_pos != list_pos:
                adj[list_pos][prev_pos] = 1.0

        # Station sequential arc: this job aggregates from the previous job on the same station
        if station in last_job_per_station:
            prev_pos = last_job_per_station[station]
            if prev_pos != list_pos:
                adj[list_pos][prev_pos] = 1.0

        if amr is not None:
            last_job_per_amr[amr] = list_pos
        last_job_per_station[station] = list_pos

    return (
        torch.tensor([amr_feat], dtype=torch.float32),
        torch.tensor([job_feat], dtype=torch.float32),
        torch.tensor([job_mask], dtype=torch.bool),
        torch.tensor([adj], dtype=torch.float32),
    )


# ===== Autoregressive Scheduling Loop =====
def solve_with_gnn(jobs, model, deterministic=True, init_state: dict = None):
    """
    Uses the Multi-Pointer GNN model to autoregressively build a schedule.
    At each step:
      1. GIN encodes job graph -> job embeddings
      2. MLP encodes machine features -> AMR embeddings
      3. Job Actor selects job j*
      4. Machine Actor selects AMR m* for j*
      5. State is updated with precise pathfinding & reservations

    Returns: Individual, (total_job_log_prob, total_machine_log_prob), solve_duration
    """
    perf_start_time = time.perf_counter()

    amr_positions, amr_availabilities, station_availabilities, amr_inventory, amr_states, reservations = initial_operation_state(
        init_state,
        precise=True,
    )
    picked_jobs_set = set()
    completed_jobs_set = set()
    carrier_map = {}
    order_seq = []
    amr_assignment_map = {}

    total_log_prob = 0.0

    if deterministic:
        model.eval()
    else:
        model.train()

    device = next(model.parameters()).device

    for step in range(2 * len(jobs)):
        # 1. State Extraction
        amr_feat, job_feat, job_mask, adj = extract_state_gnn(
            jobs, picked_jobs_set, completed_jobs_set, carrier_map, amr_positions, amr_availabilities,
            amr_inventory, station_availabilities, order_seq
        )
        amr_feat = amr_feat.to(device)
        job_feat = job_feat.to(device)
        job_mask = job_mask.to(device)
        adj = adj.to(device)
        op_mask = torch.tensor(
            [action_mask(jobs, picked_jobs_set, completed_jobs_set, carrier_map, amr_inventory)],
            dtype=torch.bool,
            device=device,
        )

        # 2. Encode
        job_embeddings = model.encode_jobs(job_feat, adj)     # (1, num_jobs, hidden)
        amr_embeddings = model.encode_amrs(amr_feat)          # (1, num_amrs, hidden)

        logits = model.forward_operation_actor(job_embeddings, amr_embeddings, op_mask)
        flat_logits = logits.view(-1)

        if deterministic:
            best_action = torch.argmax(flat_logits).item()
        else:
            log_probs = F.log_softmax(flat_logits, dim=0)
            probs = torch.exp(log_probs)
            dist = torch.distributions.Categorical(probs)
            best_action = dist.sample().item()
            total_log_prob += log_probs[best_action]

        action = decode_action_id(best_action, jobs)
        chosen_job = jobs[action.job_list_idx]
        chosen_amr = action.amr
        order_seq.append(Operation(action.job_id, action.kind))

        if action.kind == "pickup":
            pickup_location = job_pickup_location(chosen_job)
            start_time = amr_availabilities[chosen_amr]
            pickup_path = find_dynamic_path(amr_positions[chosen_amr], pickup_location, start_time, reservations, amr_states, chosen_amr)
            pickup_time = int(len(pickup_path) - 1)
            if pickup_time == 0 and amr_positions[chosen_amr] != pickup_location:
                pickup_time = MAX_DEPTH
            inbound_dock = dock_key_from_value(chosen_job.inbound_dock)
            pickup_start = max(
                start_time + pickup_time,
                float(chosen_job.arrival_time),
                station_availabilities.get(inbound_dock, 0.0),
            )
            pickup_end = pickup_start + chosen_job.duration
            for t_offset, pt in enumerate(pickup_path):
                reservations[(pt, int(start_time) + t_offset)] = chosen_amr
            for t_wait in range(int(start_time) + pickup_time, int(pickup_end) + 1):
                reservations[(pickup_location, t_wait)] = chosen_amr
            amr_states[chosen_amr] = (pickup_location, pickup_end)
            amr_availabilities[chosen_amr] = pickup_end
            amr_positions[chosen_amr] = pickup_location
            station_availabilities[inbound_dock] = pickup_end
            amr_inventory[chosen_amr][chosen_job.type_] += 1
            picked_jobs_set.add(chosen_job.idx)
            carrier_map[chosen_job.idx] = chosen_amr
            amr_assignment_map[chosen_job.idx] = chosen_amr
        else:
            target_station = STATIONS[chosen_job.station]
            travel_start = amr_availabilities[chosen_amr]
            travel_path = find_dynamic_path(amr_positions[chosen_amr], target_station, travel_start, reservations, amr_states, chosen_amr)
            travel_time = int(len(travel_path) - 1)
            if travel_time == 0 and amr_positions[chosen_amr] != target_station:
                travel_time = MAX_DEPTH
            travel_end = travel_start + travel_time
            for t_offset, pt in enumerate(travel_path):
                reservations[(pt, int(travel_start) + t_offset)] = chosen_amr
            process_start = max(travel_end, station_availabilities[chosen_job.station])
            process_end = process_start + chosen_job.duration
            for t_process in range(int(travel_end), int(process_end) + 1):
                reservations[(target_station, t_process)] = chosen_amr
            amr_states[chosen_amr] = (target_station, process_end)
            amr_availabilities[chosen_amr] = process_end
            amr_positions[chosen_amr] = target_station
            amr_inventory[chosen_amr][chosen_job.type_] -= 1
            station_availabilities[chosen_job.station] = process_end
            completed_jobs_set.add(chosen_job.idx)

    # Finalize Individual
    final_assignment = []
    for i in range(len(jobs)):
        final_assignment.append(amr_assignment_map[i])

    ind = Individual(order=order_seq, amr_assignment=final_assignment)

    solve_dur = time.perf_counter() - perf_start_time
    return ind, total_log_prob, solve_dur


if __name__ == "__main__":
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gantt", action="store_true", help="Plot Gantt Chart")
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file")
    parser.add_argument("--save_img", type=str, default="", help="Save the schedule Gantt chart to this file")
    parser.add_argument("--local_iters", type=int, default=0, help="Number of simplified local-improvement iterations")
    parser.add_argument("--collision_iters", type=int, default=0, help="Number of collision-aware local-improvement iterations")
    parser.add_argument("--output_csv", type=str, default="gnn_precise_summary_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    # 1. Setup
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. Create Model
    gnn_model = SchedulerGNN(job_in_dim=16, amr_in_dim=8, hidden_dim=128, gin_layers=3)

    weights_path = Path("gnn_precise_mpn_scheduler_best.pth")
    if not weights_path.exists():
        old_weights = Path("gnn_mpn_scheduler_best.pth")
        if old_weights.exists():
            weights_path = old_weights

    print(f"Loading trained weights from {weights_path}...")
    status = load_required_operation_checkpoint(
        gnn_model,
        weights_path,
        torch,
        required_keys=("op_emb.weight", "operation_actor.0.weight"),
    )
    print(f"GNN precise: {status}")

    # 3. Load Jobs
    if args.inbox:
        dispatch_events = load_dispatch_events(Path(args.inbox))
    else:
        dispatch_events = load_dispatch_events()
    target_index = os.environ.get(DISPATCH_EVENT_INDEX_ENV)

    output_filename = args.output_csv
    results_data = []

    print("=== Using GNN Precise (Dynamic Pathfinding + Reservations) Logic ===")

    if dispatch_events:
        if target_index is not None:
            dispatch_events = [e for e in dispatch_events if str(e["index"]) == str(target_index)]

        for event in dispatch_events:
            print(f"\n=== Processing Dispatch Event {event['index']} (Jobs: {len(event['jobs'])}) ===")

            best_ind, _, solve_dur_ns = solve_with_gnn(event["jobs"], gnn_model)

            improvement = apply_neural_local_improvement(
                best_ind,
                event["jobs"],
                simplified_iters=args.local_iters,
                collision_iters=args.collision_iters,
            )
            best_ind = improvement.individual
            solve_dur_ns += improvement.postprocess_time

            # Precise rollout already uses dynamic pathfinding and reservations.
            img_path = f"{args.save_img.split('.')[0]}_{event['index']}.png" if args.save_img else None
            makespan, computation_time = describe_solution_gnn(best_ind, event["jobs"], solve_time=solve_dur_ns, show_gantt=args.gantt, save_img=img_path)

            results_data.append([event['index'], f"{makespan:.2f}", f"{computation_time:.4f}"])
    else:
        print("No dispatch file found. Generating random jobs...")
        jobs = make_jobs()

        best_ind, _, solve_dur_ns = solve_with_gnn(jobs, gnn_model)

        improvement = apply_neural_local_improvement(
            best_ind,
            jobs,
            simplified_iters=args.local_iters,
            collision_iters=args.collision_iters,
        )
        best_ind = improvement.individual
        solve_dur_ns += improvement.postprocess_time

        # Precise rollout already uses dynamic pathfinding and reservations.
        makespan, computation_time = describe_solution_gnn(best_ind, jobs, solve_time=solve_dur_ns, show_gantt=args.gantt, save_img=args.save_img)

        results_data.append(["random", f"{makespan:.2f}", f"{computation_time:.4f}"])

    # 4. Save metrics
    if results_data:
        with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Event_Index", "Makespan", "Computation_Time"])
            writer.writerows(results_data)
        print(f"\nSummary results saved to {output_filename}")
