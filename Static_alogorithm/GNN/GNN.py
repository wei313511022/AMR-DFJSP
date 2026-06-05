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
from GA.GA import (
    Job, Individual, AMR_STARTS, AMR_KEYS, STATIONS, OBSTACLES, _GRID_POINTS,
    GRID_MIN_X, GRID_MAX_X, GRID_MIN_Y, GRID_MAX_Y, BASES, TYPE_DURATION,
    SCHEDULE_OUTBOX, DISPATCH_INBOX, DISPATCH_EVENT_INDEX_ENV,
    JOB_COUNT, routing_iters, collision_routing_iters,
    empty_count_inventory, job_pickup_location, normalize_count_inventory, paired_operation_order,
    _is_within_bounds, _DELTAS, _adjacent_points, _build_path, _manhattan_path,
    heuristic, _extend_path_log, grid_distance,
    nearest_base_to_station, _diagnose_and_print_failure, decode_schedule, decode_schedule_tick_by_tick, fitness, local_improve,
    plot_gantt, station_key_from_value, load_dispatch_events, make_jobs
)

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
    def __init__(self, job_in_dim=12, amr_in_dim=8, hidden_dim=128, gin_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Encoders ---
        self.job_emb = nn.Linear(job_in_dim, hidden_dim)
        self.amr_emb = nn.Linear(amr_in_dim, hidden_dim)

        # GIN layers for the job disjunctive graph
        self.gin_layers = nn.ModuleList([
            GINConv(hidden_dim) for _ in range(gin_layers)
        ])

        # --- Job Actor Decoder ---
        self.job_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # --- Machine Actor Decoder ---
        # Takes concatenation of [selected_job_emb, amr_emb] -> score
        self.machine_actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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
        job_embeddings: (batch, num_jobs, hidden_dim)
        job_mask: (batch, num_jobs) - True if job ALREADY ASSIGNED
        Returns: job_logits (batch, num_jobs)
        """
        logits = self.job_actor(job_embeddings).squeeze(-1)  # (batch, num_jobs)
        logits = logits.masked_fill(job_mask, float('-inf'))
        return logits

    def forward_machine_actor(self, selected_job_emb, amr_embeddings):
        """
        selected_job_emb: (batch, hidden_dim)
        amr_embeddings: (batch, num_amrs, hidden_dim)
        Returns: machine_logits (batch, num_amrs)
        """
        num_amrs = amr_embeddings.size(1)
        # Expand selected job embedding to pair with each AMR
        job_expand = selected_job_emb.unsqueeze(1).expand(-1, num_amrs, -1)  # (batch, num_amrs, hidden_dim)
        pairs = torch.cat([job_expand, amr_embeddings], dim=-1)  # (batch, num_amrs, hidden_dim*2)
        logits = self.machine_actor(pairs).squeeze(-1)  # (batch, num_amrs)
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
def extract_state_gnn(jobs, assigned_jobs_set, amr_positions, amr_availabilities,
                      amr_inventory, amr_assignment_map, station_availabilities,
                      order_seq):
    """
    Constructs the tensor inputs for the GNN model.
    Returns: amr_features, job_features, job_mask, adj_matrix

    Job Features (12):
     0:  exist_flag (1.0)
     1:  duration / 25.0
     2:  wait_time (0.0)
     3:  dest_pos_x / 10.0
     4:  dest_pos_y / 10.0
     5:  supply_pos_x / 10.0
     6:  supply_pos_y / 10.0
     7:  type A flag
     8:  type B flag
     9:  type C flag
     10: job_status (1.0 if assigned)
     11: completion_time_lower_bound / 500.0

    AMR Features (8):
     0: status_val
     1: (avail - min_avail) / 50.0
     2: inventory A ratio
     3: inventory B ratio
     4: inventory C ratio
     5: pos_x / 10.0
     6: pos_y / 10.0
     7: queue_depth / 10.0

    Adjacency Matrix (num_jobs x num_jobs):
     - "Adding arc scheme": directed edges added when jobs are scheduled
       on the same AMR or same workstation in sequence.
    """
    num_jobs = len(jobs)

    # --- AMR Features ---
    amr_feat = []
    min_avail = min(amr_availabilities.values())

    for amr in AMR_KEYS:
        pos = amr_positions[amr]
        avail = amr_availabilities[amr]
        inv = amr_inventory[amr]

        status_val = 1.0 if avail > min_avail else 0.0
        rem = avail - min_avail

        queue_depth = sum(1 for j_idx, a in amr_assignment_map.items() if a == amr) / 10.0

        feat = [
            status_val,
            rem / 50.0,
            inv.get("A", 0) / 3.0,
            inv.get("B", 0) / 3.0,
            inv.get("C", 0) / 3.0,
            pos[0] / 10.0,
            pos[1] / 10.0,
            queue_depth,
        ]
        amr_feat.append(feat)

    # --- Job Features ---
    job_feat = []
    job_mask = []

    # Compute simple completion time lower bounds for each job
    for job in jobs:
        pos = STATIONS[job.station]
        supply_pos = job_pickup_location(job)

        is_assigned = job.idx in assigned_jobs_set
        job_status = 1.0 if is_assigned else 0.0

        # LB: minimum travel from closest idle AMR + processing time
        if not is_assigned:
            lb_candidates = []
            for amr in AMR_KEYS:
                pickup_est = heuristic(amr_positions[amr], supply_pos)
                travel_est = heuristic(supply_pos, pos)
                arrival_wait = max(0.0, float(job.arrival_time) - (amr_availabilities[amr] + pickup_est))
                lb = max(amr_availabilities[amr] + pickup_est + arrival_wait + travel_est,
                         station_availabilities.get(job.station, 0.0)) + job.duration
                lb_candidates.append(lb)
            lb_val = min(lb_candidates) if lb_candidates else 0.0
        else:
            lb_val = 0.0

        feat = [
            1.0,
            job.duration / 25.0,
            0.0,
            pos[0] / 10.0,
            pos[1] / 10.0,
            supply_pos[0] / 10.0,
            supply_pos[1] / 10.0,
            1.0 if job.type_ == "A" else 0.0,
            1.0 if job.type_ == "B" else 0.0,
            1.0 if job.type_ == "C" else 0.0,
            job_status,
            lb_val / 500.0,
        ]
        job_feat.append(feat)
        job_mask.append(is_assigned)

    # --- Adjacency Matrix (Adding Arc Scheme) ---
    # Build from the currently committed order_seq and amr_assignment_map
    adj = [[0.0] * num_jobs for _ in range(num_jobs)]

    # Build index maps: job.idx -> position in jobs list
    job_idx_to_list_pos = {job.idx: i for i, job in enumerate(jobs)}

    # Track last job scheduled per AMR and per station
    last_job_per_amr = {}   # amr -> last job list index
    last_job_per_station = {}  # station -> last job list index

    job_map = {job.idx: job for job in jobs}

    for scheduled_job_idx in order_seq:
        list_pos = job_idx_to_list_pos[scheduled_job_idx]
        amr = amr_assignment_map.get(scheduled_job_idx)
        job_obj = job_map[scheduled_job_idx]
        station = job_obj.station

        # AMR sequential arc: previous job on same AMR -> this job
        if amr is not None and amr in last_job_per_amr:
            prev_pos = last_job_per_amr[amr]
            adj[prev_pos][list_pos] = 1.0

        # Station sequential arc: previous job on same station -> this job
        if station in last_job_per_station:
            prev_pos = last_job_per_station[station]
            adj[prev_pos][list_pos] = 1.0

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
      5. State is updated with fast heuristic travel and return-home time

    Returns: Individual, (total_job_log_prob, total_machine_log_prob), solve_duration
    """
    perf_start_time = time.perf_counter()

    # --- Initialize simulator state ---
    if init_state:
        amr_positions = {amr: init_state["positions"].get(amr, AMR_STARTS[amr]) for amr in AMR_KEYS}
        amr_availabilities = {amr: float(init_state["availability"].get(amr, 0.0)) for amr in AMR_KEYS}
        station_availabilities = {s: float(init_state["time"]) for s in STATIONS.keys()}
        amr_inventory = normalize_count_inventory(init_state.get("inventory", {}))
    else:
        amr_positions = {amr: AMR_STARTS[amr] for amr in AMR_KEYS}
        amr_availabilities = {amr: 0.0 for amr in AMR_KEYS}
        station_availabilities = {s: 0.0 for s in STATIONS.keys()}
        amr_inventory = empty_count_inventory()

    assigned_jobs_set = set()
    order_seq = []
    amr_assignment_map = {}

    total_job_log_prob = 0.0
    total_machine_log_prob = 0.0

    if deterministic:
        model.eval()
    else:
        model.train()

    device = next(model.parameters()).device

    for step in range(len(jobs)):
        # 1. State Extraction
        amr_feat, job_feat, job_mask, adj = extract_state_gnn(
            jobs, assigned_jobs_set, amr_positions, amr_availabilities,
            amr_inventory, amr_assignment_map, station_availabilities, order_seq
        )
        amr_feat = amr_feat.to(device)
        job_feat = job_feat.to(device)
        job_mask = job_mask.to(device)
        adj = adj.to(device)

        # 2. Encode
        job_embeddings = model.encode_jobs(job_feat, adj)     # (1, num_jobs, hidden)
        amr_embeddings = model.encode_amrs(amr_feat)          # (1, num_amrs, hidden)

        # 3. Job Actor: select job
        job_logits = model.forward_job_actor(job_embeddings, job_mask)  # (1, num_jobs)
        job_logits_flat = job_logits.view(-1)

        if deterministic:
            chosen_job_list_idx = torch.argmax(job_logits_flat).item()
        else:
            job_log_probs = F.log_softmax(job_logits_flat, dim=0)
            job_probs = torch.exp(job_log_probs)
            job_dist = torch.distributions.Categorical(job_probs)
            chosen_job_list_idx = job_dist.sample().item()
            total_job_log_prob += job_log_probs[chosen_job_list_idx]

        chosen_job = jobs[chosen_job_list_idx]
        selected_job_emb = job_embeddings[:, chosen_job_list_idx, :]  # (1, hidden)

        # 4. Machine Actor: select AMR
        machine_logits = model.forward_machine_actor(selected_job_emb, amr_embeddings)  # (1, num_amrs)
        machine_logits_flat = machine_logits.view(-1)

        if deterministic:
            chosen_amr_idx = torch.argmax(machine_logits_flat).item()
        else:
            machine_log_probs = F.log_softmax(machine_logits_flat, dim=0)
            machine_probs = torch.exp(machine_log_probs)
            machine_dist = torch.distributions.Categorical(machine_probs)
            chosen_amr_idx = machine_dist.sample().item()
            total_machine_log_prob += machine_log_probs[chosen_amr_idx]

        chosen_amr = AMR_KEYS[chosen_amr_idx]

        # 5. Record choice
        order_seq.append(chosen_job.idx)
        amr_assignment_map[chosen_job.idx] = chosen_amr
        assigned_jobs_set.add(chosen_job.idx)

        # 6. Update Internal State (Fast Heuristic Travel + Return Home)
        material = chosen_job.type_
        curr_pos = amr_positions[chosen_amr]
        avail = amr_availabilities[chosen_amr]

        pickup_location = job_pickup_location(chosen_job)
        avail = max(avail + heuristic(curr_pos, pickup_location), float(chosen_job.arrival_time))
        curr_pos = pickup_location
        amr_inventory[chosen_amr][material] = min(amr_inventory[chosen_amr][material] + 1, 3)

        # Travel to station
        target_station = STATIONS[chosen_job.station]
        avail += heuristic(curr_pos, target_station)

        # Wait for station and process
        process_start = max(avail, station_availabilities[chosen_job.station])
        process_end = process_start + chosen_job.duration
        amr_inventory[chosen_amr][material] -= 1
        station_availabilities[chosen_job.station] = process_end

        amr_availabilities[chosen_amr] = process_end
        amr_positions[chosen_amr] = target_station

    # Finalize Individual
    final_assignment = []
    for i in range(len(jobs)):
        final_assignment.append(amr_assignment_map[i])

    ind = Individual(order=paired_operation_order(order_seq), amr_assignment=final_assignment)

    solve_dur = time.perf_counter() - perf_start_time
    return ind, (total_job_log_prob, total_machine_log_prob), solve_dur


if __name__ == "__main__":
    import numpy as np
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gantt", action="store_true", help="Plot Gantt Chart")
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file")
    parser.add_argument("--save_img", type=str, default="", help="Save the schedule Gantt chart to this file")
    parser.add_argument("--collision_iters", type=int, default=collision_routing_iters, help="Number of collision routing iterations")
    parser.add_argument("--output_csv", type=str, default="gnn_summary_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    # 1. Setup
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. Create Model
    gnn_model = SchedulerGNN(job_in_dim=12, amr_in_dim=8, hidden_dim=128, gin_layers=3)

    weights_path = Path("gnn_mpn_scheduler_best.pth")
    if not weights_path.exists():
        old_weights = Path("gnn_scheduler_best.pth")
        if old_weights.exists():
            weights_path = old_weights

    if weights_path.exists():
        print(f"Loading trained weights from {weights_path}...")
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            gnn_model.load_state_dict(state_dict)
        except Exception as e:
            print(f"WARNING: Could not load weights from {weights_path}: {e}")
            print("Proceeding with randomly initialized weights.")

    # 3. Load Jobs
    if args.inbox:
        dispatch_events = load_dispatch_events(Path(args.inbox))
    else:
        dispatch_events = load_dispatch_events()
    target_index = os.environ.get(DISPATCH_EVENT_INDEX_ENV)

    output_filename = args.output_csv
    results_data = []

    print("=== Using GNN Fast Heuristic (Multi-Pointer Network) Logic ===")

    if dispatch_events:
        if target_index is not None:
            dispatch_events = [e for e in dispatch_events if str(e["index"]) == str(target_index)]

        for event in dispatch_events:
            print(f"\n=== Processing Dispatch Event {event['index']} (Jobs: {len(event['jobs'])}) ===")

            best_ind, _, solve_dur_ns = solve_with_gnn(event["jobs"], gnn_model)

            improve_start = time.perf_counter()
            best_ind = local_improve(best_ind, event["jobs"], max_iters=routing_iters)
            if args.collision_iters > 0:
                best_ind = local_improve(best_ind, event["jobs"], max_iters=args.collision_iters, check_collision=True)
            solve_dur_ns += (time.perf_counter() - improve_start)

            img_path = f"{args.save_img.split('.')[0]}_{event['index']}.png" if args.save_img else None
            makespan, computation_time = describe_solution_gnn(best_ind, event["jobs"], solve_time=solve_dur_ns, show_gantt=args.gantt, save_img=img_path)

            results_data.append([event['index'], f"{makespan:.2f}", f"{computation_time:.4f}"])
    else:
        print("No dispatch file found. Generating random jobs...")
        jobs = make_jobs()

        best_ind, _, solve_dur_ns = solve_with_gnn(jobs, gnn_model)

        improve_start = time.perf_counter()
        best_ind = local_improve(best_ind, jobs, max_iters=routing_iters)
        if args.collision_iters > 0:
            best_ind = local_improve(best_ind, jobs, max_iters=args.collision_iters, check_collision=True)
        solve_dur_ns += (time.perf_counter() - improve_start)

        makespan, computation_time = describe_solution_gnn(best_ind, jobs, solve_time=solve_dur_ns, show_gantt=args.gantt, save_img=args.save_img)

        results_data.append(["random", f"{makespan:.2f}", f"{computation_time:.4f}"])

    # 4. Save metrics
    if results_data:
        with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Event_Index", "Makespan", "Computation_Time"])
            writer.writerows(results_data)
        print(f"\nSummary results saved to {output_filename}")
