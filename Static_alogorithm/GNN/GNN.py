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
    SUPPLY_LOCATIONS, SCHEDULE_OUTBOX, DISPATCH_INBOX, DISPATCH_EVENT_INDEX_ENV,
    JOB_COUNT, MAX_DEPTH, routing_iters, collision_routing_iters,
    _is_within_bounds, _DELTAS, _adjacent_points, _build_path, _manhattan_path,
    heuristic, shortest_path, find_dynamic_path, _extend_path_log, grid_distance,
    nearest_base_to_station, _diagnose_and_print_failure, decode_schedule, decode_schedule_tick_by_tick, fitness, local_improve,
    plot_gantt, station_key_from_value, load_dispatch_events, make_jobs
)

# Overwrite describe_solution locally to pass save_img
def describe_solution_gnn(individual: Individual, jobs: List[Job], solve_time: float = None, show_gantt: bool = False, save_img: str = None) -> Tuple[float, float]:
    availability, decoded_timeline, queue_infos, path_logs, invalid_count = decode_schedule_tick_by_tick(individual, jobs, need_log=True, check_collision=True)
    makespan = max(availability.values())
    print(f"Optimal Makespan Found: {makespan:.2f}s")
    print(f"Invalid Jobs Count: {invalid_count}")
    if solve_time is not None:
        print(f"Computation Time: {solve_time:.4f}s")
    if show_gantt or save_img:
        # Generate plot
        plot_gantt(decoded_timeline, queue_infos, jobs, solve_time=solve_time, invalid_count=invalid_count, show_gantt=show_gantt, save_img=save_img)
            
    return makespan, solve_time

# ===== GNN Architecture =====
class SchedulerGNN(nn.Module):
    def __init__(self, amr_in_dim=8, job_in_dim=10, hidden_dim=128, gnn_layers=2):
        super().__init__()
        self.amr_emb = nn.Linear(amr_in_dim, hidden_dim)
        self.job_emb = nn.Linear(job_in_dim, hidden_dim)

        self.gnn_layers = gnn_layers
        # Simplified interaction using multihead attention where AMRs and Jobs attend to each other
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
            for _ in range(gnn_layers)
        ])
        
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(gnn_layers)
        ])

        # Policy Head: Takes concatenated (AMR, Job) embeddings and outputs logit
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, amr_features, job_features, job_mask):
        """
        amr_features: (batch, num_amrs, amr_in_dim)
        job_features: (batch, num_jobs, job_in_dim)
        job_mask: (batch, num_jobs) - True if job is ALREADY ASSIGNED (should be ignored)
        
        Returns logits for each valid (amr, job) pair
        Output shape: (batch, num_amrs, num_jobs)
        """
        # 1. Embeddings
        x_amr = self.amr_emb(amr_features) # (batch, num_amrs, hidden)
        x_job = self.job_emb(job_features) # (batch, num_jobs, hidden)

        # 2. Message Passing (treat as bipartite graph interaction)
        # AMRs and Jobs exchange information. We concatenate them into a single graph node list for global attention.
        num_amrs = x_amr.size(1)
        num_jobs = x_job.size(1)
        
        # Combine nodes for self-attention across the whole statespace
        # x_nodes: (batch, num_amrs + num_jobs, hidden)
        x_nodes = torch.cat([x_amr, x_job], dim=1)
        
        # Create a padding mask for self attention to prevent attention to assigned jobs
        # Attention mask: True means DO NOT attend.
        # amr nodes are never masked (False). jobs are masked if job_mask is True.
        batch_size = x_nodes.size(0)
        amr_mask = torch.zeros((batch_size, num_amrs), dtype=torch.bool, device=x_nodes.device)
        attn_key_padding_mask = torch.cat([amr_mask, job_mask], dim=1) # (batch, num_amrs + num_jobs)
        
        # If all jobs are masked (which shouldn't happen during active stepping, but just in case), 
        # prevent NaN in attention by unmasking everything temporarily
        if attn_key_padding_mask.all():
            attn_key_padding_mask = None

        for attn, fc in zip(self.attn_layers, self.fc_layers):
            # Self-attention over all valid AMRs and Jobs
            x_nodes_attn, _ = attn(x_nodes, x_nodes, x_nodes, key_padding_mask=attn_key_padding_mask)
            x_nodes = x_nodes + x_nodes_attn
            x_nodes = x_nodes + fc(x_nodes)
            
        # Split back
        x_amr = x_nodes[:, :num_amrs, :]
        x_job = x_nodes[:, num_amrs:, :]

        # 3. Policy Head (Pairwise comparisons)
        # We need an output for every combination of AMR and Job
        # x_amr_expand: (batch, num_amrs, num_jobs, hidden)
        x_amr_expand = x_amr.unsqueeze(2).expand(-1, -1, num_jobs, -1)
        # x_job_expand: (batch, num_amrs, num_jobs, hidden)
        x_job_expand = x_job.unsqueeze(1).expand(-1, num_amrs, -1, -1)
        
        # pairs: (batch, num_amrs, num_jobs, hidden * 2)
        pairs = torch.cat([x_amr_expand, x_job_expand], dim=-1)
        
        # logits: (batch, num_amrs, num_jobs)
        logits = self.policy_head(pairs).squeeze(-1)
        
        # Mask out assigned jobs by setting their logits to -inf
        # job_mask_expand: (batch, 1, num_jobs)
        job_mask_expand = job_mask.unsqueeze(1)
        logits = logits.masked_fill(job_mask_expand, float('-inf'))
        
        return logits


def extract_state(jobs, assigned_jobs_set, amr_positions, amr_availabilities, amr_inventory):
    """
    Constructs the tensor inputs for the GNN.
    Returns amr_features, job_features, job_mask defined as:
    
    AMR Features (8):
     0: x / max_x
     1: y / max_y
     2: available_time (normalized roughly to max plausible time, eg 1000)
     3: inventory A
     4: inventory B
     5: inventory C
     6: idle flag (1 if available_time is min among AMRs)
     7: identity proxy (integer id for AMR / 3.0)
     
    Job Features (10):
     0: x / max_x (station location)
     1: y / max_y
     2: duration / 30.0
     3: type A flag (1 or 0)
     4: type B flag (1 or 0)
     5: type C flag (1 or 0)
     6: identity index / limit
     7: 0.0 (placeholder for future metrics)
     8: 0.0
     9: 0.0
     
    Job Mask: boolean array of size len(jobs), True if job inside assigned_jobs
    """
    
    # AMR Features
    amr_feat = []
    min_avail = min(amr_availabilities.values())
    
    for idx, amr in enumerate(AMR_KEYS):
        pos = amr_positions[amr]
        avail = amr_availabilities[amr]
        inv = amr_inventory[amr]
        
        feat = [
            pos[0] / max(GRID_MAX_X, 1),
            pos[1] / max(GRID_MAX_Y, 1),
            avail / 1000.0,
            inv.get("A", 0) / 3.0,
            inv.get("B", 0) / 3.0,
            inv.get("C", 0) / 3.0,
            1.0 if avail == min_avail else 0.0,
            idx / len(AMR_KEYS)
        ]
        amr_feat.append(feat)
        
    # Job Features
    job_feat = []
    job_mask = []
    for job in jobs:
        pos = STATIONS[job.station]
        
        feat = [
            pos[0] / max(GRID_MAX_X, 1),
            pos[1] / max(GRID_MAX_Y, 1),
            job.duration / 30.0,
            1.0 if job.type_ == "A" else 0.0,
            1.0 if job.type_ == "B" else 0.0,
            1.0 if job.type_ == "C" else 0.0,
            job.idx / JOB_COUNT,
            0.0, 0.0, 0.0
        ]
        job_feat.append(feat)
        job_mask.append(job.idx in assigned_jobs_set)
        
    return (
        torch.tensor([amr_feat], dtype=torch.float32), 
        torch.tensor([job_feat], dtype=torch.float32),
        torch.tensor([job_mask], dtype=torch.bool)
    )

# ===== GNN Scheduling Heuristic Loop =====
def solve_with_gnn(jobs, model, deterministic=True, init_state: dict = None):
    """
    Uses the GNN to autoregressively build a schedule.
    Returns: The final Individual, the total log probabilities of the sequence, and the total execution time of the building process.
    """
    start_time = time.perf_counter()
    
    # Internal Simulator State (Tracking rough time/inventory during sequence building)
    # We use Manhattan distance here for speed. The exact simulation happens later in `decode_schedule`.
    if init_state:
        amr_positions = {amr: init_state["positions"].get(amr, AMR_STARTS[amr]) for amr in AMR_KEYS}
        amr_availabilities = {amr: float(init_state["availability"].get(amr, 0.0)) for amr in AMR_KEYS}
        station_availabilities = {s: float(init_state["time"]) for s in STATIONS.keys()}
        amr_inventory = {amr: init_state["inventory"].get(amr, {mat: 0 for mat in TYPE_DURATION.keys()}).copy() for amr in AMR_KEYS}
    else:
        amr_positions = {amr: AMR_STARTS[amr] for amr in AMR_KEYS}
        amr_availabilities = {amr: 0.0 for amr in AMR_KEYS}
        station_availabilities = {s: 0.0 for s in STATIONS.keys()}
        
        amr_inventory = {amr: {mat: 0 for mat in TYPE_DURATION.keys()} for amr in AMR_KEYS}
        amr_inventory["AMR1"]["A"] = 3
        amr_inventory["AMR2"]["B"] = 3
        amr_inventory["AMR3"]["C"] = 3
    
    assigned_jobs_set = set()
    
    # Outputs to build the Individual
    order_seq = []
    amr_assignment_map = {} # job_idx -> amr
    
    total_log_prob = 0.0
    
    # Model evaluation mode depends on deterministic flag
    if deterministic:
        model.eval()
    else:
        model.train()
        
    for step in range(len(jobs)):
        # 1. State Extraction
        amr_feat, job_feat, job_mask = extract_state(
            jobs, assigned_jobs_set, amr_positions, amr_availabilities, amr_inventory
        )
        
        # Move tensors to the model's device
        device = next(model.parameters()).device
        amr_feat = amr_feat.to(device)
        job_feat = job_feat.to(device)
        job_mask = job_mask.to(device)
        
        # 2. GNN Forward Pass
        logits = model(amr_feat, job_feat, job_mask) # shape: (1, 3, num_jobs)
        
        # 3. Action Selection
        # Flatten to 1D to pick joint (AMR, Job) action
        flat_logits = logits.view(-1)
        
        if deterministic:
            best_action = torch.argmax(flat_logits).item()
        else:
            # Stochastic sampling (e.g. for REINFORCE training)
            # Use log_softmax for numerically stable log probabilities
            log_probs = F.log_softmax(flat_logits, dim=0)
            probs = torch.exp(log_probs)
            
            # Create categorical distribution to sample from
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            best_action = action_tensor.item()
            
            # Accumulate log probability of chosen action
            total_log_prob += log_probs[best_action]
            
        # Decode action: amr_index and job_index
        num_jobs = len(jobs)
        amr_idx = best_action // num_jobs
        job_list_idx = best_action % num_jobs
        
        chosen_amr = AMR_KEYS[amr_idx]
        chosen_job = jobs[job_list_idx]
        
        # Record choice
        order_seq.append(chosen_job.idx)
        amr_assignment_map[chosen_job.idx] = chosen_amr
        assigned_jobs_set.add(chosen_job.idx)
        
        # 4. Update Internal State (Fast Approximation for the next step of GNN inference)
        mat = chosen_job.type_
        curr_pos = amr_positions[chosen_amr]
        avail = amr_availabilities[chosen_amr]
        
        # Check supply
        if amr_inventory[chosen_amr][mat] == 0:
            supply_loc = SUPPLY_LOCATIONS[mat]
            avail += heuristic(curr_pos, supply_loc)
            curr_pos = supply_loc
            amr_inventory[chosen_amr][mat] = 3
            
        # Travel to station
        target_station = STATIONS[chosen_job.station]
        avail += heuristic(curr_pos, target_station)
        
        # Wait for station
        avail = max(avail, station_availabilities[chosen_job.station])
        
        # Process
        avail += chosen_job.duration
        amr_inventory[chosen_amr][mat] -= 1
        
        # Update state dicts
        station_availabilities[chosen_job.station] = avail
        amr_availabilities[chosen_amr] = avail
        amr_positions[chosen_amr] = target_station
            
    # Finalize Individual (Order amr_assignment list by job_idx)
    final_assignment = []
    # Make sure we use the ID sequence mapped to assignments
    for i in range(len(jobs)):
        # job.idx may not perfectly sequential in all domains, but here make_jobs uses idx 0..N-1
        # Re-map so amr_assignment aligns with job_idx 0->N-1
        final_assignment.append(amr_assignment_map[i])
        
    ind = Individual(order=order_seq, amr_assignment=final_assignment)
    
    solve_dur = time.perf_counter() - start_time
    return ind, total_log_prob, solve_dur


if __name__ == "__main__":
    import numpy as np # Local import for seeds
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gantt", action="store_true", help="Plot Gantt Chart")
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file")
    parser.add_argument("--save_img", type=str, default="", help="Save the schedule Gantt chart to this file (e.g., schedule.png)")
    parser.add_argument("--collision_iters", type=int, default=collision_routing_iters, help="Number of collision routing iterations")
    parser.add_argument("--output_csv", type=str, default="GNN_summary_results.csv", help="Output CSV filename")
    args = parser.parse_args()
    
    # 1. Setup environment and seed 
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 2. Create Model
    gnn = SchedulerGNN(amr_in_dim=8, job_in_dim=10, hidden_dim=128, gnn_layers=2)
    
    weights_path = Path("gnn_scheduler_best.pth")
    if weights_path.exists():
        print(f"Loading trained weights from {weights_path}...")
        gnn.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    # 3. Load Jobs
    if args.inbox:
        dispatch_events = load_dispatch_events(Path(args.inbox))
    else:
        dispatch_events = load_dispatch_events()
    target_index = os.environ.get(DISPATCH_EVENT_INDEX_ENV)
    
    output_filename = args.output_csv
    results_data = []

    print("=== Using GNN Logic ===")

    if dispatch_events:
        if target_index is not None:
             dispatch_events = [e for e in dispatch_events if str(e["index"]) == str(target_index)]
        
        for event in dispatch_events:
            print(f"\n=== Processing Dispatch Event {event['index']} (Jobs: {len(event['jobs'])}) ===")
            
            # a. Run GNN Inference
            best_ind, _, solve_dur_ns = solve_with_gnn(event["jobs"], gnn)
            
            # b. Apply Local Improve for routing/collision adjustment exactly identically to GA.py
            improve_start = time.perf_counter()
            best_ind = local_improve(best_ind, event["jobs"], max_iters=routing_iters)
            if args.collision_iters > 0:
                best_ind = local_improve(best_ind, event["jobs"], max_iters=args.collision_iters, check_collision=True)
            solve_dur_ns += (time.perf_counter() - improve_start)
            
            # c. Evaluate with Exact GA routing logic
            img_path = f"{args.save_img.split('.')[0]}_{event['index']}.png" if args.save_img else None
            makespan, computation_time = describe_solution_gnn(best_ind, event["jobs"], solve_time=solve_dur_ns, show_gantt=args.gantt, save_img=img_path)
            
            results_data.append([event['index'], f"{makespan:.2f}", f"{computation_time:.4f}"])
    else:
        print("No dispatch file found. Generating random jobs...")
        jobs = make_jobs()
        
        # a. Run GNN Inference
        best_ind, _, solve_dur_ns = solve_with_gnn(jobs, gnn)
        
        # b. Apply Local Improve for routing/collision adjustment exactly identically to GA.py
        improve_start = time.perf_counter()
        best_ind = local_improve(best_ind, jobs, max_iters=routing_iters)
        if args.collision_iters > 0:
            best_ind = local_improve(best_ind, jobs, max_iters=args.collision_iters, check_collision=True)
        solve_dur_ns += (time.perf_counter() - improve_start)
        
        # c. Evaluate with Exact GA routing logic
        makespan, computation_time = describe_solution_gnn(best_ind, jobs, solve_time=solve_dur_ns, show_gantt=args.gantt, save_img=args.save_img)
        
        results_data.append(["random", f"{makespan:.2f}", f"{computation_time:.4f}"])

    # 4. Save metrics
    if results_data:
        with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Event_Index", "Makespan", "Computation_Time"])
            writer.writerows(results_data)
        print(f"\nSummary results saved to {output_filename}")

