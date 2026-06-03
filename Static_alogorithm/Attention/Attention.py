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
    JOB_COUNT, routing_iters, collision_routing_iters,
    _is_within_bounds, _DELTAS, _adjacent_points, _build_path, _manhattan_path,
    heuristic, _extend_path_log, grid_distance,
    nearest_base_to_station, _diagnose_and_print_failure, decode_schedule, decode_schedule_tick_by_tick, fitness, local_improve,
    plot_gantt, station_key_from_value, load_dispatch_events, make_jobs
)

# Overwrite describe_solution locally to pass save_img
def describe_solution_attention(individual: Individual, jobs: List[Job], solve_time: float = None, show_gantt: bool = False, save_img: str = None) -> Tuple[float, float]:
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

# ===== Attention Architecture =====
class SchedulerAttention(nn.Module):
    def __init__(self, amr_in_dim=8, job_in_dim=11, hidden_dim=128, attention_layers=2):
        super().__init__()
        self.amr_emb = nn.Linear(amr_in_dim, hidden_dim)
        self.job_emb = nn.Linear(job_in_dim, hidden_dim)

        self.attention_layers = attention_layers

        # Separate self-attention modules
        self.amr_self_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
            for _ in range(attention_layers)
        ])
        self.job_self_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
            for _ in range(attention_layers)
        ])

        # Separate cross-attention modules
        self.amr_to_job_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
            for _ in range(attention_layers)
        ])
        self.job_to_amr_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
            for _ in range(attention_layers)
        ])
        
        # Separate FFNs for AMRs and Jobs
        self.fc_amr = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(attention_layers)
        ])
        self.fc_job = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(attention_layers)
        ])

        # Policy Head: Takes concatenated (AMR, Job) embeddings and outputs logit
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Critic Head: estimates state value V(s_t) from pooled AMR and job embeddings.
        self.critic = nn.Sequential(
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

        num_amrs = x_amr.size(1)
        num_jobs = x_job.size(1)
        
        # Prevent NaN in key_padding_mask if all jobs are masked
        if job_mask.all():
            job_attn_mask = None
        else:
            job_attn_mask = job_mask

        # 2. Heterogeneous Message Passing
        for i in range(self.attention_layers):
            # a. Self-Attention
            amr_self, _ = self.amr_self_attn[i](x_amr, x_amr, x_amr)
            x_amr = x_amr + amr_self
            
            job_self, _ = self.job_self_attn[i](x_job, x_job, x_job, key_padding_mask=job_attn_mask)
            x_job = x_job + job_self
            
            # b. Cross-Attention (Distinct Q, K, V for each entity pair)
            # AMRs query Jobs
            amr_cross, _ = self.amr_to_job_attn[i](x_amr, x_job, x_job, key_padding_mask=job_attn_mask)
            # Jobs query AMRs
            job_cross, _ = self.job_to_amr_attn[i](x_job, x_amr, x_amr)
            
            x_amr = x_amr + amr_cross
            x_job = x_job + job_cross
            
            # c. Separate Feed-Forward Updates
            x_amr = x_amr + self.fc_amr[i](x_amr)
            x_job = x_job + self.fc_job[i](x_job)
            
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

    def forward_critic(self, amr_features, job_features, job_mask):
        """
        Returns scalar state value V(s_t).
        """
        x_amr = self.amr_emb(amr_features)
        x_job = self.job_emb(job_features)

        if job_mask.all():
            job_attn_mask = None
        else:
            job_attn_mask = job_mask

        for i in range(self.attention_layers):
            amr_self, _ = self.amr_self_attn[i](x_amr, x_amr, x_amr)
            x_amr = x_amr + amr_self

            job_self, _ = self.job_self_attn[i](x_job, x_job, x_job, key_padding_mask=job_attn_mask)
            x_job = x_job + job_self

            amr_cross, _ = self.amr_to_job_attn[i](x_amr, x_job, x_job, key_padding_mask=job_attn_mask)
            job_cross, _ = self.job_to_amr_attn[i](x_job, x_amr, x_amr)

            x_amr = x_amr + amr_cross
            x_job = x_job + job_cross

            x_amr = x_amr + self.fc_amr[i](x_amr)
            x_job = x_job + self.fc_job[i](x_job)

        mask_float = (~job_mask).float().unsqueeze(-1)
        num_unmasked = mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        job_pool = (x_job * mask_float).sum(dim=1) / num_unmasked.squeeze(-1)
        amr_pool = x_amr.mean(dim=1)

        combined = torch.cat([job_pool, amr_pool], dim=-1)
        return self.critic(combined).squeeze(-1)


def extract_state(jobs, assigned_jobs_set, amr_positions, amr_availabilities, amr_inventory, amr_assignment_map=None):
    """
    Constructs the tensor inputs for the Attention model.
    Returns amr_features, job_features, job_mask defined as:
    
    AMR Features (8):
     0: status_val (1.0 if avail > min_avail else 0.0)
     1: (avail - min_avail) / 50.0 (remaining time proxy)
     2: inventory A ratio (inv.A / 3.0)
     3: inventory B ratio (inv.B / 3.0)
     4: inventory C ratio (inv.C / 3.0)
     5: pos_x / 10.0
     6: pos_y / 10.0
     7: queue_depth (jobs assigned to this AMR / 10.0)
     
    Job Features (11):
     0: exist_flag (1.0)
     1: duration / 25.0
     2: wait_time (0.0)
     3: dest_pos_x / 10.0
     4: dest_pos_y / 10.0
     5: supply_pos_x / 10.0
     6: supply_pos_y / 10.0
     7: type A flag (1.0 or 0.0)
     8: type B flag (1.0 or 0.0)
     9: type C flag (1.0 or 0.0)
     10: job_status (1.0 if in assigned_jobs_set else 0.0)
     
    Job Mask: boolean array of size len(jobs), True if job inside assigned_jobs
    """
    
    # AMR Features
    amr_feat = []
    min_avail = min(amr_availabilities.values())
    
    for idx, amr in enumerate(AMR_KEYS):
        pos = amr_positions[amr]
        avail = amr_availabilities[amr]
        inv = amr_inventory[amr]
        
        status_val = 1.0 if avail > min_avail else 0.0
        rem = avail - min_avail
        
        if amr_assignment_map is not None:
            queue_depth = sum(1 for j_idx, a in amr_assignment_map.items() if a == amr) / 10.0
        else:
            queue_depth = 0.0
            
        feat = [
            status_val,
            rem / 50.0,
            inv.get("A", 0) / 3.0,
            inv.get("B", 0) / 3.0,
            inv.get("C", 0) / 3.0,
            pos[0] / 10.0,
            pos[1] / 10.0,
            queue_depth
        ]
        amr_feat.append(feat)
        
    # Job Features
    job_feat = []
    job_mask = []
    for job in jobs:
        pos = STATIONS[job.station]
        supply_pos = SUPPLY_LOCATIONS[job.type_]
        
        is_assigned = job.idx in assigned_jobs_set
        job_status = 1.0 if is_assigned else 0.0
        
        feat = [
            1.0,  # exist_flag
            job.duration / 25.0,
            0.0,  # wait_time
            pos[0] / 10.0,
            pos[1] / 10.0,
            supply_pos[0] / 10.0,
            supply_pos[1] / 10.0,
            1.0 if job.type_ == "A" else 0.0,
            1.0 if job.type_ == "B" else 0.0,
            1.0 if job.type_ == "C" else 0.0,
            job_status
        ]
        job_feat.append(feat)
        job_mask.append(is_assigned)
        
    return (
        torch.tensor([amr_feat], dtype=torch.float32), 
        torch.tensor([job_feat], dtype=torch.float32),
        torch.tensor([job_mask], dtype=torch.bool)
    )

# ===== Attention Scheduling Heuristic Loop =====
def solve_with_attention(jobs, model, deterministic=True, init_state: dict = None):
    """
    Uses the Attention model to autoregressively build a schedule.
    Returns: The final Individual, the total log probabilities of the sequence, and the total execution time of the building process.
    """
    start_time = time.perf_counter()
    
    # Internal Simulator State (Tracking rough time/inventory during sequence building)
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
            jobs, assigned_jobs_set, amr_positions, amr_availabilities, amr_inventory, amr_assignment_map
        )
        
        # Move tensors to the model's device
        device = next(model.parameters()).device
        amr_feat = amr_feat.to(device)
        job_feat = job_feat.to(device)
        job_mask = job_mask.to(device)
        
        # 2. Forward Pass
        logits = model(amr_feat, job_feat, job_mask) # shape: (1, 3, num_jobs)
        
        # 3. Action Selection
        flat_logits = logits.view(-1)
        
        if deterministic:
            best_action = torch.argmax(flat_logits).item()
        else:
            # Stochastic sampling (e.g. for REINFORCE training)
            log_probs = F.log_softmax(flat_logits, dim=0)
            probs = torch.exp(log_probs)
            
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
        
        # 4. Update Internal State (Fast Approximation for the next step of inference)
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
        
        # Wait for station and process
        process_start = max(avail, station_availabilities[chosen_job.station])
        process_end = process_start + chosen_job.duration
        amr_inventory[chosen_amr][mat] -= 1
        
        # Update station availability before returning home.
        station_availabilities[chosen_job.station] = process_end
        
        # Return home after each job to clear the station in the fast rollout.
        home_pos = AMR_STARTS[chosen_amr]
        return_end = process_end + heuristic(target_station, home_pos)
        amr_availabilities[chosen_amr] = return_end
        amr_positions[chosen_amr] = home_pos
            
    # Finalize Individual (Order amr_assignment list by job_idx)
    final_assignment = []
    for i in range(len(jobs)):
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
    parser.add_argument("--output_csv", type=str, default="attention_summary_results.csv", help="Output CSV filename")
    args = parser.parse_args()
    
    # 1. Setup environment and seed 
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 2. Create Model
    attention_model = SchedulerAttention(amr_in_dim=8, job_in_dim=11, hidden_dim=128, attention_layers=2)
    
    weights_path = Path("attention_scheduler_best.pth")
    if not weights_path.exists():
        # Fallback support for loading old checkpoints if they exist
        old_weights = Path("gnn_scheduler_best.pth")
        if old_weights.exists():
            weights_path = old_weights
            
    if weights_path.exists():
        print(f"Loading trained weights from {weights_path}...")
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            # Handle key renamings from SchedulerGNN checkpoints to SchedulerAttention
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("gnn_layers", "attention_layers")
                new_state_dict[new_key] = v
            attention_model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print(f"WARNING: Could not load weights from {weights_path} due to shape/feature mismatch: {e}")
            print("Proceeding with randomly initialized weights (recommend retraining with train.py).")
    
    # 3. Load Jobs
    if args.inbox:
        dispatch_events = load_dispatch_events(Path(args.inbox))
    else:
        dispatch_events = load_dispatch_events()
    target_index = os.environ.get(DISPATCH_EVENT_INDEX_ENV)
    
    output_filename = args.output_csv
    results_data = []

    print("=== Using Attention Logic ===")

    if dispatch_events:
        if target_index is not None:
             dispatch_events = [e for e in dispatch_events if str(e["index"]) == str(target_index)]
        
        for event in dispatch_events:
            print(f"\n=== Processing Dispatch Event {event['index']} (Jobs: {len(event['jobs'])}) ===")
            
            # a. Run Attention Inference
            best_ind, _, solve_dur_ns = solve_with_attention(event["jobs"], attention_model)
            
            # b. Apply Local Improve for routing/collision adjustment exactly identically to GA.py
            improve_start = time.perf_counter()
            best_ind = local_improve(best_ind, event["jobs"], max_iters=routing_iters)
            if args.collision_iters > 0:
                best_ind = local_improve(best_ind, event["jobs"], max_iters=args.collision_iters, check_collision=True)
            solve_dur_ns += (time.perf_counter() - improve_start)
            
            # c. Evaluate with Exact GA routing logic
            img_path = f"{args.save_img.split('.')[0]}_{event['index']}.png" if args.save_img else None
            makespan, computation_time = describe_solution_attention(best_ind, event["jobs"], solve_time=solve_dur_ns, show_gantt=args.gantt, save_img=img_path)
            
            results_data.append([event['index'], f"{makespan:.2f}", f"{computation_time:.4f}"])
    else:
        print("No dispatch file found. Generating random jobs...")
        jobs = make_jobs()
        
        # a. Run Attention Inference
        best_ind, _, solve_dur_ns = solve_with_attention(jobs, attention_model)
        
        # b. Apply Local Improve for routing/collision adjustment exactly identically to GA.py
        improve_start = time.perf_counter()
        best_ind = local_improve(best_ind, jobs, max_iters=routing_iters)
        if args.collision_iters > 0:
            best_ind = local_improve(best_ind, jobs, max_iters=args.collision_iters, check_collision=True)
        solve_dur_ns += (time.perf_counter() - improve_start)
        
        # c. Evaluate with Exact GA routing logic
        makespan, computation_time = describe_solution_attention(best_ind, jobs, solve_time=solve_dur_ns, show_gantt=args.gantt, save_img=args.save_img)
        
        results_data.append(["random", f"{makespan:.2f}", f"{computation_time:.4f}"])

    # 4. Save metrics
    if results_data:
        with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Event_Index", "Makespan", "Computation_Time"])
            writer.writerows(results_data)
        print(f"\nSummary results saved to {output_filename}")
