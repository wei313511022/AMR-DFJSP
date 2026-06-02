import torch
import torch.optim as optim
import time
import numpy as np
import random
import matplotlib.pyplot as plt

# Import from the Attention_precise script
from Attention_precise import (
    SchedulerAttention, solve_with_attention, extract_state,
    AMR_KEYS, STATIONS, SUPPLY_LOCATIONS, TYPE_DURATION, shortest_path, heuristic, AMR_STARTS
)
from GA.GA import make_jobs, describe_solution, local_improve, routing_iters, collision_routing_iters, find_dynamic_path, MAX_DEPTH
import torch.nn.functional as F

def evaluate_actions(jobs, model, action_seq, init_state: dict = None):
    """
    Computes the log probabilities of the specified action sequence under the current model.
    action_seq: List of selected actions (integers)
    """
    if init_state:
        amr_positions = {amr: init_state["positions"].get(amr, AMR_STARTS[amr]) for amr in AMR_KEYS}
        amr_availabilities = {amr: float(init_state["availability"].get(amr, 0.0)) for amr in AMR_KEYS}
        station_availabilities = {s: float(init_state["time"]) for s in STATIONS.keys()}
        amr_inventory = {amr: init_state["inventory"].get(amr, {mat: 0 for mat in TYPE_DURATION.keys()}).copy() for amr in AMR_KEYS}
        amr_states = {amr: (amr_positions[amr], amr_availabilities[amr]) for amr in AMR_KEYS}
        reservations = {}
    else:
        amr_positions = {amr: AMR_STARTS[amr] for amr in AMR_KEYS}
        amr_availabilities = {amr: 0.0 for amr in AMR_KEYS}
        station_availabilities = {s: 0.0 for s in STATIONS.keys()}
        amr_inventory = {amr: {mat: 0 for mat in TYPE_DURATION.keys()} for amr in AMR_KEYS}
        amr_inventory["AMR1"]["A"] = 3
        amr_inventory["AMR2"]["B"] = 3
        amr_inventory["AMR3"]["C"] = 3
        amr_states = {amr: (AMR_STARTS[amr], 0.0) for amr in AMR_KEYS}
        reservations = {}

    assigned_jobs_set = set()
    amr_assignment_map = {}
    
    log_probs = []
    
    # Must keep model in train mode to track gradients during backprop
    model.train()
    
    for step in range(len(jobs)):
        amr_feat, job_feat, job_mask = extract_state(
            jobs, assigned_jobs_set, amr_positions, amr_availabilities, amr_inventory, amr_assignment_map
        )
        
        device = next(model.parameters()).device
        amr_feat = amr_feat.to(device)
        job_feat = job_feat.to(device)
        job_mask = job_mask.to(device)
        
        logits = model(amr_feat, job_feat, job_mask) # (1, 3, num_jobs)
        flat_logits = logits.view(-1)
        
        step_log_probs = F.log_softmax(flat_logits, dim=0)
        
        chosen_action = action_seq[step]
        log_probs.append(step_log_probs[chosen_action])
        
        num_jobs = len(jobs)
        amr_idx = chosen_action // num_jobs
        job_list_idx = chosen_action % num_jobs
        
        chosen_amr = AMR_KEYS[amr_idx]
        chosen_job = jobs[job_list_idx]
        
        amr_assignment_map[chosen_job.idx] = chosen_amr
        assigned_jobs_set.add(chosen_job.idx)
        
        # Precise Dynamic Pathfinding & Reservations
        material = chosen_job.type_
        start_time = amr_availabilities[chosen_amr]
        
        # Check supply
        if amr_inventory[chosen_amr][material] == 0:
            supply_location = SUPPLY_LOCATIONS[material]
            supply_path = find_dynamic_path(amr_positions[chosen_amr], supply_location, start_time, reservations, amr_states, chosen_amr)
            supply_time = int(len(supply_path) - 1)
            supply_end = start_time + supply_time + TYPE_DURATION[material]
            if supply_time == 0 and amr_positions[chosen_amr] != supply_location:
                supply_time = MAX_DEPTH
                supply_end = start_time + supply_time
            
            # Reserve path
            for t_offset, pt in enumerate(supply_path):
                reservations[(pt, int(start_time) + t_offset)] = chosen_amr
            # Reserve supply location during loading
            for t_refill in range(int(start_time) + supply_time, int(supply_end) + 1):
                reservations[(supply_location, t_refill)] = chosen_amr
                
            amr_states[chosen_amr] = (supply_location, supply_end)
            amr_availabilities[chosen_amr] = supply_end
            amr_positions[chosen_amr] = supply_location
            start_time = supply_end
            amr_inventory[chosen_amr][material] = 3 # Refill
            
        # Travel to station
        travel_start = amr_availabilities[chosen_amr]
        travel_path = find_dynamic_path(amr_positions[chosen_amr], STATIONS[chosen_job.station], travel_start, reservations, amr_states, chosen_amr)
        travel_time = int(len(travel_path) - 1)
        if travel_time == 0 and amr_positions[chosen_amr] != STATIONS[chosen_job.station]:
            travel_time = MAX_DEPTH
        travel_end = travel_start + travel_time
        
        # Reserve path
        for t_offset, pt in enumerate(travel_path):
            reservations[(pt, int(travel_start) + t_offset)] = chosen_amr
            
        amr_availabilities[chosen_amr] = travel_end
        amr_positions[chosen_amr] = STATIONS[chosen_job.station]
        
        # Wait for station and process
        earliest_start = max(travel_end, station_availabilities[chosen_job.station])
        process_start = earliest_start
        process_end = process_start + chosen_job.duration
        
        # Reserve station during wait and process
        for t_process in range(int(travel_end), int(process_end) + 1):
            reservations[(STATIONS[chosen_job.station], t_process)] = chosen_amr
            
        amr_states[chosen_amr] = (STATIONS[chosen_job.station], process_end)
        amr_inventory[chosen_amr][material] -= 1
        station_availabilities[chosen_job.station] = process_end
        
        # Return to home base to clear the station
        return_start = process_end
        next_dest = AMR_STARTS[chosen_amr]
        return_path = find_dynamic_path(STATIONS[chosen_job.station], next_dest, return_start, reservations, amr_states, chosen_amr)
        return_time = int(len(return_path) - 1)
        if return_time == 0 and STATIONS[chosen_job.station] != next_dest:
            return_time = MAX_DEPTH
        return_end = return_start + return_time
        
        for t_offset, pt in enumerate(return_path):
            reservations[(pt, int(return_start) + t_offset)] = chosen_amr
            
        amr_states[chosen_amr] = (next_dest, return_end)
        amr_availabilities[chosen_amr] = return_end
        amr_positions[chosen_amr] = next_dest
        
    return torch.stack(log_probs).sum()

def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Hyperparameters
    num_epochs = 10000
    batch_size = 16 # Number of episodes (schedules) to sample before a weight update
    lr = 1e-3
    
    # PPO Hyperparameters
    ppo_epochs = 4
    clip_eps = 0.2
    
    # Load test case if specified
    from GA.GA import load_dispatch_events
    dispatch_events = []
    if args.inbox:
        from pathlib import Path
        inbox_path = Path(args.inbox)
        if inbox_path.exists():
            dispatch_events = load_dispatch_events(inbox_path)
            print(f"Loaded {len(dispatch_events)} dispatch events from {args.inbox}")
    
    # Initialize Model and Optimizer
    attention_model = SchedulerAttention(amr_in_dim=8, job_in_dim=11, hidden_dim=128, attention_layers=2).to(device)
    optimizer = optim.Adam(attention_model.parameters(), lr=lr)
    
    print("Starting PPO training with Self-Critical Baseline...")
    
    best_makespan = float('inf')
    losses = []
    makespans = []
    
    for epoch in range(1, num_epochs + 1):
        batch_makespans = []
        trajectories = []
        
        # 1. Rollout Phase (Data Collection)
        for b in range(batch_size):
            if dispatch_events:
                event = random.choice(dispatch_events)
                jobs = event["jobs"]
            else:
                jobs = make_jobs()
            
            # Stochastic Rollout
            ind, old_log_prob, solve_dur = solve_with_attention(jobs, attention_model, deterministic=False)
            
            # Apply Local Improve
            improve_start = time.perf_counter()
            ind = local_improve(ind, jobs, max_iters=routing_iters)
            if collision_routing_iters > 0:
                ind = local_improve(ind, jobs, max_iters=collision_routing_iters, check_collision=True)
            solve_dur += (time.perf_counter() - improve_start)
            
            stochastic_makespan, _ = describe_solution(ind, jobs, solve_time=solve_dur, show_gantt=False)
            batch_makespans.append(stochastic_makespan)
            
            # Greedy Rollout
            with torch.no_grad():
                greedy_ind, _, greedy_solve_dur = solve_with_attention(jobs, attention_model, deterministic=True)
                
                greedy_improve_start = time.perf_counter()
                greedy_ind = local_improve(greedy_ind, jobs, max_iters=routing_iters)
                if collision_routing_iters > 0:
                    greedy_ind = local_improve(greedy_ind, jobs, max_iters=collision_routing_iters, check_collision=True)
                greedy_solve_dur += (time.perf_counter() - greedy_improve_start)
                
                greedy_makespan, _ = describe_solution(greedy_ind, jobs, solve_time=greedy_solve_dur, show_gantt=False)
            
            # Advantage computation
            advantage = greedy_makespan - stochastic_makespan
            
            # Reconstruct action indices
            job_id_to_list_idx = {job.idx: idx for idx, job in enumerate(jobs)}
            action_seq = []
            for job_id, amr in zip(ind.order, ind.amr_assignment):
                job_idx = job_id_to_list_idx[job_id]
                amr_idx = AMR_KEYS.index(amr)
                action_seq.append(amr_idx * len(jobs) + job_idx)
                
            trajectories.append((jobs, action_seq, old_log_prob.detach(), advantage))
            
        # 2. PPO Optimization Phase
        epoch_loss = 0.0
        for ppo_epoch in range(ppo_epochs):
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for jobs, action_seq, old_log_prob, advantage in trajectories:
                new_log_prob = evaluate_actions(jobs, attention_model, action_seq)
                
                ratio = torch.exp(new_log_prob - old_log_prob)
                
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
                
                loss = -torch.min(surr1, surr2)
                loss_scaled = loss / batch_size
                loss_scaled.backward()
                batch_loss += loss.item()
                
            optimizer.step()
            epoch_loss += batch_loss / batch_size
            
        epoch_loss /= ppo_epochs
        
        avg_batch_makespan = sum(batch_makespans) / batch_size
        losses.append(epoch_loss)
        makespans.append(avg_batch_makespan)
        
        # Logging
        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] | Avg Makespan: {avg_batch_makespan:.2f} | Total Loss: {epoch_loss:.4f}")
            
        # Optional: Save best model
        if avg_batch_makespan < best_makespan:
            best_makespan = avg_batch_makespan
            torch.save(attention_model.state_dict(), "attention_precise_scheduler_best.pth")
            print(f"   -> Saved new best model (Makespan: {best_makespan:.2f})")
            
        # Save training loss/makespan chart periodically
        if epoch == 1 or epoch % 100 == 0 or epoch == num_epochs:
            plt.figure(figsize=(12, 5))
            
            # Panel 1: Loss
            plt.subplot(1, 2, 1)
            plt.plot(losses, color='#1f77b4', linewidth=1.5, label='Epoch Loss')
            plt.title('Attention Policy Training Loss', fontsize=12, fontweight='bold', pad=10)
            plt.xlabel('Epoch', fontsize=10)
            plt.ylabel('Loss', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            
            # Panel 2: Average Makespan
            plt.subplot(1, 2, 2)
            plt.plot(makespans, color='#ff7f0e', linewidth=1.5, label='Avg Makespan (s)')
            plt.title('Average Batch Makespan Trend', fontsize=12, fontweight='bold', pad=10)
            plt.xlabel('Epoch', fontsize=10)
            plt.ylabel('Makespan (s)', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            
            import os
            SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
            chart_path = os.path.join(SCRIPT_DIR, "attention_precise_training_metrics.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   -> Saved updated training chart to {chart_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file to train on (e.g., ../../test_case/dispatch_inbox_60.jsonl)")
    args = parser.parse_args()
    
    train(args)
