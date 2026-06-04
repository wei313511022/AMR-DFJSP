import torch
import torch.optim as optim
import time
import numpy as np
import random
import matplotlib.pyplot as plt

# Import from the Attention script
from Attention import (
    SchedulerAttention, solve_with_attention, extract_state,
    AMR_KEYS, STATIONS, SUPPLY_LOCATIONS, TYPE_DURATION, heuristic, AMR_STARTS
)
from GA.GA import make_jobs, describe_solution, local_improve, routing_iters, collision_routing_iters
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
    else:
        amr_positions = {amr: AMR_STARTS[amr] for amr in AMR_KEYS}
        amr_availabilities = {amr: 0.0 for amr in AMR_KEYS}
        station_availabilities = {s: 0.0 for s in STATIONS.keys()}
        amr_inventory = {amr: {mat: 0 for mat in TYPE_DURATION.keys()} for amr in AMR_KEYS}
        amr_inventory["AMR1"]["A"] = 3
        amr_inventory["AMR2"]["B"] = 3
        amr_inventory["AMR3"]["C"] = 3

    assigned_jobs_set = set()
    amr_assignment_map = {}
    
    log_probs = []
    values = []
    
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
        
        value = model.forward_critic(amr_feat, job_feat, job_mask)
        values.append(value.squeeze())

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
        
        mat = chosen_job.type_
        curr_pos = amr_positions[chosen_amr]
        avail = amr_availabilities[chosen_amr]
        
        if amr_inventory[chosen_amr][mat] == 0:
            supply_loc = SUPPLY_LOCATIONS[mat]
            avail += heuristic(curr_pos, supply_loc)
            curr_pos = supply_loc
            amr_inventory[chosen_amr][mat] = 3
            
        target_station = STATIONS[chosen_job.station]
        avail += heuristic(curr_pos, target_station)
        process_start = max(avail, station_availabilities[chosen_job.station])
        process_end = process_start + chosen_job.duration
        amr_inventory[chosen_amr][mat] -= 1
        
        station_availabilities[chosen_job.station] = process_end
        
        home_pos = AMR_STARTS[chosen_amr]
        return_end = process_end + heuristic(target_station, home_pos)
        amr_availabilities[chosen_amr] = return_end
        amr_positions[chosen_amr] = home_pos
        
    return torch.stack(log_probs).sum(), torch.stack(values)

def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Hyperparameters
    num_epochs = args.epochs
    batch_size = 16 # Number of episodes (schedules) to sample before a weight update
    lr = 1e-3
    
    # PPO Hyperparameters
    ppo_epochs = 4
    clip_eps = 0.2
    value_loss_coef = 0.5
    
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
    
    print("Starting PPO training with Critic-Based Advantage...")
    
    best_makespan = float('inf')
    losses_actor = []
    losses_critic = []
    losses_total = []
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
            
            # Reconstruct model-sampled action indices before local improvement mutates the schedule.
            job_id_to_list_idx = {job.idx: idx for idx, job in enumerate(jobs)}
            action_seq = []
            for job_id in ind.order:
                job_idx = job_id_to_list_idx[job_id]
                amr = ind.amr_assignment[job_id]
                amr_idx = AMR_KEYS.index(amr)
                action_seq.append(amr_idx * len(jobs) + job_idx)

            # Apply Local Improve
            improve_start = time.perf_counter()
            ind = local_improve(ind, jobs, max_iters=routing_iters)
            if collision_routing_iters > 0:
                ind = local_improve(ind, jobs, max_iters=collision_routing_iters, check_collision=True)
            solve_dur += (time.perf_counter() - improve_start)
            
            stochastic_makespan, _ = describe_solution(ind, jobs, solve_time=solve_dur, show_gantt=False)
            batch_makespans.append(stochastic_makespan)

            value_target = -stochastic_makespan
            trajectories.append((jobs, action_seq, old_log_prob.detach(), value_target))
            
        # 2. PPO Optimization Phase
        epoch_loss = 0.0
        epoch_critic_loss = 0.0
        epoch_total_loss = 0.0
        for ppo_epoch in range(ppo_epochs):
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_critic_loss = 0.0
            batch_total_loss = 0.0
            
            for jobs, action_seq, old_log_prob, value_target in trajectories:
                new_log_prob, values = evaluate_actions(jobs, attention_model, action_seq)
                value_target_tensor = torch.tensor(value_target, dtype=torch.float32, device=values.device)
                value_targets = value_target_tensor.expand_as(values)
                advantage = value_target_tensor - values.detach().mean()
                
                ratio = torch.exp(new_log_prob - old_log_prob)
                
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
                
                actor_loss = -torch.min(surr1, surr2)
                critic_loss = F.mse_loss(values, value_targets)
                total_loss = (actor_loss + value_loss_coef * critic_loss) / batch_size
                total_loss.backward()
                batch_loss += actor_loss.item()
                batch_critic_loss += critic_loss.item()
                batch_total_loss += (actor_loss + value_loss_coef * critic_loss).item()
                
            optimizer.step()
            epoch_loss += batch_loss / batch_size
            epoch_critic_loss += batch_critic_loss / batch_size
            epoch_total_loss += batch_total_loss / batch_size
            
        epoch_loss /= ppo_epochs
        epoch_critic_loss /= ppo_epochs
        epoch_total_loss /= ppo_epochs
        
        avg_batch_makespan = sum(batch_makespans) / batch_size
        losses_actor.append(epoch_loss)
        losses_critic.append(epoch_critic_loss)
        losses_total.append(epoch_total_loss)
        makespans.append(avg_batch_makespan)
        
        # Logging
        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] | Avg Makespan: {avg_batch_makespan:.2f} "
                  f"| Actor Loss: {epoch_loss:.4f} | Critic Loss: {epoch_critic_loss:.4f} "
                  f"| Total Loss: {epoch_total_loss:.4f}")
            
        # Optional: Save best model
        if avg_batch_makespan < best_makespan:
            best_makespan = avg_batch_makespan
            torch.save(attention_model.state_dict(), "attention_scheduler_best.pth")
            print(f"   -> Saved new best model (Makespan: {best_makespan:.2f})")
            
        # Save training loss/makespan chart periodically
        if epoch == 1 or epoch % 100 == 0 or epoch == num_epochs:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            axes[0, 0].plot(losses_actor, color='#e74c3c', linewidth=1.5, label='Actor Loss')
            axes[0, 0].set_title('Attention Actor Loss', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, linestyle='--', alpha=0.5)
            axes[0, 0].legend()

            axes[0, 1].plot(losses_critic, color='#2ecc71', linewidth=1.5, label='Critic Loss')
            axes[0, 1].set_title('Attention Critic Loss', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, linestyle='--', alpha=0.5)
            axes[0, 1].legend()

            axes[1, 0].plot(losses_total, color='#3498db', linewidth=1.5, label='Total PPO Loss')
            axes[1, 0].set_title('Total PPO Loss', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, linestyle='--', alpha=0.5)
            axes[1, 0].legend()

            axes[1, 1].plot(makespans, color='#ff7f0e', linewidth=1.5, label='Avg Makespan (s)')
            axes[1, 1].set_title('Average Batch Makespan Trend', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Makespan (s)')
            axes[1, 1].grid(True, linestyle='--', alpha=0.5)
            axes[1, 1].legend()
            
            import os
            SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
            chart_path = os.path.join(SCRIPT_DIR, "attention_training_metrics.png")
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   -> Saved updated training chart to {chart_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file to train on (e.g., ../../test_case/dispatch_inbox_60.jsonl)")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    args = parser.parse_args()
    
    train(args)
