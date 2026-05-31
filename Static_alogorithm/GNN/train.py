import torch
import torch.optim as optim
import time
import numpy as np
import random

# Import from the GNN script
from GNN import SchedulerGNN, solve_with_gnn
from GA.GA import make_jobs, describe_solution, local_improve, routing_iters, collision_routing_iters

def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Hyperparameters
    num_epochs = 10000
    batch_size = 16 # Number of episodes (schedules) to sample before a weight update
    lr = 1e-3
    
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
    model = SchedulerGNN(amr_in_dim=8, job_in_dim=10, hidden_dim=128, gnn_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track the moving average of the makespan to use as a baseline for REINFORCE
    baselines = {}
    alpha = 0.1 # Exponential moving average rate
    
    print("Starting REINFORCE training...")
    
    best_makespan = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        epoch_loss = 0.0
        batch_makespans = []
        
        # We will train on random dispatch events
        for b in range(batch_size):
            scenario_id = "random"
            # Generate a new random job scenario or pick from dispatch events
            if dispatch_events:
                event = random.choice(dispatch_events)
                jobs = event["jobs"]
                scenario_id = event["index"]
            else:
                jobs = make_jobs()
            
            # Forward pass: Sample a schedule
            ind, total_log_prob, solve_dur = solve_with_gnn(jobs, model, deterministic=False)
            
            # Apply Local Improve
            improve_start = time.perf_counter()
            ind = local_improve(ind, jobs, max_iters=routing_iters)
            if collision_routing_iters > 0:
                ind = local_improve(ind, jobs, max_iters=collision_routing_iters, check_collision=True)
            solve_dur += (time.perf_counter() - improve_start)
            
            # Simulator evaluation: get the exact makespan (No gantt plot during training)
            # check_collision=True is implicitly called inside describe_solution
            makespan, _ = describe_solution(ind, jobs, solve_time=solve_dur, show_gantt=False)
            
            batch_makespans.append(makespan)
            
            # --- Policy Gradient Update ---
            
            # Initialize baseline
            if scenario_id not in baselines:
                baselines[scenario_id] = makespan
                
            # Reward: Positive if we beat the baseline makespan, negative if we did worse
            reward = baselines[scenario_id] - makespan
            
            # Loss = -log_prob * reward (We want to maximize reward, so minimize -reward)
            loss = -total_log_prob * reward
            
            # Accumulate loss for the batch
            loss.backward()
            epoch_loss += loss.item()
            
            # Update the moving average baseline for this scenario
            baselines[scenario_id] = (1 - alpha) * baselines[scenario_id] + (alpha * makespan)
            
        # Update weights based on the batch gradients
        optimizer.step()
        
        avg_batch_makespan = sum(batch_makespans) / batch_size
        
        # Logging
        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] | Avg Makespan: {avg_batch_makespan:.2f} | Total Loss: {epoch_loss:.4f}")
            
        # Optional: Save best model
        if avg_batch_makespan < best_makespan:
            best_makespan = avg_batch_makespan
            torch.save(model.state_dict(), "gnn_scheduler_best.pth")
            print(f"   -> Saved new best model (Makespan: {best_makespan:.2f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file to train on (e.g., ../../test_case/dispatch_inbox_60.jsonl)")
    args = parser.parse_args()
    
    train(args)
