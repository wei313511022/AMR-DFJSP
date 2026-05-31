import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from GNN_DDQN_V7 import GridEnv, CONFIG

GLOBAL_SEED = 42

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Ensure we use the correct device
CONFIG['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG['DATASET_PATH'] = "test_dataset_r1.jsonl"

def episode_metrics(env):
    """Calculate metrics for the finished episode."""
    total_jobs = len(getattr(env, "completed_jobs", [])) + len(getattr(env, "active_jobs", [])) + len(getattr(env, "queue", []))

    if total_jobs == 0:
        return 0.0, 1000.0, 1000.0

    finished_in = sum(1 for j in env.completed_jobs if getattr(j, "finish_ts", 1000.0) <= CONFIG['SIM_TIME'])
    done_pct = (finished_in / total_jobs) * 100.0

    flows = [j.finish_ts - j.arrival_ts for j in env.completed_jobs if j.finish_ts >= 0]
    flow = float(np.mean(flows)) if flows else 1000.0

    # Makespan: last completion - first arrival (completed set)
    if env.completed_jobs:
        first_arr = min(j.arrival_ts for j in env.completed_jobs)
        last_fin  = max(j.finish_ts  for j in env.completed_jobs)
        mk = float(last_fin - first_arr)
    else:
        mk = 1000.0

    return done_pct, flow, mk

def run_one_episode_fix(env, period=5.0):
    """
    Run one episode using a fixed-period rescheduling strategy.
    
    Args:
        env: The GridEnv instance.
        period: Time in simulation seconds between reschedule attempts.
    """
    state = env.reset()
    total_ga = 0.0
    last_fix = -1e9

    while True:
        # Check if rescheduling is allowed by the environment (cooldowns, etc.)
        try:
            mask = env.get_action_mask()
            can_reschedule = mask[1] > 0.5
        except Exception:
            can_reschedule = True

        # Strategy: Reschedule if 'period' has passed since last time AND env allows it
        if (env.sim_time - last_fix >= period) and can_reschedule:
            action = 1
            last_fix = env.sim_time
        else:
            action = 0

        # Step the environment
        state, _, done, _ = env.step(action)
        
        # Accumulate GA computation time (simulated or real)
        total_ga += getattr(env, "last_ga_compute_time", 0.0)

        if done:
            break

    done_cnt, flow, mk = episode_metrics(env)
    return done_cnt, flow, mk, total_ga

def run_comparison():
    set_seed(GLOBAL_SEED)  # Global seed for reproducibility
    # Initialize Environment
    env = GridEnv()
    
    # Configuration
    PERIODS = [1.0, 5.0, 10.0, 20.0, 50.0, 70.0, 100.0, 120.0]
    # TEST_EPISODES = len(env.episodes)
    TEST_EPISODES = 10
    CSV_FILENAME = "fix_period_comparison.csv"
    PLOT_FILENAME = "fix_period_comparison_flow.png"
    
    # Storage for results
    # Structure: results[period][metric_name] = [val_ep0, val_ep1, ...]
    results = {p: {"done": [], "flow": [], "mk": [], "ga": []} for p in PERIODS}
    
    print(f"Starting comparison...")
    print(f"Periods to test: {PERIODS}")
    print(f"Episodes per period: {TEST_EPISODES}")
    print("-" * 60)
    
    for ep in range(TEST_EPISODES):
        if (ep + 1) % 5 == 0:
            print(f"Processing Episode {ep+1}/{TEST_EPISODES}...")
            
        for p in PERIODS:
            # Important: Set the episode index explicitly before reset 
            # to ensure all periods run on the exact same job scenario.
            set_seed(GLOBAL_SEED + ep * 1000 + int(p))  # Deterministic seed per episode+period
            env.ep_idx = ep
            
            done, flow, mk, ga = run_one_episode_fix(env, period=p)
            
            results[p]["done"].append(done)
            results[p]["flow"].append(flow)
            results[p]["mk"].append(mk)
            results[p]["ga"].append(ga)
        
            
    # --- 1. Save Detailed Results to CSV ---
    with open(CSV_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header: Episode, P1_Done, P1_Flow, ..., P5_Done, ...
        header = ["Episode"]
        for p in PERIODS:
            header.extend([f"P{p}_Done", f"P{p}_Flow", f"P{p}_MK", f"P{p}_GA"])
        writer.writerow(header)
        
        for i in range(TEST_EPISODES):
            row = [i]
            for p in PERIODS:
                row.extend([
                    results[p]["done"][i],
                    results[p]["flow"][i],
                    results[p]["mk"][i],
                    results[p]["ga"][i]
                ])
            writer.writerow(row)
            
    print(f"\nDetailed results saved to {CSV_FILENAME}")

    # --- 2. Print Summary Table ---
    print("\n" + "="*90)
    print(f"{'Period (s)':<12} | {'Avg Done(%)':<12} | {'Avg Flow':<12} | {'Avg Makespan':<15} | {'Avg GA Time (s)':<15}")
    print("-" * 90)
    
    for p in PERIODS:
        avg_done = np.mean(results[p]["done"])
        avg_flow = np.mean(results[p]["flow"])
        avg_mk = np.mean(results[p]["mk"])
        avg_ga = np.mean(results[p]["ga"])
        print(f"{p:<12} | {avg_done:<12.2f} | {avg_flow:<12.2f} | {avg_mk:<15.2f} | {avg_ga:<15.2f}")
    print("="*90)

    # --- 3. Generate Plot ---
    plt.figure(figsize=(12, 6))
    
    # Plot Flow Time
    for p in PERIODS:
        plt.plot(results[p]["flow"], label=f"Period={p}s", marker='o', markersize=4, alpha=0.7)
    
    plt.title(f"Flow Time Comparison over {TEST_EPISODES} Episodes")
    plt.xlabel("Episode Index")
    plt.ylabel("Average Flow Time (s)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(PLOT_FILENAME)
    print(f"Comparison plot saved to {PLOT_FILENAME}")

if __name__ == "__main__":
    run_comparison()
