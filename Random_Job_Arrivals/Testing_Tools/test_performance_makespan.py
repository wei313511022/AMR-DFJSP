import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import importlib
import sys
import math

GLOBAL_SEED = 42

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

GridEnv = None
SchedulerAgent = None
CONFIG = None


def load_agent(path):
    agent = SchedulerAgent().to(CONFIG['DEVICE'])
    agent.load_state_dict(torch.load(path, map_location=CONFIG['DEVICE']))
    agent.eval()
    return agent

def episode_metrics(env):
    total_jobs = len(getattr(env, "completed_jobs", [])) + len(getattr(env, "active_jobs", [])) + len(getattr(env, "queue", []))

    if total_jobs == 0:
        return 0.0, 1000.0, 1000.0

    finished_in = len(env.completed_jobs)
    done_pct = (finished_in / total_jobs) * 100.0

    flows = [j.finish_ts - j.arrival_ts for j in env.completed_jobs if j.finish_ts >= 0]
    flow = float(np.mean(flows)) if flows else 1000.0

    # Makespan: last completion timestamp (i.e. time when all jobs are finished)
    if env.completed_jobs:
        last_fin  = max(j.finish_ts for j in env.completed_jobs)
        mk = float(last_fin)
    else:
        mk = 10000.0

    return done_pct, flow, mk

def snapshot_env(env):
    """Return (active_ids, queue_len, done_ids)."""
    active_ids = [getattr(j, "jid", -1) for j in getattr(env, "active_jobs", [])]
    done_ids   = [getattr(j, "jid", -1) for j in getattr(env, "completed_jobs", [])]
    return active_ids, len(getattr(env, "queue", [])), done_ids


def get_local_queues(env):
    """Return list of local_queue contents per AMR. Supports both V5 and V6."""
    lqs = []
    if hasattr(env, "sim") and hasattr(env.sim, "amrs"):
        # V6 FactorySimulator
        for a in env.sim.amrs.values():
            q = [j.jid for j in getattr(a, "queue", [])]
            lqs.append(q)
    else:
        # V5 List of AMRs
        for a in getattr(env, "amrs", []):
            q = list(getattr(a, "local_queue", []))
            lqs.append(q)
    return lqs


def safe_job_mask_from_state(state, device):
    """
    Build job_mask for Attention padding / ghost job.
    Your state[1] is job features list.
    If it contains the 'ghost job' [0,0,0,...], mask them as 0.
    """
    jobs = state[1]
    if not jobs:
        # no jobs at all
        return torch.zeros((1, 1, 1), device=device)

    s_job = torch.tensor([jobs], dtype=torch.float32, device=device)
    # existence bit is feature[0] in your design
    exist = s_job[:, :, 0:1]  # [1, N, 1]
    job_mask = (exist > 0.0).float()
    return s_job, job_mask


def run_one_episode_ai(env, agent, verbose=True, print_q=False):
    """
    Runs one episode using agent policy.
    Prints:
      - [AI-DECIDE] decision-time info (before env.step)
      - [AI-AFTER]  after-step info including GA time + mask_after
    """
    state = env.reset()
    total_ga = 0.0
    step_i = 0

    while True:
        step_i += 1
        t_decide = float(env.sim_time)

        # --- build tensors ---
        s_amr = torch.tensor([state[0]], dtype=torch.float32, device=CONFIG['DEVICE'])
        # job tensor + ghost mask
        s_job, job_mask = safe_job_mask_from_state(state, CONFIG['DEVICE'])
        s_q   = torch.tensor([state[2]], dtype=torch.float32, device=CONFIG['DEVICE'])

        with torch.no_grad():
            if hasattr(agent, "q_values"):
                q = agent.q_values(s_amr, s_job, s_q, job_mask)
            else:
                q = agent(s_amr, s_job, s_q, job_mask)  # [1,2]

            # decision-time action mask (BEFORE step)
            amask = env.get_action_mask()  # [1.0, 0/1]
            if amask[1] < 0.5:
                q[0, 1] = -1e9

            action = int(q.argmax(1).item())

        # --- decision-time log (BEFORE step) ---
        if verbose:
            unstarted_now = sum(1 for j in env.active_jobs if getattr(j, "status", 0) == 1)
            if action == 1:
                msg = (f"[AI-DECIDE] t={t_decide:6.1f} step={step_i:4d} "
                       f"choose RESCHEDULE | active={len(env.active_jobs)} unstarted={unstarted_now} "
                       f"| mask={amask}")
                if print_q:
                    q_cpu = q.detach().cpu().numpy().tolist()
                    msg += f" | Q={q_cpu}"
                print(msg)

        # --- step ---
        next_state, _, done, _ = env.step(action)
        state = next_state

        # --- GA time accounting ---
        ga_sec = float(getattr(env, "last_ga_compute_time", 0.0))
        total_ga += ga_sec

        # --- AFTER-step log (show GA time + mask_after) ---
        if verbose and action == 1:
            unstarted_after = sum(1 for j in env.active_jobs if getattr(j, "status", 0) == 1)
            mask_after = env.get_action_mask()
            print(f"[AI-AFTER ] t={env.sim_time:6.1f} step={step_i:4d} "
                  f"GA={ga_sec*1000.0:7.2f} ms | active={len(env.active_jobs)} unstarted={unstarted_after} "
                  f"| mask_after={mask_after}")

        if done:
            break

    # return episode summary + total GA time (ms)
    done_cnt, flow, mk = episode_metrics(env)
    return done_cnt, flow, mk, total_ga



def run_one_episode_fix(env, period=5.0):
    state = env.reset()
    total_ga = 0.0
    last_fix = -1e9

    while True:
        try:
            can = env.get_action_mask()[1] > 0.5
        except Exception:
            can = True

        if (env.sim_time - last_fix >= period) and can:
            action = 1
            last_fix = env.sim_time
        else:
            action = 0

        state, _, done, _ = env.step(action)
        total_ga += getattr(env, "last_ga_compute_time", 0.0)

        if done:
            break

    done_cnt, flow, mk = episode_metrics(env)
    return done_cnt, flow, mk, total_ga

def run_test(model_path, output_csv):
    set_seed(GLOBAL_SEED)  # Global seed for reproducibility
    env = MakespanGridEnv()
    agent = load_agent(model_path)

    TEST_EPISODES = min(10, len(env.episodes))
    FIX_PERIOD = 10.0  # 固定時間 reschedule

    # --- CSV Setup ---
    csv_filename = output_csv
    csv_file = open(csv_filename, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["Episode", "AI_Done(%)", "Fix_Done(%)", "AI_Flow", "Fix_Flow", "AI_Makespan", "Fix_Makespan", "AI_GA_Time", "Fix_GA_Time", "Winner"])

    print("\nEp   | AI Done(%) Fix Done(%) | AI Flow  Fix Flow | AI MK    Fix MK   | AI GA(s)  Fix GA(s) | Winner")
    print("-" * 120)

    ai_flows, fix_flows = [], []
    ai_mks, fix_mks = [], []
    
    # Metrics storage
    metrics = {
        "ai_done": [], "fix_done": [],
        "ai_flow": [], "fix_flow": [],
        "ai_mk": [], "fix_mk": [],
        "ai_ga": [], "fix_ga": []
    }

    for ep in range(TEST_EPISODES):
        # --- AI ---
        set_seed(GLOBAL_SEED + ep * 1000)  # Deterministic seed per episode for AI
        env.ep_idx = ep % len(env.episodes)
        ai_done, ai_flow, ai_mk, ai_ga_ms = run_one_episode_ai(env, agent, verbose=True)

        # --- FIX (periodic reschedule) ---
        set_seed(GLOBAL_SEED + ep * 1000 + 500)  # Different but deterministic seed for Fix
        env.ep_idx = ep % len(env.episodes)
        fix_done, fix_flow, fix_mk, fix_ga_ms = run_one_episode_fix(env, period=FIX_PERIOD)

        ai_flows.append(ai_flow)
        fix_flows.append(fix_flow)
        ai_mks.append(ai_mk)
        fix_mks.append(fix_mk)
        
        metrics["ai_done"].append(ai_done)
        metrics["fix_done"].append(fix_done)
        metrics["ai_flow"].append(ai_flow)
        metrics["fix_flow"].append(fix_flow)
        metrics["ai_mk"].append(ai_mk)
        metrics["fix_mk"].append(fix_mk)
        metrics["ai_ga"].append(ai_ga_ms)
        metrics["fix_ga"].append(fix_ga_ms)

        # Winner: Makespan (Lower) > Flow (Lower) > GA overhead (Lower)
        if ai_mk < fix_mk:
            win = "AI"
        elif fix_mk < ai_mk:
            win = "Fix"
        else:
            if ai_flow < fix_flow:
                win = "AI"
            elif fix_flow < ai_flow:
                win = "Fix"
            else:
                win = "AI" if ai_ga_ms < fix_ga_ms else "Fix"

        print(f"{ep:<4} | {ai_done:<10.1f} {fix_done:<11.1f} | {ai_flow:<7.1f} {fix_flow:<8.1f} | "
              f"{ai_mk:<7.1f} {fix_mk:<8.1f} | {ai_ga_ms:<9.1f} {fix_ga_ms:<9.1f} | {win}")

        writer.writerow([ep, ai_done, fix_done, ai_flow, fix_flow, ai_mk, fix_mk, ai_ga_ms, fix_ga_ms, win])

    csv_file.close()

    # Beautiful side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Makespan
    ax1.plot(ai_mks, label="AI Makespan", marker="o", color="#1f77b4", linewidth=2)
    ax1.plot(fix_mks, label="Fix Makespan", marker="s", linestyle="--", color="#ff7f0e", linewidth=2)
    ax1.set_title("Makespan / Completion Time (Lower is Better)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Episode", fontsize=10)
    ax1.set_ylabel("Seconds", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    
    # Plot Flow Time
    ax2.plot(ai_flows, label="AI Flow Time", marker="o", color="#2ca02c", linewidth=2)
    ax2.plot(fix_flows, label="Fix Flow Time", marker="s", linestyle="--", color="#d62728", linewidth=2)
    ax2.set_title("Average Flow Time (Lower is Better)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Episode", fontsize=10)
    ax2.set_ylabel("Seconds", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    base_name = os.path.splitext(output_csv)[0]
    plot_filename = f"{base_name}.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    print(f"\nResults saved to {plot_filename}")
    print(f"Episode data saved to {csv_filename}")
    
    # Summary Table
    summary_filename = f"{base_name}_summary.csv"
    summary_file = open(summary_filename, "w", newline="")
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(["Metric", "AI Average", "Fix Average", "Diff (%)"])

    print("\n" + "="*80)
    print(f"{'Metric':<20} | {'AI Average':<15} | {'Fix Average':<15} | {'Diff (%)':<15}")
    print("-" * 80)
    
    for name, key_ai, key_fix in [
        ("Done Jobs (%)", "ai_done", "fix_done"),
        ("Flow Time", "ai_flow", "fix_flow"),
        ("Makespan", "ai_mk", "fix_mk"),
        ("GA Time (s)", "ai_ga", "fix_ga")
    ]:
        avg_ai = np.mean(metrics[key_ai])
        avg_fix = np.mean(metrics[key_fix])
        diff = ((avg_ai - avg_fix) / avg_fix) * 100 if avg_fix != 0 else 0.0
        print(f"{name:<20} | {avg_ai:<15.2f} | {avg_fix:<15.2f} | {diff:<15.2f}")
        summary_writer.writerow([name, f"{avg_ai:.2f}", f"{avg_fix:.2f}", f"{diff:.2f}"])
    print("="*80)
    summary_file.close()
    print(f"Summary table saved to {summary_filename}")
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run performance makespan test with a specific model.")
    parser.add_argument("--model", type=str, default="../models_pth/attention_ddqn_model_v7/attention_ddqn_model_v7.pth", help="Path to the model file")
    parser.add_argument("--output", type=str, default="../models_pth/attention_ddqn_model_v7/benchmark_makespan_results.csv", help="Path to the output CSV file")
    parser.add_argument("--module", type=str, default="models.Attention_DDQN_V7", help="Module to import GridEnv and SchedulerAgent from")
    args = parser.parse_args()

    # Ensure parent directory is in sys.path so we can import dynamically
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    try:
        mod = importlib.import_module(args.module)
        GridEnv = mod.GridEnv
        SchedulerAgent = mod.SchedulerAgent
        CONFIG = mod.CONFIG
    except ImportError as e:
        print(f"Error importing module {args.module}: {e}")
        sys.exit(1)

    # Subclass GridEnv to override simulation completion criteria
    class MakespanGridEnv(GridEnv):
        def step(self, action: int):
            # Call parent step
            state, reward, done, dt = super().step(action)
            # Force done to ONLY be true when all jobs are completed (or safety timeout 10000.0 is hit)
            done = (self.sim_time >= 10000.0) or (len(self.completed_jobs) >= self.total_jobs)
            return state, reward, done, dt

    CONFIG['DATASET_PATH'] = "../../test_case/dynamic/test_dataset_demo.jsonl"
    CONFIG['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_test(args.model, args.output)
