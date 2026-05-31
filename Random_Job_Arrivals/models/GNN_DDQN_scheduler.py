#!/usr/bin/env python3
# GNN-DDQN Scheduler + Visual Simulation
# - Integrates PyTorch GNN Agent to decide dispatching times.
# - Simulates a "Shop Floor" with AMRs in the background to feed the GNN.

import math, random, json, time, os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 1. GNN + DDQN Agent Model (The "Brain") ---
CONFIG = {
    'AMR_IN_DIM': 8, 'JOB_IN_DIM': 10, 'QUEUE_DIM': 3,
    'HIDDEN_DIM': 64, 'ACTION_DIM': 3, 'DEVICE': 'cpu' # CPU for viz simulation
}

class HeteroGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.msg_j2a = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim))
        self.msg_a2j = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, in_dim))
        self.update_amr = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim))
        self.update_job = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim))
        
    def forward(self, h_amr, h_job):
        if h_job.size(0) == 0: job_summary = torch.zeros_like(h_amr)
        else: job_summary = h_job.mean(dim=0, keepdim=True).expand(h_amr.shape[0], -1)
        
        if h_amr.size(0) == 0: amr_summary = torch.zeros_like(h_job)
        else: amr_summary = h_amr.mean(dim=0, keepdim=True).expand(h_job.shape[0], -1)

        out_amr = self.update_amr(h_amr + self.msg_j2a(job_summary))
        out_job = self.update_job(h_job + self.msg_a2j(amr_summary))
        return out_amr, out_job

class SchedulerAgent(nn.Module):
    def __init__(self):
        super().__init__()
        h_dim = CONFIG['HIDDEN_DIM']
        self.enc_amr = nn.Linear(CONFIG['AMR_IN_DIM'], h_dim)
        self.enc_job = nn.Linear(CONFIG['JOB_IN_DIM'], h_dim)
        self.gnn = HeteroGNNLayer(h_dim, h_dim)
        
        # Dueling Heads
        self.val = nn.Sequential(nn.Linear(h_dim + CONFIG['QUEUE_DIM'], h_dim), nn.ReLU(), nn.Linear(h_dim, 1))
        self.adv = nn.Sequential(nn.Linear(h_dim + CONFIG['QUEUE_DIM'], h_dim), nn.ReLU(), nn.Linear(h_dim, CONFIG['ACTION_DIM']))

    def forward(self, x_amr, x_job, x_queue):
        h_amr = F.relu(self.enc_amr(x_amr))
        h_job = F.relu(self.enc_job(x_job))
        h_amr, _ = self.gnn(h_amr, h_job)
        
        # Global Shop State (Pool AMRs)
        graph_emb = h_amr.mean(dim=0, keepdim=True)
        state = torch.cat([graph_emb, x_queue.unsqueeze(0)], dim=-1)
        
        val = self.val(state)
        adv = self.adv(state)
        return val + (adv - adv.mean(dim=1, keepdim=True))

# --- 2. Simulation State & Config ---
AVG_INTERARRIVAL_TIME = 3.0
SIM_SPEED_MULTIPLIER   = 1.0
UPDATE_INTERVAL_MS     = 200
DECISION_INTERVAL_S    = 1.0  # Agent decides every 1.0 sim-second

LEFT_LABEL_PAD = 5.0
VIEW_WIDTH     = 40.0
AX_Y_MIN, AX_Y_MAX = 0.0, 2.0
TOP_Y_CENTER, TOP_LANE_H = 0.6, 0.75

STATION_COUNT  = 5
JOB_TYPES = {"A": {"time": 10}, "B": {"time": 15}, "C": {"time": 20}}
JOB_TYPE_KEYS = list(JOB_TYPES.keys())
COLORS = {"A": "C0", "B": "C1", "C": "C2"}

DISPATCH_INBOX = "dispatch_inbox.jsonl"
open(DISPATCH_INBOX, "w").close() # Reset

@dataclass
class Job:
    jid: int
    jtype: str
    proc_time: float
    arrival_ts: float
    station: int
    status: int = 0 # 0: Waiting, 1: Released

@dataclass
class AMR:
    id: int
    x: float = 0.0
    y: float = 0.0
    status: int = 0 # 0: Idle, 1: Busy

# Global Sim State
simulation_time: float = 0.0
job_counter: int = 0
next_arrival_time: float = 0.0
next_decision_time: float = 0.0
is_running: bool = False

# Collections
jobs_top: List[Job] = [] # Visual Queue
rects_top: List[Rectangle] = []
texts_top: List = []

# "Shop Floor" State (Invisible logic for GNN)
jobs_bot: List[Job] = [] # Active/Released Jobs
amrs: List[AMR] = [AMR(i) for i in range(3)] # 3 Mock AMRs
current_action_name: str = "INIT"

# Agent Instance
agent = SchedulerAgent()
# agent.load_state_dict(torch.load("scheduler_weights.pth")) # Load trained weights here

# --------------------------- Helpers ---------------------------
def get_state_tensors():
    """Convert Sim State to PyTorch Tensors for the Agent"""
    # 1. AMR Tensor [N, 8] (Mocking random inventory for demo)
    amr_list = []
    for a in amrs:
        amr_list.append([
            float(a.status), 0.0, 0.0, a.x/100, a.y/100, 
            0.0, 0.0, 0.0 # Inventory (A, B, C)
        ])
    x_amr = torch.tensor(amr_list, dtype=torch.float32)

    # 2. Active Jobs Tensor [N, 10] (Jobs that have been released)
    if not jobs_bot:
        # Dummy job if empty to prevent GNN crash
        x_job = torch.zeros((1, 10), dtype=torch.float32)
    else:
        j_list = []
        for j in jobs_bot:
            # One-hot mat type
            mat = [1,0,0] if j.jtype=='A' else ([0,1,0] if j.jtype=='B' else [0,0,1])
            j_list.append([
                1.0, j.proc_time/20.0, 0.0, 
                0.5, 0.5, 0.1, 0.1, # Mock Locs
                *mat
            ])
        x_job = torch.tensor(j_list, dtype=torch.float32)

    # 3. Queue Tensor [3]
    waiting_count = len(jobs_top)
    avg_proc = sum(j.proc_time for j in jobs_top)/max(1, waiting_count)
    x_queue = torch.tensor([float(waiting_count), avg_proc, simulation_time], dtype=torch.float32)

    return x_amr, x_job, x_queue

def append_dispatch_inbox(jobs, dispatch_time):
    rec = {
        "generated_at": time.time(), "dispatch_time": float(dispatch_time),
        "jobs": [{"jid": j.jid, "type": j.jtype, "proc_time": j.proc_time, "station": j.station} for j in jobs]
    }
    with open(DISPATCH_INBOX, "a") as f: f.write(json.dumps(rec) + "\n")

def simulate_shop_floor(dt):
    """Simple background logic to update mock AMRs and clear finished jobs"""
    global jobs_bot
    # Mock: Clear active jobs randomly to simulate work completion
    if jobs_bot and random.random() < (0.1 * dt): 
        finished = jobs_bot.pop(0)
    
    # Mock: Move AMRs randomly
    for a in amrs:
        a.x = (a.x + random.uniform(-1, 1)) % 100
        a.status = 1 if jobs_bot else 0

# --------------------------- Visuals ---------------------------
def add_job_artist_top(ax, j: Job):
    x_start = LEFT_LABEL_PAD + sum(job.proc_time / 2 for job in jobs_top[:-1])
    r = Rectangle((x_start, TOP_Y_CENTER), j.proc_time / 2, TOP_LANE_H,
                  linewidth=2, edgecolor="black", facecolor=COLORS[j.jtype])
    ax.add_patch(r)
    rects_top.append(r)
    t = ax.text(x_start + j.proc_time/4, TOP_Y_CENTER + TOP_LANE_H/2, 
                f"J{j.jid}\nS{j.station}", ha="center", va="center", fontsize=8, weight="bold")
    texts_top.append(t)

def remove_artists():
    for r in rects_top: r.remove()
    for t in texts_top: t.remove()
    rects_top.clear(); texts_top.clear()

def spawn_job(ax):
    global job_counter
    jtype = random.choice(JOB_TYPE_KEYS)
    j = Job(jid=job_counter, jtype=jtype, proc_time=JOB_TYPES[jtype]["time"],
            arrival_ts=simulation_time, station=random.randint(1, STATION_COUNT))
    job_counter += 1
    jobs_top.append(j)
    add_job_artist_top(ax, j)

def dispatch(ax, count=None):
    """Release 'count' jobs from Top (Queue) to Bot (Active)"""
    global jobs_top, jobs_bot
    if not jobs_top: return 0
    
    to_move_count = len(jobs_top) if count is None else min(count, len(jobs_top))
    moving = jobs_top[:to_move_count]
    
    # Update visuals: Clear all, remove moved from list, redraw remaining
    remove_artists()
    jobs_top = jobs_top[to_move_count:] # Slice list
    for j in jobs_top: add_job_artist_top(ax, j) # Redraw remaining
    
    # Logical Move
    jobs_bot.extend(moving)
    append_dispatch_inbox(moving, simulation_time)
    return len(moving)

# --------------------------- Main Loop ---------------------------
def main():
    global is_running, simulation_time, next_arrival_time, next_decision_time, current_action_name

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_ylim(0, 2); ax.set_xlim(0, VIEW_WIDTH)
    ax.set_yticks([]); ax.set_xticks([])
    
    # Static UI
    ax.text(0.05, 0.5, "Job\nQueue", transform=ax.transAxes, ha="center", weight="bold", color="gray")
    status_txt = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", fontsize=12)
    action_txt = ax.text(0.95, 0.05, "Action: INIT", transform=ax.transAxes, ha="right", color="red", weight="bold")
    
    def schedule_next_arrival():
        global next_arrival_time
        next_arrival_time = simulation_time + random.expovariate(1.0/AVG_INTERARRIVAL_TIME)
    
    schedule_next_arrival()
    
    timer = fig.canvas.new_timer(interval=UPDATE_INTERVAL_MS)

    def tick():
        global simulation_time, next_arrival_time, next_decision_time, current_action_name
        
        if is_running:
            dt = (UPDATE_INTERVAL_MS / 1000.0) * SIM_SPEED_MULTIPLIER
            simulation_time += dt
            
            # 1. Physics: Job Arrivals
            if simulation_time >= next_arrival_time:
                spawn_job(ax)
                schedule_next_arrival()
            
            # 2. Physics: Simulate Shop Floor (AMRs working)
            simulate_shop_floor(dt)
            
            # 3. AI Decision: GNN+DDQN Control
            if simulation_time >= next_decision_time:
                # Get Tensors
                x_amr, x_job, x_q = get_state_tensors()
                
                # Forward Pass (No Grad)
                with torch.no_grad():
                    q_vals = agent(x_amr, x_job, x_q)
                    action = torch.argmax(q_vals).item() # 0=Wait, 1=Rel_1, 2=Rel_Batch
                
                # Execute Action
                if action == 0:
                    current_action_name = "WAIT"
                elif action == 1:
                    n = dispatch(ax, count=1)
                    current_action_name = f"RELEASE 1 (Sent {n})"
                elif action == 2:
                    n = dispatch(ax, count=5) # Batch size 5
                    current_action_name = f"RELEASE BATCH (Sent {n})"
                
                next_decision_time = simulation_time + DECISION_INTERVAL_S

            # UI Updates
            status_txt.set_text(f"Time: {simulation_time:.2f}s | Queue: {len(jobs_top)} | Active: {len(jobs_bot)}")
            action_txt.set_text(current_action_name)
            fig.canvas.draw_idle()

    timer.add_callback(tick)
    timer.start()

    def on_key(event):
        global is_running
        if event.key == " ": is_running = not is_running

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

if __name__ == "__main__":
    main()