import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: Running on CPU. Install PyTorch with CUDA support.")
    
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
import math
import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import copy
import random

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'DATASET_PATH': 'training_dataset.jsonl',
    'SAVE_PATH': 'gnn_ddqn_model.pth',
    
    # Physics
    'GRID_WIDTH': 10,
    'SCALE': 1.0,
    'AMR_SPEED': 1.0,
    'CAPACITY_PER_TYPE': 3,
    
    # Training
    'NUM_EPISODES': 1000,
    'BATCH_SIZE': 32,      # Increased for GPU efficiency
    'GAMMA': 0.99,
    'LR': 3e-4,
    'EPS_START': 1.0,
    'EPS_END': 0.05,
    'EPS_DECAY': 300,
    
    # Model
    'AMR_IN_DIM': 8, 
    'JOB_IN_DIM': 10, 
    'QUEUE_DIM': 4, 
    'HIDDEN_DIM': 32,     # Wider network for GPU
    'ACTION_DIM': 3 
}

# ==========================================
# 2. MAP & PATHFINDING (A*)
# ==========================================
class WarehouseMap:
    def __init__(self):
        self.barriers = set()
        for y in range(5, 15): self.barriers.add((5, y))
        for y in range(5, 15): self.barriers.add((14, y))
        for x in range(8, 12): self.barriers.add((x, 10))

    def get_true_distance(self, start_float, end_float):
        # (Same A* implementation as previous step - condensed for brevity)
        sx, sy = int(start_float[0]/CONFIG['SCALE']), int(start_float[1]/CONFIG['SCALE'])
        ex, ey = int(end_float[0]/CONFIG['SCALE']), int(end_float[1]/CONFIG['SCALE'])
        sx, sy = max(0, min(19, sx)), max(0, min(19, sy))
        ex, ey = max(0, min(19, ex)), max(0, min(19, ey))
        
        start, goal = (sx, sy), (ex, ey)
        if start == goal: return 0.0
        if goal in self.barriers: return 999.0

        frontier = [(0, start)]
        cost_so_far = {start: 0}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal: break
            x, y = current
            for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
                if 0<=nx<20 and 0<=ny<20 and (nx,ny) not in self.barriers:
                    new_cost = cost_so_far[current] + 1
                    if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                        cost_so_far[(nx, ny)] = new_cost
                        priority = new_cost + abs(nx-ex) + abs(ny-ey)
                        heapq.heappush(frontier, (priority, (nx, ny)))
                        
        return cost_so_far.get(goal, 999.0) * CONFIG['SCALE']

GLOBAL_MAP = WarehouseMap()

# ==========================================
# 3. ENVIRONMENT & LOGIC
# ==========================================

STATIONS = {
    "SUPPLY_A": (0.0, 7.0), 
    "SUPPLY_B": (0.0, 4.0), 
    "SUPPLY_C": (0.0, 1.0),
    1: (9.0, 8.0), 
    2: (9.0, 6.0), 
    3: (9.0, 4.0), 
    4: (9.0, 2.0), 
    5: (9.0, 0.0)
}

JOB_PROPS = {
    "A": {"time": 10.0, "supply": "SUPPLY_A"},
    "B": {"time": 15.0, "supply": "SUPPLY_B"},
    "C": {"time": 20.0, "supply": "SUPPLY_C"}
}

@dataclass
class Job:
    jid: int; jtype: str; material: str; arrival_ts: float; proc_time: float
    dest_pos: tuple; supply_pos: tuple; status: int = 0

@dataclass
class AMR:
    aid: int
    x: float = 10.0
    y: float = 10.0
    status: int = 0 
    remaining_time: float = 0.0
    inventory: Dict[str, int] = field(default_factory=lambda: {'A':0, 'B':0, 'C':0})
    tot_number_of_jobs: int = 0 # <--- NEW: Track total jobs handled for reward shaping
    current_job: int = -1 # <--- NEW: Track which job ID this AMR is holding

class GridEnv:
    def __init__(self):
        self.episodes = []
        with open(CONFIG['DATASET_PATH'], 'r') as f:
            for line in f: self.episodes.append(json.loads(line))
        self.ep_idx = 0

    def reset(self):
        data = self.episodes[self.ep_idx]
        self.ep_idx = (self.ep_idx + 1) % len(self.episodes)
        
        self.queue = deque()
        for raw in data['jobs']:
            props = JOB_PROPS[raw['type']]
            self.queue.append(Job(
                jid=raw['id'], jtype=raw['type'], material=raw['type'],
                arrival_ts=raw['arrival_time'], proc_time=props['time'],
                supply_pos=STATIONS[props['supply']], 
                dest_pos=STATIONS[raw['dest_station_id']]
            ))
        
        self.active_jobs = [] 
        self.completed_jobs = [] # <--- NEW: Metric Tracking
        self.amrs = [
            AMR(0, x=2, y=1), AMR(1, x=2, y=4), AMR(2, x=2, y=7)
        ]
        self.sim_time = 0.0
        return self.get_state_arrays()

    def assign_nearest(self):
        idle = [a for a in self.amrs if a.status == 0]
        # Only assign jobs that are "Waiting" (Status 1). 
        jobs = [j for j in self.active_jobs if j.status == 1]
        
        if not idle or not jobs: return
        
        cands = []
        for a in idle:
            for j in jobs:
                mat = j.material
                has_mat = a.inventory[mat] > 0
                
                # 1. Calculate Leg 1 (Where the AMR goes first)
                start_pos = (a.x, a.y)
                first_target = j.dest_pos if has_mat else j.supply_pos
                dist_leg1 = GLOBAL_MAP.get_true_distance(start_pos, first_target)
                
                # 2. Calculate Total Distance (The Logic Fix)
                if has_mat:
                    # We have it -> Drive straight to finish
                    total_dist = dist_leg1
                else:
                    # We need it -> Drive to Supply (Leg 1) + Drive to Dest (Leg 2)
                    dist_leg2 = GLOBAL_MAP.get_true_distance(j.supply_pos, j.dest_pos)
                    total_dist = dist_leg1 + dist_leg2
                
                # We sort by 'total_dist' now
                cands.append((total_dist, a, j, has_mat))
        
        # Sort by distance (Nearest Neighbor Strategy)
        cands.sort(key=lambda x: x[0])
        
        taken_a, taken_j = set(), set()
        
        for dist, a, j, has_mat in cands:
            # Ensure 1-to-1 mapping
            if a.aid in taken_a or j.jid in taken_j: continue
            
            # --- EXECUTE ASSIGNMENT ---
            j.status = 2        # Job assigned
            a.status = 1        # AMR busy
            a.current_job = j.jid 
            
            if has_mat:
                # Scenario A: Delivery Trip (We had it in stock)
                a.inventory[j.material] -= 1
                total_travel = dist
            else:
                # Scenario B: Fetch & Deliver (We go Supply -> Dest)
                # Note: We don't increment inventory because we pick up 
                # and immediately deliver in one continuous task.
                leg2 = GLOBAL_MAP.get_true_distance(j.supply_pos, j.dest_pos)
                total_travel = dist + leg2
            
            # Teleport AMR to destination (Physics engine handles the delay)
            a.x, a.y = j.dest_pos
            a.remaining_time = (total_travel / CONFIG['AMR_SPEED']) + j.proc_time
            
            taken_a.add(a.aid)
            taken_j.add(j.jid)

    def step(self, action):
        count = {0:0, 1:1, 2:5}[action]
        avail = [j for j in self.queue if j.arrival_ts <= self.sim_time]
        for _ in range(count):
            if avail:
                j = self.queue.popleft(); j.status = 1
                self.active_jobs.append(j); avail.pop(0)
        
        self.assign_nearest()
        self.sim_time += 1.0
        
        for a in self.amrs:
            if a.status == 1:
                a.remaining_time -= 1.0
                if a.remaining_time <= 0:
                    a.status = 0
                    # Find the job this AMR finished and remove it
                    finished_job_id = a.current_job
                    a.current_job = -1
                    
                    # Remove from active_jobs list
                    for idx, j in enumerate(self.active_jobs):
                        if j.jid == finished_job_id:
                            finished_job = self.active_jobs.pop(idx)
                            finished_job.finish_ts = self.sim_time # Mark finish time
                            self.completed_jobs.append(finished_job)
                            break

        q_len = len([j for j in self.queue if j.arrival_ts <= self.sim_time])
        shop_load = len(self.active_jobs)
        reward = -0.1*q_len - 0.05*shop_load
        if count>0: reward += 5.0*count
        if shop_load > 12: reward -= 5.0
        
        return self.get_state_arrays(), reward, self.sim_time>=300

    def get_state_arrays(self):
        """
        Returns raw Python lists (CPU) representing the state.
        Converting them to Tensors happens during the Training Loop or Test Loop.
        """
        # 1. AMR Features: [Status, RemTime, InvA, InvB, InvC, X, Y, 0]
        a_data = []
        for a in self.amrs:
            a_data.append([
                float(a.status), 
                a.remaining_time / 50.0, 
                a.inventory['A'] / CONFIG['CAPACITY_PER_TYPE'], 
                a.inventory['B'] / CONFIG['CAPACITY_PER_TYPE'], 
                a.inventory['C'] / CONFIG['CAPACITY_PER_TYPE'], 
                a.x / 10.0, 
                a.y / 10.0, 
                a.tot_number_of_jobs / 20.0 # <--- NEW: Normalize total jobs handled
            ])
        
        # 2. Job Features: [1, Proc, Wait, DestX, DestY, SuppX, SuppY, A, B, C]
        j_data = []
        if not self.active_jobs: 
            j_data.append([0.0] * 10) # Padding if empty
        else:
            for j in self.active_jobs:
                mat = [1,0,0] if j.material=='A' else ([0,1,0] if j.material=='B' else [0,0,1])
                j_data.append([
                    1.0, 
                    j.proc_time / 25.0, 
                    (self.sim_time - j.arrival_ts) / 100.0,
                    j.dest_pos[0] / 10.0, 
                    j.dest_pos[1] / 10.0, 
                    j.supply_pos[0] / 10.0, 
                    j.supply_pos[1] / 10.0, 
                    *mat
                ])
        
        # 3. Queue Features: [Count, Avg_Proc_Time, Var_Proc_Time]
        
        # Filter jobs that have actually arrived
        waiting_jobs = [j for j in self.queue if j.arrival_ts <= self.sim_time]
        q_len = len(waiting_jobs)
        
        if q_len > 0:
            # Extract processing times
            proc_times = [j.proc_time for j in waiting_jobs]
            
            # Calculate Average
            avg_proc = sum(proc_times) / q_len
            
            # Calculate Variance: sum((x - mean)^2) / N
            variance_proc = sum((x - avg_proc) ** 2 for x in proc_times) / q_len
            
            # Normalize Inputs for Neural Net
            # Avg is usually ~15s -> divide by 20.0 to get ~0.75
            # Variance can be ~25 -> divide by 50.0 to get ~0.5
            q_data = [
                float(q_len), 
                avg_proc / 20.0, 
                variance_proc / 50.0,
                self.sim_time / 300.0
            ]
        else:
            # Empty Queue = 0s
            q_data = [0.0, 0.0, 0.0, self.sim_time / 300.0]
        
        return a_data, j_data, q_data
# ==========================================
# 4. BATCHED GNN MODEL (THE GPU FIX)
# ==========================================
class BatchedHeteroGNN(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.upd_amr = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))
        self.upd_job = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))

    def forward(self, h_amr, h_job, job_mask):
        # h_amr: [Batch, 3, H]
        # h_job: [Batch, MaxJobs, H]
        # job_mask: [Batch, MaxJobs, 1] (1 for real job, 0 for padding)

        # 1. Pool Jobs -> Message to AMR
        # Mask out padding before mean
        masked_job = h_job * job_mask
        # Sum valid jobs and divide by count (avoid div by zero)
        job_sum = masked_job.sum(dim=1, keepdim=True) 
        job_count = job_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        job_mean = job_sum / job_count # [Batch, 1, H]
        
        # Expand to all AMRs
        msg_to_amr = job_mean.expand(-1, h_amr.size(1), -1)
        
        # 2. Pool AMRs -> Message to Job
        amr_mean = h_amr.mean(dim=1, keepdim=True) # [Batch, 1, H]
        msg_to_job = amr_mean.expand(-1, h_job.size(1), -1)

        # 3. Update
        out_amr = self.upd_amr(h_amr + msg_to_amr)
        out_job = self.upd_job(h_job + msg_to_job)
        
        return out_amr, out_job * job_mask # Re-apply mask

class SchedulerAgent(nn.Module):
    def __init__(self):
        super().__init__()
        h = CONFIG['HIDDEN_DIM']
        self.enc_amr = nn.Linear(CONFIG['AMR_IN_DIM'], h)
        self.enc_job = nn.Linear(CONFIG['JOB_IN_DIM'], h)
        self.gnn = BatchedHeteroGNN(h)
        self.head_val = nn.Sequential(nn.Linear(h+CONFIG['QUEUE_DIM'], h), nn.ReLU(), nn.Linear(h, 1))
        self.head_adv = nn.Sequential(nn.Linear(h+CONFIG['QUEUE_DIM'], h), nn.ReLU(), nn.Linear(h, CONFIG['ACTION_DIM']))

    def forward(self, x_amr, x_job, x_q, job_mask):
        # x_amr: [B, 3, 8], x_job: [B, N, 10], mask: [B, N, 1]
        h_amr = F.relu(self.enc_amr(x_amr))
        h_job = F.relu(self.enc_job(x_job))
        
        h_amr, _ = self.gnn(h_amr, h_job, job_mask)
        
        # Global Pooling
        shop_emb = h_amr.mean(dim=1) # [B, H]
        state = torch.cat([shop_emb, x_q], dim=-1)
        
        val = self.head_val(state)
        adv = self.head_adv(state)
        return val + (adv - adv.mean(dim=1, keepdim=True))

# ==========================================
# 5. BATCH PROCESSING UTILS
# ==========================================
def collate_batch(batch_list):
    """
    Takes a list of (amr, job, queue) tuples and stacks them into Tensors.
    Handles variable number of jobs via Padding.
    """
    # Unzip
    states, actions, rewards, next_states, dones = zip(*batch_list)
    
    def pad_and_stack(state_list):
        amrs, jobs, queues = zip(*state_list)
        
        # Stack AMRs (Fixed size 3)
        b_amr = torch.tensor(amrs, dtype=torch.float32, device=CONFIG['DEVICE'])
        b_q = torch.tensor(queues, dtype=torch.float32, device=CONFIG['DEVICE'])
        
        # Pad Jobs (Variable size)
        max_j = max(len(j) for j in jobs)
        b_job = torch.zeros((len(jobs), max_j, CONFIG['JOB_IN_DIM']), dtype=torch.float32, device=CONFIG['DEVICE'])
        b_mask = torch.zeros((len(jobs), max_j, 1), dtype=torch.float32, device=CONFIG['DEVICE'])
        
        for i, j_list in enumerate(jobs):
            L = len(j_list)
            if L > 0:
                tens = torch.tensor(j_list, dtype=torch.float32, device=CONFIG['DEVICE'])
                b_job[i, :L, :] = tens
                b_mask[i, :L, :] = 1.0
                
        return b_amr, b_job, b_q, b_mask

    s_amr, s_job, s_q, s_mask = pad_and_stack(states)
    ns_amr, ns_job, ns_q, ns_mask = pad_and_stack(next_states)
    
    b_a = torch.tensor(actions, device=CONFIG['DEVICE']).unsqueeze(1)
    b_r = torch.tensor(rewards, dtype=torch.float32, device=CONFIG['DEVICE']).unsqueeze(1)
    b_d = torch.tensor(dones, dtype=torch.float32, device=CONFIG['DEVICE']).unsqueeze(1)
    
    return (s_amr, s_job, s_q, s_mask), b_a, b_r, (ns_amr, ns_job, ns_q, ns_mask), b_d

# ==========================================
# 6. TRAINING
# ==========================================
class ReplayBuffer:
    def __init__(self, cap): self.buf = deque(maxlen=cap)
    def push(self, x): self.buf.append(x)
    def sample(self, n): return random.sample(self.buf, n)
    def __len__(self): return len(self.buf)

def optimize(agent, target, opt, memory):
    if len(memory) < CONFIG['BATCH_SIZE']: return 0.0
    
    # 1. Get Batch (CPU Lists)
    batch_raw = memory.sample(CONFIG['BATCH_SIZE'])
    
    # 2. Collate & Move to GPU (ONE operation per batch)
    curr_state, act, rew, next_state, done = collate_batch(batch_raw)
    
    # 3. Forward Pass (Parallel on GPU)
    # Current Q
    q_all = agent(*curr_state)
    q_curr = q_all.gather(1, act)
    
    # Target Q
    with torch.no_grad():
        ns_amr, ns_job, ns_q, ns_mask = next_state
        # Double DQN
        next_acts = agent(ns_amr, ns_job, ns_q, ns_mask).argmax(1, keepdim=True)
        next_vals = target(ns_amr, ns_job, ns_q, ns_mask).gather(1, next_acts)
        q_target = rew + (CONFIG['GAMMA'] * next_vals * (1 - done))
        
    loss = F.smooth_l1_loss(q_curr, q_target)
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    opt.step()
    
    return loss.item()

def main():
    print(f"--- GPU STATUS: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ---")
    
    env = GridEnv()
    agent = SchedulerAgent().to(CONFIG['DEVICE'])
    target = SchedulerAgent().to(CONFIG['DEVICE'])
    target.load_state_dict(agent.state_dict())
    
    opt = optim.Adam(agent.parameters(), lr=CONFIG['LR'])
    memory = ReplayBuffer(20000)
    
    for ep in range(CONFIG['NUM_EPISODES']):
        state = env.reset() # Returns CPU lists
        ep_rew, ep_loss = 0, 0
        eps = CONFIG['EPS_END'] + (CONFIG['EPS_START'] - CONFIG['EPS_END']) * math.exp(-1.*ep/CONFIG['EPS_DECAY'])
        
        while True:
            # Select Action (Single Inference)
            if random.random() < eps: action = random.randint(0, 2)
            # Inside main() while True loop:
            else:
                with torch.no_grad():
                    s_amr = torch.tensor([state[0]], dtype=torch.float32, device=CONFIG['DEVICE'])
                    s_job = torch.tensor([state[1]], dtype=torch.float32, device=CONFIG['DEVICE'])
                    s_q = torch.tensor([state[2]], dtype=torch.float32, device=CONFIG['DEVICE'])
                    
                    # LOGIC FIX: Create mask based on actual data
                    # If the first feature of the first job is 0.0, it's a Ghost Job.
                    # We create a mask of 1s, but if it's a ghost, we make it 0.
                    s_mask = torch.ones((1, s_job.size(1), 1), device=CONFIG['DEVICE'])
                    
                    # Check if 'Existence Bit' (index 0) is 0
                    if state[1][0][0] == 0.0: 
                        s_mask = torch.zeros((1, s_job.size(1), 1), device=CONFIG['DEVICE'])

                    action = agent(s_amr, s_job, s_q, s_mask).argmax(1).item()

            next_state, reward, done = env.step(action)
            memory.push((state, action, reward, next_state, done))
            state = next_state
            ep_rew += reward
            
            if ep > 20: ep_loss += optimize(agent, target, opt, memory)
            if done: break
            
        if ep % 10 == 0:
            target.load_state_dict(agent.state_dict())
            print(f"Ep {ep} | Reward: {ep_rew:.1f} | Loss: {ep_loss:.2f} | Eps: {eps:.2f}")

    torch.save(agent.state_dict(), CONFIG['SAVE_PATH'])
    print("Done.")

if __name__ == "__main__":
    main()