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
from typing import List, Set, Tuple, Dict, Optional
import copy
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Static_alogorithm/GA_code')))
from GA import evolve as ga_evolve, Job as GAJob, local_improve, routing_iters

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Static_alogorithm/GNN')))
from GNN import SchedulerGNN, solve_with_gnn




# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'DATASET_PATH': 'training_dataset_r2.jsonl',
    'SAVE_PATH': 'gnn_ddqn_model_v6/gnn_ddqn_model_v6.pth',
    
    # Physics
    'GRID_WIDTH': 10,
    'SCALE': 1.0,
    'AMR_SPEED': 1.0,
    'CAPACITY_PER_TYPE': 3,
    'SIM_TIME': 500.0,  # Max sim time per episode
    'SIM_TIME_SCALE': 25.0, # For normalizing time features
    
    # Training
    'NUM_EPISODES': 1000,
    'BATCH_SIZE': 64,
    'GAMMA': 0.99,
    'LR': 3e-4,
    'FLOW_PENALTY': 0,
    'EPS_START': 1.0,
    'EPS_END': 0.05,
    'EPS_DECAY': 200,
    
    # Model
    'AMR_IN_DIM': 8, 
    'JOB_IN_DIM': 10, 
    'QUEUE_DIM': 4, 
    'HIDDEN_DIM': 128,
    'GNN_LAYERS': 2,
    'ACTION_DIM': 2,    # 0: Wait, 1: Release
    
    # GA Hyperparameters
    'GA_POP_SIZE': 200,       # Increased from 50
    'GA_GENERATIONS': 150,    # Increased from 100
    'GA_ROUTING_ITERS': 1000,
    'GA_COLLISION_ITERS': 2000,
    'GA_ROUTING_MAX_DEPTH': 100
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

        self.W = 20
        self._precompute_all_pairs()

    def _precompute_all_pairs(self):
        W = self.W
        # dist[(sx,sy)][(ex,ey)] = steps
        self.dist = {}
        for sx in range(W):
            for sy in range(W):
                if (sx, sy) in self.barriers:
                    continue
                d = {(sx, sy): 0}
                q = deque([(sx, sy)])
                while q:
                    x, y = q.popleft()
                    for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
                        if 0<=nx<W and 0<=ny<W and (nx,ny) not in self.barriers and (nx,ny) not in d:
                            d[(nx,ny)] = d[(x,y)] + 1
                            q.append((nx,ny))
                self.dist[(sx, sy)] = d

    def get_true_distance(self, start_float, end_float):
        sx, sy = int(start_float[0]/CONFIG['SCALE']), int(start_float[1]/CONFIG['SCALE'])
        ex, ey = int(end_float[0]/CONFIG['SCALE']), int(end_float[1]/CONFIG['SCALE'])
        sx, sy = max(0, min(19, sx)), max(0, min(19, sy))
        ex, ey = max(0, min(19, ex)), max(0, min(19, ey))

        if (sx,sy) not in self.dist: 
            return 999.0
        steps = self.dist[(sx,sy)].get((ex,ey), 999.0)
        return steps * CONFIG['SCALE']

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
    "A": {"time": 5.0, "supply": "SUPPLY_A"},
    "B": {"time": 10.0, "supply": "SUPPLY_B"},
    "C": {"time": 25.0, "supply": "SUPPLY_C"}
}

@dataclass
class Job:
    jid: int; jtype: str; material: str; arrival_ts: float; proc_time: float
    dest_pos: tuple; supply_pos: tuple; status: int = 0
    finish_ts: float = -1.0

@dataclass
class AMR:
    aid: int
    x: float = 10.0
    y: float = 10.0
    status: int = 0 
    remaining_time: float = 0.0
    local_queue: deque = field(default_factory=deque)
    inventory: Dict[str, int] = field(default_factory=lambda: {'A':0, 'B':0, 'C':0})
    tot_number_of_jobs: int = 0
    current_job: int = -1


class GridEnv:
    def __init__(self):
        self.last_ga_compute_time = 0.0

        self.episodes = []
        self.last_resched_t = -999
        with open(CONFIG['DATASET_PATH'], 'r') as f:
            for line in f: self.episodes.append(json.loads(line))
        self.ep_idx = 0
        self.scheduled_queue = deque() # Stores the result of GA
        
        # Load the pretrained heuristic GNN model
        self.heuristic_gnn = SchedulerGNN(amr_in_dim=8, job_in_dim=10, hidden_dim=128, gnn_layers=2)
        weights_path = os.path.join(os.path.dirname(__file__), '../Static_alogorithm/GNN/gnn_scheduler_best.pth')
        if os.path.exists(weights_path):
            self.heuristic_gnn.load_state_dict(torch.load(weights_path, map_location=CONFIG['DEVICE']))
        self.heuristic_gnn.to(CONFIG['DEVICE'])

    def reset(self):
        self.arrival_version = 0          # 有新 job 到達就 +1
        self.last_resched_version = -1    # 上次 reschedule 時的 arrival_version
        self.last_resched_completion_count = -1

        self.completed_this_step = 0  # NEW: Track how many jobs completed in this step for reward shaping
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
        
        self._release_arrived_jobs() # Ensure jobs at t=0 are visible in initial state
        self.last_resched_t = -1e9   # ✅ critical: reset cooldown
        return self.get_state_arrays()
    
    def can_reschedule(self):
        RESCHED_COOLDOWN = 1.0
        cooldown_ok = (self.sim_time - self.last_resched_t) >= RESCHED_COOLDOWN
        has_unstarted = any(j.status == 1 for j in self.active_jobs)
        new_job_since_last = (self.arrival_version != self.last_resched_version)
        new_completion_since_last = (len(self.completed_jobs) != self.last_resched_completion_count)

        return cooldown_ok and has_unstarted and (new_job_since_last or new_completion_since_last)

    def get_action_mask(self):
        return [1.0, 1.0 if self.can_reschedule() else 0.0]


    def calculate_current_makespan(self):
        # 簡化估計：看目前所有忙碌 AMR 還要多久 + 尚未開始 jobs 的平均成本
        busy = max([a.remaining_time for a in self.amrs], default=0.0)
        unstarted = [j for j in self.active_jobs if j.status == 1]
        if not unstarted:
            return max(busy, 1.0)

        # 用 (dist to supply + dist supply->dest + proc) 當粗估
        est_costs = []
        for j in unstarted:
            # 用最近 AMR 當估計起點
            est = min(
                GLOBAL_MAP.get_true_distance((a.x, a.y), j.supply_pos) + GLOBAL_MAP.get_true_distance(j.supply_pos, j.dest_pos)
                for a in self.amrs
            ) + j.proc_time
            est_costs.append(est)

        # 三台 AMR 平行，粗估除以 3
        return max(busy + sum(est_costs) / max(len(self.amrs), 1), 1.0)

    def update_amr_tasks(self):
        """
        Decentralized Execution: Each AMR checks its own local queue.
        Now with inventory management!
        """
        for a in self.amrs:
            # If AMR is idle and has jobs assigned to it by the GA
            if a.status == 0 and len(a.local_queue) > 0:
                next_jid = a.local_queue.popleft()
                
                # Find the job object
                job = next((j for j in self.active_jobs if j.jid == next_jid), None)
                
                if job and job.status == 1:
                    # 1. Lock Job and AMR
                    job.status = 2 # Processing
                    a.status = 1
                    a.current_job = job.jid
                    material = job.material
                    
                    # 2. Check Inventory and Calculate travel + processing
                    if a.inventory.get(material, 0) == 0:
                        # Leg 1: Current Pos -> Supply | Leg 2: Supply -> Destination
                        dist_to_supply = GLOBAL_MAP.get_true_distance((a.x, a.y), job.supply_pos)
                        dist_to_dest = GLOBAL_MAP.get_true_distance(job.supply_pos, job.dest_pos)
                        travel_time = (dist_to_supply + dist_to_dest) / CONFIG['AMR_SPEED']
                        
                        # Refill inventory
                        a.inventory[material] = CONFIG['CAPACITY_PER_TYPE']
                    else:
                        # Leg 1: Current Pos -> Destination
                        dist_to_dest = GLOBAL_MAP.get_true_distance((a.x, a.y), job.dest_pos)
                        travel_time = dist_to_dest / CONFIG['AMR_SPEED']

                    # Consume one item for this job
                    a.inventory[material] -= 1
                    
                    a.remaining_time = travel_time + job.proc_time
                    
                    # 3. Teleport AMR to destination (Simulating completion)
                    a.x, a.y = job.dest_pos

                else:
                    # If job was already taken or invalid, AMR stays idle to try next tick
                    pass

    def _check_job_completions(self, dt = 1.0):
        finished_cnt = 0
        for a in self.amrs:
            if a.status == 1:  # Busy
                a.remaining_time -= dt
                if a.remaining_time <= 0:
                    a.status = 0
                    finished_jid = a.current_job
                    a.current_job = -1
                    a.tot_number_of_jobs += 1

                    # Move to completed
                    for idx, j in enumerate(self.active_jobs):
                        if j.jid == finished_jid:
                            j.status = 3
                            j.finish_ts = self.sim_time + a.remaining_time # ✅ record finish time
                            comp_j = self.active_jobs.pop(idx)
                            self.completed_jobs.append(comp_j)
                            finished_cnt += 1
                            break
        return finished_cnt

    def get_state_snapshot(self) -> dict:
        state = {
            "time": self.sim_time,
            "positions": {f"AMR{a.aid+1}": (int(a.x), int(a.y)) for a in self.amrs},
            "availability": {f"AMR{a.aid+1}": self.sim_time + max(0.0, a.remaining_time) for a in self.amrs},
            "inventory": {f"AMR{a.aid+1}": a.inventory.copy() for a in self.amrs},
            "status": {f"AMR{a.aid+1}": 1 if a.status != 0 else 0 for a in self.amrs},
            "remaining_time": {f"AMR{a.aid+1}": max(0.0, a.remaining_time) for a in self.amrs},
            "tot_number_of_jobs": {f"AMR{a.aid+1}": a.tot_number_of_jobs for a in self.amrs},
            "queues": {f"AMR{a.aid+1}": list(a.local_queue) for a in self.amrs}
        }
        return state

    def _release_arrived_jobs(self):
        moved = 0
        while self.queue and self.queue[0].arrival_ts <= self.sim_time:
            j = self.queue.popleft()
            j.status = 1
            self.active_jobs.append(j)
            moved += 1

        if moved > 0:
            self.arrival_version += 1   # 只要這一步有新 job 到達，就視為一個 arrival event
        return moved


    def step(self, action: int):
        """
        action:
        0 = wait
        1 = run GA reschedule (only if allowed)
        """
        # -------------------------------
        # 0) Release arrived jobs
        # -------------------------------
        moved = self._release_arrived_jobs()
        self.last_ga_compute_time = 0.0

        # -------------------------------
        # 1) Reward shaping params
        # -------------------------------
        DONE_REWARD = 10.0
        FLOW_PENALTY = CONFIG.get('FLOW_PENALTY', 0.1)
        EMPTY_RESCHED_PENALTY = 0.2

        reward = 0.0
        before_done = len(self.completed_jobs)
        compute_time = 0.0
        reschedule_executed = False

        # -------------------------------
        # 2) Apply action (reschedule)
        # -------------------------------
        if action == 1:
            if not self.can_reschedule():
                reward -= EMPTY_RESCHED_PENALTY
            else:
                unstarted = [j for j in self.active_jobs if j.status == 1]
                if not unstarted:
                    reward -= EMPTY_RESCHED_PENALTY
                else:
                    from GA import STATIONS
                    pos_to_station = {v: k for k, v in STATIONS.items()}
                    
                    ga_jobs = []
                    job_map = {}
                    for i, j in enumerate(unstarted):
                        st_name = pos_to_station.get((int(j.dest_pos[0]), int(j.dest_pos[1])), "M1_1")
                        ga_j = GAJob(idx=i, type_=j.material, station=st_name, duration=j.proc_time)
                        ga_jobs.append(ga_j)
                        job_map[i] = j.jid
                        
                    import time
                    start_cpu_time = time.perf_counter()
                    init_state = self.get_state_snapshot()
                    best_ind, _, compute_time = solve_with_gnn(ga_jobs, self.heuristic_gnn, deterministic=True, init_state=init_state)
                    # Apply local improve from GA to hone the schedule
                    best_ind = local_improve(best_ind, ga_jobs, max_iters=CONFIG.get('GA_ROUTING_ITERS', 1000), init_state=init_state)
                    collision_iters = CONFIG.get('GA_COLLISION_ITERS', 0)
                    if collision_iters > 0:
                        best_ind = local_improve(best_ind, ga_jobs, max_iters=collision_iters, check_collision=True, init_state=init_state)
                    
                    compute_time = (time.perf_counter() - start_cpu_time)
                    self.last_ga_compute_time = compute_time

                    # ========================================================
                    # Time paradox fix: AMRs that were busy keep working during
                    # the compute window. Complete any jobs that finish before
                    # the new schedule arrives.
                    # ========================================================
                    compute_dt = float(math.ceil(max(1.0, compute_time)))
                    self._check_job_completions(compute_dt)
                    self.sim_time += compute_dt

                    # NOW assign the new schedule (after compute delay)
                    # assign schedules to AMRs
                    for a in self.amrs:
                        a.local_queue.clear()
                    if best_ind.order:
                        for jidx in best_ind.order:
                            jid = job_map[jidx]
                            amr_str = best_ind.amr_assignment[jidx]
                            aid = int(amr_str.replace("AMR", "")) - 1
                            if 0 <= aid < len(self.amrs):
                                self.amrs[aid].local_queue.append(jid)

                    # ✅ update reschedule gate correctly (after a real schedule)
                    self.last_resched_t = self.sim_time
                    self.last_resched_version = getattr(self, "arrival_version", 0)
                    self.last_resched_completion_count = len(self.completed_jobs)

                    reschedule_executed = True

        # -------------------------------
        # 3) Execute AMR tasks & advance time
        # -------------------------------
        self.update_amr_tasks()
        
        if reschedule_executed:
            # Time was already advanced during compute window above
            # dt for reward/return reflects the real elapsed time
            dt = float(math.ceil(max(1.0, compute_time)))
        else:
            dt = float(math.ceil(max(1.0, compute_time)))
            self.sim_time += dt
            # 4) Completion: only tick when we didn't already tick during compute
            self._check_job_completions(dt)

        # -------------------------------
        # 2.5) Flow-time penalty (Calculated AFTER time advance to penalize GA delay)
        # -------------------------------
        # Penalize ALL active jobs (unstarted + assigned/processing) to minimize total flow time.
        reward -= len(self.active_jobs) * dt * FLOW_PENALTY

        # -------------------------------
        # 4) Completion reward
        # -------------------------------
        done_now = len(self.completed_jobs) - before_done
        if done_now > 0:
            reward += DONE_REWARD * done_now

        # -------------------------------
        # 5) Termination
        # -------------------------------
        done = (self.sim_time >= CONFIG['SIM_TIME'])
        return self.get_state_arrays(), reward, done, float(dt)






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
                a.remaining_time / (2*CONFIG['SIM_TIME_SCALE']), 
                a.inventory.get('A', 0) / CONFIG['CAPACITY_PER_TYPE'], 
                a.inventory.get('B', 0) / CONFIG['CAPACITY_PER_TYPE'], 
                a.inventory.get('C', 0) / CONFIG['CAPACITY_PER_TYPE'], 
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
                    j.proc_time / CONFIG['SIM_TIME_SCALE'], 
                    (self.sim_time - j.arrival_ts) / 100.0,
                    j.dest_pos[0] / 10.0, 
                    j.dest_pos[1] / 10.0, 
                    j.supply_pos[0] / 10.0, 
                    j.supply_pos[1] / 10.0, 
                    *mat
                ])
        
        # 3. Queue Features: [Count, Avg_Proc_Time, Var_Proc_Time]
        unstarted_cnt = sum(j.status == 1 for j in self.active_jobs)

        waiting_jobs = [j for j in self.queue if j.arrival_ts <= self.sim_time]
        buf_cnt = len(waiting_jobs)

        if buf_cnt > 0:
            proc_times = [j.proc_time for j in waiting_jobs]
            avg_proc = sum(proc_times) / buf_cnt
            variance_proc = sum((x - avg_proc) ** 2 for x in proc_times) / buf_cnt
            q_data = [
                float(unstarted_cnt),
                float(buf_cnt),
                avg_proc / 20.0,
                self.sim_time / CONFIG['SIM_TIME']
            ]
        else:
            q_data = [
                float(unstarted_cnt),
                0.0,
                0.0,
                self.sim_time / CONFIG['SIM_TIME']
            ]
        
        return a_data, j_data, q_data
# ==========================================
# 4. BATCHED GNN MODEL (THE GPU FIX)
# ==========================================
class BatchedHeteroGNN(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        # AMR sees: [Self, Job_Mean, Job_Max] -> 3 * h_dim
        self.upd_amr = nn.Sequential(nn.Linear(h_dim * 3, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))
        # Job sees: [Self, AMR_Mean] -> 2 * h_dim
        self.upd_job = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))

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
        
        # Max Pooling (Critical for detecting outliers like high wait times)
        # Since h_job is ReLU output (>=0), padding with 0 is safe for max
        job_max = masked_job.max(dim=1, keepdim=True)[0] # [Batch, 1, H]

        # Expand to all AMRs
        msg_mean_to_amr = job_mean.expand(-1, h_amr.size(1), -1)
        msg_max_to_amr = job_max.expand(-1, h_amr.size(1), -1)
        
        # 2. Pool AMRs -> Message to Job
        amr_mean = h_amr.mean(dim=1, keepdim=True) # [Batch, 1, H]
        msg_to_job = amr_mean.expand(-1, h_job.size(1), -1)

        # 3. Update
        # Concatenate features instead of adding them
        in_amr = torch.cat([h_amr, msg_mean_to_amr, msg_max_to_amr], dim=-1)
        out_amr = self.upd_amr(in_amr)
        
        in_job = torch.cat([h_job, msg_to_job], dim=-1)
        out_job = self.upd_job(in_job)
        
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
    states, actions, rewards, next_states, dones, cur_amasks, next_amasks, dts = zip(*batch_list)    
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
    
    b_cur_amask  = torch.tensor(cur_amasks, dtype=torch.float32, device=CONFIG['DEVICE'])   # [B,2]
    b_next_amask = torch.tensor(next_amasks, dtype=torch.float32, device=CONFIG['DEVICE'])  # [B,2]
    b_dt = torch.tensor(dts, dtype=torch.float32, device=CONFIG['DEVICE']).unsqueeze(1)

    return (s_amr, s_job, s_q, s_mask), b_a, b_r, (ns_amr, ns_job, ns_q, ns_mask), b_d, b_cur_amask, b_next_amask, b_dt

# ==========================================
# 6. TRAINING
# ==========================================
class ReplayBuffer:
    def __init__(self, cap): self.buf = deque(maxlen=cap)
    def push(self, x): self.buf.append(x)
    def sample(self, n): return random.sample(self.buf, n)
    def __len__(self): return len(self.buf)

def optimize(agent, target, opt, memory):
    if len(memory) < CONFIG['BATCH_SIZE']:
        return 0.0

    batch_raw = memory.sample(CONFIG['BATCH_SIZE'])
    curr_state, act, rew, next_state, done, cur_amask, next_amask, dt_batch = collate_batch(batch_raw)
    # Current Q
    q_all = agent(*curr_state)  # [B,2]
    # invalid -> very negative
    q_all = q_all + (cur_amask - 1.0) * 1e9
    q_curr = q_all.gather(1, act)

    with torch.no_grad():
        ns_amr, ns_job, ns_q, ns_mask = next_state

        # online net chooses next action (masked)
        q_next_online = agent(ns_amr, ns_job, ns_q, ns_mask)        # [B,2]
        q_next_online = q_next_online + (next_amask - 1.0) * 1e9
        next_acts = q_next_online.argmax(1, keepdim=True)           # [B,1]

        # target net evaluates it (masked same way也行，保險)
        q_next_target = target(ns_amr, ns_job, ns_q, ns_mask)        # [B,2]
        q_next_target = q_next_target + (next_amask - 1.0) * 1e9
        next_vals = q_next_target.gather(1, next_acts)

        # SMDP Time-Discounting: scale GAMMA by the actual elapsed time dt
        gamma_dt = CONFIG['GAMMA'] ** dt_batch
        q_target = rew + gamma_dt * next_vals * (1 - done)
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
    agent.train()
    target = SchedulerAgent().to(CONFIG['DEVICE'])
    target.load_state_dict(agent.state_dict())
    target.eval()
    
    opt = optim.Adam(agent.parameters(), lr=CONFIG['LR'])
    memory = ReplayBuffer(20000)
    
    for ep in range(CONFIG['NUM_EPISODES']):
        state = env.reset() # Returns CPU lists
        ep_rew, ep_loss, opt_steps = 0, 0, 0
        eps = CONFIG['EPS_END'] + (CONFIG['EPS_START'] - CONFIG['EPS_END']) * math.exp(-1.*ep/CONFIG['EPS_DECAY'])
        step_i = 0
        t0 = time.time()
        while True:
            step_i += 1

            
            mask = env.get_action_mask()
            # Select Action (Single Inference)
            if random.random() < eps:
                # 只從 valid actions sample
                valid_actions = [i for i, m in enumerate(mask) if m > 0.5]
                action = random.choice(valid_actions)
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

                    t_start = time.perf_counter()
                    q = agent(s_amr, s_job, s_q, s_mask)  # [1,2]

                    # action mask: invalid -> -inf
                    if mask[1] < 0.5:
                        q[0, 1] = -1e9

                    action = q.argmax(1).item()
                    action_str = "RESCHEDULE" if action == 1 else "WAIT"
                    print(f"Ep {ep} | Active Jobs: {len(env.active_jobs)} | Action: {action_str} | GNN+DDQN Time: {(time.perf_counter() - t_start) * 1000:.4f} ms")
            
            

            curr_action_mask = env.get_action_mask()  # before step

            next_state, reward, done, dt = env.step(action)
            
            if action == 1:
                print(f"Ep {ep} | Active Jobs: {len(env.active_jobs)} | Pairing Computation Time: {env.last_ga_compute_time:.2f} seconds")

            next_action_mask = env.get_action_mask()  # after step (sim_time/last_resched 已更新)

            memory.push((state, action, reward, next_state, done, curr_action_mask, next_action_mask, dt))
            state = next_state

            ep_rew += reward
            
            WARMUP_STEPS = 3000
            UPDATE_EVERY = 4

            if len(memory) > WARMUP_STEPS and (step_i % UPDATE_EVERY == 0):
                loss_val = optimize(agent, target, opt, memory)
                ep_loss += loss_val
                opt_steps += 1
            if done: break
            
        if ep % 10 == 0:
            target.load_state_dict(agent.state_dict())
            avg_loss = ep_loss / opt_steps if opt_steps > 0 else 0.0
            print(f"Ep {ep} | Reward: {ep_rew:.1f} | Avg Loss: {avg_loss:.4f} | Eps: {eps:.2f}")

        if ep % 100 == 0:
            ckpt_path = f"gnn_ddqn_model_v6/gnn_ddqn_model_v6_ep{ep}.pth"
            torch.save(agent.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    torch.save(agent.state_dict(), CONFIG['SAVE_PATH'])
    print("Done.")

if __name__ == "__main__":
    main()