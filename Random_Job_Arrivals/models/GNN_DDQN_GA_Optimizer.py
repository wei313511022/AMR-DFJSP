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
from typing import Optional

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'DATASET_PATH': 'training_dataset.jsonl',
    'SAVE_PATH': 'gnn_ddqn_model_v1.pth',
    
    # Physics
    'GRID_WIDTH': 10,
    'SCALE': 1.0,
    'AMR_SPEED': 10.0,
    'CAPACITY_PER_TYPE': 3,
    'SIM_TIME': 500.0,  # Max sim time per episode
    'SIM_TIME_SCALE': 25.0, # For normalizing time features (in get_state_arrays)
    
    # Training
    'NUM_EPISODES': 1000,
    'BATCH_SIZE': 32,      # Increased for GPU efficiency
    'GAMMA': 0.99,
    'LR': 3e-4,
    'EPS_START': 1.0,
    'EPS_END': 0.05,
    'EPS_DECAY': 200,
    
    # Model
    'AMR_IN_DIM': 8, 
    'JOB_IN_DIM': 10, 
    'QUEUE_DIM': 4, 
    'HIDDEN_DIM': 32,     # Wider network for GPU
    'ACTION_DIM': 2,    # 0: Wait, 1: Release
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
    finish_ts: float = -1.0   # ✅ add this


@dataclass
class AMR:
    aid: int
    x: float = 10.0
    y: float = 10.0
    status: int = 0 
    remaining_time: float = 0.0
    local_queue: deque = field(default_factory=deque)
    inventory: Dict[str, int] = field(default_factory=lambda: {'A':0, 'B':0, 'C':0})
    tot_number_of_jobs: int = 0 # <--- NEW: Track total jobs handled for reward shaping
    current_job: int = -1 # <--- NEW: Track which job ID this AMR is holding

class GeneticOptimizer:
    def __init__(self, pop_size=100, generations=1000):
        self.pop_size = pop_size
        self.generations = generations

    def ipox_crossover(self, p1, p2, job_ids):
        """ Improved Precedence Operation Crossover """
        if len(job_ids) < 2: return p1, p2
        ids = list(job_ids)
        random.shuffle(ids)
        set1 = set(ids[:len(ids)//2])
        # set1 = set(job_ids[:len(job_ids)//2])
        
        def breed(parent_a, parent_b):
            child = [None] * len(parent_a)
            for i, jid in enumerate(parent_a):
                if jid in set1: child[i] = jid
            ptr = 0
            for i in range(len(child)):
                if child[i] is None:
                    while ptr < len(parent_b) and parent_b[ptr] in set1:
                        ptr += 1
                    if ptr >= len(parent_b):
                        # fallback: 找一個還沒用過的 jid
                        remaining = [x for x in parent_b if x not in set(child)]
                        child[i] = remaining[0]
                    else:
                        child[i] = parent_b[ptr]
                        ptr += 1
            return child
        return breed(p1, p2), breed(p2, p1)

    def complex_crossover(self, p1, p2, job_ids):
        # p1, p2 are lists of tuples: [(jid, aid), ...]
        # 1. Extract sequences and assignments
        s1, a1 = zip(*p1); s2, a2 = zip(*p2)
        
        # 2. Sequence Crossover (IPOX)
        child_seq1, child_seq2 = self.ipox_crossover(list(s1), list(s2), job_ids)
        
        # 3. Assignment Crossover (Uniform/Point)
        # We simply swap which AMR handles which job index
        child_assign1 = list(a1)
        child_assign2 = list(a2)
        for i in range(len(a1)):
            if random.random() > 0.5:
                child_assign1[i], child_assign2[i] = a2[i], a1[i]
                
        return list(zip(child_seq1, child_assign1)), list(zip(child_seq2, child_assign2))

    def evaluate_fitness(self, chrom, amrs, job_dict, base_time):
        # base_time should be env.sim_time
        t_avail = {a.aid: base_time + max(a.remaining_time, 0.0) for a in amrs}
        pos = {a.aid: (a.x, a.y) for a in amrs}

        total_flow = 0.0

        for jid, aid in chrom:
            j = job_dict.get(jid)
            if j is None:
                return -1e18  # invalid chrom => terrible

            dist = GLOBAL_MAP.get_true_distance(pos[aid], j.supply_pos) + \
                GLOBAL_MAP.get_true_distance(j.supply_pos, j.dest_pos)
            dur = dist / CONFIG['AMR_SPEED'] + j.proc_time

            finish_t = t_avail[aid] + dur
            # arrival_ts is already absolute in your simulation
            flow = finish_t - j.arrival_ts

            total_flow += flow

            # update AMR state in simulation
            t_avail[aid] = finish_t
            pos[aid] = j.dest_pos

        # GA typically maximizes fitness, but flow time is minimize.
        # Convert: higher is better.
        # Option A (bounded, stable):
        avg_flow = total_flow / max(len(chrom), 1)
        fitness = 1.0 / (1.0 + avg_flow)
        return fitness

    def mutate_sequence(self, chrom, p=0.2):
        if random.random() > p or len(chrom) < 2:
            return chrom
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
        return chrom
    
    def mutate_assignment(self, chrom, amr_ids, p=0.1):
        if random.random() > p:
            return chrom
        idx = random.randrange(len(chrom))
        jid, aid = chrom[idx]
        new_aid = random.choice(amr_ids)
        chrom[idx] = (jid, new_aid)
        return chrom

    def solve(self, waiting_jobs, amrs, env):
        start_cpu_time = time.time()
        if not waiting_jobs:
            return [], 0.0

        job_ids = [j.jid for j in waiting_jobs]
        amr_ids = [a.aid for a in amrs]
        job_dict = {j.jid: j for j in waiting_jobs}

        # Hyperparams
        elite_n = 2
        top_k = min(20, self.pop_size)   # e.g. top 20 parents
        p_mut_seq = 0.2
        p_mut_asg = 0.2

        # init pop
        pop = []
        for _ in range(self.pop_size):
            seq = random.sample(job_ids, len(job_ids))
            assign = [random.choice(amr_ids) for _ in range(len(job_ids))]
            pop.append(list(zip(seq, assign)))

        best_chrom = None
        best_fit = -1e18

        for _ in range(self.generations):
            # 1) fitness
            fitness_scores = [
                self.evaluate_fitness(chrom, amrs, job_dict, base_time=env.sim_time)
                for chrom in pop
            ]

            order = np.argsort(fitness_scores)[::-1]
            if fitness_scores[order[0]] > best_fit:
                best_fit = fitness_scores[order[0]]
                best_chrom = pop[order[0]]

            # 2) top-k selection + elitism
            new_pop = [pop[i] for i in order[:elite_n]]
            pool = [pop[i] for i in order[:top_k]]

            # 3) reproduce
            while len(new_pop) < self.pop_size:
                p1, p2 = random.sample(pool, 2)
                c1, c2 = self.complex_crossover(p1, p2, job_ids)

                # 4) mutation
                c1 = self.mutate_sequence(c1, p=p_mut_seq)
                c2 = self.mutate_sequence(c2, p=p_mut_seq)
                c1 = self.mutate_assignment(c1, amr_ids, p=p_mut_asg)
                c2 = self.mutate_assignment(c2, amr_ids, p=p_mut_asg)

                new_pop.extend([c1, c2])

            pop = new_pop[:self.pop_size]

        compute_duration = time.time() - start_cpu_time
        return best_chrom, compute_duration


class GridEnv:
    def __init__(self):
        self.last_ga_compute_time = 0.0

        self.episodes = []
        self.last_resched_t = -999
        with open(CONFIG['DATASET_PATH'], 'r') as f:
            for line in f: self.episodes.append(json.loads(line))
        self.ep_idx = 0
        self.ga_optimizer = GeneticOptimizer()
        self.scheduled_queue = deque() # Stores the result of GA

    def reset(self):
        self.arrival_version = 0          # 有新 job 到達就 +1
        self.last_resched_version = -1    # 上次 reschedule 時的 arrival_version

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
        RESCHED_COOLDOWN = 20.0
        cooldown_ok = (self.sim_time - self.last_resched_t) >= RESCHED_COOLDOWN
        has_unstarted = any(j.status == 1 for j in self.active_jobs)
        new_job_since_last = (self.arrival_version != self.last_resched_version)

        return cooldown_ok and has_unstarted and new_job_since_last

    def get_action_mask(self):
        return [1.0, 1.0 if self.can_reschedule() else 0.0]


    def update_amr_tasks(self):
        """
        Decentralized Execution: Each AMR checks its own local queue.
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
                    
                    # 2. Calculate travel + processing
                    # Leg 1: Current Pos -> Supply | Leg 2: Supply -> Destination
                    dist_to_supply = GLOBAL_MAP.get_true_distance((a.x, a.y), job.supply_pos)
                    dist_to_dest = GLOBAL_MAP.get_true_distance(job.supply_pos, job.dest_pos)
                    
                    travel_time = (dist_to_supply + dist_to_dest) / CONFIG['AMR_SPEED']
                    a.remaining_time = travel_time + job.proc_time
                    
                    # 3. Teleport AMR to destination (Simulating completion)
                    a.x, a.y = job.dest_pos
                    # print(f"[DBG] AMR {a.aid} takes job {next_jid}, rem={a.remaining_time:.1f}")

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
        DONE_REWARD = 5.0
        WAIT_PENALTY = 0.2
        EMPTY_RESCHED_PENALTY = 0.2

        reward = 0.0
        before_done = len(self.completed_jobs)
        compute_time = 0.0

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
                    best_assignments, compute_time = self.ga_optimizer.solve(unstarted, self.amrs, self)
                    self.last_ga_compute_time = compute_time

                    # clear old queues
                    for a in self.amrs:
                        a.local_queue.clear()

                    # fill new queues
                    if best_assignments:
                        for jid, aid in best_assignments:
                            if 0 <= aid < len(self.amrs):
                                self.amrs[aid].local_queue.append(jid)

                        # ✅ update reschedule gate correctly (after a real schedule)
                        self.last_resched_t = self.sim_time
                        self.last_resched_version = getattr(self, "arrival_version", 0)

                        # # ✅ print ONCE per GA run
                        # print(f"  [DISPATCH] t={self.sim_time:6.1f} GA={compute_time*1000:7.2f} ms "
                        #     f"| queues={[list(a.local_queue) for a in self.amrs]}")

                    else:
                        # GA returns empty (rare but possible)
                        reward -= EMPTY_RESCHED_PENALTY

        # -------------------------------
        # 2.5) Waiting-time penalty (both actions)
        # -------------------------------
        unstarted = [j for j in self.active_jobs if j.status == 1]
        wait_sum = sum(max(0.0, self.sim_time - j.arrival_ts) for j in unstarted)
        reward -= WAIT_PENALTY * wait_sum

        # -------------------------------
        # 3) Execute AMR tasks & advance time
        # -------------------------------
        self.update_amr_tasks()

        # ✅ Advance time ONCE: 1 tick + GA compute time
        dt = max(1.0, compute_time)
        self.sim_time += dt

        finished = self._check_job_completions(dt)  # assumes your function uses dt

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
        return self.get_state_arrays(), reward, done






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
        msg_to_amr = job_mean.expand(-1, h_amr.size(1), -1)
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
    states, actions, rewards, next_states, dones, cur_amasks, next_amasks = zip(*batch_list)
    
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

    return (s_amr, s_job, s_q, s_mask), b_a, b_r, (ns_amr, ns_job, ns_q, ns_mask), b_d, b_cur_amask, b_next_amask

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
    curr_state, act, rew, next_state, done, cur_amask, next_amask = collate_batch(batch_raw)

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

        q_target = rew + CONFIG['GAMMA'] * next_vals * (1 - done)

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

                    q = agent(s_amr, s_job, s_q, s_mask)  # [1,2]

                    # action mask: invalid -> -inf
                    if mask[1] < 0.5:
                        q[0, 1] = -1e9

                    action = q.argmax(1).item()
            
            

            curr_action_mask = env.get_action_mask()  # before step

            next_state, reward, done = env.step(action)
            # if step_i % 20 == 0:
            #     unstarted = sum(1 for j in env.active_jobs if j.status == 1)
            #     processing = sum(1 for j in env.active_jobs if j.status == 2)
            #     busy = sum(1 for a in env.amrs if a.status == 1)
            #     rems = [a.remaining_time for a in env.amrs]
            #     print(f"[ep {ep}] step={step_i} t={env.sim_time:.1f} "
            #         f"active={len(env.active_jobs)} unstarted={unstarted} proc={processing} "
            #         f"busyAMR={busy} rem={rems}")
            #     print(f"[DBG] t={env.sim_time:.1f} qfeat={state[2][0]} "
            #         f"active={len(env.active_jobs)} unstarted={sum(j.status==1 for j in env.active_jobs)} "
            #         f"mask={env.get_action_mask()} action={action} "
            #         f"lq={[len(a.local_queue) for a in env.amrs]}")

            next_action_mask = env.get_action_mask()  # after step (sim_time/last_resched 已更新)

            memory.push((state, action, reward, next_state, done, curr_action_mask, next_action_mask))
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

    torch.save(agent.state_dict(), CONFIG['SAVE_PATH'])
    print("Done.")

if __name__ == "__main__":
    main()