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
import os
STATIC_ALGO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Static_alogorithm'))
if STATIC_ALGO_PATH not in sys.path:
    sys.path.append(STATIC_ALGO_PATH)

from GA.GA import evolve as ga_evolve, Job as GAJob, local_improve, routing_iters
from Attention.Attention import SchedulerAttention, solve_with_attention




# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'DATASET_PATH': '../../test_case/dynamic/training_dataset_r4.jsonl',
    'SAVE_PATH': '../models_pth/attention_ddqn_model_v7',
    
    # Physics
    'GRID_WIDTH': 10,
    'SCALE': 1.0,
    'AMR_SPEED': 1.0,
    'CAPACITY_PER_TYPE': 3,
    'SIM_TIME': 500.0,  # Max sim time per episode
    'SIM_TIME_SCALE': 25.0, # For normalizing time features
    'COMPUTE_TIME_SCALING': 1.0,
    
    # Training
    'NUM_EPISODES': 1000,
    'BATCH_SIZE': 64,
    'GAMMA': 0.99,
    'LR': 3e-4,
    'FLOW_PENALTY': 0.01,
    'EPS_START': 1.0,
    'EPS_END': 0.05,
    'EPS_DECAY': 200,
    'RESCHED_COOLDOWN': 1.0,
    
    # Model
    'AMR_IN_DIM': 8, 
    'JOB_IN_DIM': 11, 
    'QUEUE_DIM': 6, 
    'HIDDEN_DIM': 256,
    'ATTENTION_LAYERS': 2,
    'ACTION_DIM': 2,    # 0: Wait, 1: Release
    
    # GA Hyperparameters
    'GA_POP_SIZE': 200,       # Increased from 50
    'GA_GENERATIONS': 150,    # Increased from 100
    'GA_ROUTING_ITERS': 1000,   # WARNING: Lowered from 1000. 1000 makes RL training prohibitively slow!
    'GA_COLLISION_ITERS': 2000,  # Disabled for RL training speed
    'GA_ROUTING_MAX_DEPTH': 100
}

# ==========================================
# 2. MAP & PATHFINDING (A*)
# ==========================================
class WarehouseMap:
    def __init__(self):
        self.barriers = set()

        self.W = 10  # GRID_W
        self.H = 11  # GRID_H
        self._precompute_all_pairs()

    def _precompute_all_pairs(self):
        W, H = self.W, self.H
        # dist[(sx,sy)][(ex,ey)] = steps
        self.dist = {}
        for sx in range(W):
            for sy in range(H):
                if (sx, sy) in self.barriers:
                    continue
                d = {(sx, sy): 0}
                q = deque([(sx, sy)])
                while q:
                    x, y = q.popleft()
                    for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
                        if 0<=nx<W and 0<=ny<H and (nx,ny) not in self.barriers and (nx,ny) not in d:
                            d[(nx,ny)] = d[(x,y)] + 1
                            q.append((nx,ny))
                self.dist[(sx, sy)] = d

    def get_true_distance(self, start_float, end_float):
        sx, sy = int(start_float[0]/CONFIG['SCALE']), int(start_float[1]/CONFIG['SCALE'])
        ex, ey = int(end_float[0]/CONFIG['SCALE']), int(end_float[1]/CONFIG['SCALE'])
        sx, sy = max(0, min(self.W - 1, sx)), max(0, min(self.H - 1, sy))
        ex, ey = max(0, min(self.W - 1, ex)), max(0, min(self.H - 1, ey))

        if (sx,sy) not in self.dist: 
            return 999.0
        steps = self.dist[(sx,sy)].get((ex,ey), 999.0)
        return steps * CONFIG['SCALE']

GLOBAL_MAP = WarehouseMap()

# ==========================================
# 3. ENVIRONMENT & LOGIC
# ==========================================

STATIONS = {
    "SUPPLY_A": (0.0, 8.0), 
    "SUPPLY_B": (0.0, 5.0), 
    "SUPPLY_C": (0.0, 2.0),
    1: (9.0, 9.0), 
    2: (9.0, 7.0), 
    3: (9.0, 5.0), 
    4: (9.0, 3.0), 
    5: (9.0, 1.0)
}

JOB_PROPS = {
    "A": {"time": 5.0, "supply": "SUPPLY_A"},
    "B": {"time": 10.0, "supply": "SUPPLY_B"},
    "C": {"time": 15.0, "supply": "SUPPLY_C"}
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



# ==========================================
# 3. TICK-BY-TICK SIMULATOR (Identical to GA Collision-Free Routing)
# ==========================================
class TickSimulator:
    def __init__(self):
        from GA.GA import AMR_STARTS, TYPE_DURATION, STATIONS, SUPPLY_LOCATIONS
        self.t = 0
        self.positions = {amr: AMR_STARTS[amr] for amr in AMR_STARTS}
        self.inventory = {amr: {mat: 0 for mat in TYPE_DURATION.keys()} for amr in AMR_STARTS}
        if "AMR1" in self.inventory: self.inventory["AMR1"]["A"] = 3
        if "AMR2" in self.inventory: self.inventory["AMR2"]["B"] = 3
        if "AMR3" in self.inventory: self.inventory["AMR3"]["C"] = 3
        self.amr_states = {amr: {'mode': 'idle', 'goal': None, 'job': None, 'proc_ticks': 0} for amr in AMR_STARTS}
        self.amr_queues = {amr: deque() for amr in AMR_STARTS}
        self.station_occupied = {s: False for s in STATIONS}
        self.completed_jobs_jids = []

    def assign_schedules(self, order, amr_assignment, job_map):
        # Clear queues (but leave the ACTIVE job alone!)
        for amr in self.amr_queues:
            active_job = self.amr_states[amr]['job']
            self.amr_queues[amr].clear()
            if active_job is not None:
                self.amr_queues[amr].append(active_job)
            
        for job_idx in order:
            amr = amr_assignment[job_idx]
            jdata = job_map[job_idx] # {'jid', 'jtype', 'time', 'station'}
            # Map into a Job-like object
            import GA.GA as GA
            ga_job = GA.Job(idx=jdata['jid'], type_=jdata['jtype'], station=f"station{jdata['station']}", duration=jdata['time'])
            self.amr_queues[amr].append(ga_job)

    def get_attention_init_state(self) -> dict:
        from GA.GA import STATIONS, SUPPLY_LOCATIONS, shortest_path, TYPE_DURATION
        state = {
            "time": self.t,
            "positions": {},
            "availability": {},
            "inventory": {amr: self.inventory[amr].copy() for amr in self.inventory}
        }
        for amr, s in self.amr_states.items():
            if s['job'] is not None:
                dest = STATIONS[s['job'].station]
                state['positions'][amr] = dest
                
                # Math projection for availability
                if s['mode'] in ['processing', 'processing_old']:
                    time_left = s['proc_ticks']
                elif s['mode'] == 'loading_dock':
                    mat = s['job'].type_
                    sup = SUPPLY_LOCATIONS[mat]
                    path = shortest_path(sup, dest)
                    time_left = s['proc_ticks'] + (len(path) - 1) + s['job'].duration
                elif s['mode'] == 'moving_station':
                    path = shortest_path(self.positions[amr], dest)
                    time_left = len(path) - 1 + s['job'].duration
                elif s['mode'] == 'moving_supply':
                    mat = s['job'].type_
                    sup = SUPPLY_LOCATIONS[mat]
                    path1 = shortest_path(self.positions[amr], sup)
                    path2 = shortest_path(sup, dest)
                    time_left = (len(path1) - 1) + TYPE_DURATION[mat] + (len(path2) - 1) + s['job'].duration
                else:
                    time_left = 0

                state['availability'][amr] = self.t + time_left
                
                # Inventory projection
                if s['mode'] in ['moving_supply', 'loading_dock']:
                    state['inventory'][amr][s['job'].type_] = 2
                else:
                    state['inventory'][amr][s['job'].type_] = max(0, state['inventory'][amr][s['job'].type_] - 1)
            else:
                state['positions'][amr] = self.positions[amr]
                state['availability'][amr] = self.t
        return state

    def step(self, dt: int):
        from GA.GA import AMR_KEYS, SUPPLY_LOCATIONS, STATIONS, AMR_STARTS, shortest_path, OBSTACLES, _is_within_bounds
        import random
        # Extrapolate forward by exactly dt ticks
        for _ in range(dt):
            # 1. Transitions
            for amr in AMR_KEYS:
                s = self.amr_states[amr]
                if s['mode'] == 'idle':
                    if len(self.amr_queues[amr]) > 0:
                        s['job'] = self.amr_queues[amr][0]
                        mat = s['job'].type_
                        if self.inventory[amr][mat] == 0:
                            s['mode'] = 'moving_supply'
                            s['goal'] = SUPPLY_LOCATIONS[mat]
                        else:
                            s['mode'] = 'moving_station'
                            s['goal'] = STATIONS[s['job'].station]
                    else:
                        if self.positions[amr] != AMR_STARTS[amr]:
                            s['mode'] = 'moving_base'
                            s['goal'] = AMR_STARTS[amr]
                            s['job'] = None
                elif s['mode'] == 'processing':
                    s['proc_ticks'] -= 1
                    if s['proc_ticks'] <= 0:
                        mat = s['job'].type_
                        self.inventory[amr][mat] -= 1
                        self.station_occupied[s['job'].station] = False
                        self.completed_jobs_jids.append(s['job'].idx)
                        self.amr_queues[amr].popleft()
                        s['mode'] = 'idle'
                        s['job'] = None
                        s['goal'] = None
                elif s['mode'] == 'loading_dock':
                    s['proc_ticks'] -= 1
                    if s['proc_ticks'] <= 0:
                        mat = s['job'].type_
                        self.inventory[amr][mat] = 3
                        s['mode'] = 'idle'
                        s['goal'] = None
            
            # Additional Transition hook for immediately queued items
            for amr in AMR_KEYS:
                s = self.amr_states[amr]
                if s['mode'] == 'idle':
                    if len(self.amr_queues[amr]) > 0:
                        s['job'] = self.amr_queues[amr][0]
                        mat = s['job'].type_
                        if self.inventory[amr][mat] == 0:
                            s['mode'] = 'moving_supply'
                            s['goal'] = SUPPLY_LOCATIONS[mat]
                        else:
                            s['mode'] = 'moving_station'
                            s['goal'] = STATIONS[s['job'].station]
                    else:
                        if self.positions[amr] != AMR_STARTS[amr]:
                            s['mode'] = 'moving_base'
                            s['goal'] = AMR_STARTS[amr]
                            s['job'] = None
            
            # 2. Movement Negotiation (Collision-Free Routing Logic directly from GA)
            moves = {}
            def prio(amr):
                m = self.amr_states[amr]['mode']
                if m in ['processing', 'processing_old', 'loading_dock']: return 100
                if m == 'idle': return 10
                if m == 'moving_station': return 50
                if m == 'moving_supply': return 40
                return 30
                
            ordered = sorted(AMR_KEYS, key=prio, reverse=True)
            reserved = set()
            
            for amr in ordered:
                m = self.amr_states[amr]['mode']
                p = self.positions[amr]
                
                is_blocking_station = False
                is_blocking_highway = False
                if m == 'idle':
                    for st_pos in STATIONS.values():
                        if p == st_pos:
                            is_blocking_station = True
                            break
                    if p[0] == 2:
                        is_blocking_highway = True
                        
                if m in ['processing', 'processing_old', 'loading_dock']:
                    moves[amr] = p
                    reserved.add(p)
                elif m == 'idle' and not is_blocking_station and not is_blocking_highway and p == self.amr_states[amr].get('goal', p):
                    moves[amr] = p
                    reserved.add(p)
                elif m == 'moving_station' and p == self.amr_states[amr]['goal']:
                    moves[amr] = p
                    reserved.add(p)
                    
            for amr in ordered:
                if amr in moves: continue
                p = self.positions[amr]
                s = self.amr_states[amr]
                g = s['goal'] if s.get('goal') is not None else p
                if s.get('dodge_ticks', 0) > 0:
                    g = s['dodge_goal']
                    s['dodge_ticks'] -= 1
                    
                path = shortest_path(p, g)
                next_step = path[1] if len(path) > 1 else p
                
                if next_step in reserved:
                    moves[amr] = p
                    reserved.add(p)
                else:
                    swap = False
                    for o_amr, o_next in moves.items():
                        if o_next == p and self.positions[o_amr] == next_step:
                            swap = True
                            break
                    if swap:
                        moves[amr] = p
                        reserved.add(p)
                    else:
                        moves[amr] = next_step
                        reserved.add(next_step)
                        
                if moves[amr] == p and next_step != p:
                    s['blocked_ticks'] = s.get('blocked_ticks', 0) + 1
                    if s['blocked_ticks'] > 5 and s['mode'] in ['moving_supply', 'moving_station', 'moving_base', 'idle']:
                        if s['mode'] == 'moving_base':
                            possible_dodges = []
                            for dy in [0, 1, 8, 9]:
                                for dx in range(0, 20):
                                    dpos = (dx, dy)
                                    if dpos not in OBSTACLES: possible_dodges.append(dpos)
                            if possible_dodges:
                                s['goal'] = random.choice(possible_dodges)
                                s['blocked_ticks'] = 0
                        else:
                            possible_dodges = []
                            for dy in range(-3, 4):
                                for dx in range(-3, 4):
                                    dpos = (p[0]+dx, p[1]+dy)
                                    if _is_within_bounds(dpos) and dpos not in OBSTACLES:
                                        possible_dodges.append(dpos)
                            if possible_dodges:
                                s['dodge_goal'] = random.choice(possible_dodges)
                                s['dodge_ticks'] = 15
                else:
                    s['blocked_ticks'] = 0
                    
            # 3. Apply moves
            for amr in AMR_KEYS:
                self.positions[amr] = moves[amr]
                s = self.amr_states[amr]
                p = self.positions[amr]
                if s['mode'] == 'moving_supply' and p == s['goal']:
                    from GA.GA import TYPE_DURATION
                    mat = s['job'].type_
                    s['mode'] = 'loading_dock'
                    s['proc_ticks'] = TYPE_DURATION[mat]
                elif s['mode'] == 'moving_station' and p == s['goal']:
                    if not self.station_occupied[s['job'].station]:
                        self.station_occupied[s['job'].station] = True
                        s['mode'] = 'processing'
                        s['proc_ticks'] = s['job'].duration
                elif s['mode'] == 'moving_base' and p == s['goal']:
                    s['mode'] = 'idle'
                    
            self.t += 1

# ==========================================
class GridEnv:
    def __init__(self):
        self.last_ga_compute_time = 0.0

        self.episodes = []
        self.last_resched_t = -999
        with open(CONFIG['DATASET_PATH'], 'r') as f:
            for line in f: self.episodes.append(json.loads(line))
        self.ep_idx = 0
        self.scheduled_queue = deque() # Stores the result of GA
        
        # Load the pretrained heuristic Attention model
        self.heuristic_attention = SchedulerAttention(amr_in_dim=8, job_in_dim=16, hidden_dim=128, attention_layers=2)
        weights_path = os.path.join(os.path.dirname(__file__), '../../Static_alogorithm/Attention/attention_scheduler_best.pth')
        if os.path.exists(weights_path):
            self.heuristic_attention.load_state_dict(torch.load(weights_path, map_location=CONFIG['DEVICE']))
        self.heuristic_attention.to(CONFIG['DEVICE'])

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
            # Randomly choose processing time from [5, 10, 15] deterministically using the job ID
            jid = raw.get("id", raw.get("jid", 0))
            duration = float(random.Random(jid).choice([5, 10, 15]))
            self.queue.append(Job(
                jid=jid, jtype=raw['type'], material=raw['type'],
                arrival_ts=raw['arrival_time'], proc_time=duration,
                supply_pos=STATIONS[props['supply']], 
                dest_pos=STATIONS[raw['dest_station_id']]
            ))
        
        self.active_jobs = [] 
        self.completed_jobs = [] # <--- NEW: Metric Tracking
        self.total_jobs = len(data['jobs']) # Store target to allow early termination
        
        self.sim = TickSimulator()
        self.sim_time = 0.0
        
        self._release_arrived_jobs() # Ensure jobs at t=0 are visible in initial state
        self.last_resched_t = -1e9   # ✅ critical: reset cooldown
        return self.get_state_arrays()
    
    def can_reschedule(self):
        RESCHED_COOLDOWN = CONFIG['RESCHED_COOLDOWN']
        cooldown_ok = (self.sim_time - self.last_resched_t) >= RESCHED_COOLDOWN
        has_unstarted = any(j.status == 1 for j in self.active_jobs)
        new_job_since_last = (self.arrival_version != self.last_resched_version)
        new_completion_since_last = (len(self.completed_jobs) != self.last_resched_completion_count)

        # Allow rescheduling on new arrival OR new completion
        return cooldown_ok and has_unstarted and (new_job_since_last or new_completion_since_last)

    def get_action_mask(self):
        return [1.0, 1.0 if self.can_reschedule() else 0.0]


    def calculate_current_makespan(self):
        from GA.GA import AMR_KEYS
        busy = max([s['proc_ticks'] for s in self.sim.amr_states.values() if s['mode'] in ['processing', 'processing_old', 'loading_dock']], default=0.0)
        
        # Distinguish scheduled (in AMR queues) vs truly unscheduled jobs
        scheduled_jids = set()
        for amr in AMR_KEYS:
            s = self.sim.amr_states[amr]
            if s['job'] is not None:
                scheduled_jids.add(s['job'].idx)
            for qj in self.sim.amr_queues[amr]:
                scheduled_jids.add(qj.idx)
        
        unscheduled = [j for j in self.active_jobs if j.status == 1 and j.jid not in scheduled_jids]
        scheduled = [j for j in self.active_jobs if j.status == 1 and j.jid in scheduled_jids]
        
        if not unscheduled and not scheduled:
            return max(busy, 1.0)
        
        # Queued jobs: their routing is already decided, estimate remaining work (spent on both dock and station)
        queued_cost = sum(j.proc_time * 2.0 for j in scheduled) / max(len(self.sim.positions), 1)
        
        # Unscheduled jobs: need full travel + proc estimate (spent on both dock and station)
        unscheduled_cost = 0.0
        if unscheduled:
            for j in unscheduled:
                best_dist = min(
                    GLOBAL_MAP.get_true_distance(self.sim.positions[amr], j.supply_pos) + 
                    GLOBAL_MAP.get_true_distance(j.supply_pos, j.dest_pos)
                    for amr in self.sim.positions
                ) + j.proc_time * 2.0
                unscheduled_cost += best_dist
            unscheduled_cost /= max(len(self.sim.positions), 1)
        
        return max(busy + queued_cost + unscheduled_cost, 1.0)
        
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

        # -------------------------------
        # 2) Apply action (reschedule)
        # -------------------------------
        reschedule_executed = False

        if action == 1:
            if not self.can_reschedule():
                reward -= EMPTY_RESCHED_PENALTY
            else:
                unstarted = [j for j in self.active_jobs if j.status == 1]
                if not unstarted:
                    reward -= EMPTY_RESCHED_PENALTY
                else:
                    reschedule_executed = True
                    from GA.GA import STATIONS, AMR_KEYS
                    pos_to_station = {v: k for k, v in STATIONS.items()}
                    
                    # Snapshot makespan BEFORE rescheduling (for R3: quality reward)
                    mk_before = self.calculate_current_makespan()
                    
                    ga_jobs = []
                    job_map = {}
                    for i, j in enumerate(unstarted):
                        st_name = pos_to_station.get((int(j.dest_pos[0]), int(j.dest_pos[1])), "M1_1")
                        from GA.GA import Job as GAJob
                        ga_j = GAJob(idx=i, type_=j.material, station=st_name, duration=j.proc_time)
                        ga_jobs.append(ga_j)
                        job_map[ga_j.idx] = {'jid': j.jid, 'jtype': j.material, 'time': j.proc_time, 'station': st_name.replace('station', '')}
                    
                    import time
                    start_cpu_time = time.perf_counter()
                    init_state = self.sim.get_attention_init_state()
                    # best_ind, _, compute_time = solve_with_attention(ga_jobs, self.heuristic_attention, deterministic=True, init_state=init_state)
                    import GA.GA as GA
                    GA.POPULATION_SIZE = CONFIG.get('GA_POP_SIZE', 200)
                    GA.GENERATIONS = CONFIG.get('GA_GENERATIONS', 150)
                    
                    best_ind, _ = ga_evolve(ga_jobs, init_state=init_state)
                    
                    best_ind = local_improve(best_ind, ga_jobs, max_iters=CONFIG.get('GA_ROUTING_ITERS', 1000), init_state=init_state)
                    collision_iters = CONFIG.get('GA_COLLISION_ITERS', 2000)
                    if collision_iters > 0:
                        best_ind = local_improve(best_ind, ga_jobs, max_iters=collision_iters, check_collision=True, init_state=init_state)
                    
                    compute_time = CONFIG.get('COMPUTE_TIME_SCALING', 1.0)*(time.perf_counter() - start_cpu_time)
                    self.last_ga_compute_time = compute_time

                    # ========================================================
                    # Fix time paradox: AMRs finish current job & wait 
                    # during compute time, then get the new schedule.
                    # ========================================================
                    dt = int(math.ceil(max(1.0, compute_time)))
                    
                    # 1. Freeze unstarted jobs
                    for amr in AMR_KEYS:
                        active_job = self.sim.amr_states[amr]['job']
                        self.sim.amr_queues[amr].clear()
                        if active_job is not None:
                            self.sim.amr_queues[amr].append(active_job)
                            
                    # 2. Simulate the world while CPU was calculating (AMRs finish current, then idle)
                    self.sim.step(dt)
                    
                    # 3. Inject new schedule AFTER calculation delay
                    self.sim.assign_schedules(best_ind.order, best_ind.amr_assignment, job_map)

                    # R2: Small fixed bonus for successful reschedule
                    reward += 1.0
                    
                    # R3: Schedule-quality reward — scales with backlog size
                    # More unstarted jobs → bigger benefit from rescheduling
                    mk_after = self.calculate_current_makespan()
                    if mk_before > 0:
                        improvement = max((mk_before - mk_after) / mk_before, 0.0)
                        # Scale with log(jobs) so benefit grows with backlog but doesn't explode
                        scale = math.log(max(len(unstarted), 1) + 1)
                        reward += 2.0 * improvement * scale

                    self.last_resched_t = float(self.sim.t)
                    self.last_resched_version = getattr(self, "arrival_version", 0)
                    self.last_resched_completion_count = len(self.completed_jobs)

        # 3) Execute AMR tasks & advance time (if no reschedule happened)
        if not reschedule_executed:
            dt = int(math.ceil(max(1.0, compute_time)))
            self.sim.step(dt)
            
        self.sim_time = float(self.sim.t)
        
        # Sync active jobs (status=2) from simulator
        # Find exactly which jobs are physically being processed
        running_jids = set()
        for amr, s in self.sim.amr_states.items():
            if s['job'] is not None:
                running_jids.add(s['job'].idx)
        for j in self.active_jobs:
            if j.status == 1 and j.jid in running_jids:
                j.status = 2  # Mark as processing

        reward -= len(self.active_jobs) * dt * FLOW_PENALTY
        
        # Unscheduled-jobs pressure: penalize having unassigned work sitting around
        # This prevents the agent from ignoring a growing backlog
        from GA.GA import AMR_KEYS as _AMR_KEYS
        _sched_jids = set()
        for _amr in _AMR_KEYS:
            _s = self.sim.amr_states[_amr]
            if _s['job'] is not None:
                _sched_jids.add(_s['job'].idx)
            for _qj in self.sim.amr_queues[_amr]:
                _sched_jids.add(_qj.idx)
        n_unscheduled = sum(1 for j in self.active_jobs if j.status == 1 and j.jid not in _sched_jids)
        reward -= 0.02 * n_unscheduled * dt  # Explicit pressure to address unscheduled jobs

        # Sync completed jobs
        done_now = 0
        for jid in self.sim.completed_jobs_jids:
            for idx, j in enumerate(self.active_jobs):
                if j.jid == jid:
                    j.status = 3
                    j.finish_ts = self.sim_time
                    self.completed_jobs.append(self.active_jobs.pop(idx))
                    done_now += 1
                    break
        self.sim.completed_jobs_jids.clear()
        
        if done_now > 0:
            reward += DONE_REWARD * done_now
            
        done = (self.sim_time >= CONFIG['SIM_TIME']) or (len(self.completed_jobs) >= self.total_jobs)
        return self.get_state_arrays(), reward, done, float(dt)







    def get_state_arrays(self):
        a_data = []
        from GA.GA import AMR_KEYS
        for amr in AMR_KEYS:
            s = self.sim.amr_states[amr]
            status_val = 0.0 if s['mode'] == 'idle' else 1.0
            rem = s['proc_ticks'] if s['mode'] in ['processing', 'processing_old'] else 0.0
            # S2: Use queue depth instead of wasted 0.0
            queue_depth = len(self.sim.amr_queues[amr]) / 10.0
            a_data.append([
                status_val, 
                rem / (2*CONFIG['SIM_TIME_SCALE']), 
                self.sim.inventory[amr].get('A', 0) / CONFIG['CAPACITY_PER_TYPE'], 
                self.sim.inventory[amr].get('B', 0) / CONFIG['CAPACITY_PER_TYPE'], 
                self.sim.inventory[amr].get('C', 0) / CONFIG['CAPACITY_PER_TYPE'], 
                self.sim.positions[amr][0] / 10.0, 
                self.sim.positions[amr][1] / 10.0, 
                queue_depth  # S2: AMR queue depth
            ])
            
        j_data = []
        if not self.active_jobs: 
            j_data.append([0.0] * CONFIG['JOB_IN_DIM'])  # S4: match new dim
        else:
            # S1: Build scheduled set for status encoding
            scheduled_jids = set()
            for amr, s in self.sim.amr_states.items():
                if s['job'] is not None:
                    scheduled_jids.add(s['job'].idx)
                for qj in self.sim.amr_queues[amr]:
                    scheduled_jids.add(qj.idx)
            
            for j in self.active_jobs:
                mat = [1,0,0] if j.material=='A' else ([0,1,0] if j.material=='B' else [0,0,1])
                # S4: Encode job status (0=unstarted/unassigned, 0.5=queued, 1=processing)
                if j.status == 2:
                    j_status = 1.0
                elif j.jid in scheduled_jids:
                    j_status = 0.5
                else:
                    j_status = 0.0
                # S5: Clip wait time to prevent unbounded values
                wait_time = min((self.sim_time - j.arrival_ts) / 100.0, 5.0)
                j_data.append([
                    1.0, 
                    j.proc_time / CONFIG['SIM_TIME_SCALE'], 
                    wait_time,          # S5: clipped
                    j.dest_pos[0] / 10.0, 
                    j.dest_pos[1] / 10.0, 
                    j.supply_pos[0] / 10.0, 
                    j.supply_pos[1] / 10.0, 
                    *mat,
                    j_status            # S4: new status feature (JOB_IN_DIM=11)
                ])
                
        # S1: Count truly unscheduled jobs (not in any AMR queue or active slot)
        if not self.active_jobs:
            unstarted_cnt = 0
        else:
            s_jids = set()
            for amr, s in self.sim.amr_states.items():
                if s['job'] is not None:
                    s_jids.add(s['job'].idx)
                for qj in self.sim.amr_queues[amr]:
                    s_jids.add(qj.idx)
            unstarted_cnt = sum(1 for j in self.active_jobs if j.status == 1 and j.jid not in s_jids)
        
        waiting_jobs = [j for j in self.queue if j.arrival_ts <= self.sim_time]
        buf_cnt = len(waiting_jobs)
        avg_proc = 0.0
        if buf_cnt > 0:
            avg_proc = sum(j.proc_time for j in waiting_jobs) / buf_cnt / 20.0

        # S3: Expanded queue features (QUEUE_DIM=6)
        avg_queue_depth = sum(len(self.sim.amr_queues[amr]) for amr in AMR_KEYS) / max(len(AMR_KEYS), 1) / 10.0
        time_since_resched = min((self.sim_time - self.last_resched_t) / 100.0, 5.0)
        est_makespan = self.calculate_current_makespan() / CONFIG['SIM_TIME']
        
        q_data = [
            float(unstarted_cnt),              # [0] truly unscheduled jobs
            float(buf_cnt),                    # [1] jobs waiting in arrival queue
            avg_proc,                          # [2] avg processing time of waiting jobs
            avg_queue_depth,                   # [3] S3: avg AMR queue depth
            time_since_resched,                # [4] S3: time since last reschedule
            est_makespan,                      # [5] S3: estimated makespan (schedule quality)
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
        self.state_gnn = BatchedHeteroGNN(h)
        self.head_val = nn.Sequential(nn.Linear(h+CONFIG['QUEUE_DIM'], h), nn.ReLU(), nn.Linear(h, 1))
        self.head_adv = nn.Sequential(nn.Linear(h+CONFIG['QUEUE_DIM'], h), nn.ReLU(), nn.Linear(h, CONFIG['ACTION_DIM']))

    def forward(self, x_amr, x_job, x_q, job_mask):
        # x_amr: [B, 3, 8], x_job: [B, N, 10], mask: [B, N, 1]
        h_amr = F.relu(self.enc_amr(x_amr))
        h_job = F.relu(self.enc_job(x_job))
        
        h_amr, _ = self.state_gnn(h_amr, h_job, job_mask)
        
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
                if j_list[0][0] == 0.0:
                    b_mask[i, :L, :] = 0.0
                else:
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
                    print(f"Ep {ep} | Active Jobs: {len(env.active_jobs)} | Action: {action_str} | Attention+DDQN Time: {(time.perf_counter() - t_start) * 1000:.4f} ms")
            
            

            curr_action_mask = env.get_action_mask()  # before step

            next_state, reward, done, dt = env.step(action)
            
            if action == 1:
                print(f"Ep {ep} | Active Jobs: {len(env.active_jobs)} | Pairing Computation Time: {env.last_ga_compute_time:.2f} seconds")

            next_action_mask = env.get_action_mask()  # after step (sim_time/last_resched 已更新)

            # R4: Skip forced-Wait transitions (mask=[1,0] and action=0)
            # These provide no learning signal about when to reschedule
            is_forced_wait = (action == 0 and curr_action_mask[1] < 0.5)
            if not is_forced_wait:
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
            ckpt_path = f"{CONFIG['SAVE_PATH']}/attention_ddqn_model_v7_ep{ep}.pth"
            torch.save(agent.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    torch.save(agent.state_dict(), f"{CONFIG['SAVE_PATH']}/attention_ddqn_model_v7.pth")
    print("Done.")

if __name__ == "__main__":
    main()
