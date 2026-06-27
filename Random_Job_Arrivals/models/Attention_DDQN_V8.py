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
    'DATASET_PATH': '../../test_case/dynamic/test_dataset_demo.jsonl',
    'SAVE_PATH': '../models_pth/attention_ddqn_model_v8_demo',
    
    # Physics
    'GRID_WIDTH': 10,
    'SCALE': 1.0,
    'AMR_SPEED': 1.0,
    'CAPACITY_PER_TYPE': 3,
    'SIM_TIME': 500.0,  # Max sim time per episode
    'SIM_TIME_SCALE': 25.0, # For normalizing time features
    'COMPUTE_TIME_SCALING': 30.0,
    
    # Training
    'NUM_EPISODES': 1000,
    'BATCH_SIZE': 64,
    'GAMMA': 0.99,
    'LR': 3e-4,
    'FLOW_PENALTY': 0.01,
    'RESCHED_COOLDOWN': 1.0,
    'COMPUTE_PENALTY_WEIGHT': 0.5,  # Explicit penalty per second of GA compute
    
    # Model
    'AMR_IN_DIM': 8, 
    'JOB_IN_DIM': 11, 
    'QUEUE_DIM': 7, 
    'HIDDEN_DIM': 256,
    'ATTENTION_LAYERS': 2,
    'ACTION_DIM': 2,    # 0: Wait, 1: Release
    
    # Rainbow DQN
    'N_ATOMS': 51,
    'V_MIN': -50.0,
    'V_MAX': 200.0,
    'N_STEP': 3,
    'PER_ALPHA': 0.6,
    'PER_BETA_START': 0.4,
    'PER_BETA_END': 1.0,
    'TAU': 0.005,           # Soft update rate
    
    # GA Hyperparameters
    'GA_POP_SIZE': 200,
    'GA_GENERATIONS': 150,
    'GA_ROUTING_ITERS': 1000,
    'GA_COLLISION_ITERS': 1,
    'GA_ROUTING_MAX_DEPTH': 100
}

# ==========================================
# 2. MAP & PATHFINDING (A*)
# ==========================================
class WarehouseMap:
    def __init__(self):
        self.barriers = {
            (5, 1),(5, 2),(6, 1),(6, 2),(4, 5),(3, 5),(3, 8),
            (6, 4),(6, 5),(6, 8),(6, 9),(4, 6),(3, 1),(2, 3)
        }

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
        from GA.GA import STATIONS, SUPPLY_LOCATIONS, shortest_path
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
                elif s['mode'] == 'moving_station':
                    path = shortest_path(self.positions[amr], dest)
                    time_left = len(path) - 1 + s['job'].duration
                elif s['mode'] == 'moving_supply':
                    mat = s['job'].type_
                    sup = SUPPLY_LOCATIONS[mat]
                    path1 = shortest_path(self.positions[amr], sup)
                    path2 = shortest_path(sup, dest)
                    time_left = (len(path1) - 1) + (len(path2) - 1) + s['job'].duration
                else:
                    time_left = 0

                state['availability'][amr] = self.t + time_left
                
                # Inventory projection
                if s['mode'] == 'moving_supply':
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
                if m in ['processing', 'processing_old']: return 100
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
                        
                if m in ['processing', 'processing_old']:
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
                    mat = s['job'].type_
                    self.inventory[amr][mat] = 3
                    s['mode'] = 'idle'
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
            self.queue.append(Job(
                jid=raw['id'], jtype=raw['type'], material=raw['type'],
                arrival_ts=raw['arrival_time'], proc_time=props['time'],
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
        busy = max([s['proc_ticks'] for s in self.sim.amr_states.values() if s['mode'] in ['processing', 'processing_old']], default=0.0)
        
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
        
        # Queued jobs: their routing is already decided, estimate remaining work
        queued_cost = sum(j.proc_time for j in scheduled) / max(len(self.sim.positions), 1)
        
        # Unscheduled jobs: need full travel + proc estimate (more expensive)
        unscheduled_cost = 0.0
        if unscheduled:
            for j in unscheduled:
                best_dist = min(
                    GLOBAL_MAP.get_true_distance(self.sim.positions[amr], j.supply_pos) + 
                    GLOBAL_MAP.get_true_distance(j.supply_pos, j.dest_pos)
                    for amr in self.sim.positions
                ) + j.proc_time
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
                    best_ind, _, compute_time = solve_with_attention(ga_jobs, self.heuristic_attention, deterministic=True, init_state=init_state)
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

                    # R2: Explicit compute-time penalty (Rainbow V8 reward fix)
                    reward -= compute_time * CONFIG.get('COMPUTE_PENALTY_WEIGHT', 0.5)
                    
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

        # S3: Expanded queue features (QUEUE_DIM=7)
        avg_queue_depth = sum(len(self.sim.amr_queues[amr]) for amr in AMR_KEYS) / max(len(AMR_KEYS), 1) / 10.0
        time_since_resched = min((self.sim_time - self.last_resched_t) / 100.0, 5.0)
        est_makespan = self.calculate_current_makespan() / CONFIG['SIM_TIME']
        
        q_data = [
            float(unstarted_cnt),              # [0] truly unscheduled jobs
            float(buf_cnt),                    # [1] jobs waiting in arrival queue
            avg_proc,                          # [2] avg processing time of waiting jobs
            self.sim_time / CONFIG['SIM_TIME'], # [3] time progress
            avg_queue_depth,                   # [4] S3: avg AMR queue depth
            time_since_resched,                # [5] S3: time since last reschedule
            est_makespan,                      # [6] S3: estimated makespan (schedule quality)
        ]
        
        return a_data, j_data, q_data

# ==========================================
# 4. NOISY LINEAR LAYER (Rainbow Component 6)
# ==========================================
class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer (Fortunato et al., 2018)."""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ==========================================
# 5. BATCHED GNN MODEL (Same backbone as V7)
# ==========================================
class BatchedHeteroGNN(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.upd_amr = nn.Sequential(nn.Linear(h_dim * 3, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))
        self.upd_job = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))

    def forward(self, h_amr, h_job, job_mask):
        masked_job = h_job * job_mask
        job_sum = masked_job.sum(dim=1, keepdim=True)
        job_count = job_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        job_mean = job_sum / job_count
        job_max = masked_job.max(dim=1, keepdim=True)[0]

        msg_mean_to_amr = job_mean.expand(-1, h_amr.size(1), -1)
        msg_max_to_amr = job_max.expand(-1, h_amr.size(1), -1)

        amr_mean = h_amr.mean(dim=1, keepdim=True)
        msg_to_job = amr_mean.expand(-1, h_job.size(1), -1)

        in_amr = torch.cat([h_amr, msg_mean_to_amr, msg_max_to_amr], dim=-1)
        out_amr = self.upd_amr(in_amr)

        in_job = torch.cat([h_job, msg_to_job], dim=-1)
        out_job = self.upd_job(in_job)

        return out_amr, out_job * job_mask


# ==========================================
# 6. RAINBOW SCHEDULER AGENT (Distributional Dueling + NoisyNets)
# ==========================================
class SchedulerAgent(nn.Module):
    def __init__(self):
        super().__init__()
        h = CONFIG['HIDDEN_DIM']
        n_atoms = CONFIG['N_ATOMS']
        n_actions = CONFIG['ACTION_DIM']
        feat_dim = h + CONFIG['QUEUE_DIM']

        self.enc_amr = nn.Linear(CONFIG['AMR_IN_DIM'], h)
        self.enc_job = nn.Linear(CONFIG['JOB_IN_DIM'], h)
        self.state_gnn = BatchedHeteroGNN(h)

        # Dueling Value stream (outputs distribution over atoms)
        self.val_hidden = NoisyLinear(feat_dim, h)
        self.val_out = NoisyLinear(h, n_atoms)

        # Dueling Advantage stream (outputs distribution per action)
        self.adv_hidden = NoisyLinear(feat_dim, h)
        self.adv_out = NoisyLinear(h, n_actions * n_atoms)

        self.n_atoms = n_atoms
        self.n_actions = n_actions

        # Atom support
        self.register_buffer('support', torch.linspace(CONFIG['V_MIN'], CONFIG['V_MAX'], n_atoms))

    def forward(self, x_amr, x_job, x_q, job_mask):
        h_amr = F.relu(self.enc_amr(x_amr))
        h_job = F.relu(self.enc_job(x_job))
        h_amr, _ = self.state_gnn(h_amr, h_job, job_mask)

        shop_emb = h_amr.mean(dim=1)  # [B, H]
        state = torch.cat([shop_emb, x_q], dim=-1)  # [B, feat_dim]

        # Value distribution
        val = F.relu(self.val_hidden(state))
        val = self.val_out(val).view(-1, 1, self.n_atoms)  # [B, 1, N_ATOMS]

        # Advantage distribution
        adv = F.relu(self.adv_hidden(state))
        adv = self.adv_out(adv).view(-1, self.n_actions, self.n_atoms)  # [B, A, N_ATOMS]

        # Dueling aggregation per-atom, then softmax to get probabilities
        q_atoms = val + adv - adv.mean(dim=1, keepdim=True)  # [B, A, N_ATOMS]
        dist = F.softmax(q_atoms, dim=-1)  # [B, A, N_ATOMS]
        dist = dist.clamp(min=1e-3)  # Avoid log(0) in KL divergence

        return dist

    def q_values(self, x_amr, x_job, x_q, job_mask):
        """Compute expected Q-values from the distribution (for action selection)."""
        dist = self.forward(x_amr, x_job, x_q, job_mask)  # [B, A, N_ATOMS]
        q = (dist * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # [B, A]
        return q

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ==========================================
# 7. PRIORITIZED EXPERIENCE REPLAY (Rainbow Component 3)
# ==========================================
class SumTree:
    """Binary sum-tree for O(log n) proportional sampling."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Proportional PER with SumTree."""
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0
        self.capacity = capacity

    def push(self, transition):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size, beta=0.4):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            if data is None:
                # Fallback: resample from valid range
                s = random.uniform(0, self.tree.total())
                idx, p, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(p)

        # Importance-sampling weights
        total = self.tree.total()
        n = self.tree.n_entries
        min_prob = min(priorities) / total
        max_weight = (n * min_prob) ** (-beta) if min_prob > 0 else 1.0

        weights = []
        for p in priorities:
            prob = p / total
            w = (n * prob) ** (-beta) if prob > 0 else 1.0
            weights.append(w / max_weight)

        return batch, indices, torch.tensor(weights, dtype=torch.float32, device=CONFIG['DEVICE'])

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# ==========================================
# 8. N-STEP RETURN BUFFER (Rainbow Component 4)
# ==========================================
class NStepBuffer:
    """Accumulates n transitions and computes n-step discounted return with SMDP."""
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def append(self, transition):
        """transition = (state, action, reward, next_state, done, cur_mask, next_mask, dt)"""
        self.buffer.append(transition)

    def is_ready(self):
        return len(self.buffer) == self.n_step

    def get(self):
        """Compute n-step return and return (s0, a0, R_n, s_n, done_n, mask0, mask_n, total_dt)."""
        state, action, _, _, _, cur_mask, _, _ = self.buffer[0]

        # Compute n-step discounted return with SMDP time-discounting
        R = 0.0
        cumulative_dt = 0.0
        for i, (_, _, r, _, d, _, _, dt) in enumerate(self.buffer):
            R += (self.gamma ** cumulative_dt) * r
            cumulative_dt += dt
            if d:
                # Episode ended within n steps
                _, _, _, next_s, done, _, next_mask, _ = self.buffer[i]
                return state, action, R, next_s, done, cur_mask, next_mask, cumulative_dt

        # No terminal state within n steps
        _, _, _, next_s, done, _, next_mask, _ = self.buffer[-1]
        return state, action, R, next_s, done, cur_mask, next_mask, cumulative_dt

    def flush(self):
        """Flush remaining transitions at episode end (returns list of partial n-step transitions)."""
        results = []
        while len(self.buffer) > 0:
            state, action, _, _, _, cur_mask, _, _ = self.buffer[0]
            R = 0.0
            cumulative_dt = 0.0
            for i, (_, _, r, _, d, _, _, dt) in enumerate(self.buffer):
                R += (self.gamma ** cumulative_dt) * r
                cumulative_dt += dt
            _, _, _, next_s, done, _, next_mask, _ = self.buffer[-1]
            results.append((state, action, R, next_s, True, cur_mask, next_mask, cumulative_dt))
            self.buffer.popleft()
        return results

    def reset(self):
        self.buffer.clear()


# ==========================================
# 9. BATCH PROCESSING UTILS
# ==========================================
def collate_batch(batch_list):
    """
    Takes a list of (amr, job, queue) tuples and stacks them into Tensors.
    Handles variable number of jobs via Padding.
    """
    states, actions, rewards, next_states, dones, cur_amasks, next_amasks, dts = zip(*batch_list)
    def pad_and_stack(state_list):
        amrs, jobs, queues = zip(*state_list)
        b_amr = torch.tensor(amrs, dtype=torch.float32, device=CONFIG['DEVICE'])
        b_q = torch.tensor(queues, dtype=torch.float32, device=CONFIG['DEVICE'])
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
    b_cur_amask  = torch.tensor(cur_amasks, dtype=torch.float32, device=CONFIG['DEVICE'])
    b_next_amask = torch.tensor(next_amasks, dtype=torch.float32, device=CONFIG['DEVICE'])
    b_dt = torch.tensor(dts, dtype=torch.float32, device=CONFIG['DEVICE']).unsqueeze(1)

    return (s_amr, s_job, s_q, s_mask), b_a, b_r, (ns_amr, ns_job, ns_q, ns_mask), b_d, b_cur_amask, b_next_amask, b_dt


# ==========================================
# 10. DISTRIBUTIONAL TRAINING (Rainbow Components 1-5 combined)
# ==========================================
def project_distribution(next_dist, rewards, dones, dts, support, gamma, n_atoms, v_min, v_max):
    """
    Categorical projection of the Bellman update onto the fixed atom support.
    next_dist: [B, N_ATOMS] - target distribution for chosen next action
    rewards:   [B, 1]
    dones:     [B, 1]
    dts:       [B, 1] - SMDP elapsed time
    """
    delta_z = (v_max - v_min) / (n_atoms - 1)
    batch_size = rewards.size(0)

    rewards = rewards.squeeze(1)     # [B]
    dones = dones.squeeze(1)         # [B]
    dts = dts.squeeze(1)             # [B]

    # SMDP time-discounting
    gamma_dt = gamma ** dts  # [B]

    # Tz = r + gamma^dt * z (clipped to [V_MIN, V_MAX])
    Tz = rewards.unsqueeze(1) + gamma_dt.unsqueeze(1) * (1 - dones.unsqueeze(1)) * support.unsqueeze(0)  # [B, N_ATOMS]
    Tz = Tz.clamp(min=v_min, max=v_max)

    # Compute projection indices
    b = (Tz - v_min) / delta_z  # [B, N_ATOMS]
    l = b.floor().long()
    u = b.ceil().long()

    # Clamp indices
    l = l.clamp(0, n_atoms - 1)
    u = u.clamp(0, n_atoms - 1)

    # Distribute probability mass
    m = torch.zeros(batch_size, n_atoms, device=rewards.device)
    offset = torch.arange(batch_size, device=rewards.device).unsqueeze(1) * n_atoms

    # Lower bound contribution
    m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    # Upper bound contribution
    m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

    return m


def optimize(agent, target, opt, memory, beta):
    if len(memory) < CONFIG['BATCH_SIZE']:
        return 0.0

    n_atoms = CONFIG['N_ATOMS']
    v_min = CONFIG['V_MIN']
    v_max = CONFIG['V_MAX']
    gamma = CONFIG['GAMMA']

    batch_raw, tree_indices, is_weights = memory.sample(CONFIG['BATCH_SIZE'], beta)
    curr_state, act, rew, next_state, done, cur_amask, next_amask, dt_batch = collate_batch(batch_raw)

    # Current distribution
    curr_dist = agent(*curr_state)  # [B, A, N_ATOMS]
    act_expanded = act.unsqueeze(-1).expand(-1, -1, n_atoms)  # [B, 1, N_ATOMS]
    curr_dist_a = curr_dist.gather(1, act_expanded).squeeze(1)  # [B, N_ATOMS]
    log_curr = torch.log(curr_dist_a + 1e-8)

    with torch.no_grad():
        ns_amr, ns_job, ns_q, ns_mask = next_state

        # Double DQN: online net selects next action (masked)
        q_next_online = agent.q_values(ns_amr, ns_job, ns_q, ns_mask)  # [B, A]
        q_next_online = q_next_online + (next_amask - 1.0) * 1e9
        next_actions = q_next_online.argmax(1)  # [B]

        # Target net evaluates distribution of selected action
        next_dist_all = target(ns_amr, ns_job, ns_q, ns_mask)  # [B, A, N_ATOMS]
        next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, n_atoms)
        next_dist = next_dist_all.gather(1, next_actions_expanded).squeeze(1)  # [B, N_ATOMS]

        # Project target distribution
        target_dist = project_distribution(
            next_dist, rew, done, dt_batch,
            agent.support, gamma, n_atoms, v_min, v_max
        )

    # Cross-entropy loss with importance-sampling weights
    element_wise_loss = -(target_dist * log_curr).sum(dim=-1)  # [B]
    loss = (is_weights * element_wise_loss).mean()

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 10.0)
    opt.step()

    # Update priorities in PER (use KL-divergence as TD-error proxy)
    td_errors = element_wise_loss.detach().cpu().numpy()
    memory.update_priorities(tree_indices, td_errors)

    # Reset noise after optimization step
    agent.reset_noise()
    target.reset_noise()

    return loss.item()


# ==========================================
# 11. TRAINING LOOP (Rainbow)
# ==========================================
def main():
    print(f"--- GPU STATUS: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ---")
    print(f"--- Rainbow DQN V8 ---")

    os.makedirs(CONFIG['SAVE_PATH'], exist_ok=True)

    env = GridEnv()
    agent = SchedulerAgent().to(CONFIG['DEVICE'])
    agent.train()
    target = SchedulerAgent().to(CONFIG['DEVICE'])
    target.load_state_dict(agent.state_dict())
    target.eval()

    opt = optim.Adam(agent.parameters(), lr=CONFIG['LR'])
    memory = PrioritizedReplayBuffer(20000, alpha=CONFIG['PER_ALPHA'])
    n_step_buf = NStepBuffer(CONFIG['N_STEP'], CONFIG['GAMMA'])

    total_steps = 0
    WARMUP_STEPS = 3000
    UPDATE_EVERY = 4

    for ep in range(CONFIG['NUM_EPISODES']):
        state = env.reset()
        ep_rew, ep_loss, opt_steps = 0, 0, 0
        step_i = 0
        t0 = time.time()
        n_step_buf.reset()

        # Anneal PER beta linearly
        beta = CONFIG['PER_BETA_START'] + (CONFIG['PER_BETA_END'] - CONFIG['PER_BETA_START']) * (ep / max(CONFIG['NUM_EPISODES'] - 1, 1))

        while True:
            step_i += 1
            total_steps += 1

            mask = env.get_action_mask()

            # --- Action selection via NoisyNets (no epsilon-greedy!) ---
            with torch.no_grad():
                s_amr = torch.tensor([state[0]], dtype=torch.float32, device=CONFIG['DEVICE'])
                s_job = torch.tensor([state[1]], dtype=torch.float32, device=CONFIG['DEVICE'])
                s_q = torch.tensor([state[2]], dtype=torch.float32, device=CONFIG['DEVICE'])

                s_mask = torch.ones((1, s_job.size(1), 1), device=CONFIG['DEVICE'])
                if state[1][0][0] == 0.0:
                    s_mask = torch.zeros((1, s_job.size(1), 1), device=CONFIG['DEVICE'])

                q = agent.q_values(s_amr, s_job, s_q, s_mask)  # [1, 2]

                # Action mask: invalid -> -inf
                if mask[1] < 0.5:
                    q[0, 1] = -1e9

                action = q.argmax(1).item()
                action_str = "RESCHEDULE" if action == 1 else "WAIT"
                if action == 1:
                    print(f"Ep {ep} | Active Jobs: {len(env.active_jobs)} | Action: {action_str}")

            curr_action_mask = env.get_action_mask()
            next_state, reward, done, dt = env.step(action)

            if action == 1 and env.last_ga_compute_time > 0:
                print(f"Ep {ep} | Pairing Computation Time: {env.last_ga_compute_time:.2f}s")

            next_action_mask = env.get_action_mask()

            # Skip forced-Wait transitions (no learning signal)
            is_forced_wait = (action == 0 and curr_action_mask[1] < 0.5)
            if not is_forced_wait:
                # Feed into n-step buffer
                n_step_buf.append((state, action, reward, next_state, done, curr_action_mask, next_action_mask, dt))

                if n_step_buf.is_ready():
                    n_step_transition = n_step_buf.get()
                    memory.push(n_step_transition)

            state = next_state
            ep_rew += reward

            # Optimize
            if len(memory) > WARMUP_STEPS and (total_steps % UPDATE_EVERY == 0):
                loss_val = optimize(agent, target, opt, memory, beta)
                ep_loss += loss_val
                opt_steps += 1

                # Soft update target network
                tau = CONFIG['TAU']
                for tp, sp in zip(target.parameters(), agent.parameters()):
                    tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

            if done:
                # Flush remaining n-step transitions
                for t in n_step_buf.flush():
                    memory.push(t)
                break

        if ep % 10 == 0:
            avg_loss = ep_loss / opt_steps if opt_steps > 0 else 0.0
            elapsed = time.time() - t0
            print(f"Ep {ep} | Reward: {ep_rew:.1f} | Avg Loss: {avg_loss:.4f} | Beta: {beta:.3f} | Steps: {step_i} | Time: {elapsed:.1f}s")

        if ep % 100 == 0:
            ckpt_path = f"{CONFIG['SAVE_PATH']}/attention_ddqn_model_v8_ep{ep}.pth"
            torch.save(agent.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    torch.save(agent.state_dict(), f"{CONFIG['SAVE_PATH']}/attention_ddqn_model_v8.pth")
    print("Done.")

if __name__ == "__main__":
    main()
