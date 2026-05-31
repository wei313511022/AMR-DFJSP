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

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'DATASET_PATH': 'training_dataset.jsonl',
    'SAVE_PATH': 'gnn_ddqn_model_v5/gnn_ddqn_model_v5.pth',
    
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
    'GA_COLLISION_ITERS': 20,
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
# 2.1 GA Pathfinding Helpers (Adapted from GA.py)
# ==========================================

_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GRID_MIN_X, GRID_MAX_X = 0, 19
GRID_MIN_Y, GRID_MAX_Y = 0, 19

def _is_within_bounds(point: Tuple[int, int]) -> bool:
    x, y = point
    return GRID_MIN_X <= x <= GRID_MAX_X and GRID_MIN_Y <= y <= GRID_MAX_Y

def _adjacent_points(point: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = point
    neighbors = []
    for dx, dy in _DELTAS:
        candidate = (x + dx, y + dy)
        if not _is_within_bounds(candidate):
            continue
        if candidate in GLOBAL_MAP.barriers:
            continue
        neighbors.append(candidate)
    return neighbors

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _build_path(parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parents.get(current)
    return list(reversed(path))

def _manhattan_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = [start]
    x, y = start
    tx, ty = end
    dx = 1 if tx > x else -1
    while x != tx:
        x += dx
        path.append((x, y))
    dy = 1 if ty > y else -1
    while y != ty:
        y += dy
        path.append((x, y))
    return path

@lru_cache(maxsize=None)
def shortest_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    if start == end: return [start]
    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), 0, start))
    came_from = {start: None}
    g_score = {start: 0}
    
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == end:
            return _build_path(came_from, end)
        
        for neighbor in _adjacent_points(current):
            new_g = g + 1
            if neighbor not in g_score or new_g < g_score[neighbor]:
                g_score[neighbor] = new_g
                f = new_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f, new_g, neighbor))
                came_from[neighbor] = current
    return _manhattan_path(start, end)

def find_dynamic_path(start: Tuple[int, int], end: Tuple[int, int], start_time: float, 
                     reservations: set, amr_states: Dict[int, Tuple[Tuple[int, int], float]], 
                     active_amr: int) -> List[Tuple[int, int]]:
    t_start = int(start_time)
    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), 0, start, t_start))
    came_from = {}
    g_score = {(start, t_start): 0}
    MAX_DEPTH = CONFIG['GA_ROUTING_MAX_DEPTH']
    
    while open_set:
        _, g, current, t = heapq.heappop(open_set)
        if g > MAX_DEPTH: continue
        if current == end:
            path = []
            curr_key = (current, t)
            while curr_key in came_from:
                prev_key = came_from[curr_key]
                path.append(curr_key[0])
                curr_key = prev_key
            path.append(start)
            return list(reversed(path))
        
        moves = list(_DELTAS) + [(0, 0)]
        for dx, dy in moves:
            neighbor = (current[0] + dx, current[1] + dy)
            next_t = t + 1
            if not _is_within_bounds(neighbor): continue
            if neighbor in GLOBAL_MAP.barriers: continue
            
            if (neighbor, next_t) in reservations: continue
            
            collision_idle = False
            for other_amr, (pos, free_t) in amr_states.items():
                if other_amr == active_amr: continue
                if pos == neighbor and next_t >= free_t:
                    collision_idle = True
                    break
            if collision_idle: continue
            
            new_g = g + 1
            if (neighbor, next_t) not in g_score or new_g < g_score.get((neighbor, next_t), float('inf')):
                g_score[(neighbor, next_t)] = new_g
                f = new_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f, new_g, neighbor, next_t))
                came_from[(neighbor, next_t)] = (current, t)
    return [start]

def grid_distance(p: Tuple[int, int], q: Tuple[int, int]) -> float:
    path = shortest_path(p, q)
    return float(len(path) - 1)

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

# --- GA Data Structures & Logic ---

@dataclass
class Individual:
    order: List[int]          # permutation of job indices (0 to N-1) relative to the input list
    amr_assignment: List[int] # AMR ID assigned to each job index

class GeneticOptimizer:
    def __init__(self, pop_size=CONFIG['GA_POP_SIZE'], generations=CONFIG['GA_GENERATIONS']):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = 0.2
        self.stagnation_limit = 40
        self.routing_iters = CONFIG['GA_ROUTING_ITERS']
        self.collision_routing_iters = CONFIG['GA_COLLISION_ITERS']

    # --- Helper Functions ---
    def cluster_jobs_by_material(self, order: List[int], assignments: List[int], jobs: List[Job]) -> List[int]:
        keyed = []
        for idx, job_idx in enumerate(order):
            amr = assignments[job_idx]
            job_type = jobs[job_idx].jtype
            keyed.append((amr, job_type, idx, job_idx))
        # Sort by AMR, then Material, then original index 
        keyed.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in keyed]

    def find_adjacent_blocks(self, order: List[int], assignments: List[int], jobs: List[Job]) -> List[Tuple[int, int]]:
        job_count = len(order)
        blocks = []
        idx = 0
        while idx < job_count:
            start = idx
            current_job_idx = order[idx]
            curr_amr = assignments[current_job_idx]
            curr_type = jobs[current_job_idx].jtype
            idx += 1
            while idx < job_count:
                next_job_idx = order[idx]
                if assignments[next_job_idx] == curr_amr and jobs[next_job_idx].jtype == curr_type:
                    idx += 1
                else:
                    break
            if idx - start > 1:
                blocks.append((start, idx))
        return blocks

    def get_next_job_for_amr(self, amr_id: int, current_pos_in_order: int, order: List[int], assignments: List[int], jobs: List[Job]) -> Optional[Job]:
        # Map job index to Job object
        # Note: 'order' contains indices into 'jobs' list
        for i in range(current_pos_in_order + 1, len(order)):
            next_job_idx = order[i]
            if assignments[next_job_idx] == amr_id:
                return jobs[next_job_idx]
        return None

    def nearest_base_to_station(self, station_pos: Tuple[int, int]) -> Tuple[int, int]:
        # Use SUPPLY locations as bases
        bases = [(int(STATIONS["SUPPLY_A"][0]), int(STATIONS["SUPPLY_A"][1])),
                 (int(STATIONS["SUPPLY_B"][0]), int(STATIONS["SUPPLY_B"][1])),
                 (int(STATIONS["SUPPLY_C"][0]), int(STATIONS["SUPPLY_C"][1]))]
        return min(bases, key=lambda base: grid_distance(base, station_pos))

    def decode_schedule(self, individual: Individual, jobs: List[Job], amrs_state: List[AMR], check_collision: bool = False) -> Tuple[Dict[int, float], float]:
        # Initialize state from current environment
        availability = {}
        current_position = {}
        inventory = {}
        
        for amr in amrs_state:
            aid = amr.aid
            availability[aid] = max(0.0, amr.remaining_time)
            current_position[aid] = (int(amr.x), int(amr.y))
            inventory[aid] = amr.inventory.copy()

        # Track AMR states for collision avoidance: amr -> (position, free_time)
        amr_states = {amr.aid: (current_position[amr.aid], availability[amr.aid]) for amr in amrs_state}
        reservations = set() # (x, y, t)

        # Using dest_pos as key for stations
        station_available = {} 
        total_flow = 0.0

        order = individual.order
        
        for pos, job_idx in enumerate(order):
            job = jobs[job_idx]
            amr_id = individual.amr_assignment[job_idx]
            material = job.jtype
            
            # 1. Start Time
            start_time = availability[amr_id]
            
            # 2. Check Inventory / Supply
            if inventory[amr_id].get(material, 0) == 0:
                supply_pos = (int(job.supply_pos[0]), int(job.supply_pos[1]))
                if check_collision:
                    supply_path = find_dynamic_path(current_position[amr_id], supply_pos, start_time, reservations, amr_states, amr_id)
                else:
                    supply_path = shortest_path(current_position[amr_id], supply_pos)
                
                travel_time = float(len(supply_path) - 1)
                
                supply_end = start_time + travel_time
                if check_collision:
                    for t_offset, pt in enumerate(supply_path):
                        reservations.add((pt, int(start_time) + t_offset))
                    amr_states[amr_id] = (supply_pos, supply_end)

                availability[amr_id] = supply_end
                current_position[amr_id] = supply_pos
                start_time = supply_end
                inventory[amr_id][material] = 3 # Refill
            
            # 3. Travel to Destination
            dest_pos = (int(job.dest_pos[0]), int(job.dest_pos[1]))
            if check_collision:
                travel_path = find_dynamic_path(current_position[amr_id], dest_pos, start_time, reservations, amr_states, amr_id)
            else:
                travel_path = shortest_path(current_position[amr_id], dest_pos)
            
            travel_time = float(len(travel_path) - 1)
            travel_end = start_time + travel_time
            if check_collision:
                for t_offset, pt in enumerate(travel_path):
                    reservations.add((pt, int(start_time) + t_offset))
            
            availability[amr_id] = travel_end
            current_position[amr_id] = dest_pos
            
            # 4. Station Contention
            st_key = dest_pos
            st_avail = station_available.get(st_key, 0.0)
            earliest_start = max(travel_end, st_avail, job.arrival_ts)
            
            process_end = earliest_start + job.proc_time
            total_flow += (process_end - job.arrival_ts)
            
            if check_collision:
                for t in range(int(travel_end), int(process_end) + 1):
                    reservations.add((dest_pos, t))
                amr_states[amr_id] = (dest_pos, process_end)

            inventory[amr_id][material] -= 1
            station_available[st_key] = process_end
            
            # 5. Look-ahead Return Logic
            next_job = self.get_next_job_for_amr(amr_id, pos, order, individual.amr_assignment, jobs)
            should_return = True
            if next_job:
                if inventory[amr_id].get(next_job.jtype, 0) > 0:
                    should_return = False
            else:
                should_return = False
            
            return_start = process_end
            if next_job and should_return:
                # Return to nearest base relative to NEXT job's station
                next_job_dest = (int(next_job.dest_pos[0]), int(next_job.dest_pos[1]))
                next_base = self.nearest_base_to_station(next_job_dest)
                
                if check_collision:
                    return_path = find_dynamic_path(dest_pos, next_base, return_start, reservations, amr_states, amr_id)
                else:
                    return_path = shortest_path(dest_pos, next_base)
                
                return_time = float(len(return_path) - 1)
                return_end = return_start + return_time
                
                if check_collision:
                    for t_offset, pt in enumerate(return_path):
                        reservations.add((pt, int(return_start) + t_offset))
                    amr_states[amr_id] = (next_base, return_end)
                
                availability[amr_id] = return_end
                current_position[amr_id] = next_base
            else:
                availability[amr_id] = return_start
                current_position[amr_id] = dest_pos
                if check_collision:
                    amr_states[amr_id] = (dest_pos, return_start)

        return availability, total_flow

    def fitness(self, individual: Individual, jobs: List[Job], amrs_state: List[AMR], check_collision: bool = False) -> float:
        _, total_flow = self.decode_schedule(individual, jobs, amrs_state, check_collision=check_collision)
        return total_flow

    # --- Genetic Operators ---
    def random_individual(self, jobs: List[Job], amr_count: int) -> Individual:
        order = list(range(len(jobs)))
        random.shuffle(order)
        assign = [random.randint(0, amr_count - 1) for _ in jobs]
        return Individual(order=order, amr_assignment=assign)

    def greedy_individual(self, jobs: List[Job], amr_count: int) -> Individual:
        assign = []
        for i in range(len(jobs)):
            assign.append(i % amr_count)
        order = list(range(len(jobs)))
        random.shuffle(order)
        clustered_order = self.cluster_jobs_by_material(order, assign, jobs)
        return Individual(order=clustered_order, amr_assignment=assign)

    def order_crossover(self, parent_a: Individual, parent_b: Individual, jobs: List[Job]) -> Individual:
        size = len(parent_a.order)
        if size < 2: return parent_a
        
        a, b = sorted(random.sample(range(size), 2))
        child_order = [-1] * size
        child_order[a:b] = parent_a.order[a:b]
        
        fill = [gene for gene in parent_b.order if gene not in child_order]
        ptr = 0
        for i in range(size):
            if child_order[i] == -1:
                child_order[i] = fill[ptr]
                ptr += 1
                
        child_assign = parent_a.amr_assignment[:]
        for idx in range(size):
            if random.random() < 0.5:
                child_assign[idx] = parent_b.amr_assignment[idx]
                
        clustered_order = self.cluster_jobs_by_material(child_order, child_assign, jobs)
        return Individual(order=clustered_order, amr_assignment=child_assign)

    def smart_load_balance_mutate(self, individual: Individual, jobs: List[Job], amrs_state: List[AMR]):
        availability, _ = self.decode_schedule(individual, jobs, amrs_state, check_collision=False)
        busiest_amr = max(availability, key=availability.get)
        idlest_amr = min(availability, key=availability.get)
        if busiest_amr == idlest_amr: return
        
        busy_job_indices = [i for i, aid in enumerate(individual.amr_assignment) if aid == busiest_amr]
        if busy_job_indices:
            victim_job = random.choice(busy_job_indices)
            individual.amr_assignment[victim_job] = idlest_amr

    def mutate(self, individual: Individual, jobs: List[Job], amrs_state: List[AMR], amr_count: int) -> None:
        size = len(individual.order)
        if size < 2: return
        
        # Swap
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(size), 2)
            individual.order[i], individual.order[j] = individual.order[j], individual.order[i]
            
        # Reassign AMR
        for idx in range(size):
            if random.random() < self.mutation_rate * 0.5:
                individual.amr_assignment[idx] = random.randint(0, amr_count - 1)
                
        # Cluster Move
        if random.random() < self.mutation_rate:
            idx = random.randrange(size)
            job_idx = individual.order[idx]
            target_type = jobs[job_idx].jtype
            target_amr = individual.amr_assignment[job_idx]
            
            for target_idx, other_job_idx in enumerate(individual.order):
                if target_idx == idx: continue
                if (individual.amr_assignment[other_job_idx] == target_amr and 
                    jobs[other_job_idx].jtype == target_type):
                    individual.order.pop(idx)
                    insert_idx = target_idx if target_idx < idx else target_idx
                    individual.order.insert(insert_idx + (1 if target_idx >= idx else 0), job_idx)
                    break
                    
        # Load Balance
        if random.random() < self.mutation_rate:
            self.smart_load_balance_mutate(individual, jobs, amrs_state)

    def local_improve(self, individual: Individual, jobs: List[Job], amrs_state: List[AMR], max_iters: int = 500, check_collision: bool = False) -> Individual:
        current = Individual(order=list(individual.order), amr_assignment=list(individual.amr_assignment))
        best_score = self.fitness(current, jobs, amrs_state, check_collision=check_collision)
        job_count = len(current.order)
        if job_count < 2: return current
        
        availability, _ = self.decode_schedule(current, jobs, amrs_state, check_collision=check_collision)
        critical_amr = max(availability, key=availability.get)
        
        for _ in range(max_iters):
            improved = False
            i, j = random.sample(range(job_count), 2)
            job_i = current.order[i]
            job_j = current.order[j]
            
            if current.amr_assignment[job_i] == critical_amr or current.amr_assignment[job_j] == critical_amr:
                new_order = list(current.order)
                new_order[i], new_order[j] = new_order[j], new_order[i]
                neighbor = Individual(order=new_order, amr_assignment=list(current.amr_assignment))
                score = self.fitness(neighbor, jobs, amrs_state, check_collision=check_collision)
                
                if score < best_score:
                    current = neighbor
                    best_score = score
                    improved = True
                    availability, _ = self.decode_schedule(current, jobs, amrs_state, check_collision=check_collision)
                    critical_amr = max(availability, key=availability.get)
            
            if not improved:
                blocks = self.find_adjacent_blocks(current.order, current.amr_assignment, jobs)
                if blocks:
                    start, end = random.choice(blocks)
                    block = current.order[start:end]
                    remainder = current.order[:start] + current.order[end:]
                    insert_pos = random.randint(0, len(remainder))
                    new_order = remainder[:insert_pos] + block + remainder[insert_pos:]
                    neighbor = Individual(order=new_order, amr_assignment=list(current.amr_assignment))
                    score = self.fitness(neighbor, jobs, amrs_state, check_collision=check_collision)
                    if score < best_score:
                        current = neighbor
                        best_score = score
                        improved = True
            
            if improved: continue
            
        return current

    def solve(self, waiting_jobs, amrs, env):
        start_cpu_time = time.time()
        if not waiting_jobs:
            return [], 0.0

        amr_count = len(amrs)
        
        # Initial Population
        pop_random_count = int(self.pop_size * 0.8)
        population = [self.random_individual(waiting_jobs, amr_count) for _ in range(pop_random_count)]
        population += [self.greedy_individual(waiting_jobs, amr_count) for _ in range(self.pop_size - pop_random_count)]
        
        best_fitness = float("inf")
        archive_best = population[0]
        stagnation_counter = 0
        
        for gen in range(self.generations):
            scored = []
            for ind in population:
                m = self.fitness(ind, waiting_jobs, amrs, check_collision=False)
                scored.append((m, ind))
            
            scored.sort(key=lambda pair: pair[0])
            current_best = scored[0][1]
            f_val = scored[0][0]
            
            if f_val < best_fitness:
                best_fitness = f_val
                archive_best = current_best
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            if stagnation_counter > self.stagnation_limit:
                population = [pair[1] for pair in scored[:5]]
                population += [self.random_individual(waiting_jobs, amr_count) for _ in range(self.pop_size - 5)]
                stagnation_counter = 0
                continue
                
            new_generation = []
            # Elitism
            for _, elite_ind in scored[:2]:
                new_generation.append(Individual(order=list(elite_ind.order), amr_assignment=list(elite_ind.amr_assignment)))
            
            # Tournament Selection
            def get_parent():
                candidates = random.sample(scored, 3)
                return min(candidates, key=lambda x: x[0])[1]
            
            while len(new_generation) < self.pop_size:
                p1 = get_parent()
                p2 = get_parent()
                child = self.order_crossover(p1, p2, waiting_jobs)
                self.mutate(child, waiting_jobs, amrs, amr_count)
                new_generation.append(child)
                
            population = new_generation

        # Local Improvement
        archive_best = self.local_improve(archive_best, waiting_jobs, amrs, max_iters=self.routing_iters, check_collision=False)
        
        if self.collision_routing_iters > 0:
            archive_best = self.local_improve(archive_best, waiting_jobs, amrs, max_iters=self.collision_routing_iters, check_collision=True)
        
        # Convert best Individual to list of (jid, aid)
        # The order in 'best_assignments' must respect the execution order
        # because V2 env appends to local_queue in the order we return.
        # Individual.order gives the sequence of job indices.
        
        best_assignments = []
        for job_idx in archive_best.order:
            job = waiting_jobs[job_idx]
            aid = archive_best.amr_assignment[job_idx]
            best_assignments.append((job.jid, aid))

        compute_duration = time.time() - start_cpu_time
        return best_assignments, compute_duration


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
        DONE_REWARD = 10.0
        FLOW_PENALTY = CONFIG.get('FLOW_PENALTY', 0.1)
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
                        self.last_resched_completion_count = len(self.completed_jobs)

                    else:
                        # GA returns empty (rare but possible)
                        reward -= EMPTY_RESCHED_PENALTY

        # -------------------------------
        # 3) Execute AMR tasks & advance time
        # -------------------------------
        self.update_amr_tasks()

        # ✅ Advance time ONCE: 1 tick + GA compute time
        dt = max(1.0, compute_time)
        self.sim_time += dt

        # -------------------------------
        # 2.5) Flow-time penalty (Calculated AFTER time advance to penalize GA delay)
        # -------------------------------
        # Penalize ALL active jobs (unstarted + assigned/processing) to minimize total flow time.
        reward -= len(self.active_jobs) * dt * FLOW_PENALTY

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
                    print(f"GNN+DDQN Inference Time: {(time.perf_counter() - t_start) * 1000:.4f} ms")

                    # action mask: invalid -> -inf
                    if mask[1] < 0.5:
                        q[0, 1] = -1e9

                    action = q.argmax(1).item()
            
            

            curr_action_mask = env.get_action_mask()  # before step

            next_state, reward, done, dt = env.step(action)

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
            ckpt_path = f"gnn_ddqn_model_v5/gnn_ddqn_model_v5_ep{ep}.pth"
            torch.save(agent.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    torch.save(agent.state_dict(), CONFIG['SAVE_PATH'])
    print("Done.")

if __name__ == "__main__":
    main()