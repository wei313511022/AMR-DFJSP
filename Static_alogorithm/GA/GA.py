import csv
import json
import os
import random
import time
from collections import deque
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
except ImportError:
    plt = None
    Patch = None
    Rectangle = None
from functools import lru_cache 

# Constants Setup
AMR_STARTS = {
    "AMR1": (2, 8),
    "AMR2": (2, 5),
    "AMR3": (2, 2),
}
AMR_KEYS = list(AMR_STARTS.keys())
STATIONS = {
    "station1": (9, 9),
    "station2": (9, 7),
    "station3": (9, 5),
    "station4": (9, 3),
    "station5": (9, 1),
}
OBSTACLES = set()
BASES = list(AMR_STARTS.values()) # Parking spots are at the bases
TYPE_DURATION = {"A": 5, "B": 10, "C": 15}
SUPPLY_LOCATIONS = {"A": (0, 8), "B": (0, 5), "C": (0, 2)}
_GRID_POINTS = list(AMR_STARTS.values()) + list(STATIONS.values()) + list(SUPPLY_LOCATIONS.values())
GRID_MIN_X = min(p[0] for p in _GRID_POINTS)
GRID_MAX_X = max(p[0] for p in _GRID_POINTS)
GRID_MIN_Y = min(p[1] for p in _GRID_POINTS)
GRID_MAX_Y = max(p[1] for p in _GRID_POINTS)

SCHEDULE_OUTBOX = Path("schedule_outbox.jsonl")
DISPATCH_INBOX = Path("../../test_case/static/dispatch_inbox_60.jsonl")
DISPATCH_EVENT_INDEX_ENV = "DISPATCH_EVENT_INDEX"

JOB_COUNT = 60        
POPULATION_SIZE = 200    # number of candidate solutions
GENERATIONS = 150       
MUTATION_RATE = 0.2     
STAGNATION_LIMIT = 40   # number of convergence iterations 
routing_iters = 1000
collision_routing_iters = 1

# MAX_DEPTH limits the Space-Time A* search horizon (moves + waits).
# It prevents infinite searches in highly congested scenarios.
MAX_DEPTH = 100

@dataclass(frozen=True)
class Job:
    idx: int
    type_: str
    duration: float
    station: str

# one solution in GA
@dataclass
class Individual:
    order: List[int]          # permutation of job execution order
    amr_assignment: List[str] # job assigned to amr

# check whether routing within the bound
def _is_within_bounds(point: Tuple[int, int]) -> bool:
    x, y = point
    return GRID_MIN_X <= x <= GRID_MAX_X and GRID_MIN_Y <= y <= GRID_MAX_Y

# Pre-defined deltas to avoid recreation in loop
_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))

def _adjacent_points(point: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = point
    # right , left , up , down 
    neighbors = []
    for dx, dy in _DELTAS:
        candidate = (x + dx, y + dy)
        # check whether routing within the bound
        if not _is_within_bounds(candidate):
            continue
        # check whether collision with the obstacles
        if candidate in OBSTACLES:
            continue
        neighbors.append(candidate)
    return neighbors #return legal can move adjacent_points

#Tracing back the complete path from parents
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

# Heuristic for A* (Manhattan distance)
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Use A* to find the shortest path between start and end (Static Obstacles only)
# In this way, when the program queries the same start and end points again, it will not run BFS again, but will directly return the previously calculated path.
@lru_cache(maxsize=None)
def shortest_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    if start == end:
        return [start]
    
    # Priority queue for A*: stores (f_score, g_score, current_node)
    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), 0, start))
    
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
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

def shortest_path_avoiding(start: Tuple[int, int], end: Tuple[int, int],
                           extra_blocked: set) -> List[Tuple[int, int]]:
    """A* shortest path avoiding OBSTACLES + extra_blocked cells.
    The goal cell is never blocked so the AMR can approach it."""
    if start == end:
        return [start]
    blocked = OBSTACLES | extra_blocked
    blocked.discard(end)    # always allow reaching the goal
    blocked.discard(start)  # don't block our own position

    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), 0, start))
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    g_score = {start: 0}

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == end:
            return _build_path(came_from, end)
        for dx, dy in _DELTAS:
            neighbor = (current[0] + dx, current[1] + dy)
            if not _is_within_bounds(neighbor) or neighbor in blocked:
                continue
            new_g = g + 1
            if neighbor not in g_score or new_g < g_score[neighbor]:
                g_score[neighbor] = new_g
                f = new_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f, new_g, neighbor))
                came_from[neighbor] = current

    return [start]  # no path found — stay in place

# Dynamic A* for collision avoidance
def find_dynamic_path(start: Tuple[int, int], end: Tuple[int, int], start_time: float, 
                     reservations: Dict[Tuple[Tuple[int, int], int], str], amr_states: Dict[str, Tuple[Tuple[int, int], float]], 
                     active_amr: str) -> List[Tuple[int, int]]:
    t_start = int(start_time)
    open_set = []
    # (f_score, g_score, current_node, current_time)
    heapq.heappush(open_set, (heuristic(start, end), 0, start, t_start))
    
    came_from = {} 
    g_score = {(start, t_start): 0}
    
    while open_set:
        _, g, current, t = heapq.heappop(open_set)
        
        if g > MAX_DEPTH: continue
        if current == end:
            # Reconstruct path
            path = []
            curr_key = (current, t)
            while curr_key in came_from:
                prev_key = came_from[curr_key]
                path.append(curr_key[0])
                curr_key = prev_key
            path.append(start)
            return list(reversed(path))
            
        # Moves: adjacent + wait
        moves = list(_DELTAS) + [(0, 0)]
        for dx, dy in moves:
            neighbor = (current[0] + dx, current[1] + dy)
            next_t = t + 1
            
            if not _is_within_bounds(neighbor): continue
            if neighbor in OBSTACLES: continue
            
            # Check 1: Reserved by moving AMR
            if (neighbor, next_t) in reservations: continue
            
            # Check 1.5: Prevent edge collisions (swapping)
            if neighbor != current and (neighbor, t) in reservations: continue
            
            # Check 2: Occupied by idle AMR
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
                
    return [start] # Fallback

# Help a certain AMR accumulate the paths it has traveled.
def _extend_path_log(path_logs: Dict[str, List[Tuple[int, int]]], amr: str, segment: List[Tuple[int, int]]) -> None:
    if len(segment) <= 1:
        return
    log = path_logs[amr]
    if log and log[-1] == segment[0]:
        log.extend(segment[1:])
    else:
        log.extend(segment)

# The actual path distance between two points
def grid_distance(p: Tuple[int, int], q: Tuple[int, int]) -> float:
    # cahce's shortest_path
    path = shortest_path(p, q)
    return float(len(path) - 1)

# Reorder the job order so that Jobs on the same AMR and of the same job type are grouped together as much as possible
def cluster_jobs_by_material(order: List[int], assignments: List[str], jobs: List[Job]) -> List[int]:
    keyed = []
    for idx, job_idx in enumerate(order):
        amr = assignments[job_idx]
        job_type = jobs[job_idx].type_
        keyed.append((amr, job_type, idx, job_idx))
    # Sort by AMR, then Material, then original index 
    keyed.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in keyed]

def find_adjacent_blocks(order: List[int], assignments: List[str], jobs: List[Job]) -> List[Tuple[int, int]]:
    """Identifies blocks of consecutive jobs with same AMR and Type."""
    job_count = len(order)
    blocks = []
    idx = 0
    while idx < job_count:
        start = idx
        current_job = order[idx]
        curr_amr = assignments[current_job]
        curr_type = jobs[current_job].type_
        idx += 1
        while idx < job_count:
            next_job = order[idx]
            if assignments[next_job] == curr_amr and jobs[next_job].type_ == curr_type:
                idx += 1
            else:
                break
        if idx - start > 1:
            blocks.append((start, idx))
    return blocks

# Verify if the AMR has another task later in the sequence.If have,return the task,otherwise return null.
def get_next_job_for_amr(amr: str, current_pos_in_order: int, order: List[int], assignments: List[str], jobs: List[Job], job_map: Dict[int, Job] = None) -> Optional[Job]:
    if job_map is None:
        job_map = {job.idx: job for job in jobs}
    for i in range(current_pos_in_order + 1, len(order)):
        next_job_idx = order[i]
        if assignments[next_job_idx] == amr:
            return job_map[next_job_idx]
    return None

# Find the AMR_station closest to a certain workstation
def nearest_base_to_station(station: str) -> Tuple[int, int]:
    target = STATIONS[station]
    return min(BASES, key=lambda base: grid_distance(base, target))

def _diagnose_and_print_failure(amr: str, job_id: int, path_type: str, start_pos: Tuple[int, int], end_pos: Tuple[int, int], start_time: float, reservations: Dict, amr_states: Dict):
    """Prints a detailed reason when pathfinding fails."""
    reason = "Unknown"
    t_start_diag = int(start_time)
    next_t_diag = t_start_diag + 1
    
    is_surrounded = True
    possible_moves = list(_DELTAS) + [(0,0)]
    surrounding_reasons = []
    
    for dx, dy in possible_moves:
        neighbor = (start_pos[0] + dx, start_pos[1] + dy)
        
        if not _is_within_bounds(neighbor) or neighbor in OBSTACLES:
            continue

        if (neighbor, next_t_diag) in reservations:
            blocker_amr = reservations.get((neighbor, next_t_diag), "Unknown")
            surrounding_reasons.append(f"{neighbor} res by {blocker_amr}")
            continue

        is_blocked_by_idle = False
        for other_amr, (pos, free_t) in amr_states.items():
            if other_amr == amr: continue
            if pos == neighbor and next_t_diag >= free_t:
                surrounding_reasons.append(f"{neighbor} occ by {other_amr}")
                is_blocked_by_idle = True
                break
        if is_blocked_by_idle:
            continue

        is_surrounded = False
        break

    if is_surrounded:
        reason = f"Trapped at start. Surrounding: {', '.join(surrounding_reasons)}"
    else:
        # Check static path
        static_path = shortest_path(start_pos, end_pos)
        path_blockers = []
        # Check first few steps of static path
        check_steps = min(len(static_path), 20) 
        for i in range(1, check_steps):
            node = static_path[i]
            t = t_start_diag + i
            if (node, t) in reservations:
                path_blockers.append(f"({node}@{t} by {reservations[(node, t)]})")
            for other_amr, (pos, free_t) in amr_states.items():
                if other_amr == amr: continue
                if pos == node and t >= free_t:
                    path_blockers.append(f"({node}@{t} by idle {other_amr})")
        
        if path_blockers:
            reason = f"Static path blocked: {', '.join(path_blockers[:3])}"
        else:
            reason = f"Start not trapped, static path clear. Likely MAX_DEPTH({MAX_DEPTH}) reached or dynamic loop."

    print(f"  [WARNING] Pathfinding failed for AMR '{amr}' on Job {job_id} ({path_type} path: {start_pos} -> {end_pos} at t={start_time:.1f}).")
    print(f"  [DIAGNOSIS] Reason: {reason}")

def decode_schedule(individual: Individual, jobs: List[Job], need_log: bool = False, check_collision: bool = False, init_state: dict = None) -> Tuple[Dict[str, float], List[Tuple], List[Tuple[int, float]], Dict[str, List[Tuple[int, int]]], int]:
    job_map = {job.idx: job for job in jobs} # get job information
    timelines: List[Tuple] = []
    
    if init_state:
        availability = {amr: float(init_state["availability"].get(amr, 0.0)) for amr in AMR_STARTS}
        current_position = {amr: init_state["positions"].get(amr, AMR_STARTS[amr]) for amr in AMR_STARTS}
        amr_states = {amr: (current_position[amr], availability[amr]) for amr in AMR_STARTS}
        inventory = {amr: init_state["inventory"].get(amr, {mat: 0 for mat in TYPE_DURATION.keys()}).copy() for amr in AMR_STARTS}
    else:
        availability = {amr: 0.0 for amr in AMR_STARTS} # amr availability time
        current_position = {amr: AMR_STARTS[amr] for amr in AMR_STARTS} # current position of each AMR
        # Track AMR states for collision avoidance: amr -> (position, free_time)
        amr_states = {amr: (AMR_STARTS[amr], 0.0) for amr in AMR_STARTS}
        inventory = {amr: {mat: 0 for mat in TYPE_DURATION.keys()} for amr in AMR_STARTS} # How much material do we currently have on hand for this AMR?
        if "AMR1" in inventory: inventory["AMR1"]["A"] = 3
        if "AMR2" in inventory: inventory["AMR2"]["B"] = 3
        if "AMR3" in inventory: inventory["AMR3"]["C"] = 3
        
    path_logs = {amr: [current_position[amr]] for amr in AMR_STARTS} if need_log else {}
    reservations: Dict[Tuple[Tuple[int, int], int], str] = {} # ((x, y), t) -> amr_id
    
    station_available = {station: 0.0 for station in STATIONS} # station availability time
    order = individual.order
    queue_infos: List[Tuple[int, float]] = []
    invalid_jobs_count = 0

    for pos, job_idx in enumerate(order):
        job = job_map[job_idx]
        amr = individual.amr_assignment[job_idx]
        material = job.type_
        # AMR Start Time
        start_time = availability[amr]
        if need_log:
            queue_infos.append((job.idx, start_time))
        # fill material
        if inventory[amr][material] == 0:
            supply_location = SUPPLY_LOCATIONS[material]
            if check_collision:
                supply_path = find_dynamic_path(current_position[amr], supply_location, start_time, reservations, amr_states, amr)
            else:
                supply_path = shortest_path(current_position[amr], supply_location)
            supply_time = int(len(supply_path) - 1)
            supply_end = start_time + supply_time + TYPE_DURATION[material]
            if supply_time == 0 and current_position[amr] != supply_location:
                supply_time = MAX_DEPTH
                supply_end = start_time + supply_time
                invalid_jobs_count += 1
                if need_log and check_collision:
                    _diagnose_and_print_failure(amr, job.idx, "supply", current_position[amr], supply_location, start_time, reservations, amr_states)
            if check_collision:
                # Reserve path
                for t_offset, pt in enumerate(supply_path):
                    reservations[(pt, int(start_time) + t_offset)] = amr
                # Reserve supply location during loading
                for t in range(int(start_time) + supply_time, int(supply_end) + 1):
                    reservations[(supply_location, t)] = amr
                amr_states[amr] = (supply_location, supply_end)
            
            if need_log:
                timelines.append((amr, start_time, supply_end, "supply", f"Dock {material}"))
            availability[amr] = supply_end
            current_position[amr] = supply_location
            start_time = supply_end
            inventory[amr][material] = 3 # Refill amount
            if need_log: _extend_path_log(path_logs, amr, supply_path)

        # From your current location, navigate to job.station
        travel_start = availability[amr]
        if check_collision:
            travel_path = find_dynamic_path(current_position[amr], STATIONS[job.station], travel_start, reservations, amr_states, amr)
        else:
            travel_path = shortest_path(current_position[amr], STATIONS[job.station])
        travel_time = int(len(travel_path) - 1)
        # Penalize if pathfinding failed (returned [start] but we aren't at destination)
        if travel_time == 0 and current_position[amr] != STATIONS[job.station]:
            travel_time = MAX_DEPTH # Heavy penalty for "teleporting"
            invalid_jobs_count += 1
            if need_log and check_collision:
                _diagnose_and_print_failure(amr, job.idx, "travel", current_position[amr], STATIONS[job.station], travel_start, reservations, amr_states)

        travel_end = travel_start + travel_time
        if check_collision:
            # Reserve path
            for t_offset, pt in enumerate(travel_path):
                reservations[(pt, int(travel_start) + t_offset)] = amr
            
        if need_log and travel_time > 0:
            timelines.append((amr, travel_start, travel_end, "travel", f"Job{job.idx} trans {travel_time}s"))
        availability[amr] = travel_end
        current_position[amr] = STATIONS[job.station]
        if need_log: _extend_path_log(path_logs, amr, travel_path)

        # Wait station availability if needed
        earliest_start = max(travel_end, station_available[job.station])
        if need_log and earliest_start > travel_end:
             timelines.append((amr, travel_end, earliest_start, "wait", "Wait Stn"))
        process_start = earliest_start
        process_end = process_start + job.duration
        if need_log:
            timelines.append((amr, process_start, process_end, f"process_{job.type_}", f"Job{job.idx} {job.type_}({int(job.duration)}s)"))
        
        if check_collision:
            # Reserve station during wait and process
            for t in range(int(travel_end), int(process_end) + 1):
                reservations[(STATIONS[job.station], t)] = amr
            amr_states[amr] = (STATIONS[job.station], process_end)
            
        inventory[amr][material] -= 1
        station_available[job.station] = process_end # Occupy station
        
        # Return home after each job to match the neural schedulers' scenario.
        return_start = process_end
        next_dest = AMR_STARTS[amr]

        if check_collision:
            return_path = find_dynamic_path(STATIONS[job.station], next_dest, return_start, reservations, amr_states, amr)
        else:
            return_path = shortest_path(STATIONS[job.station], next_dest)
        return_time = int(len(return_path) - 1)
        if return_time == 0 and STATIONS[job.station] != next_dest:
            return_time = MAX_DEPTH
            invalid_jobs_count += 1
            if need_log and check_collision:
                _diagnose_and_print_failure(amr, job.idx, "return", STATIONS[job.station], next_dest, return_start, reservations, amr_states)
        return_end = return_start + return_time

        if check_collision:
            for t_offset, pt in enumerate(return_path):
                reservations[(pt, int(return_start) + t_offset)] = amr
            amr_states[amr] = (next_dest, return_end)
            
        if need_log:
            label = f"Return Home {return_time}s"
            timelines.append((amr, return_start, return_end, "return", label))
        availability[amr] = return_end
        current_position[amr] = next_dest
        if need_log: _extend_path_log(path_logs, amr, return_path)

    return availability, timelines, queue_infos, path_logs, invalid_jobs_count

# ==========================================
# True TICK-BY-TICK SIMULATION
# ==========================================
def decode_schedule_tick_by_tick(individual: Individual, jobs: List[Job], need_log: bool = False, check_collision: bool = True, init_state: dict = None):
    if not check_collision:
        return decode_schedule(individual, jobs, need_log, False, init_state=init_state)

    job_map = {job.idx: job for job in jobs}
    amr_queues = {amr: deque() for amr in AMR_STARTS}
    for job_idx in individual.order:
        amr = individual.amr_assignment[job_idx]
        amr_queues[amr].append(job_map[job_idx])
        
    t = int(init_state["time"]) if init_state else 0
    
    if init_state:
        positions = {amr: init_state["positions"].get(amr, AMR_STARTS[amr]) for amr in AMR_STARTS}
        inventory = {amr: init_state["inventory"].get(amr, {mat: 0 for mat in TYPE_DURATION.keys()}).copy() for amr in AMR_STARTS}
        amr_states = {}
        for amr in AMR_STARTS:
            avail = init_state["availability"].get(amr, float(t))
            if avail > t:
                amr_states[amr] = {'mode': 'processing_old', 'goal': positions[amr], 'job': None, 'proc_ticks': int(avail - t)}
            else:
                amr_states[amr] = {'mode': 'idle', 'goal': None, 'job': None, 'proc_ticks': 0}
    else:
        positions = {amr: AMR_STARTS[amr] for amr in AMR_STARTS}
        inventory = {amr: {mat: 0 for mat in TYPE_DURATION.keys()} for amr in AMR_STARTS}
        if "AMR1" in inventory: inventory["AMR1"]["A"] = 3
        if "AMR2" in inventory: inventory["AMR2"]["B"] = 3
        if "AMR3" in inventory: inventory["AMR3"]["C"] = 3
        amr_states = {amr: {'mode': 'idle', 'goal': None, 'job': None, 'proc_ticks': 0} for amr in AMR_STARTS}
    path_logs = {amr: [positions[amr]] for amr in AMR_STARTS} if need_log else {}
    timelines: List[Tuple] = []
    queue_infos: List[Tuple[int, float]] = []
    invalid_jobs_count = 0
    
    station_occupied = {s: False for s in STATIONS}
    
    t = 0
    while True:
        all_idle_and_empty = True
        for amr in AMR_KEYS:
            if len(amr_queues[amr]) > 0 or amr_states[amr]['mode'] != 'idle':
                all_idle_and_empty = False
                break
        if all_idle_and_empty:
            break
            
        if t > 10000:
            print(f"\\n--- DEADLOCK DETECTED at t={t} ---")
            for amr in AMR_KEYS:
                print(f"{amr}: pos={positions[amr]}, mode={amr_states[amr]['mode']}, goal={amr_states[amr]['goal']}")
            invalid_jobs_count += 1
            break
            
        # 1. Transitions
        for amr in AMR_KEYS:
            s = amr_states[amr]
            
            if s['mode'] == 'idle':
                if len(amr_queues[amr]) > 0:
                    s['job'] = amr_queues[amr][0]
                    mat = s['job'].type_
                    if inventory[amr][mat] == 0:
                        s['mode'] = 'moving_supply'
                        s['goal'] = SUPPLY_LOCATIONS[mat]
                        if need_log: 
                            queue_infos.append((s['job'].idx, t))
                            s['route_start'] = t
                    else:
                        s['mode'] = 'moving_station'
                        s['goal'] = STATIONS[s['job'].station]
                        if need_log and 'route_start' not in s: s['route_start'] = t
                else:
                    if positions[amr] != AMR_STARTS[amr]:
                        s['mode'] = 'moving_base'
                        s['goal'] = AMR_STARTS[amr]
                        if need_log: s['route_start'] = t
                        s['job'] = None
                        
            elif s['mode'] == 'processing':
                s['proc_ticks'] -= 1
                if s['proc_ticks'] <= 0:
                    mat = s['job'].type_
                    inventory[amr][mat] -= 1
                    station_occupied[s['job'].station] = False
                    
                    if need_log:
                        timelines.append((amr, t - s['job'].duration, t, f"process_{mat}", f"Job{s['job'].idx} {mat}({int(s['job'].duration)}s)"))
                        
                    amr_queues[amr].popleft()
                    if positions[amr] != AMR_STARTS[amr]:
                        s['mode'] = 'moving_base'
                        s['goal'] = AMR_STARTS[amr]
                    else:
                        s['mode'] = 'idle'
                        s['goal'] = None
                    s['job'] = None
                    if need_log: 
                        s['route_start'] = t
                        # Actually wait, we should log the NEXT job entering the "Dispatching" phase
                        if len(amr_queues[amr]) > 0:
                            queue_infos.append((amr_queues[amr][0].idx, t))

            elif s['mode'] == 'loading_dock':
                s['proc_ticks'] -= 1
                if s['proc_ticks'] <= 0:
                    mat = s['job'].type_
                    inventory[amr][mat] = 3
                    if need_log:
                        timelines.append((amr, s.get('route_start', t), t, "supply", f"Dock {mat}"))
                    s['mode'] = 'idle'
                    s['goal'] = None

            elif s['mode'] == 'processing_old':
                s['proc_ticks'] -= 1
                if s['proc_ticks'] <= 0:
                    s['mode'] = 'idle'
                    s['goal'] = None

        for amr in AMR_KEYS:
            s = amr_states[amr]
            if s['mode'] == 'idle':
                if len(amr_queues[amr]) > 0:
                    s['job'] = amr_queues[amr][0]
                    mat = s['job'].type_
                    if inventory[amr][mat] == 0:
                        s['mode'] = 'moving_supply'
                        s['goal'] = SUPPLY_LOCATIONS[mat]
                        if need_log: s['route_start'] = t
                    else:
                        s['mode'] = 'moving_station'
                        s['goal'] = STATIONS[s['job'].station]
                        if need_log and 'route_start' not in s: s['route_start'] = t
                else:
                    if positions[amr] != AMR_STARTS[amr]:
                        s['mode'] = 'moving_base'
                        s['goal'] = AMR_STARTS[amr]
                        if need_log: s['route_start'] = t
                        s['job'] = None

        # 2. Movement — priority: processing > lexicographical
        moves = {}
        occupied = set()  # cells claimed as DESTINATIONS by higher-priority AMRs

        def get_prio(amr_id):
            m = amr_states[amr_id]['mode']
            if m in ['processing', 'processing_old', 'loading_dock']: return 0
            return 1
        
        ordered_amrs = sorted(AMR_KEYS, key=lambda a: (get_prio(a), a))

        for amr in ordered_amrs:
            s = amr_states[amr]
            p = positions[amr]

            # Stationary modes: reserve position and skip
            if s['mode'] in ['processing', 'processing_old', 'loading_dock']:
                moves[amr] = p
                occupied.add(p)
                continue

            # Determine goal
            g = s.get('goal')
            if g is None:
                g = p
            if s.get('dodge_ticks', 0) > 0:
                g = s['dodge_goal']
                s['dodge_ticks'] -= 1

            # If idle at goal with nothing to do, check if blocking
            if p == g and s['mode'] == 'idle' and p not in occupied:
                is_blocking = False
                for st_pos in STATIONS.values():
                    if p == st_pos:
                        is_blocking = True
                        break
                if not is_blocking and p[0] != 2:  # not on highway
                    moves[amr] = p
                    occupied.add(p)
                    continue

            # Build blocked set:
            # - All destinations claimed by higher-priority AMRs
            # - Old positions of higher-priority AMRs that are MOVING AWAY
            #   (prevents swapping into where they came from)
            extra_blocked = occupied.copy()
            extra_blocked.discard(p)
            for o_amr in ordered_amrs:
                if o_amr == amr:
                    break
                o_old = positions[o_amr]
                o_new = moves.get(o_amr, o_old)
                if o_old != o_new:
                    extra_blocked.add(o_old)

            path = shortest_path_avoiding(p, g, extra_blocked)
            next_step = path[1] if len(path) > 1 else p

            is_swap = False
            for o_amr, o_next in moves.items():
                if o_next == p and positions[o_amr] == next_step:
                    is_swap = True
                    break

            must_dodge = False
            if is_swap or p in occupied:
                must_dodge = True
                
            if must_dodge:
                if next_step != p and next_step not in occupied and next_step not in extra_blocked:
                    pass # A* gave us a safe escape route
                else:
                    # Emergency dodge: find a safe adjacent tile
                    best_dodge = None
                    best_dist = float('inf')
                    for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                        adj = (p[0]+dx, p[1]+dy)
                        if not _is_within_bounds(adj) or adj in OBSTACLES: continue
                        if adj in occupied or adj in extra_blocked: continue
                        
                        dist = abs(adj[0] - g[0]) + abs(adj[1] - g[1])
                        if dist < best_dist:
                            best_dist = dist
                            best_dodge = adj
                    
                    if best_dodge:
                        next_step = best_dodge
                    else:
                        next_step = p # Trapped
            else:
                if next_step in occupied:
                    next_step = p
                elif next_step != p:
                    occupant = None
                    for o in AMR_KEYS:
                        if o == amr: continue
                        if o not in moves and positions[o] == next_step:
                            occupant = o
                            break
                    if occupant:
                        escape_found = False
                        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                            adj = (next_step[0]+dx, next_step[1]+dy)
                            if not _is_within_bounds(adj) or adj in OBSTACLES: continue
                            if adj in occupied or adj == p: continue
                            is_empty = True
                            for a in AMR_KEYS:
                                if a == occupant: continue
                                pos_a = moves.get(a, positions[a])
                                if pos_a == adj:
                                    is_empty = False
                                    break
                            if is_empty:
                                escape_found = True
                                break
                        if not escape_found:
                            next_step = p
            
            moves[amr] = next_step
            occupied.add(next_step)

            # Wait patiently if blocked, unless returning to base
            m = s['mode']
            if moves[amr] == p and g != p:
                s['blocked_ticks'] = s.get('blocked_ticks', 0) + 1
                if s['blocked_ticks'] > 5 and m in ['moving_base', 'moving_station', 'moving_supply']:
                    possible_dodges = []
                    for dy in [0, 1, 8, 9]:
                        for dx in range(0, GRID_MAX_X + 1):
                            dpos = (dx, dy)
                            if dpos not in OBSTACLES:
                                possible_dodges.append(dpos)
                    if possible_dodges:
                        s['dodge_goal'] = random.choice(possible_dodges)
                        s['dodge_ticks'] = 10
                        s['blocked_ticks'] = 0
            else:
                s['blocked_ticks'] = 0
                    
        # Apply moves and Evaluate arrivals
        for amr in AMR_KEYS:
            positions[amr] = moves[amr]
            if need_log and moves[amr] != path_logs[amr][-1]:
                path_logs[amr].append(moves[amr])
                
            s = amr_states[amr]
            p = positions[amr]
            
            if s['mode'] == 'moving_supply' and p == s['goal']:
                mat = s['job'].type_
                s['mode'] = 'loading_dock'
                s['proc_ticks'] = TYPE_DURATION[mat]
                
            elif s['mode'] == 'moving_station' and p == s['goal']:
                if not station_occupied[s['job'].station]:
                    station_occupied[s['job'].station] = True
                    s['mode'] = 'processing'
                    s['proc_ticks'] = s['job'].duration
                    if need_log:
                        dur = t - s.get('route_start', t)
                        if dur > 0: timelines.append((amr, s['route_start'], t, "travel", f"Job{s['job'].idx} trans {dur}s"))
                
            elif s['mode'] == 'moving_base' and p == s['goal']:
                s['mode'] = 'idle'
                if need_log:
                    dur = t - s.get('route_start', t)
                    if dur > 0: timelines.append((amr, s['route_start'], t, "return", f"Return {dur}s"))

        # End of tick
        t += 1
        
    avail = {a: float(t) for a in AMR_KEYS}
    return avail, timelines, queue_infos, path_logs, invalid_jobs_count


def fitness(individual: Individual, jobs: List[Job], check_collision: bool = False, init_state: dict = None) -> Tuple[float, List[Tuple]]:
    availability, timeline, _, _, _ = decode_schedule_tick_by_tick(individual, jobs, need_log=False, check_collision=check_collision, init_state=init_state)
    makespan = max(availability.values()) - (init_state["time"] if init_state else 0)
    #sum of all AMR finish time offset by init_score
    total_active_time = sum([ans - (init_state["time"] if init_state else 0) for ans in availability.values()])
    # Fitness=Makespan + (alpha * TotalActiveTime) ,alpha=0.001
    # Primary consider the makespan, secondary consider the total load balance 
    weighted_score = makespan + (0.001 * total_active_time)
    return weighted_score, timeline

# Random individual generation
# Can increase diversity of initial population
def random_individual(jobs: List[Job]) -> Individual:
    order = [job.idx for job in jobs]
    random.shuffle(order)
    assign = [random.choice(AMR_KEYS) for _ in jobs]
    return Individual(order=order, amr_assignment=assign)

# Creates an individual with balanced assignment and clustered order.
# This will guide the GA search space toward a certain type of solution, causing premature convergence to a suboptimal solution.
def greedy_individual(jobs: List[Job]) -> Individual:
    amrs = AMR_KEYS
    assign = []
    # Assign using round-robin: job0→AMR1, job1→AMR2, job2→AMR3, job3→AMR1… .It's about load balancing from the start.
    for i in range(len(jobs)):
        assign.append(amrs[i % len(amrs)])
    # First, shuffle the orders, then use cluster_jobs_by_material() to group jobs with the same AMR and type together as much as possible.
    order = [job.idx for job in jobs]
    random.shuffle(order)
    clustered_order = cluster_jobs_by_material(order, assign, jobs)
    return Individual(order=clustered_order, amr_assignment=assign)

def order_crossover(parent_a: Individual, parent_b: Individual, jobs: List[Job]) -> Individual:
    # Take a random segment from parent A [a:b).Place this segment unchanged into the child's position.Fill in the remaining positions in the order of parent B.
    size = len(parent_a.order)
    if size < 2: return Individual(order=list(parent_a.order), amr_assignment=list(parent_a.amr_assignment))
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
        # 50% chance to inherit AMR assignment from parent B or parent A
        if random.random() < 0.5:
            child_assign[idx] = parent_b.amr_assignment[idx]
    # Reorder the job order so that Jobs on the same AMR and of the same job type are grouped together as much as possible 
    clustered_order = cluster_jobs_by_material(child_order, child_assign, jobs)
    return Individual(order=clustered_order, amr_assignment=child_assign)

# Moves a job from the busiest AMR to the idlest AMR.
def smart_load_balance_mutate(individual: Individual, jobs: List[Job], init_state: dict = None):
    availability, _, _, _, _ = decode_schedule_tick_by_tick(individual, jobs, need_log=False, check_collision=False, init_state=init_state)
    busiest_amr = max(availability, key=availability.get) # find the busiest amr
    idlest_amr = min(availability, key=availability.get)  # find the idlest amr
    if busiest_amr == idlest_amr: return
    busy_job_indices = [i for i, amr in enumerate(individual.amr_assignment) if amr == busiest_amr]
    # Randomly select a job from busiest_amr. Reassign that job directly to idlest_amr.
    if busy_job_indices:
        victim_job = random.choice(busy_job_indices)
        individual.amr_assignment[victim_job] = idlest_amr

def mutate(individual: Individual, jobs: List[Job], init_state: dict = None) -> None:
    size = len(individual.order)
    if size < 2: 
        # Only allow AMR reassignment for single job
        if size == 1 and random.random() < MUTATION_RATE:
            individual.amr_assignment[0] = random.choice(AMR_KEYS)
        return
        
    # Swap the positions of the two jobs
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(size), 2)
        individual.order[i], individual.order[j] = individual.order[j], individual.order[i]
    # Randomly modify the AMR of certain jobs
    for idx in range(size):
        if random.random() < MUTATION_RATE * 0.5: 
            individual.amr_assignment[idx] = random.choice(AMR_KEYS)
    # Move a specific job next to a job with the same AMR and the same type.
    if random.random() < MUTATION_RATE:
        idx = random.randrange(size)
        job_idx = individual.order[idx]
        target_type = jobs[job_idx].type_
        target_amr = individual.amr_assignment[job_idx]
        for target_idx, other_job in enumerate(individual.order):
            if target_idx == idx: continue
            if (individual.amr_assignment[other_job] == target_amr and jobs[other_job].type_ == target_type):
                individual.order.pop(idx)
                insert_idx = target_idx if target_idx < idx else target_idx
                individual.order.insert(insert_idx + (1 if target_idx >= idx else 0), job_idx)
                break
    # Load Balance
    if random.random() < MUTATION_RATE:
        smart_load_balance_mutate(individual, jobs, init_state=init_state)

def local_improve(individual: Individual, jobs: List[Job], max_iters: int = routing_iters, check_collision: bool = False, init_state: dict = None) -> Individual:
    current = Individual(order=list(individual.order), amr_assignment=list(individual.amr_assignment))
    job_count = len(current.order)
    if job_count < 2: return current
    
    best_score, _ = fitness(current, jobs, check_collision=check_collision, init_state=init_state) # Current best fitness
    availability, _, _, _, _ = decode_schedule_tick_by_tick(current, jobs, need_log=False, check_collision=check_collision, init_state=init_state)
    critical_amr = max(availability, key=availability.get) # AMR with the longest makespan
    for _ in range(max_iters):
        # Each time, randomly select two job_i and job_j and try to swap them.One of them needs to belong to the critical AMR.
        improved = False
        i, j = random.sample(range(job_count), 2)
        job_i = current.order[i]
        job_j = current.order[j]
        if current.amr_assignment[job_i] == critical_amr or current.amr_assignment[job_j] == critical_amr:
            new_order = list(current.order)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            neighbor = Individual(order=new_order, amr_assignment=list(current.amr_assignment))
            score, _ = fitness(neighbor, jobs, check_collision=check_collision, init_state=init_state)
            # Only accept the neighbor if it improves the fitness
            if score < best_score:
                current = neighbor
                best_score = score
                improved = True
                availability, _, _, _, _ = decode_schedule_tick_by_tick(current, jobs, need_log=False, check_collision=check_collision, init_state=init_state)
                critical_amr = max(availability, key=availability.get)
        if not improved:
            # Find consecutive blocks with the same AMR and type in the current order.Randomly select one block.Move the entire block to another location.
            blocks = find_adjacent_blocks(current.order, current.amr_assignment, jobs)
            if blocks:
                start, end = random.choice(blocks)
                block = current.order[start:end]
                remainder = current.order[:start] + current.order[end:]
                insert_pos = random.randint(0, len(remainder))
                new_order = remainder[:insert_pos] + block + remainder[insert_pos:]
                neighbor = Individual(order=new_order, amr_assignment=list(current.amr_assignment))
                score, _ = fitness(neighbor, jobs, check_collision=check_collision, init_state=init_state)
                if score < best_score:
                    current = neighbor
                    best_score = score
                    improved = True
        if improved: continue
    return current

def get_parent_via_tournament(population_scored, k=3):
    candidates = random.sample(population_scored, k)
    winner = min(candidates, key=lambda x: x[0])
    return winner[1]

def evolve(jobs: List[Job], init_state: dict = None) -> Tuple[Individual, List[Tuple]]:
    # Initial Population = 80% random group + 20% greedy group
    pop_random_count = int(POPULATION_SIZE * 0.8)
    population = [random_individual(jobs) for _ in range(pop_random_count)]
    population += [greedy_individual(jobs) for _ in range(POPULATION_SIZE - pop_random_count)]
    # Global best solution
    archive_best: Individual = population[0]
    best_fitness = float("inf")
    best_timeline: List[Tuple] = []
    # counter the number of generations without improvement
    stagnation_counter = 0
    for gen in range(GENERATIONS):
        scored = []
        for ind in population:
            m, _ = fitness(ind, jobs, init_state=init_state)
            scored.append((m, ind))
        scored.sort(key=lambda pair: pair[0])
        current_best = scored[0][1]
        f_val = scored[0][0]
        if f_val < best_fitness:
            best_fitness = f_val
            best_timeline = fitness(current_best, jobs, init_state=init_state)[1]
            archive_best = current_best
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        if stagnation_counter > STAGNATION_LIMIT:
            # Keep the top 5 elites.Re-randomize all others.
            population = [pair[1] for pair in scored[:5]]
            population += [random_individual(jobs) for _ in range(POPULATION_SIZE - 5)]
            stagnation_counter = 0
            continue 
        # Elitism: Top 2 students directly admitted to the next generation.
        new_generation = []
        for _, elite_ind in scored[:2]:
            new_generation.append(Individual(order=list(elite_ind.order), amr_assignment=list(elite_ind.amr_assignment)))
        # Crossover+mutate
        while len(new_generation) < POPULATION_SIZE:
            parent_a = get_parent_via_tournament(scored, k=3)
            parent_b = get_parent_via_tournament(scored, k=3)
            child = order_crossover(parent_a, parent_b, jobs)
            mutate(child, jobs, init_state=init_state)
            new_generation.append(child)

        population = new_generation

    archive_best = local_improve(archive_best, jobs, max_iters=routing_iters, init_state=init_state)
    
    if collision_routing_iters > 0:
        archive_best = local_improve(archive_best, jobs, max_iters=collision_routing_iters, check_collision=True, init_state=init_state)
        
    makespan, timeline = fitness(archive_best, jobs, check_collision=True, init_state=init_state)
    return archive_best, timeline


def plot_gantt(timeline: List[Tuple], queue_infos: List[Tuple[int, float]], jobs: List[Job] = None, solve_time: float = None, invalid_count: int = 0, show_gantt: bool = True, save_img: str = None) -> None:
    if plt is None or Patch is None or Rectangle is None:
        raise RuntimeError("matplotlib is required to plot or save a Gantt chart.")

    AMR_COUNT = len(AMR_STARTS)
    AX_Y_MIN, AX_Y_MAX = 0.0, 2.0
    BOTTOM_MIN = 0.0
    BOTTOM_HEIGHT = (AX_Y_MAX - AX_Y_MIN) / 2.0 
    AMR_Y_CENTERS = [BOTTOM_MIN + (i + 0.5) * (BOTTOM_HEIGHT / AMR_COUNT) for i in range(AMR_COUNT)]
    AMR_LANE_H = BOTTOM_HEIGHT / AMR_COUNT * 0.7
    _cycle = plt.rcParams.get("axes.prop_cycle", None)
    TYPE_COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    TRANSPORT_COLOR = "lightgray"
    WAIT_COLOR = "lightgray"
    sorted_amrs = sorted(AMR_STARTS.keys())
    amr_y_map = {name: AMR_Y_CENTERS[i] for i, name in enumerate(sorted_amrs)}
    fig, ax = plt.subplots(figsize=(14, 6))
    SIDEBAR_WIDTH_FRAC = 0.12 
    def draw_static_panels(ax):
        top_panel = Rectangle((0.0, 0.5), SIDEBAR_WIDTH_FRAC, 0.5, transform=ax.transAxes,
                              fill=False, linewidth=1.5, edgecolor="black", clip_on=False, zorder=10)
        ax.add_patch(top_panel)
        tp_text = ax.text(SIDEBAR_WIDTH_FRAC * 0.5, 0.75, "Dispatching\nQueue", transform=ax.transAxes,
                          ha="center", va="center", fontsize=10, weight="bold", color="gray", zorder=11)
        tp_text.set_clip_path(top_panel)

        bot_panel = Rectangle((0.0, 0.0), SIDEBAR_WIDTH_FRAC, 0.5, transform=ax.transAxes,
                              fill=False, linewidth=1.5, edgecolor="black", clip_on=False, zorder=10)
        ax.add_patch(bot_panel)
        for i, name in enumerate(sorted_amrs):
            y_frac = (i + 0.5) / AMR_COUNT * 0.5
            txt = ax.text(SIDEBAR_WIDTH_FRAC * 0.5, y_frac, name, transform=ax.transAxes,
                          ha="center", va="center", fontsize=10, weight="bold", color="gray", zorder=11)
            txt.set_clip_path(bot_panel)
    draw_static_panels(ax)
    max_timeline_time = max([t[2] for t in timeline]) if timeline else 0.0
    total_job_duration = sum(job.duration for job in jobs) if jobs else 0.0
    max_plot_time = max(max_timeline_time, total_job_duration)
    LEFT_PAD_RATIO = 0.18 
    ax.set_xlim(-max_plot_time * LEFT_PAD_RATIO, max_plot_time * 1.05)
    ax.set_ylim(AX_Y_MIN, AX_Y_MAX)
    ax.set_yticks([])
    for entry in timeline:
        amr, start, end, kind, label = entry
        duration = end - start
        if duration <= 0: continue
        y_c = amr_y_map.get(amr, 0)
        rect_kwargs = {"linewidth": 0.5, "edgecolor": "gray", "alpha": 1.0, "clip_on": True}
        text_color = "black"
        font_weight = "bold"
        font_size = 7
        display_label = label
        if kind.startswith("process"):
            jtype = kind.split("_")[-1]
            rect_kwargs["facecolor"] = TYPE_COLORS.get(jtype, "gray")
            rect_kwargs["edgecolor"] = "black"
            rect_kwargs["linewidth"] = 1.2
            rect_kwargs["zorder"] = 5
            text_color = "white"
            try:
                parts = label.split(" ")
                jid_str = parts[0].replace("Job", "")
                jid = f"J{jid_str}"
                dur = parts[1].split("(")[1].replace("s)", "")
                station_str = ""
                if jobs:
                    j_idx = int(jid_str)
                    job_map = {j.idx: j.station for j in jobs}
                    if j_idx in job_map:
                        station_str = f"\n{job_map[j_idx]}"
                display_label = f"{jid}\n({dur})"
            except:
                pass
        elif kind in ["travel", "return", "supply"]:
            rect_kwargs["facecolor"] = TRANSPORT_COLOR
            rect_kwargs["edgecolor"] = "gray"
            rect_kwargs["hatch"] = "///"
            rect_kwargs["zorder"] = 2
            font_weight = "normal"
            if "trans" in label:
                dur = label.split(" ")[-1].replace("s", "")
                display_label = f"({dur})"
            elif "Return" in label:
                dur = label.split(" ")[-1].replace("s", "")
                display_label = f"Ret\n({dur})"
            elif "Replenish" in label or "Dock" in label:
                display_label = label
        else:
            rect_kwargs["facecolor"] = WAIT_COLOR
            rect_kwargs["hatch"] = ".."
            display_label = "W"
        r = Rectangle((start, y_c - AMR_LANE_H / 2), duration, AMR_LANE_H, **rect_kwargs)
        ax.add_patch(r)
        ax.text(start + duration / 2, y_c, display_label, ha="center", va="center", 
                fontsize=font_size, color=text_color, weight=font_weight, zorder=6)
    if queue_infos and jobs:
        queue_y_center = 1.25
        queue_h = 0.5
        job_map = {job.idx: job for job in jobs}
        sorted_queue = sorted(queue_infos, key=lambda entry: entry[0])
        current_x = 0.0
        for job_idx, _ in sorted_queue:
            job = job_map.get(job_idx)
            if not job: continue
            width = job.duration
            face = TYPE_COLORS.get(job.type_, "gray")
            r = Rectangle((current_x, queue_y_center - queue_h / 2), width, queue_h,
                          facecolor=face, edgecolor="black", linewidth=1.2, zorder=5)
            ax.add_patch(r)
            ax.text(current_x + width / 2, queue_y_center, f"J{job_idx}", 
                    ha="center", va="center", color="white", fontsize=9, weight="bold", zorder=6)
            current_x += width
    ax.set_xlabel("Time (s)", fontweight="bold")
    title_text = f"AMR Schedule (Optimized GA) | Makespan: {max_timeline_time:.1f}s"
    if solve_time is not None:
        title_text += f" | Solve: {solve_time:.4f}s"
    if invalid_count > 0:
        title_text += f" | Invalid Paths: {invalid_count}"
    ax.set_title(title_text, pad=10)
    ax.grid(True, axis="x", linestyle=":", color='gray', alpha=0.5, zorder=0)
    handles = [Patch(facecolor=TYPE_COLORS[k], edgecolor="black", label=f"Type {k}") for k in sorted(TYPE_COLORS.keys())]
    handles.append(Patch(facecolor=TRANSPORT_COLOR, edgecolor="gray", hatch="///", label="Transportation"))
    handles.append(Patch(facecolor=WAIT_COLOR, edgecolor="gray", hatch="..", label="Waiting"))
    ax.legend(handles=handles, loc="upper right", frameon=True, bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.subplots_adjust(left=0.05)
    if save_img:
        plt.savefig(save_img, dpi=300, bbox_inches='tight')
        print(f"Saved Gantt chart to {save_img}")
    if show_gantt:
        plt.show()
    else:
        plt.close(fig)
# Data Loading & Execution 
def station_key_from_value(raw_station: Optional[str]) -> Optional[str]:
    if raw_station is None: return None
    try: station_id = int(raw_station)
    except: return None
    key = f"station{station_id}"
    return key if key in STATIONS else None
def load_dispatch_events(path: Path = DISPATCH_INBOX) -> List[Dict[str, object]]:
    events = []
    if not path.exists(): return events
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for idx, payload in enumerate(lines):
        try: data = json.loads(payload)
        except: continue
        jobs = []
        for job_idx, raw_job in enumerate(data.get("jobs", [])):
            station_key = station_key_from_value(raw_job.get("station"))
            if station_key is None: continue
            
            if "proc_time" in raw_job:
                duration = float(raw_job["proc_time"])
            elif "duration" in raw_job:
                duration = float(raw_job["duration"])
            else:
                jid = raw_job.get("id", raw_job.get("jid", job_idx))
                duration = float(random.Random(jid).choice([5.0, 10.0, 15.0]))
                
            if duration == 5.0:
                type_ = "A"
            elif duration == 10.0:
                type_ = "B"
            else:
                type_ = "C"
            jobs.append(Job(idx=job_idx, type_=type_, duration=duration, station=station_key))
        if jobs:
            events.append({"index": idx, "dispatch_time": float(data.get("dispatch_time", 0.0)), "jobs": jobs})
    return events
def make_jobs() -> List[Job]:
    stations = list(STATIONS.keys())
    jobs = []
    for idx in range(JOB_COUNT):
        station = random.choice(stations)
        duration = float(random.Random(idx).choice([5.0, 10.0, 15.0]))
        if duration == 5.0:
            type_ = "A"
        elif duration == 10.0:
            type_ = "B"
        else:
            type_ = "C"
        jobs.append(Job(idx=idx, type_=type_, duration=duration, station=station))
    return jobs
def describe_solution(individual: Individual, jobs: List[Job], solve_time: float = None, show_gantt: bool = False, save_img: str = None) -> Tuple[float, float]:
    availability, decoded_timeline, queue_infos, path_logs, invalid_count = decode_schedule_tick_by_tick(individual, jobs, need_log=True, check_collision=True)
    makespan = max(availability.values())
    print(f"Optimal Makespan Found: {makespan:.2f}s")
    print(f"Invalid Jobs Count: {invalid_count}")
    if solve_time is not None:
        print(f"Computation Time: {solve_time:.4f}s")
    if show_gantt or save_img:
        plot_gantt(decoded_timeline, queue_infos, jobs, solve_time=solve_time, invalid_count=invalid_count, show_gantt=show_gantt, save_img=save_img)
    return makespan, solve_time

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gantt", action="store_true", help="Plot Gantt Chart")
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file")
    parser.add_argument("--save_img", type=str, default="", help="Save the schedule Gantt chart to this file (e.g., schedule.png)")
    args = parser.parse_args()

    random.seed(42)
    
    if args.inbox:
        dispatch_events = load_dispatch_events(Path(args.inbox))
    else:
        dispatch_events = load_dispatch_events()
        
    target_index = os.environ.get(DISPATCH_EVENT_INDEX_ENV)
    
    output_filename = "GA_summary_results.csv"
    results_data = []

    if dispatch_events:
        if target_index is not None:
             dispatch_events = [e for e in dispatch_events if str(e["index"]) == str(target_index)]
        
        for event in dispatch_events:
            print(f"\n=== Processing Dispatch Event {event['index']} (Jobs: {len(event['jobs'])}) ===")
            start_time = time.perf_counter()
            best_ind, _ = evolve(event["jobs"])
            solve_dur = time.perf_counter() - start_time
            
            img_path = f"{args.save_img.split('.')[0]}_{event['index']}.png" if args.save_img else None
            makespan, computation_time = describe_solution(best_ind, event["jobs"], solve_time=solve_dur, show_gantt=args.gantt, save_img=img_path)
            results_data.append([event['index'], f"{makespan:.2f}", f"{computation_time:.4f}" if computation_time is not None else "0.0000"])
    else:
        print("No dispatch file found. Generating random jobs...")
        jobs = make_jobs()
        start_time = time.perf_counter()
        best_ind, _ = evolve(jobs)
        solve_dur = time.perf_counter() - start_time
        
        img_path = args.save_img if args.save_img else None
        makespan, computation_time = describe_solution(best_ind, jobs, solve_time=solve_dur, show_gantt=args.gantt, save_img=img_path)
        results_data.append(["random", f"{makespan:.2f}", f"{computation_time:.4f}" if computation_time is not None else "0.0000"])

    if results_data:
        with open(output_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Event_Index", "Makespan", "Computation_Time"])
            writer.writerows(results_data)
        print(f"\nSummary results saved to {output_filename}")
