#!/usr/bin/env python3
"""
Dynamic AI-Dispatched Job Arrival + Scheduler  (Threaded Version)

Architecture:
  MAIN THREAD  — 200ms timer drives display + tick-by-tick sim advancement
  BG THREAD    — GA computation runs asynchronously; result is picked up
                 by the main thread on the next timer tick

This avoids the "display freeze" that occurs when GNN+GA blocks the
matplotlib event loop for 2-4 seconds.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json, time, math
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import torch
import numpy as np

# Import AI components
from GNN_DDQN_V7 import (
    GridEnv, SchedulerAgent, CONFIG, STATIONS, JOB_PROPS,
)
import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# ======================== Files ========================
SCHEDULE_OUTBOX = "schedule_outbox.jsonl"
DATASET_PATH = CONFIG['DATASET_PATH']
AMR_STATE_FILE = "periodic_amr_state.json"
MODEL_PATH = CONFIG['SAVE_PATH'] + '/gnn_ddqn_model_v7_ep800.pth'

FIX_PERIOD = 10.0


# Reset outbox each run
open(SCHEDULE_OUTBOX, "w").close()

def emit_assignment(amr_id: int, jid: int, jtype: str, proc_time: float, station: int):
    """Append a single assigned job to schedule_outbox.jsonl (for amr_runtime)."""
    rec = {
        "generated_at": time.time(),
        "amr": int(amr_id),
        "jid": int(jid),
        "type": str(jtype),
        "proc_time": float(proc_time),
        "station": int(station),
    }
    with open(SCHEDULE_OUTBOX, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

# ======================== Config ========================
SIM_SPEED_MULTIPLIER = 1.0     # 1.0 = real-time, 5.0 = 5x speed, etc.
UPDATE_INTERVAL_MS   = 200     # Display refresh rate — keep fast for smooth animation

LEFT_LABEL_PAD = 5.5
VIEW_WIDTH     = 40.0

AX_Y_MIN, AX_Y_MAX = 0.0, 2.0
TOP_Y_CENTER  = 1.25
TOP_LANE_H    = 0.5

AMR_COUNT     = 3
BOTTOM_MIN    = 0.0
BOTTOM_HEIGHT = (AX_Y_MAX - AX_Y_MIN) / 2.0
AMR_Y_CENTERS = [BOTTOM_MIN + (i + 0.5) * (BOTTOM_HEIGHT / AMR_COUNT)
                 for i in range(AMR_COUNT)]
AMR_LANE_H    = BOTTOM_HEIGHT / AMR_COUNT * 0.7

JOB_TYPES = ["A", "B", "C"]
_cycle = plt.rcParams.get("axes.prop_cycle", None)
_cycle_list = _cycle.by_key()["color"] if _cycle else ["C0", "C1", "C2", "C3", "C4"]
TYPE_COLORS = {jtype: _cycle_list[i % len(_cycle_list)] for i, jtype in enumerate(JOB_TYPES)}

# ======================== Data ========================
@dataclass
class VisualJob:
    jid: int
    jtype: str
    proc_time: float
    arrival_ts: float
    station: int

# ======================== State ========================
simulation_time: float = 0.0
is_running: bool = False
accumulated_sim_dt: float = 0.0   # Fractional accumulator for tick stepping

# Top queue (pending jobs)
all_visual_jobs: List[VisualJob] = []
jobs_top: List[VisualJob] = []
rects_top: List[Rectangle] = []
texts_top: List = []

# Per-AMR visual queues (bottom lanes)
amr_cursor: Dict[int, float] = {i: LEFT_LABEL_PAD for i in range(1, AMR_COUNT + 1)}
amr_rects: Dict[int, List[Rectangle]] = {i: [] for i in range(1, AMR_COUNT + 1)}
amr_texts: Dict[int, List] = {i: [] for i in range(1, AMR_COUNT + 1)}
amr_load: Dict[int, float] = {i: 0.0 for i in range(1, AMR_COUNT + 1)}

# Job stream from dataset
all_jobs_stream: List[dict] = []
next_job_idx: int = 0

# AI state
ai_env: GridEnv = None
ai_agent: SchedulerAgent = None
total_reschedule_count: int = 0
last_action_str: str = "INIT"
last_fix_trigger_t: float = -1e9  # Track when periodic reschedule was TRIGGERED (not completed)

# Threading & Multiprocessing state
ga_result_queue: Queue = Queue()   # background thread → main thread
is_computing: bool = False
visual_amr_queues: Dict = {}       # UI illusion to keep jobs visible during computation

import concurrent.futures
import multiprocessing as mp

_process_executor = None

def get_process_executor():
    global _process_executor
    if _process_executor is None:
        ctx = mp.get_context('spawn')
        _process_executor = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    return _process_executor

def _mp_bg_worker(ga_jobs, init_state, job_map, gnn_state_dict, device, routing_iters, collision_iters):
    import sys, os
    STATIC_ALGO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__) if '__file__' in globals() else '.', '../../Static_alogorithm'))
    if STATIC_ALGO_PATH not in sys.path:
        sys.path.append(STATIC_ALGO_PATH)
        
    try:
        from GNN.GNN import solve_with_gnn, SchedulerGNN
        from GA.GA import local_improve
        import time as _time
        import torch

        start = _time.perf_counter()
        
        heuristic_gnn = SchedulerGNN(amr_in_dim=8, job_in_dim=10, hidden_dim=128, gnn_layers=2)
        if gnn_state_dict:
            heuristic_gnn.load_state_dict(gnn_state_dict)
        heuristic_gnn.to(device)
        heuristic_gnn.eval()

        best_ind, _, _ = solve_with_gnn(
            ga_jobs, heuristic_gnn, deterministic=True, init_state=init_state
        )
        best_ind = local_improve(
            best_ind, ga_jobs,
            max_iters=routing_iters,
            init_state=init_state
        )
        if collision_iters > 0:
            best_ind = local_improve(
                best_ind, ga_jobs,
                max_iters=collision_iters,
                check_collision=True, init_state=init_state
            )
        elapsed = _time.perf_counter() - start
        return best_ind, job_map, elapsed
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, _time.perf_counter() - start


# ======================== Artist Helpers ========================
def remove_artists(rects, texts):
    for a in rects:
        try: a.remove()
        except Exception: pass
    for t in texts:
        try: t.remove()
        except Exception: pass
    rects.clear()
    texts.clear()

def rebuild_top_lane(ax):
    """Lay out waiting jobs left-aligned in the top lane."""
    remove_artists(rects_top, texts_top)
    x = LEFT_LABEL_PAD
    for j in jobs_top:
        w = j.proc_time / 2.0
        r = Rectangle((x, TOP_Y_CENTER), w, TOP_LANE_H,
                       linewidth=1.2, edgecolor="black",
                       facecolor=TYPE_COLORS.get(j.jtype, "gray"),
                       clip_on=True)
        ax.add_patch(r)
        rects_top.append(r)
        t = ax.text(x + w / 2.0, TOP_Y_CENTER + TOP_LANE_H / 2.0,
                    f"J_{j.jid}: S_{j.station}",
                    ha="center", va="center",
                    fontsize=9, weight="bold", clip_on=True)
        texts_top.append(t)
        x += w

def clear_amr_lanes(ax):
    """Clear all AMR lane visuals for a fresh redraw."""
    for amr_id in range(1, AMR_COUNT + 1):
        remove_artists(amr_rects[amr_id], amr_texts[amr_id])
        amr_cursor[amr_id] = LEFT_LABEL_PAD
        amr_load[amr_id] = 0.0

def draw_on_amr(ax, amr_id: int, jid: int, jtype: str, proc_time: float):
    """Draw one assigned job on an AMR lane."""
    x = amr_cursor[amr_id]
    y = AMR_Y_CENTERS[amr_id - 1]
    w = proc_time / 4.0
    r = Rectangle((x, y - AMR_LANE_H / 2.0), w, AMR_LANE_H,
                   linewidth=1.2, edgecolor="black",
                   facecolor=TYPE_COLORS.get(jtype, "gray"),
                   clip_on=True)
    ax.add_patch(r)
    amr_rects[amr_id].append(r)
    t = ax.text(x + w / 2.0, y,
                f"J_{jid}",
                ha="center", va="center",
                fontsize=9, weight="bold", clip_on=True)
    amr_texts[amr_id].append(t)
    amr_cursor[amr_id] += w

# ======================== Panel Decorations ========================
def draw_static_panels(ax):
    band_frac = 0.12
    # Top panel (Pending Queue)
    top_panel = Rectangle((0.0, 0.5), band_frac, 0.5,
                          transform=ax.transAxes, fill=False,
                          linewidth=1.8, clip_on=False, zorder=3)
    ax.add_patch(top_panel)
    tp = ax.text(band_frac * 0.5, 0.75, "New Arrival\nJobs",
                 transform=ax.transAxes, ha="center", va="center",
                 fontsize=10, weight="bold", color="gray",
                 clip_on=True, zorder=4)
    tp.set_clip_path(top_panel)

    # Bottom panel (AMRs)
    bot_panel = Rectangle((0.0, 0.0), band_frac, 0.5,
                          transform=ax.transAxes, fill=False,
                          linewidth=1.8, clip_on=False, zorder=3)
    ax.add_patch(bot_panel)
    for i in range(AMR_COUNT):
        y_frac = (i + 0.5) / AMR_COUNT * 0.5
        txt = ax.text(band_frac * 0.5, y_frac, f"AMR {i+1}",
                      transform=ax.transAxes,
                      ha="center", va="center",
                      fontsize=10, weight="bold", color="gray",
                      clip_on=True, zorder=4)
        txt.set_clip_path(bot_panel)

    # Legend
    handles = [Patch(facecolor=TYPE_COLORS[k], edgecolor="black", label=f"Type {k}") for k in JOB_TYPES]
    ax.legend(handles=handles, loc="upper right", frameon=True)

def update_title(ax):
    status = "RUNNING" if is_running else "PAUSED"
    pending = len(jobs_top)
    active = len(ai_env.active_jobs) if ai_env else 0
    completed = len(ai_env.completed_jobs) if ai_env else 0
    computing_tag = " ⏳" if is_computing else ""
    ax.set_title(
        f"Periodic Rescheduling (Period={FIX_PERIOD}s)\n"
        f"t={simulation_time:.1f}s — {status} | "
        # f"AI: {last_action_str}{computing_tag} | "
        f"New={pending}  Active={active}  Done={completed}"
        # f"Reschedules={total_reschedule_count}"
    )

# ======================== AI Integration (Threaded) ========================
def init_ai():
    """Initialize the GridEnv and load the trained DDQN agent."""
    global ai_env, ai_agent

    CONFIG['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    CONFIG['DATASET_PATH'] = DATASET_PATH

    ai_env = GridEnv()
    ai_env.reset()

    model_path = MODEL_PATH
    ai_agent = SchedulerAgent().to(CONFIG['DEVICE'])
    if os.path.exists(model_path):
        ai_agent.load_state_dict(torch.load(model_path, map_location=CONFIG['DEVICE']))
        print(f"[AI] Loaded model from {model_path}")
    else:
        print(f"[AI] WARNING: Model not found at {model_path}, using random weights")
    ai_agent.eval()

def load_jobs_from_dataset():
    """Load the first episode's jobs from the dataset JSONL."""
    global all_jobs_stream
    path = DATASET_PATH
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found: {path}")
        return
    with open(path, "r") as f:
        first_line = f.readline().strip()
    if not first_line:
        print("[ERROR] Dataset is empty")
        return
    episode = json.loads(first_line)
    all_jobs_stream = sorted(episode["jobs"], key=lambda j: j["arrival_time"])
    print(f"[DATA] Loaded {len(all_jobs_stream)} jobs from first episode")

def spawn_arrived_jobs(ax):
    """Check if any jobs from the dataset have arrived and add them to the visual queue."""
    global next_job_idx
    spawned = 0
    while next_job_idx < len(all_jobs_stream):
        raw = all_jobs_stream[next_job_idx]
        if raw["arrival_time"] > simulation_time:
            break
        props = JOB_PROPS[raw["type"]]
        vj = VisualJob(
            jid=raw["id"],
            jtype=raw["type"],
            proc_time=props["time"],
            arrival_ts=raw["arrival_time"],
            station=raw["dest_station_id"],
        )
        all_visual_jobs.append(vj)
        next_job_idx += 1
        spawned += 1
    return spawned

def get_ai_action():
    """Run DDQN forward pass — returns 0 (WAIT) or 1 (RESCHEDULE)."""
    state = ai_env.get_state_arrays()

    s_amr = torch.tensor([state[0]], dtype=torch.float32, device=CONFIG['DEVICE'])
    s_job = torch.tensor([state[1]], dtype=torch.float32, device=CONFIG['DEVICE'])
    exist = s_job[:, :, 0:1]
    job_mask = (exist > 0.0).float()
    s_q = torch.tensor([state[2]], dtype=torch.float32, device=CONFIG['DEVICE'])

    with torch.no_grad():
        q_vals = ai_agent(s_amr, s_job, s_q, job_mask)
        amask = ai_env.get_action_mask()
        if amask[1] < 0.5:
            q_vals[0, 1] = -1e9
        action = int(q_vals.argmax(1).item())

    return action

def start_background_ga():
    """Freeze AMR queues, prepare GA input, launch background thread."""
    global is_computing, last_action_str

    from GA.GA import STATIONS as GA_STATIONS, AMR_KEYS, Job as GAJob

    unstarted = [j for j in ai_env.active_jobs if j.status == 1]
    if not unstarted:
        return

    pos_to_station = {v: k for k, v in GA_STATIONS.items()}

    ga_jobs = []
    job_map = {}
    for i, j in enumerate(unstarted):
        st_name = pos_to_station.get(
            (int(j.dest_pos[0]), int(j.dest_pos[1])), "M1_1"
        )
        ga_j = GAJob(idx=i, type_=j.material, station=st_name, duration=j.proc_time)
        ga_jobs.append(ga_j)
        job_map[ga_j.idx] = {
            'jid': j.jid, 'jtype': j.material,
            'time': j.proc_time,
            'station': st_name.replace('station', '')
        }

    # ----- Freeze queues: keep only active job, remove unstarted -----
    global visual_amr_queues
    visual_amr_queues = {amr: list(ai_env.sim.amr_queues[amr]) for amr in AMR_KEYS}

    for amr in AMR_KEYS:
        active_job = ai_env.sim.amr_states[amr]['job']
        ai_env.sim.amr_queues[amr].clear()
        if active_job is not None:
            ai_env.sim.amr_queues[amr].append(active_job)

    # ----- Snapshot state for GA (immutable once captured) -----
    init_state = ai_env.sim.get_gnn_init_state()
    heuristic_gnn = ai_env.heuristic_gnn

    is_computing = True
    last_action_str = "COMPUTING..."
    
    # Record trigger time for periodic interval measurement (matches test_performance's last_fix)
    global last_fix_trigger_t
    last_fix_trigger_t = ai_env.sim_time
    
    print(f"[AI] t={ai_env.sim_time:.1f}  Started GA in background | "
          f"{len(unstarted)} unstarted jobs")

    # ---- Snapshot PyTorch Weights Safely ----
    gnn_state = None
    if getattr(ai_env, 'heuristic_gnn', None) is not None:
        gnn_state = {k: v.cpu() for k, v in ai_env.heuristic_gnn.state_dict().items()}
    device_name = CONFIG.get('DEVICE', 'cpu')
    routing_iters = CONFIG.get('GA_ROUTING_ITERS', 1000)
    collision_iters = CONFIG.get('GA_COLLISION_ITERS', 2000)

    # ---- Submit to Process Pool ----
    future = get_process_executor().submit(
        _mp_bg_worker, 
        ga_jobs, 
        init_state, 
        job_map, 
        gnn_state, 
        device_name, 
        routing_iters, 
        collision_iters
    )

    # ---- Thread bridging to GUI queue ----
    def _wait_and_forward(f):
        try:
            res = f.result()
            ga_result_queue.put(res)
        except Exception as e:
            print(f"[ERROR] Background worker failed: {e}")
            ga_result_queue.put((None, None, 0.0))
        
    threading.Thread(target=_wait_and_forward, args=(future,), daemon=True).start()

def advance_simulation_one_tick():
    """Advance the TickSimulator by exactly 1 tick and sync GridEnv state."""
    # Release arrived jobs into active_jobs
    ai_env._release_arrived_jobs()

    # Advance physics
    ai_env.sim.step(1)
    ai_env.sim_time = float(ai_env.sim.t)

    # Sync running jobs (status 1 → 2)
    running_jids = set()
    for amr, s in ai_env.sim.amr_states.items():
        if s['job'] is not None:
            running_jids.add(s['job'].idx)
    for j in ai_env.active_jobs:
        if j.status == 1 and j.jid in running_jids:
            j.status = 2

    # Sync completed jobs (→ status 3)
    for jid in ai_env.sim.completed_jobs_jids:
        for idx, j in enumerate(ai_env.active_jobs):
            if j.jid == jid:
                j.status = 3
                j.finish_ts = ai_env.sim_time
                ai_env.completed_jobs.append(ai_env.active_jobs.pop(idx))
                break
    ai_env.sim.completed_jobs_jids.clear()

def update_amr_lanes_from_sim(ax):
    """Redraw AMR lanes based on the current TickSimulator queues."""
    from GA.GA import AMR_KEYS
    clear_amr_lanes(ax)

    done_jids = {j.jid for j in ai_env.completed_jobs} if ai_env else set()

    for amr in AMR_KEYS:
        amr_id = int(amr.replace("AMR", ""))
        state = ai_env.sim.amr_states[amr]

        if is_computing and amr in visual_amr_queues:
            queue = [qj for qj in visual_amr_queues[amr] if qj.idx not in done_jids]
        else:
            queue = ai_env.sim.amr_queues[amr]

        # Draw the currently active job (if any)
        if state['job'] is not None:
            j = state['job']
            draw_on_amr(ax, amr_id, j.idx, j.type_, j.duration)

        # Draw queued jobs
        for qj in queue:
            if state['job'] is not None and qj.idx == state['job'].idx:
                continue
            draw_on_amr(ax, amr_id, qj.idx, qj.type_, qj.duration)

def write_schedule_outbox():
    """Write the current AMR queues to schedule_outbox.jsonl for amr_runtime."""
    from GA.GA import AMR_KEYS

    # Truncate the file. amr_runtime will detect len(lines) < _lines_consumed 
    # and reset its own queues.
    open(SCHEDULE_OUTBOX, "w").close()

    for amr in AMR_KEYS:
        amr_id = int(amr.replace("AMR", ""))
        state = ai_env.sim.amr_states[amr]
        queue = ai_env.sim.amr_queues[amr]

        if state['job'] is not None:
            j = state['job']
            station_num = int(j.station.replace("station", ""))
            emit_assignment(amr_id, j.idx, j.type_, j.duration, station_num)

        for qj in queue:
            if state['job'] is not None and qj.idx == state['job'].idx:
                continue
            station_num = int(qj.station.replace("station", ""))
            emit_assignment(amr_id, qj.idx, qj.type_, qj.duration, station_num)


def write_amr_state():
    """Write authoritative AMR state to JSON for Routing_Demo viewer."""
    from GA.GA import AMR_KEYS, shortest_path_avoiding
    if ai_env is None:
        return
    sim = ai_env.sim
    state = {"sim_time": float(sim.t)}
    amrs_data = {}
    all_positions = {amr: sim.positions[amr] for amr in AMR_KEYS}
    for amr in sorted(AMR_KEYS):
        amr_id = amr.replace("AMR", "")
        s = sim.amr_states[amr]
        pos = sim.positions[amr]
        job_info = {}
        if s['job'] is not None:
            j = s['job']
            station_num = int(j.station.replace("station", ""))
            job_info = {
                "job_idx": j.idx,
                "job_type": j.type_,
                "job_station": station_num,
                "job_duration": j.duration,
            }
        mode = s['mode']
        if mode == 'moving_supply':
            phase = "supply"
        elif mode in ('moving_station', 'moving_base'):
            phase = "deliver"
        else:
            phase = None
        goal = s.get('goal')
        goal_xy = list(goal) if goal else None
        # Compute A* path avoiding other AMRs
        path = []
        if goal is not None and mode in ('moving_supply', 'moving_station', 'moving_base'):
            other_amr_positions = {all_positions[o] for o in AMR_KEYS if o != amr}
            path = [list(p) for p in shortest_path_avoiding(pos, goal, other_amr_positions)]
        queue_jids = []
        for qj in sim.amr_queues[amr]:
            if s['job'] is not None and qj.idx == s['job'].idx:
                continue
            queue_jids.append(qj.idx)
        amrs_data[amr_id] = {
            "x": pos[0], "y": pos[1],
            "mode": mode,
            "phase": phase,
            "goal": goal_xy,
            "path": path,
            "proc_ticks": s.get('proc_ticks', 0),
            "inventory": sim.inventory[amr].copy(),
            "queue_jids": queue_jids,
            **job_info,
        }
    state["amrs"] = amrs_data
    tmp_path = AMR_STATE_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    os.replace(tmp_path, AMR_STATE_FILE)


# ======================== Main ========================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_pos", type=str, default=None)
    parser.add_argument("--state_file", type=str, default="periodic_amr_state.json")
    parser.add_argument("--sync_file", type=str, default=None)
    args = parser.parse_args()

    global AMR_STATE_FILE
    AMR_STATE_FILE = args.state_file

    global is_running, simulation_time, accumulated_sim_dt
    global is_computing, total_reschedule_count, last_action_str

    # Initialize
    load_jobs_from_dataset()
    init_ai()

    fig, ax = plt.subplots(figsize=(13, 4.8))
    fig.canvas.manager.set_window_title(f"Periodic Rescheduling (Period={FIX_PERIOD}s)")
    
    if args.window_pos:
        try:
            fig.canvas.manager.window.geometry(args.window_pos)
        except Exception:
            pass

    ax.set_ylim(AX_Y_MIN, AX_Y_MAX)
    ax.set_xlim(0.0, VIEW_WIDTH)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_visible(True)

    draw_static_panels(ax)
    update_title(ax)
    
    # Initialize the amr_state.json with t=0 before the demo begins
    write_amr_state()

    timer = fig.canvas.new_timer(interval=UPDATE_INTERVAL_MS)

    def tick():
        global simulation_time, is_running, accumulated_sim_dt
        global is_computing, total_reschedule_count, last_action_str

        if args.sync_file:
            try:
                with open(args.sync_file, "r") as f:
                    is_running = (f.read().strip() == "1")
            except Exception:
                pass

        if is_running:
            # 1) Accumulate real time → sim time
            real_dt = UPDATE_INTERVAL_MS / 1000.0
            accumulated_sim_dt += real_dt * SIM_SPEED_MULTIPLIER

            # 2) Step simulation for each full second accumulated
            steps_this_tick = 0
            while accumulated_sim_dt >= 1.0:
                advance_simulation_one_tick()
                accumulated_sim_dt -= 1.0
                steps_this_tick += 1

            simulation_time = ai_env.sim_time

            # 3) Spawn visual jobs that have arrived
            spawn_arrived_jobs(ax)

            # 4) Check for completed GA results from background thread
            try:
                best_ind, job_map, elapsed = ga_result_queue.get_nowait()
                if best_ind is None:
                    print("[AI] Background task failed. Resuming simulation without updates.")
                    is_computing = False
                else:
                    # Apply the new schedule
                    ai_env.sim.assign_schedules(
                        best_ind.order, best_ind.amr_assignment, job_map
                    )
                    ai_env.last_resched_t = float(ai_env.sim.t)
                    ai_env.last_resched_version = ai_env.arrival_version
                    ai_env.last_resched_completion_count = len(ai_env.completed_jobs)
                    is_computing = False
                    total_reschedule_count += 1
                    last_action_str = f"RESCHEDULE (GA={elapsed*1000:.0f}ms)"
                    write_schedule_outbox()
                    print(f"[AI] t={ai_env.sim_time:.1f}  RESCHEDULE #{total_reschedule_count} "
                          f"| active={len(ai_env.active_jobs)} "
                          f"| done={len(ai_env.completed_jobs)} "
                          f"| GA={elapsed*1000:.0f}ms")
            except Empty:
                pass

            # 5) Fixed Period Reschedule — only when not already computing and sim stepped
            #    Use last_fix_trigger_t (trigger time) to match test_performance's behavior
            if not is_computing and steps_this_tick > 0:
                unstarted = any(j.status == 1 for j in ai_env.active_jobs)
                if unstarted and (last_fix_trigger_t < 0 or (ai_env.sim_time - last_fix_trigger_t) >= FIX_PERIOD):
                    start_background_ga()
                else:
                    last_action_str = "WAIT"

            # 6) Rebuild top visual queue based on current unscheduled jobs
            scheduled_jids = set()
            if ai_env:
                for amr_id in ai_env.sim.amr_queues:
                    amr_state = ai_env.sim.amr_states[amr_id]
                    if amr_state['job'] is not None:
                        scheduled_jids.add(amr_state['job'].idx)
                    for qj in ai_env.sim.amr_queues[amr_id]:
                        scheduled_jids.add(qj.idx)
                
                # Keep frozen jobs visually "scheduled" so they don't bounce to the top panel
                if is_computing:
                    for amr, vq in visual_amr_queues.items():
                        for qj in vq:
                            scheduled_jids.add(qj.idx)
                
                done_jids = {j.jid for j in ai_env.completed_jobs}
                exclude_jids = scheduled_jids.union(done_jids)
                
                jobs_top_jids_before = {j.jid for j in jobs_top}
                jobs_top[:] = [vj for vj in all_visual_jobs if vj.jid not in exclude_jids]
                jobs_top_jids_after = {j.jid for j in jobs_top}
                
                if jobs_top_jids_before != jobs_top_jids_after:
                    rebuild_top_lane(ax)

            # 7) Refresh AMR lane visuals (shows jobs completing in real time)
            if steps_this_tick > 0:
                update_amr_lanes_from_sim(ax)

            # 8) Check end condition
            done = (ai_env.sim_time >= CONFIG['SIM_TIME']) or \
                   (len(ai_env.completed_jobs) >= ai_env.total_jobs)
            if done:
                is_running = False
                print(f"\n[DONE] Simulation Complete at t={simulation_time:.1f}s")
                print(f"  Completed: {len(ai_env.completed_jobs)} jobs")
                print(f"  Reschedules: {total_reschedule_count}")

            update_title(ax)
            write_amr_state()
            fig.canvas.draw_idle()

        timer.start()

    timer.add_callback(tick)
    timer.start()

    def on_key(event):
        global is_running
        if event.key == " ":
            if args.sync_file:
                try:
                    with open(args.sync_file, "r") as f:
                        val = f.read().strip()
                    new_val = "1" if val == "0" else "0"
                    with open(args.sync_file, "w") as f:
                        f.write(new_val)
                except Exception:
                    pass
            else:
                is_running = not is_running
            update_title(ax)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
