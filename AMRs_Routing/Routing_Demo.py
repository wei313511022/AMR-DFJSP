#!/usr/bin/env python3
"""
AMR State Viewer — Pure renderer driven by amr_state.json

Reads the authoritative AMR state written by Dynamic_Pairing_Demo.py or
Fix_Pairing_Demo.py and renders positions, A* routes, and status.
No physics simulation is performed here; this is a "dumb view".

Controls:
  SPACE  — pause/resume rendering (the producer still runs)
"""

import os
import json
import math
import argparse
from typing import Dict, Tuple, Optional, Set, List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# ---------- Files ----------
AMR_STATE_FILE = "../Random_Job_Arrivals/models/dynamic_amr_state.json"

# ---------- Grid / Layout ----------
GRID_W, GRID_H = 10, 11

START_POS: Dict[int, Tuple[int, int]] = {
    3: (2, 2),
    2: (2, 5),
    1: (2, 8),
}

# Production stations (right side).
STATION_POS: Dict[int, Tuple[int, int]] = {
    5: (9, 1),
    4: (9, 3),
    3: (9, 5),
    2: (9, 7),
    1: (9, 9),
}

# Material stations (left side) by job type
MAT_POS: Dict[str, Tuple[int, int]] = {
    "C": (0, 2),
    "B": (0, 5),
    "A": (0, 8),
}

# Static obstacles
OBSTACLES: Set[Tuple[int, int]] = {
    (5, 1),(5, 2),(6, 1),(6, 2),(4, 5),(3, 5),(3,8),
    (6, 4),(6, 5),(6, 8),(6, 9),(4,6),(3,1), (2, 3)
}

# ---------- Config ----------
UPDATE_INTERVAL_MS = 200   # poll rate in ms
MATERIAL_CAPACITY = 3
AMR_COUNT = 3

# ---------- AMR Visual State ----------
class AMRVisual:
    def __init__(self, amr_id: int, x: float, y: float):
        self.amr_id = amr_id
        # Authoritative position (from JSON) — no interpolation
        self.x = x
        self.y = y
        # State fields
        self.mode = "idle"
        self.phase = None
        self.goal = None
        self.path: List[List[int]] = []  # Full A* path
        self.proc_ticks = 0
        self.inventory = {"A": 0, "B": 0, "C": 0}
        self.job_idx = None
        self.job_type = None
        self.job_station = None
        self.queue_jids: list = []
        # Matplotlib artists
        self.marker: Optional[Circle] = None
        self.label: Optional[plt.Text] = None
        self.route_line: Optional[plt.Line2D] = None


# ---------- Drawing ----------

def draw_static(ax):
    ax.set_xlim(-0.5, GRID_W - 0.5)
    ax.set_ylim(-0.5, GRID_H - 0.5)
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.grid(True, which="both", linewidth=0.4, color="black", alpha=0.4)

    # obstacles
    for (x, y) in OBSTACLES:
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=(0.8, 0.8, 0.8),
                edgecolor="none", zorder=1,
            )
        )

    # production stations (red)
    for sid, (sx, sy) in STATION_POS.items():
        ax.add_patch(
            Rectangle(
                (sx - 0.5, sy - 0.5), 1, 1,
                facecolor="none", edgecolor="tab:red",
                linewidth=2.0, zorder=2,
            )
        )
        ax.text(
            sx - 0.4, sy + 0.15, f"S{sid}",
            fontsize=9, color="tab:red", weight="bold", zorder=3,
        )

    # material stations (blue)
    for jt, (mx, my) in MAT_POS.items():
        ax.add_patch(
            Rectangle(
                (mx - 0.5, my - 0.5), 1, 1,
                facecolor="none", edgecolor="tab:blue",
                linewidth=2.0, zorder=2,
            )
        )
        ax.text(
            mx - 0.4, my + 0.15, f"M{jt}",
            fontsize=9, color="tab:blue", weight="bold", zorder=3,
        )


def create_amrs(ax) -> Dict[int, AMRVisual]:
    """Create AMR visual objects at start positions."""
    amrs: Dict[int, AMRVisual] = {}
    color_map = {
        1: "tab:red",
        2: "tab:green",
        3: "tab:purple",
    }

    for i in range(1, AMR_COUNT + 1):
        x, y = START_POS.get(i, (1, 1))
        vis = AMRVisual(amr_id=i, x=float(x), y=float(y))

        mk = Circle(
            (x, y), radius=0.35,
            facecolor="white", edgecolor="black",
            linewidth=1.8, zorder=5,
        )
        ax.add_patch(mk)
        vis.marker = mk

        vis.label = ax.text(
            x, y + 0.55, f"AMR{i}",
            fontsize=9, ha="center", va="bottom", zorder=6,
        )

        # Route line (dashed, colored per AMR)
        (line,) = ax.plot([], [], linestyle="--", linewidth=1.5,
                          alpha=0.7, zorder=4, color=color_map.get(i, "black"))
        vis.route_line = line

        amrs[i] = vis
    return amrs


def update_route_line(vis: AMRVisual):
    """Draw the full A* path as a dashed line."""
    if vis.path and len(vis.path) > 1:
        xs = [p[0] for p in vis.path]
        ys = [p[1] for p in vis.path]
        vis.route_line.set_data(xs, ys)
    else:
        vis.route_line.set_data([], [])


# ---------- State reader ----------

def read_amr_state() -> Optional[dict]:
    """Read the shared amr_state.json file. Returns None on any error."""
    try:
        with open(AMR_STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        return None


def apply_state(amrs: Dict[int, AMRVisual], state: dict):
    """Apply the JSON state to the AMR visuals."""
    amrs_data = state.get("amrs", {})
    for amr_id_str, data in amrs_data.items():
        amr_id = int(amr_id_str)
        if amr_id not in amrs:
            continue
        vis = amrs[amr_id]

        # Snap to authoritative position (no interpolation — prevents visual collisions)
        vis.x = float(data["x"])
        vis.y = float(data["y"])

        # State fields
        vis.mode = data.get("mode", "idle")
        vis.phase = data.get("phase")
        vis.goal = tuple(data["goal"]) if data.get("goal") else None
        vis.path = data.get("path", [])
        vis.proc_ticks = data.get("proc_ticks", 0)
        vis.inventory = data.get("inventory", {"A": 0, "B": 0, "C": 0})
        vis.job_idx = data.get("job_idx")
        vis.job_type = data.get("job_type")
        vis.job_station = data.get("job_station")
        vis.queue_jids = data.get("queue_jids", [])


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="AMR Routing Demo")
    parser.add_argument("--state_file", type=str, default="../Random_Job_Arrivals/models/dynamic_amr_state.json", help="Path to amr_state.json")
    parser.add_argument("--window_pos", type=str, default=None, help="Window geometry e.g., +100+100")
    parser.add_argument("--title", type=str, default="Route Map", help="Title of the window and plot")
    parser.add_argument("--sync_file", type=str, default=None)
    args = parser.parse_args()

    global AMR_STATE_FILE
    AMR_STATE_FILE = args.state_file

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.2)
    fig.canvas.manager.set_window_title(args.title)
    
    if args.window_pos:
        try:
            # For TkAgg
            fig.canvas.manager.window.geometry(args.window_pos)
        except Exception:
            pass

    draw_static(ax)
    amrs = create_amrs(ax)

    # Timer text above the map
    timer_text = fig.text(
        0.5, 0.9, args.title,
        ha="center", va="bottom", fontsize=16, weight="bold",
        transform=fig.transFigure,
    )

    # Status lines under the map for each AMR
    status_texts: Dict[int, plt.Text] = {}
    base_y = 0.03
    line_dy = 0.03
    for k in sorted(amrs.keys()):
        status_texts[k] = fig.text(
            0.5, base_y + (4 - k) * line_dy,
            f"AMR{k}: waiting for state...",
            fontsize=11, ha="center", va="bottom",
            transform=fig.transFigure,
        )

    is_running = True
    last_sim_time = 0.0

    timer = fig.canvas.new_timer(interval=UPDATE_INTERVAL_MS)

    def tick():
        nonlocal is_running, last_sim_time

        if args.sync_file:
            try:
                with open(args.sync_file, "r") as f:
                    is_running = (f.read().strip() == "1")
            except Exception:
                pass

        if not is_running:
            timer.start()
            return

        # 1) Read state from JSON
        state = read_amr_state()
        if state is not None:
            apply_state(amrs, state)
            last_sim_time = state.get("sim_time", last_sim_time)

        # 2) Render
        for k, vis in amrs.items():
            # Update marker position (snap — no interpolation)
            if vis.marker:
                vis.marker.center = (vis.x, vis.y)

            # Update label
            if vis.label:
                vis.label.set_position((vis.x, vis.y + 0.55))

            # Update A* route line
            update_route_line(vis)

            # Build status string
            inv = vis.inventory
            inv_str = (
                f"A{inv.get('A', 0)} "
                f"B{inv.get('B', 0)} "
                f"C{inv.get('C', 0)}"
            )
            phase_str = vis.phase if vis.phase else "-"
            job_str = f"Job{vis.job_idx}" if vis.job_idx is not None else "-"
            if vis.mode == "processing":
                status_str = (
                    f"AMR{k}: processing ({vis.proc_ticks}s left), "
                    f"{job_str}, {inv_str}, "
                    f"(x:{vis.x:.0f} y:{vis.y:.0f})"
                )
            elif vis.mode in ("moving_supply", "moving_station", "moving_base"):
                
                status_str = (
                    f"AMR{k}: {vis.mode}, {phase_str}, {job_str}, "
                    f"{inv_str}, (x:{vis.x:.0f} y:{vis.y:.0f})"
                )
            else:
                status_str = (
                    f"AMR{k}: {vis.mode}, {phase_str}, {inv_str}, "
                    f"(x:{vis.x:.0f} y:{vis.y:.0f})"
                )

            # if vis.queue_jids:
            #     status_str += f" | queue={vis.queue_jids}"

            status_texts[k].set_text(status_str)

        # 3) Update timer display
        timer_text.set_text(f"{args.title}  —  Sim Time: {last_sim_time:.0f}s")

        fig.canvas.draw_idle()
        timer.start()

    def on_key(e):
        nonlocal is_running
        if e.key == " ":
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

    fig.canvas.mpl_connect("key_press_event", on_key)
    timer.add_callback(tick)
    timer.start()
    plt.show()


if __name__ == "__main__":
    main()
