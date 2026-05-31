#!/usr/bin/env python3
import os
import json
import time
from dataclasses import dataclass
from typing import List, Dict

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# ===================== MILP CONFIG / LOGIC (from MILP.py) =====================

TIME_LIMIT = 10
GRID_SIZE = 10
M_BIG = 100.0
P_NODES = {1, 6, 10}  # pickup candidate nodes (grid indices)
JSON_STATION_MAPPING = {1: 93, 2: 96, 3: 98}  # station id -> delivery node
M_SET = range(1, 4)   # 3 AGVs (1,2,3)
S_m = {1: 1, 2: 6, 3: 10}  # AGV start nodes (grid indices)

# Paths (keep MILP's paths)
INBOX = "Random_Job_Arrivals/dispatch_inbox.jsonl"
SCHEDULE_OUTBOX = "Random_Job_Arrivals/schedule_outbox.jsonl"

POLL_INTERVAL = 1.0  # seconds (used to re-arm timer; not blocking sleep here)


def calculate_distance(node1, node2, grid_size=GRID_SIZE):
    r1, c1 = (node1 - 1) // grid_size, (node1 - 1) % grid_size
    r2, c2 = (node2 - 1) // grid_size, (node2 - 1) % grid_size
    return abs(r1 - r2) + abs(c1 - c2)


def solve_vrp_from_jobs(jobs, time_limit=TIME_LIMIT):
    """MILP dispatching logic from MILP.py, returning sequence_map & makespan."""
    if not jobs:
        return None

    L_REAL = [j["jid"] for j in jobs]
    n_tasks = len(L_REAL)
    L_SET = range(1, n_tasks + 1)
    VIRTUAL_END = n_tasks + 1
    L_PRIME = range(0, VIRTUAL_END + 1)

    L_REAL_MAP = {idx: L_REAL[idx - 1] for idx in L_SET}
    TASK_DATA = {}
    for idx in L_SET:
        job = next(j for j in jobs if j["jid"] == L_REAL_MAP[idx])
        TASK_DATA[idx] = {
            "E_l": float(job["proc_time"]),
            "g_l": int(job["g_l"]),
            "type": job.get("type", "?"),
            "arrival_time": float(job.get("arrival_time", 0.0)),
        }

    # event-specific Big-M
    delivery_nodes = [TASK_DATA[i]["g_l"] for i in L_SET]
    D_max = max(calculate_distance(p, g) for p in P_NODES for g in delivery_nodes)
    S_max = max(calculate_distance(s, p) for s in S_m.values() for p in P_NODES)
    E_sum = sum(TASK_DATA[i]["E_l"] for i in L_SET)
    M_local = max(1.0, float(S_max + n_tasks * D_max + E_sum + 5.0))

    model = gp.Model("Reschedule_event")
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 0.0
    model.Params.OutputFlag = 0

    Y = model.addVars(L_SET, M_SET, vtype=GRB.BINARY, name="Y")
    W = model.addVars(L_PRIME, L_PRIME, M_SET, vtype=GRB.BINARY, name="W")
    A_P = model.addVars(L_SET, P_NODES, vtype=GRB.BINARY, name="A")
    T_Pick = model.addVars(L_SET, lb=0.0, name="T_pick")
    T_Del = model.addVars(L_SET, lb=0.0, name="T_del")
    T_End = model.addVars(L_SET, lb=0.0, name="T_end")
    T_makespan = model.addVar(lb=0.0, name="T_makespan")

    model.addConstrs((T_makespan >= T_End[l] for l in L_SET), name="makespan_link")
    model.setObjective(T_makespan, GRB.MINIMIZE)

    model.addConstrs(
        (gp.quicksum(A_P[l, p] for p in P_NODES) == 1 for l in L_SET),
        name="one_pick",
    )
    model.addConstrs(
        (gp.quicksum(Y[l, m] for m in M_SET) == 1 for l in L_SET),
        name="assign",
    )

    for l in L_SET:
        model.addConstr(
            gp.quicksum(W[lp, l, m] for lp in L_PRIME if lp != l for m in M_SET) == 1,
            name=f"pred_{l}",
        )
        model.addConstr(
            gp.quicksum(W[l, ln, m] for ln in L_PRIME if ln != l for m in M_SET) == 1,
            name=f"succ_{l}",
        )
        model.addConstrs(
            (W[lp, l, m] <= Y[l, m] for lp in L_PRIME if lp != l for m in M_SET),
            name=f"link_in_{l}",
        )
        model.addConstrs(
            (W[l, ln, m] <= Y[l, m] for ln in L_PRIME if ln != l for m in M_SET),
            name=f"link_out_{l}",
        )

    for m in M_SET:
        # allow AMR m to be idle if no job is assigned to it
        model.addConstr(
            gp.quicksum(W[0, l, m] for l in L_SET) <= 1, name=f"start_{m}"
        )
        model.addConstr(
            gp.quicksum(W[l, VIRTUAL_END, m] for l in L_SET) <= 1, name=f"end_{m}"
        )


    model.addConstrs((W[l, 0, m] == 0 for l in L_SET for m in M_SET), name="no_ret0")
    model.addConstrs(
        (T_End[l] == T_Del[l] + TASK_DATA[l]["E_l"] for l in L_SET),
        name="exec",
    )
    model.addConstrs(
        (T_Pick[l] >= TASK_DATA[l]["arrival_time"] for l in L_SET),
        name="arrival",
    )

    for l in L_SET:
        g_l = TASK_DATA[l]["g_l"]
        min_d = min(calculate_distance(pp, g_l) for pp in P_NODES)
        for p in P_NODES:
            d_pg = calculate_distance(p, g_l)
            model.addConstr(
                T_Del[l] >= T_Pick[l] + d_pg - M_BIG * (1 - A_P[l, p]),
                name=f"trans_{l}_{p}",
            )
        model.addConstr(T_Del[l] >= T_Pick[l] + min_d, name=f"trans_lb_{l}")

    for l in L_SET:
        for lp in L_SET:
            if lp == l:
                continue
            g_lp = TASK_DATA[lp]["g_l"]
            for p in P_NODES:
                d_dp = calculate_distance(g_lp, p)
                model.addConstrs(
                    (
                        T_Pick[l]
                        >= T_End[lp]
                        + d_dp
                        - M_local * (2 - W[lp, l, m] - A_P[l, p])
                        for m in M_SET
                    ),
                    name=f"seq_{lp}_{l}_{p}",
                )
        for m in M_SET:
            Snode = S_m[m]
            for p in P_NODES:
                d_sp = calculate_distance(Snode, p)
                model.addConstr(
                    T_Pick[l]
                    >= d_sp - M_local * (2 - W[0, l, m] - A_P[l, p]),
                    name=f"start_seq_{l}_{m}_{p}",
                )

    model.optimize()
    if model.SolCount == 0:
        return None

    # pull solution values
    A_vals       = model.getAttr("X", A_P)
    Y_vals       = model.getAttr("X", Y)
    T_pick_vals  = model.getAttr("X", T_Pick)
    T_del_vals   = model.getAttr("X", T_Del)
    T_end_vals   = model.getAttr("X", T_End)

    # build sequences per AMR directly from Y + times
    seq_map = {m: [] for m in M_SET}

    for l in L_SET:
        # which AGV?
        assigned_m = None
        for m in M_SET:
            if Y_vals.get((l, m), 0.0) >= 0.5:
                assigned_m = m
                break
        if assigned_m is None:
            # shouldn't happen, but be safe
            continue

        # times
        pick_t = float(T_pick_vals.get(l, 0.0))
        del_t  = float(T_del_vals.get(l, 0.0))
        end_t  = float(T_end_vals.get(l, 0.0))

        # chosen pickup node
        chosen_pick = None
        for p in P_NODES:
            if A_vals.get((l, p), 0.0) >= 0.5:
                chosen_pick = p
                break

        delivery_node = TASK_DATA[l]["g_l"]

        seq_map[assigned_m].append(
            {
                "idx": l,
                "jid": L_REAL_MAP[l],
                "assigned_agv": int(assigned_m),
                "type": TASK_DATA[l].get("type", "?"),
                "pickup_node": int(chosen_pick) if chosen_pick is not None else None,
                "delivery_node": int(delivery_node),
                "pick_time": pick_t,
                "del_time": del_t,
                "end_time": end_t,
                "proc_time": float(TASK_DATA[l]["E_l"]),
            }
        )

    # sort tasks on each AMR by pick_time, then jid (for stable output)
    for m in M_SET:
        seq_map[m].sort(key=lambda j: (j["pick_time"], j["jid"]))

    makespan = max(float(v) for v in T_end_vals.values()) if T_end_vals else 0.0
    return {"sequence_map": seq_map, "makespan": float(makespan)}



# ===================== VISUAL CONFIG / STATE (from amr_scheduler.py) =====================

# Visual config
AMR_COUNT          = 3
UPDATE_INTERVAL_MS = 250

LEFT_LABEL_PAD = 5.5
VIEW_WIDTH     = 40.0

AX_Y_MIN, AX_Y_MAX = 0.0, 2.0
TOP_Y_CENTER  = 1.25
TOP_LANE_H    = 0.5

BOTTOM_MIN    = 0.0
BOTTOM_HEIGHT = (AX_Y_MAX - AX_Y_MIN) / 2.0
AMR_Y_CENTERS = [BOTTOM_MIN + (i + 0.5) * (BOTTOM_HEIGHT / AMR_COUNT)
                 for i in range(AMR_COUNT)]
AMR_LANE_H    = BOTTOM_HEIGHT / AMR_COUNT * 0.7

_cycle = plt.rcParams.get("axes.prop_cycle", None)
_cycle_list = _cycle.by_key()["color"] if _cycle else ["C0", "C1", "C2", "C3"]
TYPE_COLORS = {"A": _cycle_list[0], "B": _cycle_list[1], "C": _cycle_list[2]}

@dataclass
class JobVisual:
    jid: int
    jtype: str
    proc_time: float
    station: str

# Top lane (dispatching queue) state
waiting: List[JobVisual] = []
rects_top: List[Rectangle] = []
texts_top: List = []

# Bottom AMR lanes
amr_cursor: Dict[int, float] = {i: LEFT_LABEL_PAD for i in range(1, AMR_COUNT+1)}
amr_rects: Dict[int, List[Rectangle]] = {i: [] for i in range(1, AMR_COUNT+1)}
amr_texts: Dict[int, List]           = {i: [] for i in range(1, AMR_COUNT+1)}
amr_load:  Dict[int, float] = {i: 0.0 for i in range(1, AMR_COUNT+1)}

# File tail progress
_lines_consumed = 0


def remove_artists(rects, texts):
    for a in rects:
        try:
            a.remove()
        except Exception:
            pass
    for t in texts:
        try:
            t.remove()
        except Exception:
            pass
    rects.clear()
    texts.clear()


def rebuild_top_lane(ax):
    """Show the current event's jobs in the top dispatching lane."""
    remove_artists(rects_top, texts_top)
    x = LEFT_LABEL_PAD
    for j in waiting:
        r = Rectangle(
            (x, TOP_Y_CENTER),
            j.proc_time / 2.0,
            TOP_LANE_H,
            linewidth=1.2,
            edgecolor="black",
            facecolor=TYPE_COLORS.get(j.jtype, "C3"),
            clip_on=True,
        )
        ax.add_patch(r)
        rects_top.append(r)
        t = ax.text(
            x + j.proc_time / 4.0,
            TOP_Y_CENTER + TOP_LANE_H / 2.0,
            f"J_{j.jid}",
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            clip_on=True,
        )
        texts_top.append(t)
        x += j.proc_time / 2.0


def draw_on_amr(ax, amr_id: int, j: JobVisual):
    """Draw one assigned job on an AMR lane at that lane's cursor."""
    x = amr_cursor[amr_id]
    y = AMR_Y_CENTERS[amr_id - 1]
    r = Rectangle(
        (x, y - AMR_LANE_H / 2.0),
        j.proc_time / 4.0,
        AMR_LANE_H,
        linewidth=1.2,
        edgecolor="black",
        facecolor=TYPE_COLORS.get(j.jtype, "C3"),
        clip_on=True,
    )
    ax.add_patch(r)
    amr_rects[amr_id].append(r)
    t = ax.text(
        x + j.proc_time / 8.0,
        y,
        f"J_{j.jid}",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        clip_on=True,
    )
    amr_texts[amr_id].append(t)
    amr_cursor[amr_id] += j.proc_time / 4.0
    amr_load[amr_id] += j.proc_time


def draw_static_panels(ax):
    band_frac = 0.12
    # Top panel
    top_panel = Rectangle(
        (0.0, 0.5),
        band_frac,
        0.5,
        transform=ax.transAxes,
        fill=False,
        linewidth=1.8,
        clip_on=False,
        zorder=3,
    )
    ax.add_patch(top_panel)
    tp = ax.text(
        band_frac * 0.5,
        0.75,
        "Dispatching\nQueue",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        color="gray",
        clip_on=True,
        zorder=4,
    )
    tp.set_clip_path(top_panel)

    # Bottom panel (AMRs)
    bot_panel = Rectangle(
        (0.0, 0.0),
        band_frac,
        0.5,
        transform=ax.transAxes,
        fill=False,
        linewidth=1.8,
        clip_on=False,
        zorder=3,
    )
    ax.add_patch(bot_panel)

    for i in range(AMR_COUNT):
        y_frac = (i + 0.5) / AMR_COUNT * 0.5
        txt = ax.text(
            band_frac * 0.5,
            y_frac,
            f"AMR {i+1}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            color="gray",
            clip_on=True,
            zorder=4,
        )
        txt.set_clip_path(bot_panel)

    handles = [
        Patch(facecolor=TYPE_COLORS[k], edgecolor="black", label=f"Type {k}")
        for k in TYPE_COLORS
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)


def update_title(ax):
    total_wait = sum(j.proc_time for j in waiting)
    loads = ", ".join(f"{k}:{amr_load[k]:.0f}" for k in sorted(amr_load))
    ax.set_title(
        f"AMR Scheduler with Gurobi"
    )


# ===================== INBOX INGEST + MILP + VISUAL =====================

def process_event_line_visual(line: str, ax, out_f):
    """Parse one dispatch event line, show top lane, run MILP, draw assignments, write schedule_outbox."""
    global waiting

    try:
        data = json.loads(line)
    except Exception:
        return

    dispatch_time = float(data.get("dispatch_time", 0.0))
    jobs_raw = data.get("jobs", [])
    if not jobs_raw:
        return

    # Build waiting list for top lane (show the event's jobs)
    waiting = []
    for j in jobs_raw:
        waiting.append(
            JobVisual(
                jid=int(j.get("jid")),
                jtype=str(j.get("type", "A")),
                proc_time=float(j.get("proc_time", 0.0)),
                station=str(j.get("station")),
            )
        )
    rebuild_top_lane(ax)

    # Prepare jobs for MILP solver
    jobs_for_milp = []
    for j in jobs_raw:
        st = j.get("station")
        g_l = JSON_STATION_MAPPING.get(st)
        if g_l is None:
            continue
        jobs_for_milp.append(
            {
                "jid": int(j.get("jid")),
                "type": j.get("type"),
                "proc_time": float(j.get("proc_time", 0.0)),
                "station": int(st),
                "g_l": int(g_l),
                "arrival_time": 0.0,
            }
        )
    if not jobs_for_milp:
        return

    res = solve_vrp_from_jobs(jobs_for_milp)
    if res is None:
        return

    # Flatten MILP result jobs by pick_time then jid
    all_jobs = []
    for m in M_SET:
        all_jobs.extend(res["sequence_map"].get(m, []))
    all_jobs.sort(key=lambda j: (j.get("pick_time", 0.0), j.get("jid", 0)))

    # Draw assignments + write schedule_outbox.jsonl
    for job in all_jobs:
        amr = job.get("assigned_agv")
        jid = job.get("jid")
        jtype = job.get("type") or "?"
        proc_time = float(job.get("proc_time", 0.0))
        delivery_node = job.get("delivery_node")

        # recover station from delivery node
        station = None
        for st, del_node in JSON_STATION_MAPPING.items():
            if del_node == delivery_node:
                station = st
                break

        # draw on AMR lane (graphic)
        vjob = JobVisual(
            jid=int(jid),
            jtype=jtype,
            proc_time=proc_time,
            station=str(station) if station is not None else "?",
        )
        draw_on_amr(ax, int(amr), vjob)

        # write schedule record (MILP-based dispatching result)
        rec = {
            "generated_at": dispatch_time,
            "amr": int(amr),
            "jid": int(jid),
            "type": jtype,
            "proc_time": proc_time,
            "station": str(station) if station is not None else "?",
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()


# ===================== MAIN (timer + tail loop with graphics) =====================

def main():
    global _lines_consumed

    # Clear old schedule outbox
    os.makedirs(os.path.dirname(SCHEDULE_OUTBOX), exist_ok=True)
    open(SCHEDULE_OUTBOX, "w", encoding="utf-8").close()

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_ylim(AX_Y_MIN, AX_Y_MAX)
    ax.set_xlim(0.0, VIEW_WIDTH)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)

    draw_static_panels(ax)
    update_title(ax)

    timer = fig.canvas.new_timer(interval=UPDATE_INTERVAL_MS)

    def tick():
        global _lines_consumed
        changed = False

        try:
            with open(INBOX, "r", encoding="utf-8") as f_in, \
                 open(SCHEDULE_OUTBOX, "a", encoding="utf-8") as f_out:
                lines = f_in.readlines()
                if _lines_consumed < len(lines):
                    new_lines = lines[_lines_consumed:]
                    _lines_consumed = len(lines)
                    for line in new_lines:
                        if line.strip():
                            process_event_line_visual(line, ax, f_out)
                            changed = True
        except FileNotFoundError:
            pass

        if changed:
            update_title(ax)
            fig.canvas.draw_idle()

        timer.start()

    timer.add_callback(tick)
    timer.start()
    plt.show()


if __name__ == "__main__":
    main()
