#!/usr/bin/env python3
"""
Visualization module for AMR Scheduler.
Handles all matplotlib-based graphics and GUI components.
"""
import os
import json
from dataclasses import dataclass
from typing import List, Dict

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# ===================== VISUAL CONFIG =====================

# Visual config
AMR_COUNT          = 3
UPDATE_INTERVAL_MS = 250

DISPATCHING_LEFT_LABEL_PAD = 7.25
QUEUE_LEFT_LABEL_PAD = 5.25
VIEW_WIDTH     = 60.0

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
TRANSPORT_COLOR = "lightgray"  # Color for transportation time segments
WAIT_COLOR = TRANSPORT_COLOR    # Reuse same color; differentiate by hatch

@dataclass
class JobVisual:
    """Visual representation of a job."""
    jid: int
    jtype: str
    proc_time: float

# ===================== VISUAL STATE =====================

# Top lane (dispatching queue) state
waiting: List[JobVisual] = []
rects_top: List[Rectangle] = []
texts_top: List = []

# Bottom AMR lanes
amr_rects: Dict[int, List[Rectangle]] = {i: [] for i in range(1, AMR_COUNT+1)}
amr_texts: Dict[int, List]           = {i: [] for i in range(1, AMR_COUNT+1)}
amr_load:  Dict[int, float] = {i: 0.0 for i in range(1, AMR_COUNT+1)}

# File tail progress
_lines_consumed = 0
current_makespan = 0.0  # Store the latest makespan
total_solve_time = 0.0  # Sum of solver runtimes across processed events (seconds)

# ===================== VISUALIZATION FUNCTIONS =====================

def remove_artists(rects, texts):
    """Remove matplotlib artists (rectangles and texts) from the plot."""
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
    x = QUEUE_LEFT_LABEL_PAD
    for j in waiting:
        r = Rectangle(
            (x + 2.0, TOP_Y_CENTER),
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
            x + j.proc_time / 4.0 + 2.0,
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


def draw_on_amr(
    ax,
    amr_id: int,
    j: JobVisual,
    transport_time: float = 0.0,
    pick_time: float = 0.0,
    del_time: float = 0.0,
    end_time: float = 0.0,
    pickup_node: int = None,
    delivery_node: int = None,
    wait_time: float = 0.0,
    route_nodes=None,
    route_legs=None,
):
    """Draw one assigned job on an AMR lane (time-aligned only)."""
    y = AMR_Y_CENTERS[amr_id - 1]

    base_x = DISPATCHING_LEFT_LABEL_PAD
    scale = 1.0 / 2.5

    prev_end_time = float(getattr(j, "prev_end_time", 0.0))
    prev_node = getattr(j, "prev_node", None)
    to_pick_travel = float(getattr(j, "to_pick_travel", 0.0))
    idle_before_pick = float(getattr(j, "idle_before_pick", 0.0))

    pick_time = float(pick_time)
    del_time = float(del_time)
    end_time = float(end_time)
    transport_time = float(transport_time)
    wait_time = float(wait_time)

    # If a multi-leg route is provided, render each leg explicitly.
    if route_nodes is None:
        route_nodes = getattr(j, "route_nodes", None)
    if route_legs is None:
        route_legs = getattr(j, "route_legs", None)

    has_route = (
        isinstance(route_nodes, (list, tuple))
        and isinstance(route_legs, (list, tuple))
        and len(route_nodes) >= 2
        and len(route_legs) == (len(route_nodes) - 1)
    )

    # Build a clean, non-overlapping timeline on this AMR lane.
    t0 = prev_end_time
    t_depart = pick_time
    t4 = del_time  # arrive/enter station
    t5 = end_time  # finish processing

    if has_route:
        # Idle before departure
        t_idle_end = max(t0, t_depart)

        def _rect(t_start: float, t_end: float, *, face, hatch, alpha, z, label=None):
            w = max(0.0, (t_end - t_start)) * scale
            if w <= 0:
                return
            x0 = base_x + t_start * scale
            rseg = Rectangle(
                (x0, y - AMR_LANE_H / 2.0),
                w,
                AMR_LANE_H,
                linewidth=0.0,
                edgecolor="gray",
                facecolor=face,
                hatch=hatch,
                clip_on=True,
            )
            rseg.set_alpha(1.0)
            rseg.set_zorder(z)
            ax.add_patch(rseg)
            amr_rects[amr_id].append(rseg)
            if label and w > 1.0:
                t = ax.text(
                    x0 + w / 2.0,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                    weight="bold",
                    clip_on=True,
                )
                t.set_zorder(4)
                amr_texts[amr_id].append(t)

        _rect(
            t0,
            t_idle_end,
            face=WAIT_COLOR,
            hatch="..",
            alpha=1.0,
            z=1,
            label=f"W\n({(t_idle_end - t0):.0f})" if (t_idle_end - t0) > 0 else None,
        )

        # Transport legs
        t_cur = t_idle_end
        for i, leg in enumerate(route_legs):
            a = route_nodes[i]
            b = route_nodes[i + 1]
            t_next = t_cur + max(0.0, float(leg))
            _rect(
                t_cur,
                t_next,
                face=TRANSPORT_COLOR,
                hatch="///",
                alpha=1.0,
                z=2,
                label=f"{int(a)}->{int(b)}\n({(t_next - t_cur):.0f})" if (t_next - t_cur) > 0 else None,
            )
            t_cur = t_next

        # Station wait (queue)
        t_arrive = t_cur
        _rect(
            t_arrive,
            t4,
            face=WAIT_COLOR,
            hatch="..",
            alpha=1.0,
            z=2,
            label=f"W\n({(t4 - t_arrive):.0f})" if (t4 - t_arrive) > 0 else None,
        )

        # Processing block (same as before)
        proc_w = max(0.0, (t5 - t4)) * scale
        x_proc = base_x + t4 * scale
        r = Rectangle(
            (x_proc, y - AMR_LANE_H / 2.0),
            proc_w,
            AMR_LANE_H,
            linewidth=1.8,
            edgecolor="black",
            facecolor=TYPE_COLORS.get(j.jtype, "C3"),
            clip_on=True,
        )
        r.set_zorder(5)
        ax.add_patch(r)
        amr_rects[amr_id].append(r)

        node_label = f"D:{delivery_node}" if delivery_node else ""
        label_text = (
            f"J_{j.jid}\n({float(j.proc_time):.0f})\n{node_label}"
            if node_label
            else f"J_{j.jid}\n({float(j.proc_time):.0f})"
        )
        t = ax.text(
            x_proc + proc_w / 2.0,
            y,
            label_text,
            ha="center",
            va="center",
            fontsize=6,
            weight="bold",
            color="white",
            clip_on=True,
        )
        t.set_zorder(6)
        amr_texts[amr_id].append(t)

        amr_load[amr_id] += max(0.0, t5 - t0)
        return

    # --------- Backward-compatible single-pickup rendering ---------
    t1 = t0 + max(0.0, to_pick_travel)  # arrive pickup
    t2 = pick_time  # start pick
    t3 = t2 + max(0.0, transport_time)  # finish travel to delivery

    # Guard against small numerical inversions
    t1 = min(t1, t2)
    t3 = min(t3, t4)

    def _rect(t_start: float, t_end: float, *, face, hatch, alpha, z, label=None):
        w = max(0.0, (t_end - t_start)) * scale
        if w <= 0:
            return
        x0 = base_x + t_start * scale
        rseg = Rectangle(
            (x0, y - AMR_LANE_H / 2.0),
            w,
            AMR_LANE_H,
            linewidth=0.0,
            edgecolor="gray",
            facecolor=face,
            hatch=hatch,
            clip_on=True,
        )
        rseg.set_alpha(1.0)
        rseg.set_zorder(z)
        ax.add_patch(rseg)
        amr_rects[amr_id].append(rseg)
        if label and w > 1.0:
            t = ax.text(
                x0 + w / 2.0,
                y,
                label,
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                weight="bold",
                clip_on=True,
            )
            t.set_zorder(4)
            amr_texts[amr_id].append(t)

    # Pre-pick travel + idle
    _rect(
        t0,
        t1,
        face=TRANSPORT_COLOR,
        hatch="///",
        alpha=1.0,
        z=1,
        label=(
            (
                f"{int(prev_node)}->{int(pickup_node)}\n({(t1 - t0):.0f})"
                if (prev_node is not None and pickup_node is not None)
                else f"({(t1 - t0):.0f})"
            )
            if (t1 - t0) > 0
            else None
        ),
    )
    _rect(
        t1,
        t2,
        face=WAIT_COLOR,
        hatch="..",
        alpha=1.0,
        z=1,
        label=f"W\n({(t2 - t1):.0f})" if (t2 - t1) > 0 else None,
    )

    # Pick->delivery travel + station-wait
    _rect(
        t2,
        t3,
        face=TRANSPORT_COLOR,
        hatch="///",
        alpha=1.0,
        z=2,
        label=(
            (
                f"{int(pickup_node)}->{int(delivery_node)}\n({(t3 - t2):.0f})"
                if (pickup_node is not None and delivery_node is not None)
                else f"({(t3 - t2):.0f})"
            )
            if (t3 - t2) > 0
            else None
        ),
    )
    _rect(
        t3,
        t4,
        face=WAIT_COLOR,
        hatch="..",
        alpha=1.0,
        z=2,
        label=f"W\n({(t4 - t3):.0f})" if (t4 - t3) > 0 else None,
    )

    # Processing block (on top)
    proc_w = max(0.0, (t5 - t4)) * scale
    x_proc = base_x + t4 * scale
    r = Rectangle(
        (x_proc, y - AMR_LANE_H / 2.0),
        proc_w,
        AMR_LANE_H,
        linewidth=1.8,
        edgecolor="black",
        facecolor=TYPE_COLORS.get(j.jtype, "C3"),
        clip_on=True,
    )
    r.set_zorder(5)
    ax.add_patch(r)
    amr_rects[amr_id].append(r)

    node_label = f"D:{delivery_node}" if delivery_node else ""
    label_text = (
        f"J_{j.jid}\n({float(j.proc_time):.0f})\n{node_label}"
        if node_label
        else f"J_{j.jid}\n({float(j.proc_time):.0f})"
    )
    t = ax.text(
        x_proc + proc_w / 2.0,
        y,
        label_text,
        ha="center",
        va="center",
        fontsize=6,
        weight="bold",
        color="white",
        clip_on=True,
    )
    t.set_zorder(6)
    amr_texts[amr_id].append(t)

    amr_load[amr_id] += (
        float(j.proc_time)
        + transport_time
        + wait_time
        + max(0.0, to_pick_travel)
        + max(0.0, idle_before_pick)
    )


def draw_static_panels(ax):
    """Draw the static side panels for dispatching queue and AMRs."""
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
    handles.append(Patch(facecolor=TRANSPORT_COLOR, edgecolor="gray", hatch="///", label="Transportation"))
    handles.append(Patch(facecolor=WAIT_COLOR, edgecolor="gray", hatch="..", label="Waiting"))
    ax.legend(handles=handles, loc="upper right", frameon=True)


def update_title(ax):
    """Update the figure title with current makespan and solve time."""
    ax.set_title(
        f"AMR Scheduler with Gurobi | Makespan: {current_makespan:.1f} | Solve: {total_solve_time:.2f}s"
    )


def setup_figure():
    """Create and setup the matplotlib figure."""
    fig, ax = plt.subplots(figsize=(16, 5.2))
    ax.set_ylim(AX_Y_MIN, AX_Y_MAX)
    ax.set_xlim(0.0, VIEW_WIDTH)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)

    draw_static_panels(ax)
    update_title(ax)
    
    return fig, ax


def setup_interactions(fig, ax):
    """Setup mouse and keyboard interactions for the figure."""
    # Simple scrolling with mouse wheel and keyboard
    def on_scroll(event):
        if event.inaxes != ax:
            return
        xlim = ax.get_xlim()
        xrange = xlim[1] - xlim[0]
        # Scroll to move horizontally
        if event.button == 'up':
            shift = -xrange * 0.1  # Move left
        elif event.button == 'down':
            shift = xrange * 0.1   # Move right
        else:
            return
        ax.set_xlim([xlim[0] + shift, xlim[1] + shift])
        fig.canvas.draw_idle()
    
    def on_key(event):
        xlim = ax.get_xlim()
        xrange = xlim[1] - xlim[0]
        shift = 0
        
        if event.key == 'left':
            shift = -xrange * 0.15  # Move left
        elif event.key == 'right':
            shift = xrange * 0.15   # Move right
        elif event.key == 'home':
            # Go to beginning
            ax.set_xlim([0, xrange])
            fig.canvas.draw_idle()
            return
        else:
            return
        
        ax.set_xlim([xlim[0] + shift, xlim[1] + shift])
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Enable panning (dragging to scroll)
    pan_data = {'pressed': False, 'xpress': None, 'xlim': None}
    
    def on_press(event):
        if event.inaxes != ax:
            return
        pan_data['pressed'] = True
        pan_data['xpress'] = event.xdata
        pan_data['xlim'] = list(ax.get_xlim())
    
    def on_release(event):
        pan_data['pressed'] = False
        pan_data['xpress'] = None
    
    def on_motion(event):
        if not pan_data['pressed'] or event.inaxes != ax or pan_data['xpress'] is None:
            return
        dx = event.xdata - pan_data['xpress']
        cur_xlim = ax.get_xlim()
        new_xlim = [pan_data['xlim'][0] - dx, pan_data['xlim'][1] - dx]
        ax.set_xlim(new_xlim)
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)


def reset_state():
    """Reset visualization state for new event."""
    global waiting, _lines_consumed, current_makespan, total_solve_time
    waiting = []
    _lines_consumed = 0
    current_makespan = 0.0
    total_solve_time = 0.0
    for m in amr_rects:
        amr_rects[m].clear()
        amr_texts[m].clear()
        amr_load[m] = 0.0


def save_dpr_schedule_image(
    schedule,
    image_path,
    *,
    rule_name="DPR",
    instance_index=0,
    makespan=None,
):
    """Render a DPR schedule to a PNG using the AMR timeline visualization."""
    global current_makespan, total_solve_time

    if makespan is None:
        makespan = max((float(rec.get("end", 0.0)) for rec in schedule), default=0.0)

    out_dir = os.path.dirname(str(image_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    reset_state()
    current_makespan = float(makespan)
    total_solve_time = 0.0

    fig, ax = setup_figure()
    ax.set_title(
        f"{rule_name} | Instance {int(instance_index)} | Makespan: {float(makespan):.1f}"
    )

    scale = 1.0 / 2.5
    max_x = DISPATCHING_LEFT_LABEL_PAD + max(float(makespan), 1.0) * scale + 10.0
    fig.set_size_inches(min(max(16.0, max_x / 22.0), 96.0), 5.2)
    ax.set_xlim(0.0, max(VIEW_WIDTH, max_x))

    for rec in sorted(
        schedule,
        key=lambda r: (
            int(r.get("amr", 1)),
            float(r.get("depart", r.get("start", 0.0))),
            float(r.get("start", 0.0)),
        ),
    ):
        amr_id = int(rec.get("amr", 1))
        if amr_id < 1 or amr_id > AMR_COUNT:
            continue

        job = JobVisual(
            jid=int(rec.get("job", 0)),
            jtype=str(rec.get("material", "A")),
            proc_time=float(rec.get("processing", 0.0)),
        )
        job.prev_end_time = float(rec.get("amr_ready", rec.get("depart", 0.0)))
        job.prev_node = rec.get("amr_prev_node")
        job.route_nodes = rec.get("route_nodes")
        job.route_legs = rec.get("route_legs")

        draw_on_amr(
            ax,
            amr_id,
            job,
            transport_time=float(rec.get("transport", 0.0)),
            pick_time=float(rec.get("depart", rec.get("start", 0.0))),
            del_time=float(rec.get("start", 0.0)),
            end_time=float(rec.get("end", 0.0)),
            pickup_node=rec.get("pickup_node"),
            delivery_node=rec.get("delivery_node"),
            wait_time=float(rec.get("machine_wait", 0.0)),
            route_nodes=rec.get("route_nodes"),
            route_legs=rec.get("route_legs"),
        )

    fig.savefig(str(image_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
