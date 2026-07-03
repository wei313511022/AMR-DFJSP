from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox


def format_station_label(station: Any) -> str:
    if station is None:
        return ""
    s = str(station)
    if not s:
        return ""
    if s.startswith("S"):
        return s
    return f"S{s}"


def format_job_label(jid: Any, station: Any) -> str:
    station = format_station_label(station)
    station_tag = f"@{station}" if station else ""
    return f"J_{jid}{station_tag}"


def format_inventory(inv: Dict[str, int]) -> str:
    return f"A:{inv.get('A',0)} B:{inv.get('B',0)} C:{inv.get('C',0)}"


def build_robot_labels(
    inventories: Optional[List[Dict[str, int]]], num_robots: int = 5
) -> List[str]:
    if inventories:
        num_robots = max(num_robots, len(inventories))
    base = [f"AMR{i+1}" for i in range(num_robots)]
    if not inventories:
        return base
    labels: List[str] = []
    for i, name in enumerate(base):
        inv = inventories[i] if i < len(inventories) else {}
        labels.append(f"{name} [{format_inventory(inv)}]")
    return labels


def split_transport_intervals(item: dict, seg: dict) -> Dict[str, List[Tuple[float, float]]]:
    """
    Split a transport segment into four interval groups:
      - pickup_move: moving toward material source
      - pickup_wait: waiting while still on the way to material source
      - station_move: moving toward workstation
      - station_wait: waiting while on the way to workstation
    """
    s = float(seg.get("start", 0.0))
    e = float(seg.get("end", s))
    dur = e - s
    if dur <= 1e-9:
        return {
            "pickup_move": [],
            "pickup_wait": [],
            "station_move": [],
            "station_wait": [],
        }

    path = item.get("transport_path", []) or []
    if len(path) < 2:
        return {
            "pickup_move": [],
            "pickup_wait": [],
            "station_move": [(s, dur)],
            "station_wait": [],
        }

    step_count = len(path) - 1
    if step_count <= 0:
        return {
            "pickup_move": [],
            "pickup_wait": [],
            "station_move": [(s, dur)],
            "station_wait": [],
        }

    dt = dur / float(step_count)
    pickup_move_raw: List[Tuple[float, float]] = []
    pickup_wait_raw: List[Tuple[float, float]] = []
    station_move_raw: List[Tuple[float, float]] = []
    station_wait_raw: List[Tuple[float, float]] = []

    need_pickup = bool(item.get("need_pickup", False))
    pickup_idx: Optional[int] = None
    if need_pickup:
        pickup = tuple(item.get("pickup", ()))
        if len(pickup) == 2:
            for i, c in enumerate(path):
                if tuple(c) == pickup:
                    pickup_idx = i
                    break

    for i in range(step_count):
        t0 = s + dt * i
        t1 = t0 + dt
        if t1 <= t0 + 1e-9:
            continue
        c0 = path[i]
        c1 = path[i + 1]
        is_wait = tuple(c0) == tuple(c1)
        on_pickup_leg = bool(need_pickup and (pickup_idx is not None) and (i < pickup_idx))
        if on_pickup_leg:
            if is_wait:
                pickup_wait_raw.append((t0, t1 - t0))
            else:
                pickup_move_raw.append((t0, t1 - t0))
        else:
            if is_wait:
                station_wait_raw.append((t0, t1 - t0))
            else:
                station_move_raw.append((t0, t1 - t0))

    def merge_adjacent(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        out: List[Tuple[float, float]] = []
        cur_s, cur_d = intervals[0]
        cur_e = cur_s + cur_d
        for x_s, x_d in intervals[1:]:
            x_e = x_s + x_d
            if abs(x_s - cur_e) <= 1e-9:
                cur_e = x_e
            else:
                out.append((cur_s, cur_e - cur_s))
                cur_s = x_s
                cur_e = x_e
        out.append((cur_s, cur_e - cur_s))
        return out

    return {
        "pickup_move": merge_adjacent(pickup_move_raw),
        "pickup_wait": merge_adjacent(pickup_wait_raw),
        "station_move": merge_adjacent(station_move_raw),
        "station_wait": merge_adjacent(station_wait_raw),
    }


def draw_dispatch_queue(
    ax, trace: List[dict], show_labels: bool = True, current_t: Optional[float] = None
) -> None:
    ax.set_title("Dispatching Queue")
    ax.set_xlabel("Time (s)")
    if not trace:
        ax.set_yticks([0.5])
        ax.set_yticklabels(["Dispatching\nQueue"])
        ax.text(0.5, 0.5, "No trace to plot.", transform=ax.transAxes, ha="center", va="center")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    type_to_color = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}

    y0, h = 10, 6
    cursor = 0.0
    for item in trace:
        proc = float(item.get("proc_time", 0.0))
        jtype = item.get("type")
        jid = item.get("jid", item.get("seq", ""))
        label = format_job_label(jid, item.get("dst"))
        ax.broken_barh([(cursor, proc)], (y0, h), facecolors=type_to_color.get(jtype, "tab:gray"))
        if show_labels:
            ax.text(cursor + proc / 2, y0 + h / 2, label, ha="center", va="center", fontsize=9, color="black")
        cursor += proc

    ax.set_yticks([y0 + h / 2])
    ax.set_yticklabels(["Dispatching\nQueue"])
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    leg = [
        mpatches.Patch(color=type_to_color["A"], label="Type A"),
        mpatches.Patch(color=type_to_color["B"], label="Type B"),
        mpatches.Patch(color=type_to_color["C"], label="Type C"),
    ]
    ax.legend(handles=leg, loc="upper right", frameon=True)

    if current_t is not None:
        ax.axvline(current_t, color="red", linestyle="--", linewidth=1)


def plot_dispatch_queue(
    trace: List[dict], save_path: Optional[str] = None, title_info: Optional[str] = None
):
    if not trace:
        print("No trace to plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 2.2))
    draw_dispatch_queue(ax, trace)

    if title_info:
        fig.suptitle(title_info)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def build_arrivals(input_source: Union[dict, List[dict]]) -> Tuple[List[dict], float]:
    arrivals: List[dict] = []
    times: List[float] = []
    if isinstance(input_source, dict) and "jobs" in input_source:
        dt = float(input_source.get("dispatch_time", 0.0))
        times.append(dt)
        for job in input_source.get("jobs", []):
            item = dict(job)
            item["arrival_t"] = dt
            arrivals.append(item)
    elif isinstance(input_source, list):
        if len(input_source) > 0 and isinstance(input_source[0], dict) and "jobs" in input_source[0]:
            for rec in input_source:
                dt = float(rec.get("dispatch_time", 0.0))
                times.append(dt)
                for job in rec.get("jobs", []):
                    item = dict(job)
                    item["arrival_t"] = dt
                    arrivals.append(item)
        else:
            for job in input_source:
                item = dict(job)
                item["arrival_t"] = 0.0
                arrivals.append(item)
            if input_source:
                times.append(0.0)

    t0 = min(times) if times else 0.0
    if t0 != 0.0:
        for item in arrivals:
            item["arrival_t"] = float(item.get("arrival_t", 0.0)) - t0
    return arrivals, t0


def draw_input_queue(
    ax,
    input_source: Union[dict, List[dict]],
    show_labels: bool = True,
    current_t: Optional[float] = None,
) -> None:
    ax.set_title("Input Queue")
    ax.set_xlabel("Time (s)")
    arrivals, _t0 = build_arrivals(input_source)
    if not arrivals:
        ax.set_yticks([0.5])
        ax.set_yticklabels(["Input Queue"])
        ax.text(0.5, 0.5, "No jobs to plot.", transform=ax.transAxes, ha="center", va="center")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        if current_t is not None:
            ax.axvline(current_t, color="red", linestyle="--", linewidth=1)
            ax.set_xlim(0.0, max(1.0, float(current_t)))
        return

    type_to_color = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}

    buckets: Dict[int, List[dict]] = {}
    for item in arrivals:
        sec = int(float(item.get("arrival_t", 0.0)))
        buckets.setdefault(sec, []).append(item)

    max_slots = max((len(v) for v in buckets.values()), default=1)
    lane_h = 54
    lane_gap = 20
    arrival_width = 10
    max_sec = max(buckets.keys()) if buckets else 0

    for sec in sorted(buckets.keys()):
        for slot_idx, item in enumerate(buckets[sec]):
            y0 = slot_idx * (lane_h + lane_gap)
            jtype = str(item.get("type", "")).upper()
            jid = item.get("jid", "")
            label = format_job_label(jid, item.get("station"))
            ax.broken_barh(
                [(sec, arrival_width)],
                (y0, lane_h),
                facecolors=type_to_color.get(jtype, "tab:gray"),
                edgecolors="white",
            )
            if show_labels:
                ax.text(sec + arrival_width / 2, y0 + lane_h / 2, label, ha="center", va="center", fontsize=12, color="black")

    total_h = max_slots * (lane_h + lane_gap) - lane_gap
    if total_h <= 0:
        total_h = lane_h

    ax.set_ylim(-lane_gap, total_h + lane_gap)
    ax.set_yticks([total_h / 2])
    ax.set_yticklabels(["Input Queue"])
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    if current_t is not None:
        ax.axvline(current_t, color="red", linestyle="--", linewidth=1)
        max_x = max(max_sec + arrival_width, float(current_t))
    else:
        max_x = max_sec + arrival_width
    ax.set_xlim(0.0, max_x)

    leg = [
        mpatches.Patch(color=type_to_color["A"], label="Type A"),
        mpatches.Patch(color=type_to_color["B"], label="Type B"),
        mpatches.Patch(color=type_to_color["C"], label="Type C"),
    ]
    ax.legend(handles=leg, loc="upper right", frameon=True)


def plot_input_queue(
    input_source: Union[dict, List[dict]],
    save_path: Optional[str] = None,
    title_info: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(11, 4.0))
    draw_input_queue(ax, input_source)

    if title_info:
        fig.suptitle(title_info)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def draw_amr_schedule(
    ax,
    trace: List[dict],
    makespan: Optional[float],
    show_labels: bool = True,
    current_t: Optional[float] = None,
    inventories: Optional[List[Dict[str, int]]] = None,
) -> None:
    if makespan is None:
        title = "AMR Schedule (DDQN)"
    else:
        title = f"AMR Schedule (DDQN) | Makespan: {makespan:.1f}s"
    ax.set_title(title)
    ax.set_xlabel("Time (s)")

    if not trace:
        ax.text(0.5, 0.5, "No trace to plot.", transform=ax.transAxes, ha="center", va="center")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        return

    type_to_color = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    robot_names = build_robot_labels(inventories)

    lane_h = 16
    lane_gap = 8

    for item in trace:
        rid = item["robot"]
        lane_y = rid * (lane_h + lane_gap)

        jtype = item["type"]
        jid = item.get("jid", item.get("seq", ""))
        label_job = format_job_label(jid, item.get("dst"))

        for seg in item["segments"]:
            s = seg["start"]
            e = seg["end"]
            dur = e - s
            if dur <= 1e-9:
                continue

            if seg["kind"] == "transport":
                split_map = split_transport_intervals(item, seg)
                pickup_move = split_map["pickup_move"]
                pickup_wait = split_map["pickup_wait"]
                station_move = split_map["station_move"]
                station_wait = split_map["station_wait"]

                if pickup_move:
                    ax.broken_barh(
                        pickup_move,
                        (lane_y, lane_h),
                        facecolors="#e6f0ff",
                        hatch="\\\\\\",
                        edgecolors="#4c72b0",
                    )
                if station_move:
                    ax.broken_barh(
                        station_move,
                        (lane_y, lane_h),
                        facecolors="lightgray",
                        hatch="///",
                        edgecolors="gray",
                    )
                all_wait_intervals = pickup_wait + station_wait
                if all_wait_intervals:
                    ax.broken_barh(
                        all_wait_intervals,
                        (lane_y, lane_h),
                        facecolors="lightgray",
                        hatch="...",
                        edgecolors="gray",
                    )
            elif seg["kind"] == "wait":
                ax.broken_barh([(s, dur)], (lane_y, lane_h), facecolors="lightgray", hatch="...", edgecolors="gray")
            else:
                ax.broken_barh([(s, dur)], (lane_y, lane_h), facecolors=type_to_color.get(jtype, "tab:gray"))
                if show_labels:
                    ax.text(s + dur / 2, lane_y + lane_h / 2, label_job, ha="center", va="center", fontsize=9, color="black")

    yticks = [i * (lane_h + lane_gap) + lane_h / 2 for i in range(3)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(robot_names)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    leg = [
        mpatches.Patch(color=type_to_color["A"], label="Type A"),
        mpatches.Patch(color=type_to_color["B"], label="Type B"),
        mpatches.Patch(color=type_to_color["C"], label="Type C"),
        mpatches.Patch(
            facecolor="#e6f0ff",
            edgecolor="#4c72b0",
            hatch="\\\\\\",
            label="To Material",
        ),
        mpatches.Patch(facecolor="lightgray", edgecolor="gray", hatch="///", label="To Station"),
        mpatches.Patch(facecolor="lightgray", edgecolor="gray", hatch="...", label="Waiting"),
    ]
    ax.legend(handles=leg, loc="upper right", frameon=True)

    if current_t is not None:
        ax.axvline(current_t, color="red", linestyle="--", linewidth=1)


def plot_amr_schedule(
    trace: List[dict],
    makespan: float,
    save_path: Optional[str] = None,
    inventories: Optional[List[Dict[str, int]]] = None,
    title_info: Optional[str] = None,
):
    if not trace:
        print("No trace to plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 4.5))
    draw_amr_schedule(ax, trace, makespan, inventories=inventories)

    if title_info:
        fig.suptitle(title_info)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def show_interactive_schedule(
    trace: List[dict],
    input_source: Union[dict, List[dict]],
    makespan: Optional[float],
    inventories: Optional[List[Dict[str, int]]] = None,
    window: float = 60.0,
    play_step: float = 0.5,
    play_interval_ms: int = 100,
    title_info: Optional[str] = None,
    initial_t: float = 0.0,
):
    """Interactive viewer: pan/zoom with mouse, or use controls to jump in time."""
    if not trace:
        print("No trace to plot.")
        return

    total_proc = sum(float(item.get("proc_time", 0.0)) for item in trace)
    arrivals, _t0 = build_arrivals(input_source)
    max_arrival = max((float(item.get("arrival_t", 0.0)) for item in arrivals), default=0.0)
    max_t = max(makespan or 0.0, total_proc, max_arrival + 1.0, 1.0)
    window = min(window, max_t)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.5, bottom=0.22, top=0.9)
    if title_info:
        fig.suptitle(title_info, y=0.98)

    draw_dispatch_queue(axes[0], trace)
    draw_amr_schedule(axes[1], trace, makespan, inventories=inventories)
    draw_input_queue(axes[2], input_source, current_t=initial_t)

    vline = axes[1].axvline(0.0, color="red", linewidth=1)

    ax_center = fig.add_axes([0.12, 0.12, 0.58, 0.03])
    ax_window = fig.add_axes([0.12, 0.07, 0.58, 0.03])
    ax_timebox = fig.add_axes([0.74, 0.115, 0.1, 0.04])
    ax_button = fig.add_axes([0.86, 0.115, 0.1, 0.04])

    s_center = Slider(ax_center, "Center t", 0.0, max_t, valinit=0.0, valstep=0.1)
    s_window = Slider(ax_window, "Window", 1.0, max_t, valinit=window, valstep=0.5)
    time_box = TextBox(ax_timebox, "t", initial="0.0")
    btn = Button(ax_button, "Play")

    state = {"updating": False, "playing": False}

    def update_view(center: float) -> None:
        half = s_window.val / 2.0
        left = max(0.0, center - half)
        right = min(max_t, center + half)
        for ax in axes:
            ax.set_xlim(left, right)
        vline.set_xdata([center, center])
        fig.canvas.draw_idle()

    def set_center(val: float) -> None:
        state["updating"] = True
        s_center.set_val(val)
        time_box.set_val(f"{val:.1f}")
        state["updating"] = False

    def on_center_change(val: float) -> None:
        if not state["updating"]:
            time_box.set_val(f"{val:.1f}")
        update_view(val)

    def on_window_change(_val: float) -> None:
        update_view(s_center.val)

    def on_time_submit(text: str) -> None:
        if state["updating"]:
            return
        try:
            val = float(text)
        except ValueError:
            return
        val = max(0.0, min(max_t, val))
        set_center(val)
        update_view(val)

    s_center.on_changed(on_center_change)
    s_window.on_changed(on_window_change)
    time_box.on_submit(on_time_submit)

    timer = fig.canvas.new_timer(interval=play_interval_ms)

    def on_timer():
        next_val = s_center.val + play_step
        if next_val > max_t:
            next_val = 0.0
        set_center(next_val)
        update_view(next_val)

    timer.add_callback(on_timer)

    def on_toggle(_event):
        state["playing"] = not state["playing"]
        if state["playing"]:
            btn.label.set_text("Pause")
            timer.start()
        else:
            btn.label.set_text("Play")
            timer.stop()
        fig.canvas.draw_idle()

    btn.on_clicked(on_toggle)
    update_view(0.0)
    plt.show()
