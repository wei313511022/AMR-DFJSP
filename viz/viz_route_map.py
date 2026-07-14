import json
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from core.env import Coord, TaskSchedulingEnv
from viz.viz_matplotlib import format_trace_job_label


def _segment_map(item: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for seg in item.get("segments", []):
        kind = seg.get("kind")
        if kind:
            out[kind] = seg
    return out


def _path_prefix(path: List[Coord], elapsed: float) -> List[Tuple[float, float]]:
    if not path:
        return []
    if elapsed <= 0:
        x, y = path[0]
        return [(float(x), float(y))]

    max_step = len(path) - 1
    if elapsed >= max_step:
        return [(float(x), float(y)) for x, y in path]

    i = int(np.floor(elapsed))
    frac = float(elapsed - i)

    pts: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in path[: i + 1]]
    if i < len(path) - 1 and frac > 1e-9:
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        pts.append((x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac))
    return pts


def _machine_states_at_time(trace: List[dict], t: float) -> List[dict]:
    """Per-machine occupancy at time t under the single-slot buffer:
    phase "processing" while the operation runs, then phase "blocked"
    (done, still occupying the machine) until an AMR picks the part up."""
    from viz.viz_matplotlib import build_removal_times

    removal = build_removal_times(trace)
    states: List[dict] = []
    for item in trace:
        if item.get("is_delivery"):
            continue
        segs = _segment_map(item)
        seg_p = segs.get("process")
        if seg_p is None:
            continue
        s = float(seg_p.get("start", 0.0))
        e = float(seg_p.get("end", s))
        if removal:
            rem = removal.get((item.get("jid"), int(item.get("op_index", 0))), float("inf"))
        else:
            # Trace carries no pickup info at all (legacy / contract runs
            # without delivery semantics): show only the processing phase
            # instead of flagging every finished operation BLOCKED forever.
            rem = e
        if not (s - 1e-9 <= t < rem - 1e-9):
            continue
        phase = "processing" if t < e - 1e-9 else "blocked"
        states.append(
            {
                "station": str(item.get("dst", "")),
                "jid": item.get("jid"),
                "op_index": item.get("op_index"),
                "num_ops": item.get("num_ops"),
                "type": str(item.get("type", "")),
                "phase": phase,
                "proc_elapsed": min(max(0.0, t - s), max(0.0, e - s)),
                "proc_total": max(0.0, e - s),
                "proc_remaining": max(0.0, e - t) if phase == "processing" else 0.0,
                "blocked_for": max(0.0, t - e) if phase == "blocked" else 0.0,
            }
        )
    return states


def _robot_snapshot_at_time(
    env: TaskSchedulingEnv, trace: List[dict], t: float
) -> Tuple[List[dict], List[dict], float]:
    max_t = 0.0
    actions_by_robot: Dict[int, List[dict]] = {rid: [] for rid in range(env.num_robots)}

    for item in trace:
        rid = int(item.get("robot", -1))
        if rid < 0 or rid >= env.num_robots:
            continue
        segs = _segment_map(item)
        if "transport" not in segs or "process" not in segs:
            continue
        max_t = max(max_t, float(segs["process"].get("end", 0.0)))
        actions_by_robot[rid].append(
            {
                "jid": item.get("jid"),
                "type": str(item.get("type", "")),
                "op_index": item.get("op_index"),
                "num_ops": item.get("num_ops"),
                "is_delivery": bool(item.get("is_delivery", False)),
                "need_pickup": bool(item.get("need_pickup", False)),
                "pickup": tuple(item.get("pickup", ()) or ()),
                "replenish": int(item.get("replenish", 0)),
                "replenish_plan": dict(item.get("replenish_plan", {})),
                "dst": item.get("dst"),
                "path": [tuple(p) for p in item.get("transport_path", [])],
                "post_pos": tuple(item.get("post_pos", item.get("drop", (0, 0)))),
                "transport": segs.get("transport"),
                "wait": segs.get("wait"),
                "process": segs.get("process"),
            }
        )

    for rid in actions_by_robot:
        actions_by_robot[rid].sort(
            key=lambda a: float(a["transport"].get("start", 0.0)) if a.get("transport") else 0.0
        )

    snapshots: List[dict] = []
    for rid in range(env.num_robots):
        pos_x, pos_y = env.initial_robot_positions[rid]
        pos = (float(pos_x), float(pos_y))
        inv = {k: 0 for k in env.material_types}
        status = "idle"
        mode = "idle"
        jid = None
        dst = None
        current_route: List[Tuple[float, float]] = []
        proc_elapsed = 0.0
        proc_total = 0.0
        proc_remaining = 0.0
        carrying = False
        wait_kind: Optional[str] = None
        carry_label = ""
        carry_type = ""
        is_delivery_leg = False
        added_total = {k: 0 for k in env.material_types}
        consumed_total = {k: 0 for k in env.material_types}
        last_inventory_event: Optional[dict] = None

        for action in actions_by_robot[rid]:
            seg_t = action["transport"]
            seg_w = action.get("wait")
            seg_p = action.get("process")
            if seg_t is None:
                continue

            t0 = float(seg_t.get("start", 0.0))
            t1 = float(seg_t.get("end", t0))
            tw = float(seg_w.get("end", t1)) if seg_w else t1
            tp = float(seg_p.get("end", tw)) if seg_p else tw
            path = action.get("path", [])
            if not path:
                path = [(int(round(pos[0])), int(round(pos[1])))]

            if t >= t0:
                jtype = action["type"]
                plan_raw = action.get("replenish_plan", {})
                if isinstance(plan_raw, dict) and len(plan_raw) > 0:
                    add_plan = {
                        k: max(0, int(plan_raw.get(k, 0))) for k in env.material_types
                    }
                else:
                    # Backward-compatible fallback for old traces.
                    add = max(0, int(action.get("replenish", 0)))
                    add_plan = {k: 0 for k in env.material_types}
                    if jtype in add_plan:
                        add_plan[jtype] = add
                add_total = int(sum(add_plan.values()))
                before_inv = {k: int(inv.get(k, 0)) for k in env.material_types}
                after_replenish = dict(before_inv)
                for k in env.material_types:
                    add_k = int(add_plan.get(k, 0))
                    if add_k > 0:
                        after_replenish[k] = min(
                            env.capacity_per_type, after_replenish.get(k, 0) + add_k
                        )
                after_consume = dict(after_replenish)
                after_consume[jtype] = max(0, after_consume.get(jtype, 0) - 1)

                inv = {k: int(after_consume.get(k, 0)) for k in env.material_types}
                for k in env.material_types:
                    added_total[k] = int(added_total.get(k, 0)) + int(add_plan.get(k, 0))
                consumed_total[jtype] = int(consumed_total.get(jtype, 0)) + 1
                last_inventory_event = {
                    "event": "dispatch_start_bookkeeping",
                    "rule": "+replenish then -1 for dispatched job at transport start",
                    "t_start": float(t0),
                    "jid": action.get("jid"),
                    "jtype": jtype,
                    "replenish_add": add_total,
                    "replenish_plan": dict(add_plan),
                    "consume": 1,
                    "before": before_inv,
                    "after_replenish": after_replenish,
                    "after_consume": dict(inv),
                }

            if t < t0:
                break

            if t < t1:
                prefix = _path_prefix(path, t - t0)
                current_route = prefix
                if prefix:
                    pos = prefix[-1]
                elapsed = max(0.0, t - t0)
                step_idx = int(np.floor(elapsed + 1e-9))
                is_wait_step = False
                if len(path) >= 2 and step_idx < len(path) - 1:
                    c0 = path[step_idx]
                    c1 = path[step_idx + 1]
                    is_wait_step = (int(c0[0]) == int(c1[0])) and (int(c0[1]) == int(c1[1]))
                status = "wait" if is_wait_step else "move"
                mode = "supply" if int(action["replenish"]) > 0 else "deliver"
                jid = action["jid"]
                dst = action["dst"]

                # Cargo / wait-kind classification: where is the pickup point
                # along the path, and has the loading run finished? Before
                # that the AMR travels empty (or queues/loads at the dock);
                # after it the AMR carries the part, and any hold step means
                # it is holding the part at the machine (buffer full).
                need_pickup = bool(action.get("need_pickup"))
                pickup_cell = tuple(action.get("pickup", ()) or ())
                pickup_idx = None
                if need_pickup and len(pickup_cell) == 2:
                    for pi, c in enumerate(path):
                        if tuple(c) == pickup_cell:
                            pickup_idx = pi
                            break
                service_end = 0
                if pickup_idx is not None:
                    service_end = pickup_idx
                    while (
                        service_end < len(path) - 1
                        and tuple(path[service_end]) == pickup_cell
                        and tuple(path[service_end + 1]) == pickup_cell
                    ):
                        service_end += 1
                carrying = (not need_pickup) or (
                    pickup_idx is not None and step_idx >= service_end
                )
                if is_wait_step:
                    on_pickup_side = need_pickup and (
                        pickup_idx is None or step_idx < service_end
                    )
                    wait_kind = "dock" if on_pickup_side else "buffer"
                else:
                    wait_kind = None
                carry_label = format_trace_job_label(action)
                carry_type = str(action.get("type", ""))
                is_delivery_leg = bool(action.get("is_delivery"))
                break

            pos = (float(path[-1][0]), float(path[-1][1]))

            # Drop-off semantics: the AMR is free the moment it arrives (t1);
            # the part queues and is processed at the machine on its own
            # (see _machine_states_at_time). The robot idles at post_pos.
            post_pos = tuple(action.get("post_pos", path[-1]))
            pos = (float(post_pos[0]), float(post_pos[1]))

        snapshots.append(
            {
                "rid": rid,
                "pos": pos,
                "inv": inv,
                "inventory_net": dict(inv),
                "inventory_semantics": {
                    "definition": (
                        "Net onboard inventory after dispatch bookkeeping, not physical loading progress."
                    ),
                    "rule": "+replenish then -1 at transport start time",
                    "cumulative_added": {k: int(added_total.get(k, 0)) for k in env.material_types},
                    "cumulative_consumed": {
                        k: int(consumed_total.get(k, 0)) for k in env.material_types
                    },
                    "last_event": last_inventory_event,
                },
                "status": status,
                "mode": mode,
                "jid": jid,
                "dst": dst,
                "route": current_route,
                "proc_elapsed": proc_elapsed,
                "proc_total": proc_total,
                "proc_remaining": proc_remaining,
                # Cargo state: is the AMR holding a part right now, and if it
                # is waiting — at the dock (queue/loading) or at the machine
                # (buffer full, waiting for the occupant to finish/be taken).
                "carrying": carrying,
                "carry_label": carry_label,
                "carry_type": carry_type,
                "wait_kind": wait_kind,
                "is_delivery": is_delivery_leg,
            }
        )

    return snapshots, _machine_states_at_time(trace, t), max_t


def _draw_base_map(ax, env: TaskSchedulingEnv) -> None:
    ax.set_xlim(-0.5, env.W - 0.5)
    ax.set_ylim(-0.5, env.H - 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(env.W))
    ax.set_yticks(range(env.H))
    ax.grid(True, color="#A0A0A0", linewidth=0.6, alpha=0.5)

    for ox, oy in env.obstacles:
        rect = patches.Rectangle(
            (ox - 0.5, oy - 0.5),
            1.0,
            1.0,
            facecolor="#BDBDBD",
            edgecolor="none",
            alpha=0.8,
            zorder=1,
        )
        ax.add_patch(rect)

    dock_locs = getattr(env, "dock_locs", None) or {
        f"M{k}": v for k, v in env.source_locs.items()
    }
    for dkey, (sx, sy) in dock_locs.items():
        rect = patches.Rectangle(
            (sx - 0.45, sy - 0.45),
            0.9,
            0.9,
            fill=False,
            edgecolor="#1f77b4",
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(sx, sy, dkey, ha="center", va="center", fontsize=12, color="#1f77b4", weight="bold")

    for sname, (sx, sy) in env.station_locs.items():
        rect = patches.Rectangle(
            (sx - 0.45, sy - 0.45),
            0.9,
            0.9,
            fill=False,
            edgecolor="#d62728",
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(sx, sy, sname, ha="center", va="center", fontsize=12, color="#d62728", weight="bold")

    for tname, (sx, sy) in getattr(env, "output_locs", {}).items():
        rect = patches.Rectangle(
            (sx - 0.45, sy - 0.45),
            0.9,
            0.9,
            fill=False,
            edgecolor="#ff7f0e",
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(sx, sy, tname, ha="center", va="center", fontsize=12, color="#ff7f0e", weight="bold")


def _normalize_snapshot_for_draw(snap: dict) -> dict:
    rid = int(snap.get("rid", -1))

    if "pos" in snap and isinstance(snap.get("pos"), (list, tuple)) and len(snap.get("pos")) >= 2:
        pos = (float(snap["pos"][0]), float(snap["pos"][1]))
    else:
        pos = (float(snap.get("x", 0.0)), float(snap.get("y", 0.0)))

    inv_raw = snap.get("inv", snap.get("inventory_net", snap.get("inventory", {})))
    inv = {
        "A": int(inv_raw.get("A", 0)) if isinstance(inv_raw, dict) else 0,
        "B": int(inv_raw.get("B", 0)) if isinstance(inv_raw, dict) else 0,
        "C": int(inv_raw.get("C", 0)) if isinstance(inv_raw, dict) else 0,
    }

    route_raw = snap.get("route", []) or []
    route: List[Tuple[float, float]] = []
    for p in route_raw:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            route.append((float(p[0]), float(p[1])))

    jid_raw = snap.get("jid", None)
    jid = None if jid_raw is None else int(jid_raw)

    return {
        "rid": rid,
        "pos": pos,
        "inv": inv,
        "status": str(snap.get("status", "idle")),
        "mode": str(snap.get("mode", "idle")),
        "jid": jid,
        "dst": snap.get("dst", None),
        "route": route,
        "proc_elapsed": float(snap.get("proc_elapsed", 0.0)),
        "proc_total": float(snap.get("proc_total", 0.0)),
        "proc_remaining": float(snap.get("proc_remaining", 0.0)),
        "carrying": bool(snap.get("carrying", False)),
        "carry_label": str(snap.get("carry_label", "") or ""),
        "carry_type": str(snap.get("carry_type", "") or ""),
        "wait_kind": snap.get("wait_kind", None),
        "is_delivery": bool(snap.get("is_delivery", False)),
    }


def _draw_route_map_from_snapshots(
    ax,
    env: TaskSchedulingEnv,
    snapshots: List[dict],
    current_t: float,
    max_t: float,
    machine_states: Optional[List[dict]] = None,
) -> List[str]:
    ax.clear()
    _draw_base_map(ax, env)
    ax.set_title(f"Route Map | t={current_t:.1f}s / {max_t:.1f}s")

    # Strong, fixed color identity per AMR.
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]
    status_lines: List[str] = []

    # Machine occupancy (single-slot buffer): colored highlight while
    # processing; gray highlight when done but still awaiting pickup.
    # A station can carry two tags at once after a deadlock override
    # (old part awaiting pickup + forced new part) — stack them vertically.
    mat_colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    tags_per_station: Dict[str, int] = {}
    # station -> occupant info (labelled) for the AMR status lines below;
    # when two states coexist (deadlock-override anomaly) keep the blocked
    # one — that pickup is what a holding AMR is actually waiting for.
    machine_by_station: Dict[str, dict] = {}
    for m in machine_states or []:
        st = str(m.get("station", ""))
        _jid = m.get("jid")
        _op = m.get("op_index")
        _nops = m.get("num_ops")
        if _op is not None and _nops:
            _lbl = f"J{_jid}({int(_op) + 1}/{int(_nops)})"
        else:
            _lbl = f"J{_jid}"
        if st not in machine_by_station or str(m.get("phase")) == "blocked":
            machine_by_station[st] = {**m, "label": _lbl}
        if st not in env.station_locs:
            continue
        sx, sy = env.station_locs[st]
        phase = str(m.get("phase", "processing"))
        mcolor = mat_colors.get(str(m.get("type", "")).upper(), "#7f7f7f")
        hl_color = mcolor if phase == "processing" else "#888888"
        hl = patches.Rectangle(
            (sx - 0.45, sy - 0.45),
            0.9,
            0.9,
            facecolor=hl_color,
            edgecolor=hl_color,
            alpha=0.25,
            linewidth=1.2,
            zorder=2.5,
        )
        ax.add_patch(hl)
        jid = m.get("jid")
        op = m.get("op_index")
        nops = m.get("num_ops")
        if op is not None and nops:
            job_txt = f"J{jid}({int(op) + 1}/{int(nops)})"
        else:
            job_txt = f"J{jid}" + (f".{op}" if op is not None else "")
        if phase == "processing":
            left = float(m.get("proc_remaining", 0.0))
            tag = f"{job_txt} left {left:.1f}s"
            status_lines.append(
                f"{st}: processing {job_txt} ({m.get('type','')}), "
                f"{float(m.get('proc_elapsed', 0.0)):.1f}/{float(m.get('proc_total', 0.0)):.1f}s "
                f"(left {left:.1f}s)"
            )
        else:
            blocked = float(m.get("blocked_for", 0.0))
            tag = f"{job_txt} await pickup"
            status_lines.append(
                f"{st}: BLOCKED — {job_txt} ({m.get('type','')}) done, "
                f"awaiting pickup for {blocked:.1f}s"
            )
        slot = tags_per_station.get(st, 0)
        tags_per_station[st] = slot + 1
        ax.text(
            sx,
            sy - 0.62 - 0.42 * slot,
            tag,
            ha="center",
            va="top",
            fontsize=8,
            color=hl_color,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            zorder=6,
        )

    norm_snaps = [_normalize_snapshot_for_draw(s) for s in snapshots]
    norm_snaps.sort(key=lambda s: int(s.get("rid", -1)))

    for snap in norm_snaps:
        rid = int(snap.get("rid", -1))
        if rid < 0:
            continue
        color = colors[rid % len(colors)]
        route = snap.get("route", [])
        if len(route) >= 2:
            xs = [p[0] for p in route]
            ys = [p[1] for p in route]
            line_style = "--" if snap.get("mode") == "supply" else "-"
            ax.plot(xs, ys, linestyle=line_style, linewidth=2.2, color=color, alpha=0.9, zorder=3)

        x, y = snap["pos"]
        # Filled circle = the AMR is carrying a part (tinted by material);
        # hollow circle = travelling empty / idle.
        carry_tints = {"A": "#aec7e8", "B": "#ffce9e", "C": "#b5e3b5"}
        face = (
            carry_tints.get(snap.get("carry_type", "").upper(), "#e0e0e0")
            if snap.get("carrying")
            else "white"
        )
        circ = patches.Circle((x, y), 0.32, facecolor=face, edgecolor=color, linewidth=2.4, zorder=5)
        ax.add_patch(circ)
        ax.text(
            x,
            y + 0.5,
            f"AMR{rid+1}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
            zorder=6,
        )

        if snap.get("status") == "process":
            dst = str(snap.get("dst", ""))
            if dst in env.station_locs:
                sx, sy = env.station_locs[dst]
                hl = patches.Rectangle(
                    (sx - 0.45, sy - 0.45),
                    0.9,
                    0.9,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.18,
                    linewidth=1.2,
                    zorder=2.5,
                )
                ax.add_patch(hl)
                left = float(snap.get("proc_remaining", 0.0))
                ax.text(
                    sx,
                    sy - 0.62,
                    f"left {left:.1f}s",
                    ha="center",
                    va="top",
                    fontsize=9,
                    color=color,
                    weight="bold",
                )

        inv = snap["inv"]
        inv_txt = f"A{inv.get('A',0)} B{inv.get('B',0)} C{inv.get('C',0)}"
        pos_txt = f"(x:{x:.1f} y:{y:.1f})"
        dst = str(snap.get("dst", "") or "")
        label = snap.get("carry_label") or (
            f"J{snap['jid']}" if snap.get("jid") is not None else ""
        )
        status = snap.get("status")
        wait_kind = snap.get("wait_kind")

        # Spell the AMR's situation out: empty vs carrying, and — when
        # waiting — whether it queues at the dock or holds the part at a
        # machine whose occupant is still processing / awaiting pickup.
        if status == "wait" and wait_kind == "buffer":
            occ = machine_by_station.get(dst)
            if occ is not None:
                if str(occ.get("phase")) == "processing":
                    reason = (
                        f"machine busy: {occ['label']} processing, "
                        f"left {float(occ.get('proc_remaining', 0.0)):.1f}s"
                    )
                else:
                    reason = f"waiting pickup of {occ['label']}"
            else:
                reason = "buffer full"
            act = f"HOLDING {label} — {reason}"
        elif status == "wait" and wait_kind == "dock":
            act = f"queue/loading @ dock for {label}"
        elif status == "wait":
            act = f"waiting ({label})"
        elif status == "move":
            if snap.get("carrying"):
                act = f"carrying {label}"
            else:
                act = f"empty, to pick {label}"
        elif status == "process":
            pe = float(snap.get("proc_elapsed", 0.0))
            pt = float(snap.get("proc_total", 0.0))
            pr = float(snap.get("proc_remaining", 0.0))
            act = f"process {label}, {pe:.1f}/{pt:.1f}s (left {pr:.1f}s)"
        else:
            act = "idle"
        status_lines.append(f"AMR{rid+1}: {act}, {inv_txt}, {pos_txt}")

    return status_lines


def draw_route_map(ax, env: TaskSchedulingEnv, trace: List[dict], current_t: float) -> List[str]:
    snapshots, machine_states, max_t = _robot_snapshot_at_time(env, trace, current_t)
    return _draw_route_map_from_snapshots(
        ax, env, snapshots, current_t, max_t, machine_states=machine_states
    )


def _load_route_jsonl_frames(route_jsonl_path: str) -> List[dict]:
    frames: List[dict] = []
    with open(route_jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            txt = line.strip()
            if not txt:
                continue
            try:
                rec = json.loads(txt)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid route jsonl at line {ln} in '{route_jsonl_path}': {e.msg}"
                ) from e

            t = float(rec.get("t", len(frames)))
            amrs = rec.get("amrs", [])
            if not isinstance(amrs, list):
                amrs = []
            frames.append({"t": t, "amrs": amrs})

    frames.sort(key=lambda r: float(r.get("t", 0.0)))
    return frames


def show_route_map_replay(
    env: TaskSchedulingEnv,
    trace: List[dict],
    initial_t: float = 0.0,
    play_step: float = 0.5,
    play_interval_ms: int = 120,
) -> None:
    if not trace:
        print("No trace to render on route map.")
        return
    backend = str(plt.get_backend()).lower()
    if "inline" in backend:
        print(
            "Route replay controls may be non-interactive on inline backend. "
            "Use `%matplotlib qt` or `%matplotlib widget`."
        )

    _, _, max_t = _robot_snapshot_at_time(env, trace, initial_t)
    max_t = max(1.0, max_t)
    initial_t = max(0.0, min(max_t, initial_t))

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.subplots_adjust(bottom=0.2, top=0.92)
    status_text = fig.text(0.02, 0.03, "", fontsize=11, ha="left", va="bottom")

    ax_slider = fig.add_axes([0.16, 0.11, 0.56, 0.03])
    ax_button = fig.add_axes([0.76, 0.102, 0.15, 0.05])
    slider = Slider(ax_slider, "t", 0.0, max_t, valinit=initial_t, valstep=0.1)
    btn = Button(ax_button, "Play")

    state = {"playing": False, "updating": False}

    def redraw(t: float) -> None:
        lines = draw_route_map(ax, env, trace, t)
        status_text.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    def set_time(t: float) -> None:
        state["updating"] = True
        slider.set_val(t)
        state["updating"] = False

    def on_slider(val: float) -> None:
        if state["updating"]:
            return
        redraw(float(val))

    slider.on_changed(on_slider)
    timer = fig.canvas.new_timer(interval=play_interval_ms)

    def on_timer():
        nxt = float(slider.val) + play_step
        if nxt > max_t:
            nxt = 0.0
        set_time(nxt)
        redraw(nxt)

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
    redraw(initial_t)
    plt.show()


def show_route_map_replay_from_jsonl(
    env: TaskSchedulingEnv,
    route_jsonl_path: str,
    initial_t: float = 0.0,
    play_interval_ms: int = 120,
) -> None:
    backend = str(plt.get_backend()).lower()
    if "inline" in backend:
        print(
            "Route replay controls may be non-interactive on inline backend. "
            "Use `%matplotlib qt` or `%matplotlib widget`."
        )
    frames = _load_route_jsonl_frames(route_jsonl_path)
    if not frames:
        print(f"No frames in route jsonl: {route_jsonl_path}")
        return

    times = [float(f.get("t", 0.0)) for f in frames]
    max_t = max(1.0, max(times))
    idx0 = int(np.argmin(np.abs(np.asarray(times, dtype=np.float64) - float(initial_t))))

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.subplots_adjust(bottom=0.2, top=0.92)
    status_text = fig.text(0.02, 0.03, "", fontsize=11, ha="left", va="bottom")

    ax_slider = fig.add_axes([0.16, 0.11, 0.56, 0.03])
    ax_button = fig.add_axes([0.76, 0.102, 0.15, 0.05])
    slider = Slider(ax_slider, "frame", 0, len(frames) - 1, valinit=idx0, valstep=1)
    btn = Button(ax_button, "Play")

    state = {"playing": False, "updating": False}

    def redraw_by_idx(i: int) -> None:
        idx = int(max(0, min(len(frames) - 1, i)))
        frame = frames[idx]
        t = float(frame.get("t", 0.0))
        snapshots = frame.get("amrs", [])
        lines = _draw_route_map_from_snapshots(ax, env, snapshots, t, max_t)
        ax.set_title(f"Route Map | t={t:.1f}s / {max_t:.1f}s | frame={idx+1}/{len(frames)}")
        status_text.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    def set_idx(i: int) -> None:
        state["updating"] = True
        slider.set_val(int(max(0, min(len(frames) - 1, i))))
        state["updating"] = False

    def on_slider(val: float) -> None:
        if state["updating"]:
            return
        redraw_by_idx(int(round(float(val))))

    slider.on_changed(on_slider)
    timer = fig.canvas.new_timer(interval=play_interval_ms)

    def on_timer():
        nxt = int(round(float(slider.val))) + 1
        if nxt > len(frames) - 1:
            nxt = 0
        set_idx(nxt)
        redraw_by_idx(nxt)

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
    redraw_by_idx(idx0)
    plt.show()
