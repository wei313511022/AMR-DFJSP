"""Folder-based test evaluation with schedule/route-map video export.

Point `evaluate_test_folder()` at a directory of user-provided .jsonl/.json
test scenarios; it runs a greedy episode per scenario, reports makespan and
the contract objective, writes a summary CSV, and (optionally) renders one
video per scenario that combines the AMR Gantt schedule (top, with a moving
time cursor) and the field route map (bottom, AMR positions/routes over time).

Scenario discovery rules (per file, records loaded via core.data_io.load_records):
  - any record has "dispatch_time"  -> the whole file is ONE stream scenario
    (dynamic dispatch batches, scripts/random_job_gen.py format)
  - otherwise                       -> each record is an independent scenario
    (FJSSP instances, scripts/Generate_training_data.py format)

Videos are saved as .mp4 when ffmpeg is available (matplotlib FFMpegWriter),
otherwise as .gif via PillowWriter.
"""
from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from core.data_io import load_records
from core.env import TaskSchedulingEnv
from core.features import flatten_jobs
from training.rollout import run_greedy_episode
from viz.viz_matplotlib import _station_sort_key, draw_amr_schedule, draw_machine_schedule
from viz.viz_route_map import draw_route_map

Scenario = Union[dict, List[dict]]


def discover_test_scenarios(test_data_dir: str) -> List[Tuple[str, Scenario]]:
    """Scan a folder for .jsonl/.json files and split them into named scenarios."""
    out: List[Tuple[str, Scenario]] = []
    root = Path(test_data_dir)
    if not root.is_dir():
        return out

    files = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in (".jsonl", ".json")
    )
    for path in files:
        records = load_records(str(path))
        if not records:
            print(f"[TEST-DATA] {path.name}: empty file, skipped")
            continue
        is_stream = any(isinstance(r, dict) and "dispatch_time" in r for r in records)
        if is_stream:
            out.append((path.stem, records))
        elif len(records) == 1:
            out.append((path.stem, records[0]))
        else:
            for i, rec in enumerate(records):
                out.append((f"{path.stem}_{i:03d}", rec))
    return out


def _make_video_writer(fps: int):
    """Prefer mp4: use ffmpeg from PATH, else the binary bundled with the
    optional `imageio-ffmpeg` package (pip install imageio-ffmpeg); fall back
    to animated GIF when neither is available."""
    if not animation.FFMpegWriter.isAvailable():
        try:
            import imageio_ffmpeg
            import matplotlib

            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if animation.FFMpegWriter.isAvailable():
        return animation.FFMpegWriter(fps=fps), ".mp4"
    return animation.PillowWriter(fps=fps), ".gif"


def record_schedule_video(
    env: TaskSchedulingEnv,
    trace: List[dict],
    makespan: float,
    out_path_base: str,
    title_info: str = "",
    fps: int = 10,
    max_frames: int = 300,
    dpi: int = 100,
    inventories: Optional[List[dict]] = None,
) -> Optional[str]:
    """Render the finished episode (env.trace) into one combined video:
    top = machine Gantt (classic FJSSP schedule), middle = AMR Gantt
    (transport/wait activity), bottom = field route map — all sharing a
    moving time cursor. Returns the written file path, or None."""
    if not trace:
        print(f"[VIDEO] {out_path_base}: empty trace, skipped")
        return None

    horizon = max(1.0, float(makespan))
    n_frames = int(min(max(2, max_frames), math.ceil(horizon) + 1))
    frame_times = np.linspace(0.0, horizon, n_frames)

    writer, ext = _make_video_writer(fps)
    out_path = out_path_base + ext

    station_keys = sorted(env.station_locs.keys(), key=_station_sort_key)

    fig = Figure(figsize=(12.5, 15.5))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.9], hspace=0.3)
    ax_machine = fig.add_subplot(gs[0])
    ax_amr = fig.add_subplot(gs[1])
    ax_route = fig.add_subplot(gs[2])
    status_text = fig.text(0.02, 0.005, "", fontsize=9, ha="left", va="bottom")

    t0 = time.perf_counter()
    with writer.saving(fig, out_path, dpi):
        for t in frame_times:
            ax_machine.clear()
            draw_machine_schedule(
                ax_machine,
                trace,
                makespan,
                station_keys=station_keys,
                show_labels=True,  # J{jid}(op/total) progress on each bar
                current_t=float(t),
            )
            ax_amr.clear()
            draw_amr_schedule(
                ax_amr,
                trace,
                makespan,
                show_labels=False,
                current_t=float(t),
                inventories=inventories,
            )
            # draw_route_map clears ax_route itself.
            lines = draw_route_map(ax_route, env, trace, current_t=float(t))
            status_text.set_text("\n".join(lines))
            fig.suptitle(
                f"{title_info} | t={float(t):.1f}s / {makespan:.1f}s", fontsize=13
            )
            writer.grab_frame()

    dt = time.perf_counter() - t0
    print(f"[VIDEO] saved {out_path} ({n_frames} frames, {dt:.1f}s)")
    return out_path


def evaluate_test_folder(
    env: TaskSchedulingEnv,
    policy_net: nn.Module,
    device: torch.device,
    test_data_dir: str,
    output_dir: str = os.path.join("results", "test_runs"),
    collision_avoidance: bool = True,
    max_scenarios: Optional[int] = None,
    save_videos: bool = True,
    video_max_scenarios: Optional[int] = None,
    video_fps: int = 10,
    video_max_frames: int = 300,
    video_dpi: int = 100,
) -> List[dict]:
    """Evaluate every scenario found in `test_data_dir` and export videos.

    Returns the per-scenario result rows (also written to summary.csv).
    Videos are rendered for the first `video_max_scenarios` scenarios
    (None = all evaluated scenarios).
    """
    scenarios = discover_test_scenarios(test_data_dir)
    if not scenarios:
        print(
            f"[TEST-DATA] no .jsonl/.json scenarios found in '{test_data_dir}' — "
            "nothing to evaluate."
        )
        return []
    if max_scenarios is not None:
        scenarios = scenarios[: max(0, int(max_scenarios))]

    os.makedirs(output_dir, exist_ok=True)
    mode = "collision-aware" if collision_avoidance else "fast estimate (no CA)"
    print(
        f"\n=== TEST DATASET ({len(scenarios)} scenarios from '{test_data_dir}', {mode}) ==="
    )

    prev_ca = env.enable_collision_avoidance
    env.enable_collision_avoidance = collision_avoidance
    w = float(getattr(env, "objective_load_balance_weight", 0.0))
    rows: List[dict] = []
    try:
        for i, (name, scenario) in enumerate(scenarios):
            job_count = len(flatten_jobs(scenario))
            t0 = time.perf_counter()
            mk = run_greedy_episode(env, policy_net, scenario, device)
            elapsed = time.perf_counter() - t0
            finish_sum = float(sum(env.robot_free_times))
            objective = mk + w * finish_sum
            rows.append(
                {
                    "scenario": name,
                    "jobs": job_count,
                    "makespan": round(mk, 3),
                    "finish_sum": round(finish_sum, 3),
                    "objective": round(objective, 3),
                    "eval_seconds": round(elapsed, 2),
                }
            )
            print(
                f"[{i + 1}/{len(scenarios)}] {name}: jobs={job_count} "
                f"makespan={mk:.2f}s objective={objective:.2f} ({elapsed:.1f}s)"
            )

            want_video = save_videos and (
                video_max_scenarios is None or i < int(video_max_scenarios)
            )
            if want_video:
                record_schedule_video(
                    env,
                    env.trace,
                    mk,
                    out_path_base=os.path.join(output_dir, name),
                    title_info=f"{name} | jobs={job_count}",
                    fps=video_fps,
                    max_frames=video_max_frames,
                    dpi=video_dpi,
                    inventories=env.robot_inventory,
                )
    finally:
        env.enable_collision_avoidance = prev_ca

    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mks = [r["makespan"] for r in rows]
    print(
        f"=== TEST SUMMARY: {len(rows)} scenarios | "
        f"makespan mean={np.mean(mks):.2f} min={np.min(mks):.2f} "
        f"max={np.max(mks):.2f} | results -> {summary_path} ==="
    )
    return rows
