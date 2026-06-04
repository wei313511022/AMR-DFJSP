#!/usr/bin/env python3
"""
Headless hybrid-trigger timing policy for dynamic AMR rescheduling.

This mirrors the Attention_DDQN_V7 environment, but there is no DDQN and no GUI.
At each decision point it chooses:

    action = RESCHEDULE

only when all AMRs are busy, unstarted work exists, the normal cooldown/event
gate permits rescheduling, and the estimated scheduling compute time fits inside
the current all-AMR busy window.
"""

import argparse
import csv
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from Attention_DDQN_V7 import CONFIG, GridEnv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "test_case" / "dynamic" / "test_dataset_demo.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "hybrid_trigger_results.csv"
DEFAULT_SUMMARY = Path(__file__).resolve().parent / "hybrid_trigger_summary.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure(args: argparse.Namespace) -> None:
    CONFIG["DEVICE"] = args.device
    CONFIG["DATASET_PATH"] = str(Path(args.dataset).resolve())
    CONFIG["SIM_TIME"] = float(args.sim_time)
    CONFIG["RESCHED_COOLDOWN"] = float(args.cooldown)
    CONFIG["COMPUTE_TIME_SCALING"] = float(args.compute_time_scaling)
    CONFIG["GA_ROUTING_ITERS"] = int(args.routing_iters)
    CONFIG["GA_COLLISION_ITERS"] = int(args.collision_iters)


def job_station_pos(job):
    from GA.GA import STATIONS

    if hasattr(job, "station") and job.station in STATIONS:
        return STATIONS[job.station]
    if hasattr(job, "dest_pos"):
        return tuple(int(v) for v in job.dest_pos)
    return None


def amr_busy_window(env: GridEnv, amr: str) -> float:
    """
    Estimate seconds until this AMR can reasonably need a new queue decision.

    This is intentionally conservative and cheap. It is only used for trigger
    timing, while the environment still performs the actual simulation.
    """
    from GA.GA import SUPPLY_LOCATIONS, TYPE_DURATION, heuristic

    sim = env.sim
    state = sim.amr_states[amr]
    mode = state.get("mode")
    pos = sim.positions[amr]
    job = state.get("job")

    if mode == "idle":
        return 0.0

    if mode in {"processing", "processing_old"}:
        return float(max(state.get("proc_ticks", 0), 0))

    if mode == "loading_dock":
        remaining = float(max(state.get("proc_ticks", 0), 0))
        if job is not None:
            dest = job_station_pos(job)
            if dest is not None:
                remaining += heuristic(pos, dest) + float(job.duration)
        return remaining

    goal = state.get("goal")
    travel_remaining = float(heuristic(pos, goal)) if goal is not None else 0.0

    if mode == "moving_supply" and job is not None:
        supply_time = TYPE_DURATION.get(job.type_, 0)
        dest = job_station_pos(job)
        station_leg = heuristic(goal, dest) if goal is not None and dest is not None else 0.0
        return travel_remaining + supply_time + station_leg + float(job.duration)

    if mode == "moving_station" and job is not None:
        return travel_remaining + float(job.duration)

    if mode == "moving_base":
        return travel_remaining

    return travel_remaining


def all_amrs_busy_window(env: GridEnv) -> float:
    from GA.GA import AMR_KEYS

    windows = [amr_busy_window(env, amr) for amr in AMR_KEYS]
    if not windows or any(window <= 0.0 for window in windows):
        return 0.0
    return min(windows)


def has_unstarted_work(env: GridEnv) -> bool:
    return any(job.status == 1 for job in env.active_jobs)


def compute_required_window(env: GridEnv, args: argparse.Namespace) -> float:
    observed = float(getattr(env, "last_ga_compute_time", 0.0) or 0.0)
    expected = max(float(args.expected_compute_time), observed)
    return max(expected + float(args.safety_margin), float(args.min_busy_window))


def choose_hybrid_action(env: GridEnv, args: argparse.Namespace) -> Tuple[int, str, float, float]:
    """
    Returns action, reason, busy_window, required_window.
    """
    # Make arrivals visible before deciding; env.step() will call this again,
    # but no extra jobs will be released for the same sim time.
    env._release_arrived_jobs()

    busy_window = all_amrs_busy_window(env)
    required_window = compute_required_window(env, args)

    if not has_unstarted_work(env):
        return 0, "no_unstarted_work", busy_window, required_window
    if not env.can_reschedule():
        return 0, "reschedule_gate_closed", busy_window, required_window
    if busy_window <= 0.0:
        return 0, "not_all_amrs_busy", busy_window, required_window
    if busy_window < required_window:
        return 0, "busy_window_too_short", busy_window, required_window

    return 1, "hybrid_trigger", busy_window, required_window


def run_episode(env: GridEnv, episode_idx: int, args: argparse.Namespace) -> Dict[str, float]:
    state = env.reset()
    _ = state
    total_reward = 0.0
    steps = 0
    wait_actions = 0
    reschedule_actions = 0
    reschedule_compute_times: List[float] = []
    trigger_reasons: Dict[str, int] = {}
    episode_wall_start = time.perf_counter()

    while True:
        steps += 1
        action, reason, busy_window, required_window = choose_hybrid_action(env, args)
        trigger_reasons[reason] = trigger_reasons.get(reason, 0) + 1

        if action == 1:
            reschedule_actions += 1
        else:
            wait_actions += 1

        _, reward, done, dt = env.step(action)
        total_reward += float(reward)

        if action == 1 and env.last_ga_compute_time > 0:
            reschedule_compute_times.append(float(env.last_ga_compute_time))

        if args.verbose:
            action_name = "RESCHEDULE" if action == 1 else "WAIT"
            print(
                f"Ep {episode_idx} step {steps:04d} t={env.sim_time:7.2f} "
                f"{action_name:10s} reason={reason:22s} "
                f"busy={busy_window:6.2f} required={required_window:6.2f} "
                f"dt={dt:.1f} completed={len(env.completed_jobs)}/{env.total_jobs}"
            )

        if done:
            break

    wall_time = time.perf_counter() - episode_wall_start
    completed = len(env.completed_jobs)
    total_jobs = max(env.total_jobs, 1)
    mean_compute = statistics.fmean(reschedule_compute_times) if reschedule_compute_times else 0.0
    total_compute = sum(reschedule_compute_times)

    return {
        "episode": episode_idx,
        "sim_time": float(env.sim_time),
        "wall_time": wall_time,
        "completed_jobs": completed,
        "total_jobs": env.total_jobs,
        "completion_ratio": completed / total_jobs,
        "total_reward": total_reward,
        "steps": steps,
        "wait_actions": wait_actions,
        "reschedules": reschedule_actions,
        "mean_reschedule_compute_time": mean_compute,
        "total_reschedule_compute_time": total_compute,
        "last_reschedule_compute_time": float(getattr(env, "last_ga_compute_time", 0.0) or 0.0),
        "trigger_reasons": ";".join(f"{key}:{value}" for key, value in sorted(trigger_reasons.items())),
    }


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "episode",
        "sim_time",
        "wall_time",
        "completed_jobs",
        "total_jobs",
        "completion_ratio",
        "total_reward",
        "steps",
        "wait_actions",
        "reschedules",
        "mean_reschedule_compute_time",
        "total_reschedule_compute_time",
        "last_reschedule_compute_time",
        "trigger_reasons",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    numeric_fields = [
        "sim_time",
        "wall_time",
        "completed_jobs",
        "completion_ratio",
        "total_reward",
        "steps",
        "reschedules",
        "mean_reschedule_compute_time",
        "total_reschedule_compute_time",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        for field in numeric_fields:
            values = [float(row[field]) for row in rows]
            mean = statistics.fmean(values) if values else 0.0
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            writer.writerow([field, f"{mean:.6f}", f"{std:.6f}"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run headless hybrid-trigger rescheduling.")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--sim_time", type=float, default=CONFIG["SIM_TIME"])
    parser.add_argument("--cooldown", type=float, default=CONFIG["RESCHED_COOLDOWN"])
    parser.add_argument("--expected_compute_time", type=float, default=0.25)
    parser.add_argument("--min_busy_window", type=float, default=1.0)
    parser.add_argument("--safety_margin", type=float, default=0.25)
    parser.add_argument("--routing_iters", type=int, default=0)
    parser.add_argument("--collision_iters", type=int, default=0)
    parser.add_argument("--compute_time_scaling", type=float, default=CONFIG["COMPUTE_TIME_SCALING"])
    parser.add_argument("--device", type=str, default=CONFIG["DEVICE"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary_csv", type=str, default=str(DEFAULT_SUMMARY))
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    configure(args)

    env = GridEnv()
    available_episodes = len(env.episodes)
    episode_count = min(int(args.episodes), available_episodes)
    if episode_count <= 0:
        raise RuntimeError(f"No episodes found in dataset: {CONFIG['DATASET_PATH']}")

    print("=== Headless Hybrid Trigger Timing ===")
    print(f"Dataset: {CONFIG['DATASET_PATH']}")
    print(f"Episodes: {episode_count}")
    print(f"Device: {CONFIG['DEVICE']}")
    print(f"Routing iters: {CONFIG['GA_ROUTING_ITERS']} | Collision iters: {CONFIG['GA_COLLISION_ITERS']}")

    rows = []
    for episode_idx in range(episode_count):
        row = run_episode(env, episode_idx, args)
        rows.append(row)
        print(
            f"Ep {episode_idx:03d} | sim_time={row['sim_time']:.2f} "
            f"| completed={row['completed_jobs']}/{row['total_jobs']} "
            f"| reschedules={row['reschedules']} "
            f"| mean_compute={row['mean_reschedule_compute_time']:.4f}s "
            f"| reward={row['total_reward']:.2f}"
        )

    output_path = Path(args.output_csv)
    summary_path = Path(args.summary_csv)
    write_rows(output_path, rows)
    write_summary(summary_path, rows)
    print(f"\nDetailed results saved to {output_path}")
    print(f"Summary results saved to {summary_path}")


if __name__ == "__main__":
    main()
