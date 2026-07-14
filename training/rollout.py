import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from core.data_io import poll_live_job_file
from core.env import TaskSchedulingEnv
from core.features import (
    env_action_candidates,
    env_selection_bias,
    q_values_batch,
    select_action_index,
)
from viz.viz_matplotlib import draw_amr_schedule, draw_dispatch_queue, draw_input_queue


def _greedy_action(
    env: TaskSchedulingEnv, policy_net: nn.Module, state: np.ndarray, device: torch.device
) -> Tuple[int, Union[int, Dict[str, int], None]]:
    """One greedy decision: candidates + features + Q + biased argmax."""
    rid = env.current_robot
    actions, feats = env_action_candidates(env, rid)
    with torch.no_grad():
        q_all = q_values_batch(policy_net, state, feats, device)
        best_idx = select_action_index(
            q_values=q_all,
            actions=actions,
            tasks=env.available_tasks,
            inventory=env.robot_inventory[rid],
            capacity_per_type=env.capacity_per_type,
            action_feats=feats,
            **env_selection_bias(env),
        )
    return actions[best_idx]


def run_greedy_episode(
    env: TaskSchedulingEnv,
    policy_net: nn.Module,
    scenario: Union[dict, List[dict]],
    device: torch.device,
) -> float:
    prev_mode = policy_net.training
    policy_net.eval()
    try:
        _ = env.reset(scenario)
        s = np.array(env._get_state(), dtype=np.float32)
        done = False
        while not done:
            action = _greedy_action(env, policy_net, s, device)
            sp, _, done = env.step(action)
            s = np.array(sp, dtype=np.float32) if sp is not None else None

        return env.makespan()
    finally:
        policy_net.train(prev_mode)


def run_greedy_episode_live(
    env: TaskSchedulingEnv,
    policy_net: nn.Module,
    scenario: Union[dict, List[dict]],
    device: torch.device,
    pause: float = 0.3,
    record_dir: Optional[str] = None,
    record_every: int = 1,
    record_dpi: int = 120,
    make_gif: bool = False,
    gif_path: str = "live_schedule.gif",
) -> float:
    prev_mode = policy_net.training
    policy_net.eval()
    try:
        _ = env.reset(scenario)
        s = np.array(env._get_state(), dtype=np.float32)
        input_source = scenario
        done = False
        step = 0

        frame_paths: List[str] = []
        if make_gif and record_dir is None:
            record_dir = "live_frames"

        gif_enabled = False
        image_cls = None
        if make_gif:
            try:
                from PIL import Image as _Image

                image_cls = _Image
                gif_enabled = True
            except ImportError:
                print("PIL not available; GIF recording disabled.")
                make_gif = False

        if record_dir:
            os.makedirs(record_dir, exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        fig.subplots_adjust(hspace=0.5, top=0.9)

        while not done:
            action = _greedy_action(env, policy_net, s, device)
            sp, _, done = env.step(action)
            s = np.array(sp, dtype=np.float32) if sp is not None else None
            step += 1

            for ax in axes:
                ax.clear()

            draw_dispatch_queue(axes[0], env.trace, current_t=env.t)
            draw_amr_schedule(
                axes[1],
                env.trace,
                env.makespan(),
                current_t=env.t,
                inventories=env.robot_inventory,
            )
            draw_input_queue(axes[2], input_source, current_t=env.t)

            fig.suptitle(f"t={env.t:.1f}s | step={step} | available={len(env.available_tasks)}")
            fig.canvas.draw()
            fig.canvas.flush_events()

            if record_dir and step % max(1, record_every) == 0:
                frame_path = os.path.join(record_dir, f"frame_{step:05d}.png")
                fig.savefig(frame_path, dpi=record_dpi)
                frame_paths.append(frame_path)

            plt.pause(pause)

        if make_gif and gif_enabled and frame_paths and image_cls is not None:
            images = [image_cls.open(p) for p in frame_paths]
            if images:
                duration = max(1, int(pause * 1000))
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=duration,
                    loop=0,
                )
                for img in images:
                    img.close()
                print(f"Saved GIF to {gif_path}")

        return env.makespan()
    finally:
        policy_net.train(prev_mode)


def run_greedy_episode_live_stream(
    env: TaskSchedulingEnv,
    policy_net: nn.Module,
    scenario: Union[dict, List[dict]],
    device: torch.device,
    live_job_file: Optional[str] = None,
    start_at_end: bool = True,
    poll_interval: float = 0.5,
    idle_sleep: float = 0.1,
    max_steps: Optional[int] = None,
    max_sim_time: Optional[float] = None,
    pause: float = 0.3,
    record_dir: Optional[str] = None,
    record_every: int = 1,
    record_dpi: int = 120,
) -> float:
    prev_mode = policy_net.training
    policy_net.eval()
    try:
        _ = env.reset(scenario)
        s = np.array(env._get_state(), dtype=np.float32)
        step = 0

        live_records: List[dict] = []
        if isinstance(scenario, dict) and "jobs" in scenario:
            live_records.append(scenario)
        elif (
            isinstance(scenario, list)
            and len(scenario) > 0
            and isinstance(scenario[0], dict)
            and "jobs" in scenario[0]
        ):
            live_records = list(scenario)

        fh = None
        if live_job_file:
            with open(live_job_file, "a", encoding="utf-8"):
                pass
            fh = open(live_job_file, "r", encoding="utf-8")
            if start_at_end:
                fh.seek(0, os.SEEK_END)

        last_poll = 0.0

        if record_dir:
            os.makedirs(record_dir, exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        fig.subplots_adjust(hspace=0.5, top=0.9)
        draw_dispatch_queue(axes[0], env.trace, current_t=env.t)
        draw_amr_schedule(
            axes[1], env.trace, env.makespan(), current_t=env.t, inventories=env.robot_inventory
        )
        draw_input_queue(axes[2], live_records, current_t=env.t)
        fig.suptitle(f"t={env.t:.1f}s | step={step} | available={len(env.available_tasks)}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(pause)

        while True:
            now_wall = time.time()
            if fh and (now_wall - last_poll) >= poll_interval:
                new_records = poll_live_job_file(fh)
                for rec in new_records:
                    jobs = rec.get("jobs", []) if isinstance(rec, dict) else []
                    if not isinstance(jobs, list) or not jobs:
                        continue
                    dt = rec.get("dispatch_time", None) if isinstance(rec, dict) else None
                    env.enqueue_jobs(jobs, dispatch_time=dt)
                    if isinstance(rec, dict) and "dispatch_time" in rec:
                        live_records.append(rec)
                    else:
                        live_records.append({"dispatch_time": env.t, "jobs": jobs})
                last_poll = now_wall

            env._release_until(env.t)
            idle = [i for i in range(env.num_robots) if env.robot_free_times[i] <= env.t + 1e-9]

            if idle and env.available_tasks:
                # Mirror env._advance_to_decision_point: if EVERY available
                # task is buffer-blocked and a future release can unblock the
                # chain, advance time instead of forcing a blocked dispatch.
                if (
                    env.enable_buffer_blocking
                    and len(env.blocked_task_indices()) == len(env.available_tasks)
                ):
                    next_rel = env._next_release_time()
                    if math.isfinite(next_rel) and next_rel > env.t + 1e-9:
                        env.t = max(env.t, next_rel)
                        continue

                env.current_robot = min(idle, key=lambda i: (env.robot_free_times[i], i))
                s = np.array(env._get_state(), dtype=np.float32)
                action = _greedy_action(env, policy_net, s, device)
                sp, _, _done = env.step(action)
                s = np.array(sp, dtype=np.float32) if sp is not None else s
                step += 1

                for ax in axes:
                    ax.clear()

                draw_dispatch_queue(axes[0], env.trace, current_t=env.t)
                draw_amr_schedule(
                    axes[1],
                    env.trace,
                    env.makespan(),
                    current_t=env.t,
                    inventories=env.robot_inventory,
                )
                draw_input_queue(axes[2], live_records, current_t=env.t)

                fig.suptitle(f"t={env.t:.1f}s | step={step} | available={len(env.available_tasks)}")
                fig.canvas.draw()
                fig.canvas.flush_events()

                if record_dir and step % max(1, record_every) == 0:
                    frame_path = os.path.join(record_dir, f"frame_{step:05d}.png")
                    fig.savefig(frame_path, dpi=record_dpi)

                plt.pause(pause)

            else:
                if env.available_tasks and not idle:
                    env.t = max(env.t, min(env.robot_free_times))
                elif env.release_idx < len(env.release_events):
                    env.t = max(env.t, env.release_events[env.release_idx][0])
                else:
                    if max_steps is not None and step >= max_steps:
                        break
                    if max_sim_time is not None and env.t >= max_sim_time:
                        break
                    if fh is None:
                        break
                    time.sleep(idle_sleep)

            if max_steps is not None and step >= max_steps:
                break
            if max_sim_time is not None and env.t >= max_sim_time:
                break

        if fh:
            fh.close()

        return env.makespan()
    finally:
        policy_net.train(prev_mode)
