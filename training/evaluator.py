import os
from typing import List, Optional, Union

import torch
import torch.nn as nn

from core.data_io import load_records
from core.env import TaskSchedulingEnv
from core.features import flatten_jobs
from training.rollout import run_greedy_episode, run_greedy_episode_live, run_greedy_episode_live_stream
from viz.viz_matplotlib import (
    plot_amr_schedule,
    plot_dispatch_queue,
    plot_input_queue,
    show_interactive_schedule,
)
from viz.viz_plotly import show_interactive_schedule_plotly
from viz.viz_route_map import show_route_map_replay


def run_test_and_plot(
    env: TaskSchedulingEnv,
    policy_net: nn.Module,
    device: torch.device,
    scenario_list: List[Union[dict, List[dict]]],
    test_scenario_file: Optional[str] = None,
    show_live: bool = False,
    show_live_stream: bool = False,
    show_interactive: bool = False,
    show_route_map: bool = False,
    show_plotly: bool = True,
    show_test_plots: bool = True,
    live_pause: float = 0.05,
    live_job_file: str = "live_jobs.jsonl",
    live_start_at_end: bool = True,
    live_poll_interval: float = 0.5,
    live_idle_sleep: float = 0.1,
    live_max_steps: int = 100,
    live_max_sim_time: Optional[float] = None,
    live_init_scenario: Optional[List[dict]] = None,
    live_record_dir: Optional[str] = None,
    live_record_every: int = 5,
    live_record_dpi: int = 140,
    live_make_gif: bool = False,
    live_gif_path: str = "live_schedule.gif",
    route_play_step: float = 0.5,
    route_play_interval_ms: int = 120,
) -> float:
    print("\n=== TEST & PLOT ===")
    if test_scenario_file:
        test_records = load_records(test_scenario_file)
        if not test_records:
            raise RuntimeError(f"No records found in {test_scenario_file}")
        test_scenario = test_records[0] if len(test_records) == 1 else test_records
    else:
        test_scenario = scenario_list[0]

    test_is_stream = (
        isinstance(test_scenario, list)
        and len(test_scenario) > 0
        and isinstance(test_scenario[0], dict)
        and "jobs" in test_scenario[0]
    )
    test_tag = f"stream:{len(test_scenario)}" if test_is_stream else "0"
    test_dispatch_time = (
        float(test_scenario[0].get("dispatch_time", 0.0))
        if test_is_stream
        else float(test_scenario.get("dispatch_time", 0.0))
        if isinstance(test_scenario, dict)
        else 0.0
    )
    test_job_count = len(flatten_jobs(test_scenario))
    test_title = f"TEST scenario={test_tag} dt={test_dispatch_time:.2f} jobs={test_job_count}"

    if live_init_scenario is None:
        live_init_scenario = []

    if show_live_stream:
        mk = run_greedy_episode_live_stream(
            env,
            policy_net,
            live_init_scenario,
            device,
            live_job_file=live_job_file,
            start_at_end=live_start_at_end,
            poll_interval=live_poll_interval,
            idle_sleep=live_idle_sleep,
            max_steps=live_max_steps,
            max_sim_time=live_max_sim_time,
            pause=live_pause,
            record_dir=live_record_dir,
            record_every=live_record_every,
            record_dpi=live_record_dpi,
        )
    elif show_live:
        mk = run_greedy_episode_live(
            env,
            policy_net,
            test_scenario,
            device,
            pause=live_pause,
            record_dir=live_record_dir,
            record_every=live_record_every,
            record_dpi=live_record_dpi,
            make_gif=live_make_gif,
            gif_path=live_gif_path,
        )
    else:
        mk = run_greedy_episode(env, policy_net, test_scenario, device)

    if show_interactive:
        show_interactive_schedule(
            env.trace,
            test_scenario,
            mk,
            inventories=env.robot_inventory,
            window=80.0,
            title_info=test_title,
            initial_t=env.t,
        )
    if show_route_map:
        show_route_map_replay(
            env,
            env.trace,
            initial_t=0.0,
            play_step=route_play_step,
            play_interval_ms=route_play_interval_ms,
        )
    if show_plotly:
        show_interactive_schedule_plotly(
            env.trace,
            test_scenario,
            mk,
            inventories=env.robot_inventory,
            window=80.0,
            step=5.0,
            title_info=test_title,
        )
    print(f"Test makespan = {mk:.2f}s")

    if show_test_plots:
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        plot_dispatch_queue(
            env.trace,
            save_path=os.path.join(out_dir, "dispatch_queue.png"),
            title_info=test_title,
        )
        plot_amr_schedule(
            env.trace,
            makespan=mk,
            save_path=os.path.join(out_dir, "amr_schedule.png"),
            inventories=env.robot_inventory,
            title_info=test_title,
        )
        plot_input_queue(
            test_scenario,
            save_path=os.path.join(out_dir, "input_queue.png"),
            title_info=test_title,
        )

    return mk


def print_batch_results(
    env: TaskSchedulingEnv,
    policy_net: nn.Module,
    device: torch.device,
    scenario_list: List[Union[dict, List[dict]]],
    max_scenarios: Optional[int] = 10,
    collision_avoidance: bool = False,
) -> None:
    """Batch evaluation summary.

    Runs with collision avoidance OFF by default: the collision-aware
    simulation costs minutes per FJSSP instance, so evaluating a full dataset
    with it can take hours. `max_scenarios=None` evaluates everything.
    """
    subset = scenario_list if max_scenarios is None else scenario_list[: max(0, int(max_scenarios))]
    mode = "collision-aware" if collision_avoidance else "fast estimate (no CA)"
    print(f"\n=== BATCH RESULTS ({len(subset)}/{len(scenario_list)} scenarios, {mode}) ===")

    prev_ca = env.enable_collision_avoidance
    env.enable_collision_avoidance = collision_avoidance
    try:
        for i, rec in enumerate(subset):
            if isinstance(rec, list) and len(rec) > 0 and isinstance(rec[0], dict) and "jobs" in rec[0]:
                job_count = len(flatten_jobs(rec))
                mk_i = run_greedy_episode(env, policy_net, rec, device)
                print(f"[STREAM {i}] batches={len(rec)} jobs={job_count} makespan={mk_i:.2f}s")
            else:
                dt = float(rec.get("dispatch_time", 0.0)) if isinstance(rec, dict) else 0.0
                job_count = len(flatten_jobs(rec))
                mk_i = run_greedy_episode(env, policy_net, rec, device)
                print(f"[BATCH {i}] dispatch_time={dt:.2f} jobs={job_count} makespan={mk_i:.2f}s")
        if len(subset) < len(scenario_list):
            print(
                f"... {len(scenario_list) - len(subset)} more scenarios skipped "
                "(raise max_scenarios or pass None to evaluate all)"
            )
    finally:
        env.enable_collision_avoidance = prev_ca
