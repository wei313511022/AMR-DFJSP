import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from core.data_io import load_records
from core.env import TaskSchedulingEnv
from core.features import (
    env_action_candidates,
    env_selection_bias,
    flatten_jobs,
    q_values_batch,
    select_action_index,
)
from training.replay import NStepAccumulator, PrioritizedReplayBuffer, ReplayTransition
from viz.viz_matplotlib import draw_amr_schedule, draw_dispatch_queue, draw_input_queue
from viz.viz_route_map import draw_route_map


def sample_scenario(records: List[dict], mode: str, window_size: int, subset_size: int):
    n = len(records)
    if n == 0:
        return records
    if mode == "window":
        if window_size <= 0 or window_size >= n:
            return records
        start = random.randint(0, n - window_size)
        return records[start : start + window_size]
    if mode == "subset":
        k = min(subset_size, n) if subset_size > 0 else n
        chosen = random.sample(records, k)
        chosen.sort(key=lambda r: r.get("dispatch_time", 0.0))
        return chosen
    return records


def prepare_scenarios(
    task_file: str,
    train_data_dir: str,
    auto_generate_data: bool,
    gen_batches: int,
    gen_size: Union[int, None],
    gen_min_size: int,
    gen_max_size: int,
    gen_arrival_mean: float,
    gen_seed: Union[int, None],
    multi_streams: bool,
    num_streams: int,
    base_seed: Union[int, None],
    stream_file_template: str,
) -> List[Union[dict, List[dict]]]:
    def collect_stream_paths() -> List[str]:
        paths: List[str] = []

        if train_data_dir:
            if "{i}" in stream_file_template:
                glob_pattern = stream_file_template.replace("{i}", "*")
            else:
                glob_pattern = stream_file_template
            discovered = sorted(Path(train_data_dir).glob(glob_pattern))
            for p in discovered:
                if p.is_file() and p.suffix.lower() == ".jsonl":
                    paths.append(str(p))

        if not paths:
            for i in range(num_streams):
                fname = stream_file_template.format(i=i)
                fpath = os.path.join(train_data_dir, fname) if train_data_dir else fname
                if os.path.exists(fpath):
                    paths.append(fpath)

        return paths

    if train_data_dir:
        os.makedirs(train_data_dir, exist_ok=True)

    generated_paths: List[str] = []
    if auto_generate_data:
        from scripts import random_job_gen as rjg

        if multi_streams:
            if train_data_dir:
                if "{i}" in stream_file_template:
                    glob_pattern = stream_file_template.replace("{i}", "*")
                else:
                    glob_pattern = stream_file_template
                for p in Path(train_data_dir).glob(glob_pattern):
                    if p.is_file() and p.suffix.lower() == ".jsonl":
                        p.unlink()

            for i in range(num_streams):
                out_file = (
                    os.path.join(train_data_dir, stream_file_template.format(i=i))
                    if train_data_dir
                    else stream_file_template.format(i=i)
                )
                rjg.generate_data(
                    num_batches=gen_batches,
                    fixed_size=gen_size,
                    min_size=gen_min_size,
                    max_size=gen_max_size,
                    arrival_mean=gen_arrival_mean,
                    output_file=out_file,
                    station_count=6,
                    seed=(base_seed + i) if base_seed is not None else None,
                )
                generated_paths.append(out_file)
        else:
            out_file = os.path.join(train_data_dir, task_file) if train_data_dir else task_file
            rjg.generate_data(
                num_batches=gen_batches,
                fixed_size=gen_size,
                min_size=gen_min_size,
                max_size=gen_max_size,
                arrival_mean=gen_arrival_mean,
                output_file=out_file,
                station_count=6,
                seed=gen_seed,
            )
            generated_paths.append(out_file)

    if multi_streams:
        scenario_list = []
        stream_paths = [p for p in generated_paths if os.path.exists(p)] if generated_paths else collect_stream_paths()
        for fpath in stream_paths:
            recs = load_records(fpath)
            if recs:
                scenario_list.append(recs)
        if not scenario_list:
            raise RuntimeError(
                "No records found in streams "
                f"(dir={train_data_dir}, template={stream_file_template})"
            )
    else:
        if generated_paths:
            path = generated_paths[0]
        else:
            path = os.path.join(train_data_dir, task_file) if train_data_dir else task_file
        records = load_records(path)
        if not records:
            raise RuntimeError(f"No records found in {task_file}")
        scenario_list = [records]

    return scenario_list


def train_ddqn(
    env: TaskSchedulingEnv,
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scenario_list: List[Union[dict, List[dict]]],
    device: torch.device,
    num_episodes: int,
    batch_size: int,
    gamma: float,
    epsilon: float,
    epsilon_end: float,
    epsilon_decay: float,
    sampling_mode: str,
    window_size: int,
    subset_size: int,
    show_train_schedule: bool,
    train_schedule_every_episodes: int,
    train_schedule_every_steps: int,
    train_schedule_window: float,
    train_schedule_window_all_axes: bool,
    train_schedule_pause: float,
    train_schedule_show_labels: bool,
    train_schedule_figsize: tuple,
    show_train_route_map: bool,
    train_route_map_every_episodes: int,
    train_route_map_every_steps: int,
    train_route_map_pause: float,
    train_route_map_figsize: tuple,
    train_route_map_animate: bool,
    train_route_map_time_step: float,
    train_route_map_max_frames_per_update: int,
    train_route_map_delay_seconds: float,
    enable_profile: bool,
    profile_cuda_sync: bool,
    replay_capacity: int = 50000,
    rainbow_n_step: int = 3,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_end: float = 1.0,
    priority_eps: float = 1e-5,
    grad_clip_norm: Optional[float] = 10.0,
    use_noisy_exploration: Optional[bool] = None,
    min_replay_size: Optional[int] = None,
    train_every_steps: int = 1,
    target_sync_steps: int = 0,
    sample_with_replacement: Optional[bool] = None,
) -> Dict[str, Any]:
    del criterion  # Rainbow uses per-sample losses; classic path computes MSE manually.

    use_rainbow = bool(getattr(policy_net, "is_rainbow", False))
    noisy_exploration = (
        bool(getattr(policy_net, "uses_noisy_layers", False))
        if use_noisy_exploration is None
        else bool(use_noisy_exploration)
    )
    if hasattr(policy_net, "set_noise_enabled"):
        policy_net.set_noise_enabled(noisy_exploration)
    if hasattr(target_net, "set_noise_enabled"):
        target_net.set_noise_enabled(noisy_exploration)

    train_every_steps = max(1, int(train_every_steps))
    target_sync_steps = max(0, int(target_sync_steps))
    replay_start_size = max(batch_size, int(min_replay_size) if min_replay_size is not None else batch_size)
    replay_with_replacement = bool(use_rainbow) if sample_with_replacement is None else bool(sample_with_replacement)

    memory = PrioritizedReplayBuffer(
        capacity=replay_capacity,
        alpha=per_alpha if use_rainbow else 0.0,
        priority_eps=priority_eps,
        sample_with_replacement=replay_with_replacement,
    )
    n_step_acc = NStepAccumulator(rainbow_n_step if use_rainbow else 1, gamma)
    print(
        "[TRAIN] replay_start="
        f"{replay_start_size} train_every_steps={train_every_steps} "
        f"target_sync_steps={'episode(5)' if target_sync_steps <= 0 else target_sync_steps} "
        f"sample_with_replacement={replay_with_replacement}"
    )

    makespans: List[float] = []

    policy_net.train()
    target_net.train()

    plt.ion()
    fig_train, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax_mk = axes[0, 0]
    ax_loss = axes[0, 1]
    ax_eps = axes[1, 0]
    ax_dummy = axes[1, 1]
    ax_dummy_right = ax_dummy.twinx()

    mk_history: List[float] = []
    mk_ma50_history: List[float] = []
    loss_history: List[float] = []
    loss_ma100_history: List[float] = []
    eps_history: List[float] = []
    mk_per_job_history: List[float] = []
    mk_ratio_history: List[float] = []
    mk_per_job_ma50: List[float] = []
    mk_ratio_ma50: List[float] = []

    prof: Dict[str, float] = {}

    def prof_add(key: str, dt: float) -> None:
        prof[key] = prof.get(key, 0.0) + dt

    def sync_cuda() -> None:
        if profile_cuda_sync and device.type == "cuda":
            torch.cuda.synchronize()

    train_sched_fig = None
    train_sched_axes = None
    train_route_fig = None
    train_route_ax = None
    train_route_text = None

    def moving_avg(x: List[float], w: int) -> float:
        if len(x) == 0:
            return float("nan")
        if len(x) < w:
            return float(sum(x) / len(x))
        return float(sum(x[-w:]) / w)

    def set_auto_ylim(
        ax,
        series_list: List[List[float]],
        margin_ratio: float = 0.08,
        min_span: float = 1e-6,
    ) -> None:
        vals: List[float] = []
        for s in series_list:
            if not s:
                continue
            vals.extend([float(v) for v in s if np.isfinite(v)])
        if not vals:
            return

        y_min = float(min(vals))
        y_max = float(max(vals))
        span = y_max - y_min
        if span < min_span:
            center = 0.5 * (y_min + y_max)
            base = max(abs(center), 1.0)
            pad = max(min_span, base * margin_ratio)
            ax.set_ylim(center - pad, center + pad)
            return

        pad = span * margin_ratio
        ax.set_ylim(y_min - pad, y_max + pad)

    def update_train_plot(ep: int):
        ax_mk.clear()
        ax_mk.plot(mk_history, linewidth=1, label="makespan")
        ax_mk.plot(mk_ma50_history, linewidth=2, label="MA-50")
        ax_mk.set_title("Makespan per episode (lower is better)")
        ax_mk.set_xlabel("Episode")
        ax_mk.set_ylabel("Makespan")
        ax_mk.grid(True, linestyle="--", alpha=0.4)
        set_auto_ylim(ax_mk, [mk_history, mk_ma50_history])
        ax_mk.legend(loc="upper right")

        ax_loss.clear()
        if len(loss_history) > 0:
            ax_loss.plot(loss_history, linewidth=1, label="loss")
            ax_loss.plot(loss_ma100_history, linewidth=2, label="MA-100")
            set_auto_ylim(ax_loss, [loss_history, loss_ma100_history])
            ax_loss.legend(loc="upper right")
        ax_loss.set_title("Training loss")
        ax_loss.set_xlabel("Update step")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, linestyle="--", alpha=0.4)

        ax_eps.clear()
        ax_eps.plot(eps_history, linewidth=2)
        ax_eps.set_title("Exploration rate")
        ax_eps.set_xlabel("Episode")
        ax_eps.set_ylabel("epsilon")
        ax_eps.set_ylim(0.0, 1.05)
        ax_eps.grid(True, linestyle="--", alpha=0.4)

        ax_dummy.clear()
        ax_dummy_right.clear()
        left_lines = []
        right_lines = []
        if len(mk_per_job_history) > 0:
            left_lines.extend(
                ax_dummy.plot(mk_per_job_history, linewidth=1, color="tab:blue", label="mk/job")
            )
            left_lines.extend(
                ax_dummy.plot(mk_per_job_ma50, linewidth=2, color="tab:orange", label="mk/job MA-50")
            )
            set_auto_ylim(ax_dummy, [mk_per_job_history, mk_per_job_ma50])
        if len(mk_ratio_history) > 0:
            right_lines.extend(
                ax_dummy_right.plot(mk_ratio_history, linewidth=1, color="tab:green", label="mk/proc")
            )
            right_lines.extend(
                ax_dummy_right.plot(mk_ratio_ma50, linewidth=2, color="tab:red", label="mk/proc MA-50")
            )
            set_auto_ylim(ax_dummy_right, [mk_ratio_history, mk_ratio_ma50])
        if left_lines or right_lines:
            handles = left_lines + right_lines
            labels = [h.get_label() for h in handles]
            ax_dummy.legend(handles, labels, loc="upper right")
        ax_dummy.set_title("Normalized Makespan")
        ax_dummy.set_xlabel("Episode")
        ax_dummy.set_ylabel("mk/job")
        ax_dummy_right.set_ylabel("mk/proc")
        ax_dummy.grid(True, linestyle="--", alpha=0.4)

        fig_train.suptitle(f"Training Monitor | EP={ep+1}", fontsize=14)
        fig_train.tight_layout()
        fig_train.canvas.draw()
        fig_train.canvas.flush_events()

    def beta_by_episode(ep_idx: int) -> float:
        if num_episodes <= 1:
            return float(per_beta_end)
        frac = float(ep_idx) / float(max(1, num_episodes - 1))
        return float(per_beta_start + (per_beta_end - per_beta_start) * frac)

    support = None
    delta_z = None
    if use_rainbow:
        support = policy_net.net.support.to(device)
        delta_z = float((policy_net.v_max - policy_net.v_min) / max(1, policy_net.num_atoms - 1))

    def distributional_projection(
        reward: float,
        discount: float,
        next_prob: Optional[torch.Tensor],
        done: bool,
    ) -> torch.Tensor:
        assert support is not None and delta_z is not None
        target = torch.zeros_like(support)
        if done or next_prob is None:
            tz = torch.tensor(float(reward), device=device, dtype=support.dtype)
            tz = torch.clamp(tz, policy_net.v_min, policy_net.v_max)
            b = (tz - policy_net.v_min) / delta_z
            l = int(torch.floor(b).item())
            u = int(torch.ceil(b).item())
            if l == u:
                target[l] = 1.0
            else:
                target[l] = float(u) - float(b)
                target[u] = float(b) - float(l)
            return target

        tz = torch.clamp(float(reward) + float(discount) * support, policy_net.v_min, policy_net.v_max)
        b = (tz - policy_net.v_min) / delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        for atom in range(policy_net.num_atoms):
            lower = int(l[atom].item())
            upper = int(u[atom].item())
            prob = float(next_prob[atom].item())
            if lower == upper:
                target[lower] += prob
            else:
                target[lower] += prob * float(upper - b[atom].item())
                target[upper] += prob * float(b[atom].item() - lower)

        total = target.sum()
        if total > 0:
            target = target / total
        return target

    def rainbow_target_distribution(exp: ReplayTransition) -> torch.Tensor:
        assert support is not None
        if exp.done or exp.next_state is None or len(exp.next_action_feats) == 0:
            return distributional_projection(exp.reward, 0.0, None, True)

        next_state_t = torch.from_numpy(exp.next_state.astype(np.float32)).to(device)
        next_feats_t = torch.from_numpy(exp.next_action_feats.astype(np.float32)).to(device)

        with torch.no_grad():
            next_q = policy_net.action_values(next_state_t, next_feats_t)
            best_idx = int(torch.argmax(next_q).item())
            next_logits = target_net.chosen_action_logits(next_state_t, next_feats_t, best_idx)
            next_prob = torch.softmax(next_logits, dim=-1)
        return distributional_projection(exp.reward, exp.discount, next_prob, False)

    train_wall_start = time.perf_counter()
    update_step = 0
    env_step_total = 0

    for ep in range(num_episodes):
        scenario_idx = random.randrange(len(scenario_list))
        base_scenario = scenario_list[scenario_idx]
        if sampling_mode != "full" and isinstance(base_scenario, list):
            scenario = sample_scenario(base_scenario, sampling_mode, window_size, subset_size)
        else:
            scenario = base_scenario
        is_stream = (
            isinstance(scenario, list)
            and len(scenario) > 0
            and isinstance(scenario[0], dict)
            and "jobs" in scenario[0]
        )
        scenario_tag = f"stream:{len(scenario)}" if is_stream else str(scenario_idx)
        dispatch_time = (
            float(scenario[0].get("dispatch_time", 0.0))
            if is_stream
            else float(scenario.get("dispatch_time", 0.0))
            if isinstance(scenario, dict)
            else 0.0
        )
        job_count = len(flatten_jobs(scenario))
        s = np.array(env.reset(scenario), dtype=np.float32)
        release_t0 = env.release_events[0][0] if env.release_events else float("nan")
        show_schedule_this_ep = show_train_schedule and (
            train_schedule_every_episodes <= 1 or (ep % train_schedule_every_episodes == 0)
        )
        show_route_map_this_ep = show_train_route_map and (
            train_route_map_every_episodes <= 1 or (ep % train_route_map_every_episodes == 0)
        )
        if show_schedule_this_ep and train_sched_fig is None:
            train_sched_fig, train_sched_axes = plt.subplots(3, 1, figsize=train_schedule_figsize)
            train_sched_fig.subplots_adjust(hspace=0.5, top=0.9)
        if show_route_map_this_ep and train_route_fig is None:
            train_route_fig, train_route_ax = plt.subplots(figsize=train_route_map_figsize)
            train_route_fig.subplots_adjust(bottom=0.2, top=0.92)
            train_route_text = train_route_fig.text(0.02, 0.02, "", fontsize=10, ha="left", va="bottom")
        route_last_draw_t = float(env.t)
        step = 0

        done = False
        ep_reward = 0.0

        while not done:
            rid = env.current_robot

            if enable_profile:
                t0 = time.perf_counter()
            actions, feats = env_action_candidates(env, rid)
            k = len(actions)
            if k == 0:
                raise RuntimeError("No valid actions available during training step.")

            if enable_profile:
                prof_add("action_features", time.perf_counter() - t0)

            if noisy_exploration and hasattr(policy_net, "reset_noise"):
                policy_net.reset_noise()

            if (not noisy_exploration) and random.random() < epsilon:
                a_idx = random.randrange(k)
            else:
                with torch.no_grad():
                    if enable_profile:
                        sync_cuda()
                        t_q = time.perf_counter()
                    q_all = q_values_batch(policy_net, s, feats, device)
                    if enable_profile:
                        sync_cuda()
                        prof_add("q_select", time.perf_counter() - t_q)
                    a_idx = select_action_index(
                        q_values=q_all,
                        actions=actions,
                        tasks=env.available_tasks,
                        inventory=env.robot_inventory[rid],
                        capacity_per_type=env.capacity_per_type,
                        action_feats=feats,
                        **env_selection_bias(env),
                    )
            a_feat = feats[a_idx].copy()

            action = actions[a_idx]
            if enable_profile:
                t_env = time.perf_counter()
            sp, r, done = env.step(action)
            if enable_profile:
                prof_add("env_step", time.perf_counter() - t_env)
            ep_reward += r

            sp_arr = np.array(sp, dtype=np.float32) if sp is not None else None
            if not done:
                _next_actions, next_feats = env_action_candidates(env, env.current_robot)
            else:
                next_feats = np.zeros((0, 4), dtype=np.float32)

            base_transition = ReplayTransition(
                state=s.copy(),
                action_index=int(a_idx),
                action_feat=a_feat.copy(),
                action_feats=feats.copy(),
                reward=float(r),
                next_state=sp_arr.copy() if sp_arr is not None else None,
                next_action_feats=next_feats.copy(),
                done=bool(done),
                discount=float(gamma),
            )
            for ready in n_step_acc.append(base_transition):
                memory.add(ready)

            s = sp_arr
            step += 1
            env_step_total += 1

            if show_schedule_this_ep and train_sched_axes is not None:
                if step % max(1, train_schedule_every_steps) == 0 or done:
                    for ax in train_sched_axes:
                        ax.clear()

                    draw_dispatch_queue(
                        train_sched_axes[0],
                        env.trace,
                        show_labels=train_schedule_show_labels,
                        current_t=env.t,
                    )
                    draw_amr_schedule(
                        train_sched_axes[1],
                        env.trace,
                        env.makespan(),
                        show_labels=train_schedule_show_labels,
                        current_t=env.t,
                        inventories=env.robot_inventory,
                    )
                    draw_input_queue(
                        train_sched_axes[2],
                        scenario,
                        show_labels=train_schedule_show_labels,
                        current_t=env.t,
                    )

                    if train_schedule_window and train_schedule_window > 0:
                        center = env.t
                        half = train_schedule_window / 2.0
                        left = max(0.0, center - half)
                        right = center + half
                        if train_schedule_window_all_axes:
                            for ax in train_sched_axes:
                                ax.set_xlim(left, right)
                        else:
                            train_sched_axes[1].set_xlim(left, right)

                    train_sched_fig.suptitle(
                        f"TRAIN t={env.t:.1f}s | ep={ep+1} step={step} | "
                        f"scenario={scenario_tag} dt={dispatch_time:.2f} jobs={job_count}"
                    )
                    train_sched_fig.canvas.draw()
                    train_sched_fig.canvas.flush_events()
                    plt.pause(train_schedule_pause)

            if show_route_map_this_ep and train_route_ax is not None and train_route_text is not None:
                if step % max(1, train_route_map_every_steps) == 0 or done:
                    t_sched = float(env.t)
                    delay_s = max(0.0, float(train_route_map_delay_seconds))
                    t_target = max(0.0, t_sched - delay_s)
                    if t_target < route_last_draw_t:
                        route_last_draw_t = t_target
                    frame_times: List[float]
                    frame_pause = train_route_map_pause
                    if train_route_map_animate and t_target > route_last_draw_t:
                        dt = max(1e-6, float(train_route_map_time_step))
                        n_frames_raw = int(np.ceil((t_target - route_last_draw_t) / dt))
                        n_frames = max(1, min(train_route_map_max_frames_per_update, n_frames_raw))
                        if n_frames_raw > n_frames and train_route_map_pause > 0:
                            frame_pause = train_route_map_pause * (n_frames_raw / n_frames)
                        frame_times = list(np.linspace(route_last_draw_t, t_target, n_frames + 1))[1:]
                    else:
                        frame_times = [t_target]

                    for t_frame in frame_times:
                        lines = draw_route_map(train_route_ax, env, env.trace, current_t=float(t_frame))
                        train_route_text.set_text("\n".join(lines))
                        train_route_fig.suptitle(
                            f"TRAIN Route | ep={ep+1} step={step} | "
                            f"scenario={scenario_tag} dt={dispatch_time:.2f} jobs={job_count} | "
                            f"sched_t={t_sched:.1f}s route_t={float(t_frame):.1f}s",
                            fontsize=12,
                        )
                        train_route_fig.canvas.draw()
                        train_route_fig.canvas.flush_events()
                        if frame_pause > 0:
                            plt.pause(frame_pause)
                    route_last_draw_t = t_target

            if len(memory) >= replay_start_size and (env_step_total % train_every_steps == 0):
                if enable_profile:
                    t_upd = time.perf_counter()

                beta = beta_by_episode(ep) if use_rainbow else 1.0
                idxs, batch, is_weights = memory.sample(batch_size, beta=beta)
                weight_t = torch.from_numpy(is_weights).to(device)

                if noisy_exploration and hasattr(policy_net, "reset_noise"):
                    policy_net.reset_noise()
                if noisy_exploration and hasattr(target_net, "reset_noise"):
                    target_net.reset_noise()

                if use_rainbow:
                    per_sample_losses: List[torch.Tensor] = []
                    for exp in batch:
                        state_t = torch.from_numpy(exp.state.astype(np.float32)).to(device)
                        cur_feats_t = torch.from_numpy(exp.action_feats.astype(np.float32)).to(device)
                        logits = policy_net.chosen_action_logits(state_t, cur_feats_t, exp.action_index)
                        log_prob = torch.log_softmax(logits, dim=-1)
                        target_dist = rainbow_target_distribution(exp)
                        per_sample_losses.append(-(target_dist * log_prob).sum())

                    loss_vec = torch.stack(per_sample_losses)
                    loss = torch.mean(weight_t * loss_vec)
                    priorities = (loss_vec.detach().cpu().numpy() + priority_eps).tolist()
                else:
                    q_pred_list: List[torch.Tensor] = []
                    y_list: List[float] = []

                    for exp in batch:
                        state_t = torch.from_numpy(exp.state.astype(np.float32)).to(device)
                        cur_feats_t = torch.from_numpy(exp.action_feats.astype(np.float32)).to(device)
                        q_pred = policy_net.chosen_action_value(state_t, cur_feats_t, exp.action_index)
                        q_pred_list.append(q_pred)

                        if exp.done or exp.next_state is None or len(exp.next_action_feats) == 0:
                            y = float(exp.reward)
                        else:
                            with torch.no_grad():
                                q_next_all = q_values_batch(
                                    policy_net,
                                    exp.next_state.astype(np.float32),
                                    exp.next_action_feats,
                                    device,
                                )
                                best_i = int(torch.argmax(q_next_all).item())
                                next_state_t = torch.from_numpy(exp.next_state.astype(np.float32)).to(device)
                                next_feats_t = torch.from_numpy(exp.next_action_feats.astype(np.float32)).to(device)
                                q_t = target_net.chosen_action_value(next_state_t, next_feats_t, best_i).item()
                                y = float(exp.reward + exp.discount * q_t)
                        y_list.append(y)

                    q_pred_t = torch.stack(q_pred_list).reshape(-1)
                    y_t = torch.tensor(y_list, dtype=torch.float32, device=device)
                    td = y_t - q_pred_t
                    loss_vec = td.pow(2)
                    loss = torch.mean(weight_t * loss_vec)
                    priorities = (td.detach().abs().cpu().numpy() + priority_eps).tolist()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), float(grad_clip_norm))
                optimizer.step()

                memory.update_priorities(idxs, priorities)

                loss_val = float(loss.detach().cpu().item())
                loss_history.append(loss_val)
                loss_ma100_history.append(moving_avg(loss_history, 100))
                update_step += 1

                if enable_profile:
                    sync_cuda()
                    prof_add("ddqn_update", time.perf_counter() - t_upd)

            if target_sync_steps > 0 and (env_step_total % target_sync_steps == 0):
                target_net.load_state_dict(policy_net.state_dict())
                target_net.train()

        mk = -ep_reward
        makespans.append(mk)

        mk_history.append(mk)
        mk_per_job = mk / max(1, job_count)
        total_proc = 0.0
        for j in flatten_jobs(scenario):
            ops = j if isinstance(j, list) else j.get("operations") if isinstance(j, dict) else None
            if ops:
                # FJSSP job: lower-bound processing = fastest feasible machine per op.
                total_proc += sum(
                    min(float(c.get("processing", 0.0)) for c in op) for op in ops if op
                )
            elif isinstance(j, dict):
                total_proc += float(j.get("proc_time", 0.0))
        mk_ratio = mk / max(1e-6, total_proc)
        mk_per_job_history.append(mk_per_job)
        mk_ratio_history.append(mk_ratio)
        mk_per_job_ma50.append(moving_avg(mk_per_job_history, 50))
        mk_ratio_ma50.append(moving_avg(mk_ratio_history, 50))
        mk_ma50_history.append(moving_avg(mk_history, 50))

        if noisy_exploration:
            eps_history.append(0.0)
        else:
            eps_history.append(epsilon)
            if epsilon > epsilon_end:
                epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if target_sync_steps <= 0 and (ep + 1) % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_net.train()

        avg_mk = moving_avg(mk_history, 50)
        explore_value = 0.0 if noisy_exploration else epsilon
        print(
            f"[EP {ep+1}] batch={scenario_tag} dispatch_time={dispatch_time:.2f} "
            f"release_t0={release_t0:.2f} jobs={job_count} MA-50 mk: {avg_mk:.2f}, "
            f"mk/job: {mk_per_job:.3f}, mk/proc: {mk_ratio:.3f}, "
            f"eps={explore_value:.3f}, last_mk={mk:.2f}, "
            f"buffer_overrides={int(getattr(env, 'buffer_override_count', 0))}"
        )
        update_train_plot(ep)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.train()

    if enable_profile:
        train_wall_end = time.perf_counter()
        total = train_wall_end - train_wall_start
        accounted = sum(prof.values())
        print("\n=== PROFILING (TRAIN) ===")
        for k, v in sorted(prof.items(), key=lambda x: -x[1]):
            pct = (v / total * 100.0) if total > 0 else 0.0
            print(f"{k:>16}: {v:8.2f}s  ({pct:5.1f}%)")
        other = max(0.0, total - accounted)
        print(
            f"{'other':>16}: {other:8.2f}s  "
            f"({(other/total*100.0) if total>0 else 0.0:5.1f}%)"
        )
        print(f"{'total':>16}: {total:8.2f}s")

    return {
        "makespans": makespans,
        "mk_history": mk_history,
        "loss_history": loss_history,
        "epsilon_final": 0.0 if noisy_exploration else epsilon,
        "profile": prof,
        "rainbow_enabled": use_rainbow,
        "updates": update_step,
        "env_steps": env_step_total,
    }
