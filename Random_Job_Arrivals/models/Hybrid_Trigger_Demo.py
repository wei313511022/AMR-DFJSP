#!/usr/bin/env python3
"""
Hybrid trigger rescheduling demo.

This uses the same visualization and background Attention scheduler as
Periodic_Pairing_Demo, but replaces DDQN/fixed-period timing with:

  all AMRs busy
  + waiting/unstarted work exists
  + reschedule cooldown/event gate passes
  + estimated scheduler time fits before the earliest AMR becomes idle

The default is pure Attention rescheduling with no post-local improvement so
the trigger can exploit the short Attention.py inference time.
"""

import argparse
import os
import math

import matplotlib.pyplot as plt

import Periodic_Pairing_Demo as base


HYBRID_STATE_FILE = "hybrid_amr_state.json"
DEFAULT_EXPECTED_COMPUTE_TIME = 0.25
DEFAULT_MIN_BUSY_WINDOW = 1.0
DEFAULT_SAFETY_MARGIN = 0.25


def _amr_busy_window(amr: str) -> float:
    """Estimated sim-time seconds until this AMR can request new work."""
    from GA.GA import heuristic

    sim = base.ai_env.sim
    state = sim.amr_states[amr]
    mode = state.get("mode")

    if mode == "idle":
        return 0.0

    if mode in {"processing", "processing_old", "loading_dock"}:
        return float(max(state.get("proc_ticks", 0), 0))

    goal = state.get("goal")
    if goal is None:
        return 0.0

    return float(heuristic(sim.positions[amr], goal))


def all_amrs_busy_window() -> float:
    from GA.GA import AMR_KEYS

    windows = [_amr_busy_window(amr) for amr in AMR_KEYS]
    if not windows or any(window <= 0.0 for window in windows):
        return 0.0
    return min(windows)


def has_waiting_or_unstarted_jobs() -> bool:
    return any(job.status == 1 for job in base.ai_env.active_jobs)


def estimated_compute_window(args: argparse.Namespace) -> float:
    observed = float(getattr(base.ai_env, "last_ga_compute_time", 0.0) or 0.0)
    expected = max(float(args.expected_compute_time), observed)
    return expected * float(base.SIM_SPEED_MULTIPLIER) + float(args.safety_margin)


def should_hybrid_reschedule(args: argparse.Namespace):
    if base.is_computing:
        return False, "already computing"
    if not has_waiting_or_unstarted_jobs():
        return False, "no waiting jobs"
    if not base.ai_env.can_reschedule():
        return False, "cooldown/no new event"

    busy_window = all_amrs_busy_window()
    required_window = estimated_compute_window(args)
    required_window = max(required_window, float(args.min_busy_window))

    if busy_window < required_window:
        return False, f"busy window too short ({busy_window:.2f}s < {required_window:.2f}s)"

    return True, f"busy window {busy_window:.2f}s"


def init_hybrid_ai(args: argparse.Namespace) -> None:
    """Initialize GridEnv and Attention model without loading a DDQN agent."""
    base.CONFIG["DEVICE"] = "cuda" if base.torch.cuda.is_available() else "cpu"
    base.CONFIG["DATASET_PATH"] = base.DATASET_PATH
    base.CONFIG["GA_ROUTING_ITERS"] = int(args.routing_iters)
    base.CONFIG["GA_COLLISION_ITERS"] = int(args.collision_iters)

    base.ai_env = base.GridEnv()
    base.ai_env.reset()
    base.ai_agent = None
    base.ai_env.last_ga_compute_time = float(args.expected_compute_time)


def update_title(ax):
    status = "RUNNING" if base.is_running else "PAUSED"
    pending = len(base.jobs_top)
    active = len(base.ai_env.active_jobs) if base.ai_env else 0
    completed = len(base.ai_env.completed_jobs) if base.ai_env else 0
    busy_window = all_amrs_busy_window() if base.ai_env else 0.0
    ax.set_title(
        "Hybrid Busy-Window Rescheduling\n"
        f"t={base.simulation_time:.1f}s | {status} | "
        f"New={pending} Active={active} Done={completed} | "
        f"BusyWindow={busy_window:.1f}s Reschedules={base.total_reschedule_count}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_pos", type=str, default=None)
    parser.add_argument("--state_file", type=str, default=HYBRID_STATE_FILE)
    parser.add_argument("--sync_file", type=str, default=None)
    parser.add_argument("--expected_compute_time", type=float, default=DEFAULT_EXPECTED_COMPUTE_TIME)
    parser.add_argument("--min_busy_window", type=float, default=DEFAULT_MIN_BUSY_WINDOW)
    parser.add_argument("--safety_margin", type=float, default=DEFAULT_SAFETY_MARGIN)
    parser.add_argument("--routing_iters", type=int, default=0)
    parser.add_argument("--collision_iters", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()

    base.AMR_STATE_FILE = args.state_file

    base.load_jobs_from_dataset()
    init_hybrid_ai(args)

    fig, ax = plt.subplots(figsize=(13, 4.8))
    fig.canvas.manager.set_window_title("Hybrid Busy-Window Rescheduling")

    if args.window_pos:
        try:
            fig.canvas.manager.window.geometry(args.window_pos)
        except Exception:
            pass

    ax.set_ylim(base.AX_Y_MIN, base.AX_Y_MAX)
    ax.set_xlim(0.0, base.VIEW_WIDTH)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)

    base.draw_static_panels(ax)
    update_title(ax)
    base.write_amr_state()

    timer = fig.canvas.new_timer(interval=base.UPDATE_INTERVAL_MS)

    def tick():
        if args.sync_file:
            try:
                with open(args.sync_file, "r", encoding="utf-8") as f:
                    base.is_running = f.read().strip() == "1"
            except Exception:
                pass

        if base.is_running:
            real_dt = base.UPDATE_INTERVAL_MS / 1000.0
            base.accumulated_sim_dt += real_dt * base.SIM_SPEED_MULTIPLIER

            steps_this_tick = 0
            while base.accumulated_sim_dt >= 1.0:
                base.advance_simulation_one_tick()
                base.accumulated_sim_dt -= 1.0
                steps_this_tick += 1

            base.simulation_time = base.ai_env.sim_time
            base.spawn_arrived_jobs(ax)

            try:
                best_ind, job_map, elapsed = base.ga_result_queue.get_nowait()
                if best_ind is None:
                    print("[HYBRID] Background task failed. Resuming simulation without updates.")
                    base.is_computing = False
                else:
                    base.ai_env.sim.assign_schedules(best_ind.order, best_ind.amr_assignment, job_map)
                    base.ai_env.last_resched_t = float(base.ai_env.sim.t)
                    base.ai_env.last_resched_version = base.ai_env.arrival_version
                    base.ai_env.last_resched_completion_count = len(base.ai_env.completed_jobs)
                    base.ai_env.last_ga_compute_time = float(elapsed)
                    base.is_computing = False
                    base.total_reschedule_count += 1
                    base.last_action_str = f"RESCHEDULE ({elapsed * 1000:.0f}ms)"
                    base.write_schedule_outbox()
                    print(
                        f"[HYBRID] t={base.ai_env.sim_time:.1f} "
                        f"RESCHEDULE #{base.total_reschedule_count} | "
                        f"active={len(base.ai_env.active_jobs)} "
                        f"done={len(base.ai_env.completed_jobs)} | "
                        f"Attention={elapsed * 1000:.0f}ms"
                    )
            except base.Empty:
                pass

            if not base.is_computing and steps_this_tick > 0:
                trigger, reason = should_hybrid_reschedule(args)
                if trigger:
                    print(f"[HYBRID] t={base.ai_env.sim_time:.1f} trigger: {reason}")
                    base.start_background_ga()
                else:
                    base.last_action_str = f"WAIT ({reason})"

            scheduled_jids = set()
            if base.ai_env:
                for amr_id in base.ai_env.sim.amr_queues:
                    amr_state = base.ai_env.sim.amr_states[amr_id]
                    if amr_state["job"] is not None:
                        scheduled_jids.add(amr_state["job"].idx)
                    for qj in base.ai_env.sim.amr_queues[amr_id]:
                        scheduled_jids.add(qj.idx)

                if base.is_computing:
                    for _, visual_queue in base.visual_amr_queues.items():
                        for qj in visual_queue:
                            scheduled_jids.add(qj.idx)

                done_jids = {j.jid for j in base.ai_env.completed_jobs}
                exclude_jids = scheduled_jids.union(done_jids)

                before = {j.jid for j in base.jobs_top}
                base.jobs_top[:] = [vj for vj in base.all_visual_jobs if vj.jid not in exclude_jids]
                after = {j.jid for j in base.jobs_top}
                if before != after:
                    base.rebuild_top_lane(ax)

            if steps_this_tick > 0:
                base.update_amr_lanes_from_sim(ax)

            done = (
                base.ai_env.sim_time >= base.CONFIG["SIM_TIME"]
                or len(base.ai_env.completed_jobs) >= base.ai_env.total_jobs
            )
            if done:
                base.is_running = False
                print(f"\n[DONE] Simulation Complete at t={base.simulation_time:.1f}s")
                print(f"  Completed: {len(base.ai_env.completed_jobs)} jobs")
                print(f"  Reschedules: {base.total_reschedule_count}")

            update_title(ax)
            base.write_amr_state()
            fig.canvas.draw_idle()

        timer.start()

    timer.add_callback(tick)
    timer.start()

    def on_key(event):
        if event.key == " ":
            if args.sync_file:
                try:
                    with open(args.sync_file, "r", encoding="utf-8") as f:
                        val = f.read().strip()
                    with open(args.sync_file, "w", encoding="utf-8") as f:
                        f.write("1" if val == "0" else "0")
                except Exception:
                    pass
            else:
                base.is_running = not base.is_running
            update_title(ax)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    executor = getattr(base, "_process_executor", None)
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
