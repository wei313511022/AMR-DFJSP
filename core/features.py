from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from core.env import Coord, TaskSchedulingEnv


def flatten_jobs(scenario: Union[dict, List[dict]]) -> List[dict]:
    if isinstance(scenario, dict) and "jobs" in scenario:
        jobs = scenario.get("jobs", [])
        return jobs if isinstance(jobs, list) else []
    if isinstance(scenario, list):
        if len(scenario) == 0:
            return []
        if isinstance(scenario[0], dict) and "jobs" in scenario[0]:
            out: List[dict] = []
            for rec in scenario:
                jobs = rec.get("jobs", [])
                if isinstance(jobs, list):
                    out.extend(jobs)
            return out
        return scenario
    return []


def decode_robot_pos(state_vec: np.ndarray, num_robots: int, rid: int) -> Coord:
    """
    state layout (M robots, S stations, D docks):
      [onehot(M), n_tasks(1), t(1), robots(6*M), stations(S), docks(D)]
    robots block starts at idx = M + 2
    each robot: [free_time, x, y, invA, invB, invC]
    """
    base = num_robots + 2 + rid * 6
    x = int(state_vec[base + 1])
    y = int(state_vec[base + 2])
    return (x, y)


def decode_robot_free_time(state_vec: np.ndarray, rid: int, num_robots: int = 5) -> float:
    base = num_robots + 2 + rid * 6
    return float(state_vec[base + 0])


def decode_robot_inventory(state_vec: np.ndarray, num_robots: int, rid: int) -> Dict[str, int]:
    base = num_robots + 2 + rid * 6
    inv_a = int(state_vec[base + 3])
    inv_b = int(state_vec[base + 4])
    inv_c = int(state_vec[base + 5])
    return {"A": inv_a, "B": inv_b, "C": inv_c}


def decode_now_t(state_vec: np.ndarray, num_robots: int = 5) -> float:
    return float(state_vec[num_robots + 1])


def decode_station_busy(
    state_vec: np.ndarray, num_stations: int = 5, num_docks: int = 5
) -> Dict[str, float]:
    start = -(num_stations + num_docks)
    end = -num_docks if num_docks > 0 else None
    vals = state_vec[start:end]
    return {f"S{i+1}": float(v) for i, v in enumerate(vals)}


def decode_dock_busy(state_vec: np.ndarray, num_docks: int = 5) -> Dict[str, float]:
    vals = state_vec[-num_docks:]
    return {f"D{i+1}": float(v) for i, v in enumerate(vals)}

#build the action of AMR list
def build_actions_for_tasks(
    tasks: List[dict],  #the available task
    inventory: Dict[str, int],  #kept for signature compatibility (unused)
    capacity_per_type: int,  #kept for signature compatibility (unused)
    allow_proactive_replenish: bool = False,  #kept for signature compatibility (unused)
) -> List[Tuple[int, Dict[str, int]]]:
    """
    Dock-per-job action space (Phase III contract): one action per available
    task. Each task always performs exactly one pickup at its own dock, so the
    replenish plan carries a single unit of the task's material.
    """
    _ = inventory, capacity_per_type, allow_proactive_replenish
    mat_types = ["A", "B", "C"]
    actions: List[Tuple[int, Dict[str, int]]] = []
    for idx, task in enumerate(tasks):
        jtype = str(task.get("type", "")).upper()
        if jtype not in mat_types:
            continue
        plan = {t: 0 for t in mat_types}
        plan[jtype] = 1
        actions.append((idx, plan))
    return actions


def action_features_from_snapshot(
    env: TaskSchedulingEnv,
    robot_pos: Coord,
    now_t: float,
    station_busy: Dict[str, float],
    inventory: Dict[str, int],
    task: dict,
    replenish: Union[int, Dict[str, int], None],
    dock_busy: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Quick estimate of (travel, wait, proc, dock_wait) from a state snapshot,
    matching the dock-per-job timeline used by env.action_features.
    """
    _ = inventory, replenish
    dock = task["pickup"]
    dock_key = str(task.get("dock", ""))
    drop = task["drop"]
    station = task["station"]
    proc = float(task["proc_time"])

    arrive_dock = now_t + float(env._dist(robot_pos, dock))
    dock_free = float((dock_busy or {}).get(dock_key, 0.0))
    dock_wait = max(0.0, dock_free - arrive_dock)
    pickup_end = arrive_dock + dock_wait + proc

    travel = (pickup_end - now_t) + float(env._dist(dock, drop))
    arrive = now_t + travel
    wait = max(0.0, station_busy[station] - arrive)

    return np.array([travel, wait, proc, dock_wait], dtype=np.float32)


def q_values_batch(
    policy_net: nn.Module, state: np.ndarray, feats: np.ndarray, device: torch.device
) -> torch.Tensor:
    """
    state: (state_dim,)
    feats: (K, action_dim)
    return: q (K,) on device
    """
    if feats.shape[0] == 0:
        return torch.empty(0, dtype=torch.float32, device=device)

    if hasattr(policy_net, "action_values"):
        state_t = torch.from_numpy(state.astype(np.float32)).to(device)
        feats_t = torch.from_numpy(feats.astype(np.float32)).to(device)
        q = policy_net.action_values(state_t, feats_t)
        return q.reshape(-1)

    k = feats.shape[0]
    s_mat = np.repeat(state[None, :], k, axis=0).astype(np.float32)
    sa = np.concatenate([s_mat, feats], axis=1).astype(np.float32)
    sa_t = torch.from_numpy(sa).to(device)
    q = policy_net(sa_t).squeeze(1)
    return q


def _same_type_remaining(tasks: List[dict], task_idx: int, jtype: str) -> int:
    cnt = 0
    for i, t in enumerate(tasks):
        if i == task_idx:
            continue
        if str(t.get("type", "")).upper() == jtype:
            cnt += 1
    return cnt


def proactive_replenish_coverage_gain(
    tasks: List[dict],
    task_idx: int,
    replenish: Union[int, Dict[str, int], None],
    inventory: Dict[str, int],
    capacity_per_type: int,
) -> float:
    """
    Heuristic gain: how many extra same-type future jobs can be covered
    by replenishing now (after consuming current job's one item).
    """
    if task_idx < 0 or task_idx >= len(tasks):
        return 0.0

    task = tasks[task_idx]
    jtype = str(task.get("type", "")).upper()
    if not jtype:
        return 0.0

    same_type_future = _same_type_remaining(tasks, task_idx, jtype)
    if same_type_future <= 0:
        return 0.0

    if isinstance(replenish, dict):
        add_main = int(max(0, replenish.get(jtype, 0)))
    else:
        add_main = int(max(0, replenish or 0))

    inv = int(inventory.get(jtype, 0))
    cap = max(0, int(capacity_per_type))

    # Inventory after completing current task (with / without proactive add).
    post_base = max(0, min(cap, inv) - 1)
    post_now = max(0, min(cap, inv + add_main) - 1)

    cover_base = min(post_base, same_type_future)
    cover_now = min(post_now, same_type_future)
    return float(max(0, cover_now - cover_base))


def select_action_index(
    q_values: torch.Tensor,
    actions: List[Tuple[int, Dict[str, int]]],
    tasks: List[dict],
    inventory: Dict[str, int],
    capacity_per_type: int,
    proactive_replenish_bias_weight: float = 0.0,
    action_feats: Optional[np.ndarray] = None,
    full_load_bias_weight: float = 0.0,
    waiting_replenish_bias_weight: float = 0.0,
) -> int:
    """
    Select action index by Q-value, optionally adding a small proactive
    replenishment bias to reduce future source round-trips.
    """
    if len(actions) == 0:
        raise ValueError("actions is empty.")

    if (
        proactive_replenish_bias_weight <= 0.0
        and full_load_bias_weight <= 0.0
        and waiting_replenish_bias_weight <= 0.0
    ):
        return int(torch.argmax(q_values).item())

    best_idx = 0
    best_score = float("-inf")
    w_cover = float(proactive_replenish_bias_weight)
    w_load = float(full_load_bias_weight)
    w_wait = float(waiting_replenish_bias_weight)
    for i, (task_idx, replenish_plan) in enumerate(actions):
        task_idx_i = int(task_idx)
        q_i = float(q_values[i].item())

        task = tasks[task_idx_i] if 0 <= task_idx_i < len(tasks) else {}
        jtype = str(task.get("type", "")).upper()
        if isinstance(replenish_plan, dict):
            plan = {k: int(v) for k, v in dict(replenish_plan).items()}
        else:
            plan = {jtype: int(max(0, replenish_plan or 0))}
        repl_main = int(max(0, plan.get(jtype, 0)))
        repl_total = int(sum(max(0, int(v)) for v in plan.values()))

        total_headroom = 0
        for t in ["A", "B", "C"]:
            total_headroom += max(0, int(capacity_per_type) - int(inventory.get(t, 0)))
        load_ratio = (float(repl_total) / float(total_headroom)) if total_headroom > 0 else 0.0

        gain = proactive_replenish_coverage_gain(
            tasks=tasks,
            task_idx=task_idx_i,
            replenish=plan,
            inventory=inventory,
            capacity_per_type=capacity_per_type,
        )

        wait = 0.0
        if action_feats is not None and i < len(action_feats):
            # action feature layout = [travel, wait, proc, replenish]
            wait = float(action_feats[i][1])

        wait_factor = min(max(wait, 0.0), 30.0) / 30.0
        score = q_i + w_cover * gain + w_load * load_ratio
        if repl_total > 0 and wait > 1e-6:
            # If station is occupied, favor using that waiting window to pre-load.
            score += w_wait * load_ratio * (1.0 + wait_factor)

        if score > best_score:
            best_score = score
            best_idx = i
        elif abs(score - best_score) <= 1e-9:
            # Tie-break: prefer larger replenish to reduce future round-trips.
            best_task_idx, best_plan = actions[best_idx]
            if isinstance(best_plan, dict):
                best_plan_dict = dict(best_plan)
            else:
                best_task = tasks[int(best_task_idx)] if 0 <= int(best_task_idx) < len(tasks) else {}
                best_jtype_local = str(best_task.get("type", "")).upper()
                best_plan_dict = {best_jtype_local: int(max(0, best_plan or 0))}
            best_total = int(sum(max(0, int(v)) for v in best_plan_dict.values()))
            if repl_total > best_total:
                best_idx = i
            elif repl_total == best_total:
                best_task = tasks[int(best_task_idx)] if 0 <= int(best_task_idx) < len(tasks) else {}
                best_jtype = str(best_task.get("type", "")).upper()
                best_main = int(max(0, best_plan_dict.get(best_jtype, 0)))
                if repl_main > best_main:
                    best_idx = i
    return int(best_idx)
