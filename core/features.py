from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


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


#build the action of AMR list
def build_actions_for_tasks(
    tasks: List[dict],  #the available task
    inventory: Dict[str, int],  #usable onboard stock per material
    capacity_per_type: int,
    allow_proactive_replenish: bool = True,
    max_add: Optional[int] = None,
) -> List[Tuple[int, Optional[Dict[str, int]]]]:
    """
    Batch-pickup action space (proactive replenishment restored, adapted to
    the material-dock field):

      - transfer operations (op_index > 0) fetch a specific part at the
        previous station: exactly one action, no replenish choice.
      - dock operations (op_index == 0) choose how many units of the job's
        material to pick up at its dock (Score(i) = Q + cover/load/wait bonuses
        compares these quantity options):
          inventory == 0 -> add in 1..cap_i   (must visit the dock)
          inventory >  0 -> add = 0 (deliver from onboard stock, skip the dock)
                            plus add in 1..cap_i when proactive top-up
                            is enabled (visit the dock on the way).

    cap_i masks useless pickups: never carry more units of a material than the
    currently visible same-material demand (this job + other pending dock ops).
    Future arrivals are not counted — conservative, avoids dead stock at the
    end of an episode. `max_add` additionally caps units per dock visit
    (inference uses max_add=1 when batching would be unfaithful).
    """
    mat_types = ["A", "B", "C"]
    actions: List[Tuple[int, Optional[Dict[str, int]]]] = []
    for idx, task in enumerate(tasks):
        if int(task.get("op_index", 0)) > 0:
            actions.append((idx, None))
            continue
        jtype = str(task.get("type", "")).upper()
        if jtype not in mat_types:
            continue
        inv = int(inventory.get(jtype, 0))
        headroom = max(0, int(capacity_per_type) - inv)
        # Other pending dock operations of the same material (siblings deduped).
        others = _same_type_remaining(tasks, idx, jtype)

        adds: List[int] = []
        if inv > 0:
            adds.append(0)
            if allow_proactive_replenish:
                # After consuming one unit for this job, carried stock should
                # not exceed the remaining visible demand.
                cap_i = min(headroom, max(0, others - inv + 1))
                if max_add is not None:
                    cap_i = min(cap_i, int(max_add))
                adds.extend(range(1, cap_i + 1))
        else:
            cap_i = min(headroom, others + 1)
            if max_add is not None:
                cap_i = min(cap_i, int(max_add))
            cap_i = max(1, cap_i)  # must at least pick this job's own unit
            adds.extend(range(1, cap_i + 1))
        for add in adds:
            plan = {t: 0 for t in mat_types}
            plan[jtype] = int(add)
            actions.append((idx, plan))
    return actions


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
    """Count other pending dock operations (op_index == 0) of the same material.
    Machine-choice siblings (same op_uid) count once; transfer operations are
    excluded because they never consume dock material."""
    chosen_uid = (
        tasks[task_idx].get("op_uid") if 0 <= task_idx < len(tasks) else None
    )
    seen = set()
    for i, t in enumerate(tasks):
        if i == task_idx:
            continue
        if int(t.get("op_index", 0)) > 0:
            continue
        if str(t.get("type", "")).upper() != jtype:
            continue
        uid = t.get("op_uid")
        if uid is None:
            uid = ("idx", i)
        if uid == chosen_uid:
            continue
        seen.add(uid)
    return len(seen)


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

    The bias only compares options OF THE SAME TASK (which quantity to pick
    up); it is zero-based per task so it never shifts the ranking BETWEEN
    tasks — cross-task preferences stay purely Q-driven. Without this, tasks
    at busy stations would systematically collect wait/load bonuses.
    """
    if len(actions) == 0:
        raise ValueError("actions is empty.")

    if (
        proactive_replenish_bias_weight <= 0.0
        and full_load_bias_weight <= 0.0
        and waiting_replenish_bias_weight <= 0.0
    ):
        return int(torch.argmax(q_values).item())

    w_cover = float(proactive_replenish_bias_weight)
    w_load = float(full_load_bias_weight)
    w_wait = float(waiting_replenish_bias_weight)

    bonuses: List[float] = []
    repl_totals: List[int] = []
    repl_mains: List[int] = []
    for i, (task_idx, replenish_plan) in enumerate(actions):
        task_idx_i = int(task_idx)
        task = tasks[task_idx_i] if 0 <= task_idx_i < len(tasks) else {}
        jtype = str(task.get("type", "")).upper()
        if isinstance(replenish_plan, dict):
            plan = {k: int(v) for k, v in dict(replenish_plan).items()}
        else:
            plan = {jtype: int(max(0, replenish_plan or 0))}
        repl_main = int(max(0, plan.get(jtype, 0)))
        repl_total = int(sum(max(0, int(v)) for v in plan.values()))
        repl_mains.append(repl_main)
        repl_totals.append(repl_total)

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
        bonus = w_cover * gain + w_load * load_ratio
        if repl_total > 0 and wait > 1e-6:
            # If the station is occupied anyway, favor using that waiting
            # window to pre-load extra units.
            bonus += w_wait * load_ratio * (1.0 + wait_factor)
        bonuses.append(bonus)

    # Zero-base the bonus within each task group.
    min_bonus_by_task: Dict[int, float] = {}
    for (task_idx, _plan), bonus in zip(actions, bonuses):
        key = int(task_idx)
        cur = min_bonus_by_task.get(key)
        if cur is None or bonus < cur:
            min_bonus_by_task[key] = bonus

    best_idx = 0
    best_score = float("-inf")
    for i, (task_idx, _plan) in enumerate(actions):
        q_i = float(q_values[i].item())
        repl_main = repl_mains[i]
        repl_total = repl_totals[i]
        score = q_i + bonuses[i] - min_bonus_by_task[int(task_idx)]

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
