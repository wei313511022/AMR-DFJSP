from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from GA.GA import (
    AMR_KEYS,
    PICKUP,
    UNLOAD,
    Individual,
    Job,
    Operation,
    decode_schedule_tick_by_tick,
    load_dispatch_events,
    repair_operation_order,
)
from operation_policy import OperationAction, action_id
from dispatching_rules import dispatching_rules as dr


DEFAULT_BASELINE_RULE = "earliest_completion_job+earliest_completion"


@dataclass(frozen=True)
class BaselineComparison:
    baseline_individual: Individual
    baseline_makespan: float
    sampled_makespan: float
    step_advantages: List[float]
    improvement: float
    win: bool
    sampled_invalid_jobs: int
    baseline_invalid_jobs: int


def parse_rule_name(rule_name: str) -> Tuple[str, str]:
    if "+" not in rule_name:
        raise ValueError(
            f"Baseline rule must use 'job_rule+amr_rule' format, got: {rule_name}"
        )
    job_rule, amr_rule = (part.strip() for part in rule_name.split("+", 1))
    dr.parse_rule_list(job_rule, dr.JOB_RULES, "job")
    dr.parse_rule_list(amr_rule, dr.AMR_RULES, "AMR")
    return job_rule, amr_rule


def operation_order_from_individual(individual: Individual, jobs: Sequence[Job]) -> List[Operation]:
    return repair_operation_order(list(individual.order), list(jobs))


def load_training_events(inbox: str = "", inboxes: str = ""):
    events = []
    raw_paths = []
    if inbox:
        raw_paths.append(inbox)
    if inboxes:
        raw_paths.extend(path.strip() for path in inboxes.split(",") if path.strip())

    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.exists():
            loaded = load_dispatch_events(path)
            events.extend(loaded)
            print(f"Loaded {len(loaded)} dispatch events from {path}")
        else:
            print(f"Warning: dispatch inbox not found: {path}")
    return events


def evaluate_makespan(
    individual: Individual,
    jobs: Sequence[Job],
    check_collision: bool = True,
) -> Tuple[float, int]:
    availability, _, _, _, invalid_count = decode_schedule_tick_by_tick(
        individual,
        list(jobs),
        need_log=False,
        check_collision=check_collision,
    )
    return max(availability.values()), invalid_count


def complete_with_dispatch_rule(
    jobs: Sequence[Job],
    prefix_operations: Sequence[Operation],
    prefix_assignment: Dict[int, str],
    baseline_rule: str = DEFAULT_BASELINE_RULE,
    seed: int = 42,
) -> Individual:
    job_rule, amr_rule = parse_rule_name(baseline_rule)
    rng = random.Random(seed)
    state = dr.initial_state()
    job_map = {job.idx: job for job in jobs}
    job_id_to_list_idx = {job.idx: idx for idx, job in enumerate(jobs)}
    max_job_idx = max(job_map) if job_map else -1
    assignment = [""] * (max_job_idx + 1)
    order: List[Operation] = []

    for op in prefix_operations:
        job_idx = op.job_idx
        if job_idx not in job_map:
            raise ValueError(f"Prefix contains unknown job id: {job_idx}")
        if op.kind == PICKUP and job_idx not in prefix_assignment:
            raise ValueError(f"Prefix assignment missing AMR for job id: {job_idx}")

        if op.kind == PICKUP:
            chosen_amr = prefix_assignment[job_idx]
        elif op.kind == UNLOAD:
            chosen_amr = state.carrier_map.get(job_idx)
        else:
            raise ValueError(f"Prefix contains unsupported operation kind: {op.kind}")

        if chosen_amr not in dr.AMR_KEYS:
            raise ValueError(f"Unknown AMR in prefix assignment: {chosen_amr}")

        job_list_idx = job_id_to_list_idx[job_idx]
        action = OperationAction(
            kind=op.kind,
            job_list_idx=job_list_idx,
            job_id=job_idx,
            amr=chosen_amr,
            amr_idx=AMR_KEYS.index(chosen_amr),
            action_id=action_id(op.kind, AMR_KEYS.index(chosen_amr), job_list_idx, len(jobs)),
        )
        dr.apply_operation(action, state, jobs)
        order.append(Operation(job_idx, op.kind))
        assignment[job_idx] = chosen_amr

    while len(state.completed_jobs) < len(jobs):
        action, _ = dr.choose_operation(jobs, state, job_rule, amr_rule, rng)
        order.append(Operation(action.job_id, action.kind))
        if action.kind == PICKUP:
            assignment[action.job_id] = action.amr
        dr.apply_operation(action, state, jobs)

    return Individual(order=order, amr_assignment=assignment)


def compute_dispatch_baseline_comparison(
    jobs: Sequence[Job],
    sampled_individual: Individual,
    baseline_rule: str = DEFAULT_BASELINE_RULE,
    baseline_mode: str = "stepwise",
    seed: int = 42,
    check_collision: bool = True,
) -> BaselineComparison:
    if baseline_mode not in {"stepwise", "episode"}:
        raise ValueError("baseline_mode must be 'stepwise' or 'episode'")

    full_baseline = complete_with_dispatch_rule(
        jobs,
        prefix_operations=[],
        prefix_assignment={},
        baseline_rule=baseline_rule,
        seed=seed,
    )
    baseline_makespan, baseline_invalid = evaluate_makespan(
        full_baseline, jobs, check_collision=check_collision
    )
    sampled_makespan, sampled_invalid = evaluate_makespan(
        sampled_individual, jobs, check_collision=check_collision
    )

    episode_advantage = baseline_makespan - sampled_makespan
    sampled_operation_order = operation_order_from_individual(sampled_individual, jobs)
    if baseline_mode == "episode":
        step_advantages = [episode_advantage for _ in sampled_operation_order]
    else:
        step_advantages = []
        prefix_operations: List[Operation] = []
        prefix_assignment: Dict[int, str] = {}

        for op in sampled_operation_order:
            rule_next_individual = complete_with_dispatch_rule(
                jobs,
                prefix_operations=prefix_operations,
                prefix_assignment=prefix_assignment,
                baseline_rule=baseline_rule,
                seed=seed,
            )
            rule_next_makespan, _ = evaluate_makespan(
                rule_next_individual, jobs, check_collision=check_collision
            )

            sampled_prefix_assignment = dict(prefix_assignment)
            if op.kind == PICKUP:
                sampled_prefix_assignment[op.job_idx] = sampled_individual.amr_assignment[op.job_idx]
            sampled_next_individual = complete_with_dispatch_rule(
                jobs,
                prefix_operations=[*prefix_operations, op],
                prefix_assignment=sampled_prefix_assignment,
                baseline_rule=baseline_rule,
                seed=seed,
            )
            sampled_next_makespan, _ = evaluate_makespan(
                sampled_next_individual, jobs, check_collision=check_collision
            )

            step_advantages.append(rule_next_makespan - sampled_next_makespan)
            prefix_operations.append(op)
            if op.kind == PICKUP:
                prefix_assignment[op.job_idx] = sampled_individual.amr_assignment[op.job_idx]

    return BaselineComparison(
        baseline_individual=full_baseline,
        baseline_makespan=baseline_makespan,
        sampled_makespan=sampled_makespan,
        step_advantages=step_advantages,
        improvement=episode_advantage,
        win=sampled_makespan < baseline_makespan,
        sampled_invalid_jobs=sampled_invalid,
        baseline_invalid_jobs=baseline_invalid,
    )


def normalize_advantage_batches(
    batch_advantages: Sequence[Sequence[float]],
    enabled: bool = True,
    eps: float = 1e-8,
) -> List[List[float]]:
    nested = [list(values) for values in batch_advantages]
    if not enabled:
        return nested

    flat = [value for values in nested for value in values]
    if not flat:
        return nested

    mean = sum(flat) / len(flat)
    variance = sum((value - mean) ** 2 for value in flat) / len(flat)
    std = variance ** 0.5
    if std < eps:
        return [[0.0 for _ in values] for values in nested]

    return [[(value - mean) / (std + eps) for value in values] for values in nested]
