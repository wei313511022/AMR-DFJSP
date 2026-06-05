from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from GA.GA import (
    PICKUP,
    Individual,
    Job,
    decode_schedule_tick_by_tick,
    load_dispatch_events,
    paired_operation_order,
    repair_operation_order,
)
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


def job_order_from_individual(individual: Individual, jobs: Sequence[Job]) -> List[int]:
    return [
        op.job_idx
        for op in repair_operation_order(list(individual.order), list(jobs))
        if op.kind == PICKUP
    ]


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
    prefix_order: Sequence[int],
    prefix_assignment: Dict[int, str],
    baseline_rule: str = DEFAULT_BASELINE_RULE,
    seed: int = 42,
) -> Individual:
    job_rule, amr_rule = parse_rule_name(baseline_rule)
    rng = random.Random(seed)
    state = dr.initial_state()
    job_map = {job.idx: job for job in jobs}
    unscheduled = list(jobs)
    max_job_idx = max(job_map) if job_map else -1
    assignment = [""] * (max_job_idx + 1)
    order: List[int] = []

    for job_idx in prefix_order:
        if job_idx not in job_map:
            raise ValueError(f"Prefix contains unknown job id: {job_idx}")
        if job_idx not in prefix_assignment:
            raise ValueError(f"Prefix assignment missing AMR for job id: {job_idx}")

        job = job_map[job_idx]
        chosen_amr = prefix_assignment[job_idx]
        if chosen_amr not in dr.AMR_KEYS:
            raise ValueError(f"Unknown AMR in prefix assignment: {chosen_amr}")

        estimate = dr.estimate_assignment(job, chosen_amr, state)
        dr.apply_assignment(job, estimate, state)
        order.append(job_idx)
        assignment[job_idx] = chosen_amr
        unscheduled = [remaining for remaining in unscheduled if remaining.idx != job_idx]

    while unscheduled:
        job = dr.choose_job(unscheduled, state, job_rule, rng)
        estimate = dr.choose_amr(job, state, amr_rule, rng)
        order.append(job.idx)
        assignment[job.idx] = estimate.amr
        dr.apply_assignment(job, estimate, state)
        unscheduled.remove(job)

    return Individual(order=paired_operation_order(order), amr_assignment=assignment)


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
        prefix_order=[],
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
    sampled_job_order = job_order_from_individual(sampled_individual, jobs)
    if baseline_mode == "episode":
        step_advantages = [episode_advantage for _ in sampled_job_order]
    else:
        step_advantages = []
        prefix_order: List[int] = []
        prefix_assignment: Dict[int, str] = {}

        for job_idx in sampled_job_order:
            chosen_amr = sampled_individual.amr_assignment[job_idx]

            rule_next_individual = complete_with_dispatch_rule(
                jobs,
                prefix_order=prefix_order,
                prefix_assignment=prefix_assignment,
                baseline_rule=baseline_rule,
                seed=seed,
            )
            rule_next_makespan, _ = evaluate_makespan(
                rule_next_individual, jobs, check_collision=check_collision
            )

            sampled_prefix_assignment = dict(prefix_assignment)
            sampled_prefix_assignment[job_idx] = chosen_amr
            sampled_next_individual = complete_with_dispatch_rule(
                jobs,
                prefix_order=[*prefix_order, job_idx],
                prefix_assignment=sampled_prefix_assignment,
                baseline_rule=baseline_rule,
                seed=seed,
            )
            sampled_next_makespan, _ = evaluate_makespan(
                sampled_next_individual, jobs, check_collision=check_collision
            )

            step_advantages.append(rule_next_makespan - sampled_next_makespan)
            prefix_order.append(job_idx)
            prefix_assignment[job_idx] = chosen_amr

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
