from __future__ import annotations

import math
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


DEFAULT_BASELINE_RULE = "milk_run+earliest_completion"

# Learning-rate schedule shared by ALL trainers, so an architecture comparison
# is not confounded by one model getting a better training loop than another.
# Previously only extend_GNN annealed its LR; GNN and Attention ran a constant
# rate for the whole run.
DEFAULT_LR = 3e-4
DEFAULT_LR_MIN = 3e-5


def cosine_actor_lr(epoch: int, epochs: int, lr_max: float, lr_min: float) -> float:
    """Half-cosine decay from `lr_max` at epoch 1 to `lr_min` at epoch `epochs`.

    Nearly flat at both ends and steepest in the middle: the run keeps a high
    rate long enough to explore, then finishes at a stable low rate instead of
    one that is still falling underneath it.

    `lr_min` is a FLOOR, not zero. In RL the data distribution is the policy, so
    it keeps moving; annealing to zero freezes the model mid-drift rather than
    at a solution. Setting lr_min == lr_max makes the schedule constant, which
    is the clean way to disable it for an ablation.

    Note the schedule is defined against the DECLARED `epochs`, not wall-clock:
    stopping early means never reaching the low-rate phase, and two runs with
    different epoch counts follow different trajectories and are not comparable.
    """
    if epochs <= 1:
        return lr_max
    progress = (epoch - 1) / (epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine


def score_solution(
    individual: Individual,
    jobs: Sequence[Job],
    check_collision: bool = True,
) -> Tuple[float, int]:
    """Single scoring entry point for the counterfactual baseline.

    Returns (makespan, nu). The objective is pure makespan under the executor;
    the caller charges nu separately. The v2 `use_objective` switch, which mixed
    shipment tardiness into the credit assigned to each decision, is gone along
    with deadlines.
    """
    return evaluate_makespan(individual, jobs, check_collision)


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
    invalid_penalty: float = 0.0,
) -> BaselineComparison:
    if baseline_mode not in {"stepwise", "episode"}:
        raise ValueError("baseline_mode must be 'stepwise' or 'episode'")
    if invalid_penalty < 0:
        raise ValueError("invalid_penalty must be non-negative")

    full_baseline = complete_with_dispatch_rule(
        jobs,
        prefix_operations=[],
        prefix_assignment={},
        baseline_rule=baseline_rule,
        seed=seed,
    )
    baseline_makespan, baseline_invalid = score_solution(
        full_baseline, jobs, check_collision=check_collision
    )
    sampled_makespan, sampled_invalid = score_solution(
        sampled_individual, jobs, check_collision=check_collision
    )

    episode_advantage = baseline_makespan - sampled_makespan
    # Advantages (not the logged improvement/win metrics) additionally charge
    # invalid jobs, so training optimizes the same trade-off validation scores.
    penalized_episode_advantage = (
        (baseline_makespan + invalid_penalty * baseline_invalid)
        - (sampled_makespan + invalid_penalty * sampled_invalid)
    )
    sampled_operation_order = operation_order_from_individual(sampled_individual, jobs)
    if baseline_mode == "episode":
        step_advantages = [penalized_episode_advantage for _ in sampled_operation_order]
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
            rule_next_makespan, rule_next_invalid = score_solution(
                rule_next_individual, jobs, check_collision=check_collision,
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
            sampled_next_makespan, sampled_next_invalid = score_solution(
                sampled_next_individual, jobs, check_collision=check_collision,
            )

            step_advantages.append(
                (rule_next_makespan + invalid_penalty * rule_next_invalid)
                - (sampled_next_makespan + invalid_penalty * sampled_next_invalid)
            )
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


@dataclass(frozen=True)
class MultiSampleComparison:
    """Result of scoring K samples of the SAME instance against each other."""

    step_advantages: List[List[float]]
    sampled_makespans: List[float]
    sampled_invalid_jobs: List[int]
    scores: List[float]
    baseline_makespan: float
    baseline_invalid_jobs: int
    improvements: List[float]
    wins: List[bool]
    spread_cv: float


def compute_multisample_baseline_comparison(
    jobs: Sequence[Job],
    sampled_individuals: Sequence[Individual],
    baseline_rule: str = DEFAULT_BASELINE_RULE,
    seed: int = 42,
    check_collision: bool = True,
    invalid_penalty: float = 0.0,
) -> MultiSampleComparison:
    """Baseline each sample with the MEAN SCORE OF ITS OWN INSTANCE GROUP.

    The dispatch-rule baseline in `compute_dispatch_baseline_comparison` is a
    per-instance constant, so the advantage it produces decomposes as

        A_k = (h_i - E[s_i]) + (E[s_i] - s_k)
               instance offset      true rollout quality

    Only the second term says anything about the actions that were taken; the
    first is contamination that survives batch normalization because it differs
    from instance to instance. Measured on this environment the offset carries
    ~27% of the advantage variance, and the sampling noise it cannot reach is
    ~86% of the total score variance -- so a per-instance baseline, heuristic or
    greedy, is addressing the small half of the problem.

    Drawing K samples of ONE instance and centring them on their own mean
    removes the offset by construction: the advantage IS the quality term.

        b   = mean(s_1 .. s_K)
        A_k = b - s_k

    This is POMO's shared-mean baseline (Kwon et al. 2020). The leave-one-out
    form, b_k = mean of the other K-1, is the same vector scaled by K/(K-1):

        A_k^LOO = (K/(K-1)) * A_k^shared

    so with `normalize_advantage_batches` enabled the two produce identical
    gradients. The shared mean is used here because it is cheaper and its
    advantages sum to exactly zero within each group.

    Unbiasedness: `b` depends on the sampled scores, which is what makes it
    O(1/K)-biased in the strict sense, exactly as in POMO. The leave-one-out
    form is unbiased; since normalization makes them equivalent, nothing is
    lost by taking the simpler one.

    The dispatch rule is still solved ONCE per group so the logged Baseline /
    Improvement / Win Rate columns stay comparable with runs that used the
    heuristic baseline for gradients. It no longer touches the gradient.
    """
    if not sampled_individuals:
        raise ValueError("compute_multisample_baseline_comparison needs at least one sample")
    if invalid_penalty < 0:
        raise ValueError("invalid_penalty must be non-negative")

    full_baseline = complete_with_dispatch_rule(
        jobs,
        prefix_operations=[],
        prefix_assignment={},
        baseline_rule=baseline_rule,
        seed=seed,
    )
    baseline_makespan, baseline_invalid = score_solution(
        full_baseline, jobs, check_collision=check_collision
    )

    makespans: List[float] = []
    invalids: List[int] = []
    scores: List[float] = []
    for individual in sampled_individuals:
        makespan, invalid = score_solution(individual, jobs, check_collision=check_collision)
        makespans.append(makespan)
        invalids.append(invalid)
        scores.append(makespan + invalid_penalty * invalid)

    mean_score = sum(scores) / len(scores)
    variance = sum((value - mean_score) ** 2 for value in scores) / len(scores)
    spread_cv = (variance ** 0.5) / mean_score if mean_score else 0.0

    step_advantages: List[List[float]] = []
    for individual, score in zip(sampled_individuals, scores):
        advantage = mean_score - score
        num_steps = len(operation_order_from_individual(individual, jobs))
        step_advantages.append([advantage for _ in range(num_steps)])

    return MultiSampleComparison(
        step_advantages=step_advantages,
        sampled_makespans=makespans,
        sampled_invalid_jobs=invalids,
        scores=scores,
        baseline_makespan=baseline_makespan,
        baseline_invalid_jobs=baseline_invalid,
        improvements=[baseline_makespan - value for value in makespans],
        wins=[value < baseline_makespan for value in makespans],
        spread_cv=spread_cv,
    )


def validate_sampling_args(
    batch_size: int, samples_per_instance: int, baseline_mode: str
) -> None:
    """Reject sampling settings that would silently train on nothing."""
    if samples_per_instance < 1:
        raise SystemExit("--samples_per_instance must be >= 1")
    if batch_size % samples_per_instance != 0:
        raise SystemExit(
            f"--batch_size ({batch_size}) must be a multiple of "
            f"--samples_per_instance ({samples_per_instance})"
        )
    if baseline_mode == "multisample" and samples_per_instance < 2:
        raise SystemExit(
            "--baseline_mode multisample needs --samples_per_instance >= 2: a "
            "group of one sample IS its own mean, so every advantage would be "
            "exactly zero and the run would train on nothing"
        )


@dataclass(frozen=True)
class GroupSample:
    """One sampled rollout, scored under whichever baseline is configured."""

    step_advantages: List[float]
    sampled_makespan: float
    baseline_makespan: float
    improvement: float
    win: bool
    sampled_invalid_jobs: int
    baseline_invalid_jobs: int


def score_instance_group(
    jobs: Sequence[Job],
    sampled_individuals: Sequence[Individual],
    *,
    baseline_rule: str = DEFAULT_BASELINE_RULE,
    baseline_mode: str = "stepwise",
    seed: int = 42,
    invalid_penalty: float = 0.0,
    check_collision: bool = True,
) -> Tuple[List[GroupSample], float]:
    """Score K samples of ONE instance under the configured baseline.

    Single entry point shared by every trainer so the five REINFORCE loops
    cannot drift apart -- the whole point of the architecture comparison is
    that only the architecture differs.

    Returns the per-sample records plus the group's score coefficient of
    variation. Watch that CV: a multisample baseline centres each sample on
    its group, so if the group ever collapses to a single value the advantages
    all go to zero and the gradient dies. Measured healthy range on this
    environment is 6-8%; drifting under ~1-2% is the early warning.

    With `samples_per_instance == 1` and a dispatch-rule mode this reduces
    exactly to the previous per-sample behaviour, so old runs reproduce.
    """
    if baseline_mode == "multisample":
        group = compute_multisample_baseline_comparison(
            jobs,
            sampled_individuals,
            baseline_rule=baseline_rule,
            seed=seed,
            check_collision=check_collision,
            invalid_penalty=invalid_penalty,
        )
        samples = [
            GroupSample(
                step_advantages=group.step_advantages[idx],
                sampled_makespan=group.sampled_makespans[idx],
                baseline_makespan=group.baseline_makespan,
                improvement=group.improvements[idx],
                win=group.wins[idx],
                sampled_invalid_jobs=group.sampled_invalid_jobs[idx],
                baseline_invalid_jobs=group.baseline_invalid_jobs,
            )
            for idx in range(len(sampled_individuals))
        ]
        return samples, group.spread_cv

    samples = []
    scores = []
    for individual in sampled_individuals:
        comparison = compute_dispatch_baseline_comparison(
            jobs,
            individual,
            baseline_rule=baseline_rule,
            baseline_mode=baseline_mode,
            seed=seed,
            check_collision=check_collision,
            invalid_penalty=invalid_penalty,
        )
        samples.append(
            GroupSample(
                step_advantages=comparison.step_advantages,
                sampled_makespan=comparison.sampled_makespan,
                baseline_makespan=comparison.baseline_makespan,
                improvement=comparison.improvement,
                win=comparison.win,
                sampled_invalid_jobs=comparison.sampled_invalid_jobs,
                baseline_invalid_jobs=comparison.baseline_invalid_jobs,
            )
        )
        scores.append(
            comparison.sampled_makespan + invalid_penalty * comparison.sampled_invalid_jobs
        )

    mean_score = sum(scores) / len(scores) if scores else 0.0
    if len(scores) > 1 and mean_score:
        variance = sum((value - mean_score) ** 2 for value in scores) / len(scores)
        spread_cv = (variance ** 0.5) / mean_score
    else:
        spread_cv = 0.0
    return samples, spread_cv


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
