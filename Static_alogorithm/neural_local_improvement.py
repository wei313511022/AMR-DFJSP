"""Shared post-processing for neural AMR-DFJSP schedulers.

The neural schedulers produce an ``Individual`` and then optionally refine it
with the GA neighbourhood search.  Keeping the two refinement stages here
ensures that all entrypoints use the same order, collision mode, cloning, and
timing behaviour.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List, Optional

from GA.GA import Individual, Job, local_improve


@dataclass(frozen=True)
class NeuralLocalImprovementResult:
    individual: Individual
    simplified_time: float
    collision_time: float

    @property
    def postprocess_time(self) -> float:
        return self.simplified_time + self.collision_time


def clone_individual(individual: Individual) -> Individual:
    """Return a shallow structural copy suitable for local search."""

    return Individual(
        order=list(individual.order),
        amr_assignment=list(individual.amr_assignment),
    )


def apply_neural_local_improvement(
    individual: Individual,
    jobs: List[Job],
    simplified_iters: int = 0,
    collision_iters: int = 0,
    *,
    init_state: dict = None,
    simplified_seed: Optional[int] = None,
    collision_seed: Optional[int] = None,
) -> NeuralLocalImprovementResult:
    """Clone and refine a neural schedule with two ordered search stages.

    Explicit seeds are isolated from the caller's global random stream.  This
    lets benchmark budgets share identical random prefixes without making one
    configuration influence the next configuration.
    """

    if simplified_iters < 0 or collision_iters < 0:
        raise ValueError("Local-improvement iteration counts must be non-negative")

    current = clone_individual(individual)
    simplified_time = 0.0
    collision_time = 0.0
    preserve_random_state = simplified_seed is not None or collision_seed is not None
    random_state = random.getstate() if preserve_random_state else None

    try:
        if simplified_iters > 0:
            if simplified_seed is not None:
                random.seed(simplified_seed)
            started = time.perf_counter()
            current = local_improve(
                current,
                jobs,
                max_iters=simplified_iters,
                check_collision=False,
                init_state=init_state,
            )
            simplified_time = time.perf_counter() - started

        if collision_iters > 0:
            if collision_seed is not None:
                random.seed(collision_seed)
            started = time.perf_counter()
            current = local_improve(
                current,
                jobs,
                max_iters=collision_iters,
                check_collision=True,
                init_state=init_state,
            )
            collision_time = time.perf_counter() - started
    finally:
        if random_state is not None:
            random.setstate(random_state)

    return NeuralLocalImprovementResult(
        individual=current,
        simplified_time=simplified_time,
        collision_time=collision_time,
    )
