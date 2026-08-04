"""Idealised evaluator and the execution-error metrics -- the paper's core measurement.

Implements Section III-C of paper/problem_formulation.tex:

  eq. (5)  C~_k = sum_i ( Delta_i^k + p(u_i^k) ) + Delta( l(u_nk^k), b_k )
           C~   = max_k C~_k
  eq. (6)  Lambda = ( C_max^Phi - C~ ) / C~
  eq. (7)  Omega_q = sum_k Q_k / sum_k C~_k ,  Omega_r = sum_k R_k / sum_k C~_k

Why this file exists
--------------------
The comparison the paper makes is against *the model the FJSP-T and learned
dispatcher literature actually uses*: a fixed travel-time matrix, service
starting on arrival, no queueing, no collisions. Nothing in this repository
computed that. The old "congestion tax" in health_check_v2 compared the
executor against `decode_schedule(check_collision=False)`, which still enforces
dock exclusivity, waiting-line reservations and upstream holds -- it already
CONTAINS the queueing, so it subtracts out precisely the term prior work omits,
and understates the abstraction error. That reference is not used here.

Delta is free-space shortest-path distance (GA.grid_distance). On the open v3
floor this equals Manhattan distance; using grid_distance keeps eq. (5) correct
if obstacles are ever reintroduced.

What Lambda is and is not
-------------------------
Lambda is a RELATIVE VALUE ERROR of an abstraction, measured on one fixed
schedule. Omega_q and Omega_r diagnose where the robot-time inflation lives.
They are neither independent causal effects nor an additive decomposition of
makespan: Lambda is a max over robots, the Omegas are fleet-summed ratios, and
delays on a non-critical robot move the Omegas without moving Lambda.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import GA.GA as GA
from GA.GA import (
    PICKUP,
    Individual,
    Job,
    decode_schedule,
    grid_distance,
    job_pickup_location,
    repair_operation_order,
)

# Executor timeline kinds, split the way SCENARIO_SPEC_v3 Section 6 prescribes.
QUEUE_KINDS = ("wait_inbound_line", "wait_outbound_line", "hold_upstream")
TRAVEL_KINDS = ("travel", "return")


def _operation_location(op, job_map: Dict[int, Job]) -> Tuple[int, int]:
    """l(u): where operation u is served."""
    job = job_map[op.job_idx]
    if op.kind == PICKUP:
        return job_pickup_location(job)
    return GA.STATIONS[job.station]


def per_robot_ideal(individual: Individual, jobs: Sequence[Job]) -> Dict[str, float]:
    """C~_k for every robot, eq. (5).

    Queue-free and interference-free: each robot walks its own committed
    sequence at free-space distance, starts service the instant it arrives, and
    returns to its dedicated bay. A robot with no work still contributes C~_k=0,
    which is what makes C~ = max_k C~_k a valid makespan for the abstraction.
    """
    job_map = {job.idx: job for job in jobs}
    operations = repair_operation_order(individual.order, list(jobs))

    per_robot: Dict[str, List] = {amr: [] for amr in GA.AMR_KEYS}
    for op in operations:
        amr = individual.amr_assignment[op.job_idx]
        per_robot.setdefault(amr, []).append(op)

    ideal: Dict[str, float] = {}
    for amr, ops in per_robot.items():
        base = GA.AMR_STARTS[amr]
        here = base
        total = 0.0
        for op in ops:
            dest = _operation_location(op, job_map)
            total += grid_distance(here, dest) + float(job_map[op.job_idx].duration)
            here = dest
        total += grid_distance(here, base)   # F_k includes the return leg
        ideal[amr] = total
    return ideal


def per_robot_ideal_travel(individual: Individual, jobs: Sequence[Job]) -> Dict[str, float]:
    """The travel component of eq. (5) alone: sum_i Delta_i^k + return leg.

    This is the free-space baseline that route delay R_k is measured against.
    """
    job_map = {job.idx: job for job in jobs}
    operations = repair_operation_order(individual.order, list(jobs))

    per_robot: Dict[str, List] = {amr: [] for amr in GA.AMR_KEYS}
    for op in operations:
        per_robot.setdefault(individual.amr_assignment[op.job_idx], []).append(op)

    travel: Dict[str, float] = {}
    for amr, ops in per_robot.items():
        base = GA.AMR_STARTS[amr]
        here = base
        total = 0.0
        for op in ops:
            dest = _operation_location(op, job_map)
            total += grid_distance(here, dest)
            here = dest
        total += grid_distance(here, base)
        travel[amr] = total
    return travel


def _union_length(intervals: Iterable[Tuple[float, float]]) -> float:
    """Total length of a union of intervals -- overlapping time counted ONCE.

    The paper requires this explicitly. Without it a robot that is logged as
    both holding upstream and waiting in a line over the same seconds would
    have that second charged twice, and Omega_q could exceed the robot's own
    elapsed time.
    """
    spans = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not spans:
        return 0.0
    total = 0.0
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total


def _by_robot(timeline, kinds) -> Dict[str, List[Tuple[float, float]]]:
    wanted = set(kinds)
    out: Dict[str, List[Tuple[float, float]]] = {amr: [] for amr in GA.AMR_KEYS}
    for entry in timeline:
        amr, start, end, kind = entry[0], entry[1], entry[2], entry[3]
        if kind in wanted:
            out.setdefault(amr, []).append((start, end))
    return out


def evaluate(individual: Individual, jobs: Sequence[Job]) -> Dict[str, float]:
    """Run both evaluators on one schedule and return every quantity the paper reports.

    Returns
    -------
    executed      C_max^Phi, the collision-aware makespan (eq. 8)
    ideal         C~, the idealised makespan (eq. 5)
    penalty       Lambda, the relative value error (eq. 6)
    omega_q       fleet queueing ratio (eq. 7)
    omega_r       fleet route-delay ratio (eq. 7)
    nu            routing failures, reported SEPARATELY
    routable      1.0 iff nu == 0 -- filter every aggregate on this

    A schedule with nu > 0 still returns numbers, but `executed` is not a
    duration: the decoder charges MAX_DEPTH per failed leg. Never average
    `executed` or `penalty` over rows with routable == 0.
    """
    jobs = list(jobs)

    availability, timeline, _, _, invalid_count = decode_schedule(
        individual, jobs, need_log=True, check_collision=True,
    )
    executed = max(availability.values())

    ideal_by_robot = per_robot_ideal(individual, jobs)
    ideal_travel = per_robot_ideal_travel(individual, jobs)
    ideal = max(ideal_by_robot.values()) if ideal_by_robot else 0.0

    queue_spans = _by_robot(timeline, QUEUE_KINDS)
    travel_spans = _by_robot(timeline, TRAVEL_KINDS)

    total_q = sum(_union_length(spans) for spans in queue_spans.values())
    total_r = 0.0
    for amr, spans in travel_spans.items():
        executed_travel = _union_length(spans)
        # Route delay is time spent moving BEYOND the free-space path: detours
        # around other robots, and the extra legs through a waiting slot. It can
        # be negative only through rounding, so it is floored at zero per robot
        # rather than allowed to cancel against another robot's delay.
        total_r += max(0.0, executed_travel - ideal_travel.get(amr, 0.0))

    denom = sum(ideal_by_robot.values())

    return {
        "executed": float(executed),
        "ideal": float(ideal),
        "penalty": float((executed - ideal) / ideal) if ideal > 0 else 0.0,
        "omega_q": float(total_q / denom) if denom > 0 else 0.0,
        "omega_r": float(total_r / denom) if denom > 0 else 0.0,
        "nu": float(invalid_count),
        "routable": 1.0 if invalid_count == 0 else 0.0,
    }


def aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean over CLEANLY ROUTED rows only, plus the failure rate over all rows.

    This is the guard the spec calls for. On a failed leg the decoder does
    `availability[amr] += MAX_DEPTH` (=100), a penalty constant rather than
    elapsed time, so a mean taken over failed runs is not a duration. Several
    early "congestion" spikes in this project were exactly that arithmetic:
    33.9% collapsed to 16.3% once restricted to cleanly-routed runs.
    """
    rows = list(rows)
    clean = [r for r in rows if r.get("routable", 0.0) >= 1.0]
    out = {
        "instances": float(len(rows)),
        "clean_instances": float(len(clean)),
        "nu_per_episode": (sum(r["nu"] for r in rows) / len(rows)) if rows else 0.0,
    }
    for key in ("executed", "ideal", "penalty", "omega_q", "omega_r"):
        out[key] = (sum(r[key] for r in clean) / len(clean)) if clean else float("nan")
    # Share of the abstraction's error attributable to queueing -- the quantity
    # the paper's motivation rests on, and the one an assignment policy controls.
    q, r = out["omega_q"], out["omega_r"]
    out["queue_share"] = (q / (q + r)) if (q + r) > 0 else float("nan")
    return out
