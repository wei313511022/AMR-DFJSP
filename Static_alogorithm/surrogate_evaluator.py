"""Calibrated surrogate evaluator Psi-hat -- the cheap model a deployment would actually use.

Implements the third evaluator of `experiments/surrogate_fidelity`:

    C~      ideal_evaluator.per_robot_ideal   free-space travel, service on arrival,
                                              no queueing, no interference        (eq. 5)
    Psi-hat THIS FILE                         calibrated travel `0.892*d + 4.49`,
                                              fitted queue-depth penalty, no routing
    Phi     GA.decode_schedule                space-time A*, exclusivity, waiting
                                              lines, collisions                   (eq. 8)

Why this file exists
--------------------
`train_extend_gnn.py --train_evaluator ideal` prices the training return with C~, which
is the transport model of the fixed-travel-time literature. Measured on 1080 schedules at
n=20/60/100 (`experiments/surrogate_fidelity/fidelity_summary.txt`), C~ is a weak ranker:

    n=60   value error   ranking tau   regret of its argmin
    C~          15.3%          0.454                  4.96%
    Psi-hat      7.0%          0.736                  1.24%

So an arm that loses to the executor-trained arm may be losing to the STRAWMAN of C~
rather than to the absence of the executor. Psi-hat is the honest cheap opponent: it is
what `apply_fast_action` already advances during every rollout, it costs 0.3 ms against
the executor's 85 ms per schedule at n=60 (290x), and it is what the policy conditions on
through its own decision-state features.

Definition
----------
The Psi-hat makespan of a COMPLETE schedule is what the fast state machine reaches after
replaying every operation in order -- there is no separate closed form. `apply_fast_action`
is therefore the definition, and this replay is the only way to score a schedule the
rollout did not itself produce (a validation decode, a rule baseline, a GA solution).

Like C~, Psi-hat does no routing, so it cannot fail and cannot report an invalid count.
Any comparison of a Psi-hat-trained arm against an executor-trained one therefore bundles
two differences -- the fidelity of the makespan model, and the absence of the routing
failure signal. That is the same bundle the C~ arms carry, which is what keeps the 2x3
factorial's evaluator axis internally consistent.
"""

from __future__ import annotations

from typing import Sequence

import operation_policy as op


def surrogate_makespan(individual, jobs: Sequence) -> float:
    """Psi-hat makespan: replay a complete schedule exactly as the rollout advances state.

    Reads `jobs` without mutating it, so the same list can be scored repeatedly and can
    be the same list a rollout just consumed.
    """
    positions, availabilities, station_availabilities, inventory = op.initial_operation_state(None)
    picked: set = set()
    done: set = set()
    carrier: dict = {}
    events = op.empty_dock_service_events()
    for action_id in op.operation_sequence_from_individual(individual, jobs):
        op.apply_fast_action(
            op.decode_action_id(action_id, jobs),
            jobs,
            picked,
            done,
            carrier,
            positions,
            availabilities,
            station_availabilities,
            inventory,
            dock_service_events=events,
        )
    return max(availabilities.values()) if availabilities else 0.0
