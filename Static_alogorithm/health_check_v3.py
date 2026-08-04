"""Scenario-v3 health checks. Run before spending GPU time.

Confirms the instances contain the decisions the paper claims to make well.
Replaces health_check_v2.py, and changes the headline measurement:

    v2:  "congestion tax" = (executed - collision_free_decode) / collision_free
    v3:  Lambda           = (executed - IDEALISED) / idealised          (eq. 6)

The v2 reference was `decode_schedule(check_collision=False)`, which still
enforces dock exclusivity, waiting-slot reservations and upstream holds. It
already contained the queueing, so it subtracted out exactly the term the
FJSP-T literature omits and understated the abstraction error. Lambda is
measured against the model prior work actually uses -- fixed travel times,
service on arrival, no queues, no interference.

Targets
-------
    Lambda (execution penalty)   13-18%   below ~10% the claim is not credible
    queueing share of the error  > 50%    the part an assignment policy controls
    AMR-rule spread              > 5%     the assignment decision is worth making
    capacity mask rate           > 5%     the rack actually binds
    unroutable / episode         < 0.1    executor-relative feasibility holds

Usage
    python health_check_v3.py --inbox ../test_case/v3/instances_60.jsonl --events 10
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

STATIC_DIR = os.path.abspath(os.path.dirname(__file__))
if STATIC_DIR not in sys.path:
    sys.path.insert(0, STATIC_DIR)

import GA.GA as GA  # noqa: E402
import scenario_v3 as sc  # noqa: E402
import ideal_evaluator as ie  # noqa: E402
from GA.GA import load_dispatch_events  # noqa: E402
from reinforce_baseline import complete_with_dispatch_rule  # noqa: E402
from operation_policy import legal_actions  # noqa: E402
from dispatching_rules import dispatching_rules as dr  # noqa: E402

AMR_RULES = ["earliest_available", "earliest_completion", "least_loaded",
             "material_match", "nearest_amr", "random"]

# `random` is a floor, not a candidate policy. Including it in the AMR-rule
# spread turns the spread into "how bad can you do", which is always large and
# says nothing about whether the assignment decision is worth learning. Spread
# is therefore reported over the sensible rules, with the full range shown
# alongside it.
SENSIBLE_AMR_RULES = [r for r in AMR_RULES if r != "random"]

# The spec asks for the penalty to be confirmed separately for the batching rule
# and the single-trip rules, which stress the doors differently: milk_run holds
# one door across consecutive services, lpt takes one parcel per trip.
DEFAULT_JOB_RULES = ("milk_run", "lpt")


def capacity_mask_rate(jobs, rule: str, seed: int = 42) -> float:
    """Fraction of pickup decisions at which at least one AMR was slot-blocked.

    If this is near zero the rack never binds, and any claim about learned
    consolidation is unsupported regardless of how well motivated the rack is.

    MUST be measured under `milk_run`. Every other rule carries exactly 1.00
    parcel at a time and never approaches the rack limit, which is why Q=3/3/3
    and Q=2/2/2 once produced byte-identical results.
    """
    import random as _r

    rng = _r.Random(seed)
    job_rule, amr_rule = rule.split("+")
    state = dr.initial_state()
    blocked_steps = total_steps = 0

    while len(state.completed_jobs) < len(jobs):
        legal = legal_actions(jobs, state.picked_jobs, state.completed_jobs,
                              state.carrier_map, state.inventory)
        pickup_candidates = [a for a in legal if a.kind == GA.PICKUP]
        if pickup_candidates:
            total_steps += 1
            unpicked = [j for j in jobs
                        if j.idx not in state.picked_jobs and j.idx not in state.completed_jobs]
            possible = len(unpicked) * len(GA.AMR_KEYS)
            if len(pickup_candidates) < possible:
                blocked_steps += 1
        action, _ = dr.choose_operation(jobs, state, job_rule, amr_rule, rng)
        dr.apply_operation(action, state, jobs)

    return blocked_steps / max(1, total_steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", type=str, default="../test_case/v3/instances_60.jsonl")
    ap.add_argument("--events", type=int, default=10)
    ap.add_argument("--amrs", type=int, default=sc.NUM_AMRS)
    ap.add_argument("--job_rules", type=str, default=",".join(DEFAULT_JOB_RULES),
                    help="comma-separated job rules to check separately")
    args = ap.parse_args()

    sc.apply_layout(num_amrs=args.amrs)
    events = load_dispatch_events(Path(args.inbox))[: args.events]
    if not events:
        raise SystemExit(f"no instances found in {args.inbox}")
    jobs0 = events[0]["jobs"]
    job_rules = [r.strip() for r in args.job_rules.split(",") if r.strip()]

    print(f"scenario  {GA.GRID_MAX_X + 1}x{GA.GRID_MAX_Y} open floor, "
          f"{len(GA.INBOUND_DOCK_LOCATIONS)} doors/side, {len(GA.AMR_KEYS)} AMRs, "
          f"eta = {sc.contention_ratio(len(GA.AMR_KEYS)):.2f}")
    print(f"rack      {GA.SLOT_CAPACITY}, suffix caps {GA.SUFFIX_CAP}")
    print(f"instances {len(events)} x {len(jobs0)} parcels, all released at t = 0")

    def verdict(ok):
        return "PASS" if ok else "CHECK"

    by_job_rule = {}
    for job_rule in job_rules:
        print(f"\n--- job rule: {job_rule} ---")
        print(f"  {'AMR rule':<20} {'ideal':>8} {'exec':>8} {'Lambda':>8} "
              f"{'Om_q':>7} {'Om_r':>7} {'nu':>6}  clean")
        rows = {}
        for amr_rule in AMR_RULES:
            per_instance = []
            for ev in events:
                jobs = ev["jobs"]
                ind = complete_with_dispatch_rule(
                    jobs, [], {}, baseline_rule=f"{job_rule}+{amr_rule}", seed=42)
                per_instance.append(ie.evaluate(ind, jobs))
            agg = ie.aggregate(per_instance)
            rows[amr_rule] = agg
            print(f"  {amr_rule:<20} {agg['ideal']:>8.1f} {agg['executed']:>8.1f} "
                  f"{100 * agg['penalty']:>7.1f}% {100 * agg['omega_q']:>6.1f}% "
                  f"{100 * agg['omega_r']:>6.1f}% {agg['nu_per_episode']:>6.2f}  "
                  f"{int(agg['clean_instances'])}/{int(agg['instances'])}")
        by_job_rule[job_rule] = rows

    print(f"\n{'job rule':<12}{'Lambda':>9}{'q-share':>10}{'spread*':>10}"
          f"{'spread(all)':>13}{'nu/ep':>8}   status")
    all_lambdas, all_qshares, all_nus = [], [], []
    for job_rule, rows in by_job_rule.items():
        usable = {k: v for k, v in rows.items() if v["clean_instances"] > 0}
        if not usable:
            print(f"{job_rule:<12}  every AMR rule failed to route on every instance")
            continue
        lam = 100 * statistics.mean([v["penalty"] for v in usable.values()])
        q_share = 100 * statistics.mean(
            [v["queue_share"] for v in usable.values() if v["queue_share"] == v["queue_share"]]
        )
        nu_mean = statistics.mean([v["nu_per_episode"] for v in rows.values()])
        sensible = [usable[r]["executed"] for r in SENSIBLE_AMR_RULES if r in usable]
        every = [v["executed"] for v in usable.values()]
        spread = 100 * (max(sensible) - min(sensible)) / min(sensible) if sensible else float("nan")
        spread_all = 100 * (max(every) - min(every)) / min(every)
        all_lambdas.append(lam); all_qshares.append(q_share); all_nus.append(nu_mean)
        print(f"{job_rule:<12}{lam:>8.1f}%{q_share:>9.1f}%{spread:>9.2f}%"
              f"{spread_all:>12.2f}%{nu_mean:>8.2f}   "
              f"{verdict(lam >= 13 and q_share > 50 and nu_mean < 0.1)}")

    mask_rate = capacity_mask_rate(jobs0, "milk_run+earliest_completion")

    # The spec's 13-18% band was calibrated against the SUPERSEDED reference
    # (collision-free decode), which already contained the queueing and so read
    # low. What the claim actually needs is a floor: below roughly 10% the
    # execution-aware premise is not credible. The check enforces the floor and
    # reports the value; re-derive the band from a 30-instance paired run.
    print(f"\n{'metric':<28}{'value':>10}   {'target':<16}status")
    lam = statistics.mean(all_lambdas)
    q_share = statistics.mean(all_qshares)
    nu_mean = statistics.mean(all_nus)
    print(f"{'Lambda (exec penalty)':<28}{lam:>9.1f}%   {'>= 13% (floor)':<16}{verdict(lam >= 13)}")
    print(f"{'queueing share of error':<28}{q_share:>9.1f}%   {'> 50%':<16}{verdict(q_share > 50)}")
    print(f"{'capacity mask rate':<28}{100 * mask_rate:>9.1f}%   {'> 5%':<16}{verdict(mask_rate > 0.05)}")
    print(f"{'unroutable / episode':<28}{nu_mean:>10.2f}   {'< 0.1':<16}{verdict(nu_mean < 0.1)}")

    print("\n* spread is over SENSIBLE AMR rules (random excluded). Including "
          "`random` measures\n  how bad you can do, not whether the assignment "
          "decision is worth learning.\n"
          "  Lambda and the Omegas are means over cleanly-routed instances only; "
          "nu is over all.\n"
          "  Omega_q + Omega_r exceeds Lambda by construction: Lambda is a max "
          "over robots,\n  the Omegas are fleet-summed ratios.")


if __name__ == "__main__":
    main()
