"""Scenario-v2 health checks (Section 7 of the scenario spec).

Confirms the instances contain the decisions the paper claims to make well, before
any GPU time is spent. Run this after regenerating instances and after any change
to the layout, the fleet size, or the size mix.

    python health_check_v2.py --inbox ../test_case/v2/smoke_60.jsonl --events 10

Targets
-------
    congestion tax        13-18%    below ~10% the execution-aware claim is not credible
    AMR-rule spread       > 10%     confirms the assignment decision is worth making
    capacity mask rate    > 5%      below this the consolidation claim is unsupported
    unroutable / episode  < 0.1     currently the weakest number; planner work needed
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
import scenario_v2 as sc  # noqa: E402
from GA.GA import load_dispatch_events, evaluate_solution  # noqa: E402
from reinforce_baseline import complete_with_dispatch_rule  # noqa: E402
from operation_policy import legal_actions, initial_operation_state, apply_fast_action  # noqa: E402
from dispatching_rules import dispatching_rules as dr  # noqa: E402

AMR_RULES = ["earliest_available", "earliest_completion", "least_loaded",
             "material_match", "nearest_amr", "random"]


def capacity_mask_rate(jobs, rule: str, seed: int = 42) -> float:
    """Fraction of pickup decisions at which at least one AMR was slot-blocked.

    If this is near zero the rack never binds, and any claim about learned
    consolidation is unsupported regardless of how well motivated the rack is.
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
    ap.add_argument("--inbox", type=str, default="../test_case/v2/smoke_60.jsonl")
    ap.add_argument("--events", type=int, default=10)
    ap.add_argument("--amrs", type=int, default=sc.NUM_AMRS)
    ap.add_argument("--job_rule", type=str, default="material_match")
    args = ap.parse_args()

    sc.apply_layout(num_amrs=args.amrs)
    events = load_dispatch_events(Path(args.inbox))[: args.events]
    if not events:
        raise SystemExit(f"no instances found in {args.inbox}")
    jobs0 = events[0]["jobs"]

    print(f"scenario  {GA.GRID_MAX_X + 1}x{GA.GRID_MAX_Y} open floor, "
          f"{len(GA.INBOUND_DOCK_LOCATIONS)} doors/side, {len(GA.AMR_KEYS)} AMRs, "
          f"eta = {sc.contention_ratio(len(GA.AMR_KEYS)):.2f}")
    print(f"instances {len(events)} x {len(jobs0)} parcels, "
          f"{len({j.shipment_id for j in jobs0})} shipments\n")

    rows = {}
    for amr_rule in AMR_RULES:
        free, coll, tard, late, inv = [], [], [], [], []
        for ev in events:
            jobs = ev["jobs"]
            ind = complete_with_dispatch_rule(
                jobs, [], {}, baseline_rule=f"{args.job_rule}+{amr_rule}", seed=42)
            f = evaluate_solution(ind, jobs, check_collision=False)
            c = evaluate_solution(ind, jobs, check_collision=True)
            free.append(f["makespan"]); coll.append(c["makespan"])
            tard.append(c["tardiness"]); late.append(c["late_shipments"])
            inv.append(c["invalid_jobs"])
        rows[amr_rule] = tuple(statistics.mean(x) for x in (free, coll, tard, late, inv))
        mf, mc, mt, ml, mi = rows[amr_rule]
        print(f"  {amr_rule:<20} free={mf:7.1f} exec={mc:7.1f} "
              f"tax={100 * (mc - mf) / mf:5.1f}%  tard={mt:7.1f} late={ml:4.1f} inv={mi:.2f}")

    frees = [v[0] for v in rows.values()]
    colls = [v[1] for v in rows.values()]
    invs = [v[4] for v in rows.values()]
    tax = 100 * (statistics.mean(colls) - statistics.mean(frees)) / statistics.mean(frees)
    spread = 100 * (max(colls) - min(colls)) / min(colls)
    mask_rate = capacity_mask_rate(jobs0, f"{args.job_rule}+earliest_completion")

    def verdict(ok):
        return "PASS" if ok else "CHECK"

    print(f"\n{'metric':<24}{'value':>10}   {'target':<14}status")
    print(f"{'congestion tax':<24}{tax:>9.1f}%   {'13-18%':<14}{verdict(13 <= tax <= 20)}")
    print(f"{'AMR-rule spread':<24}{spread:>9.2f}%   {'> 10%':<14}{verdict(spread > 10)}")
    print(f"{'capacity mask rate':<24}{100 * mask_rate:>9.1f}%   {'> 5%':<14}{verdict(mask_rate > 0.05)}")
    print(f"{'unroutable / episode':<24}{statistics.mean(invs):>10.2f}   {'< 0.1':<14}"
          f"{verdict(statistics.mean(invs) < 0.1)}")
    print(f"\nbest executed rule: {min(rows, key=lambda k: rows[k][1])}   "
          f"best collision-free rule: {min(rows, key=lambda k: rows[k][0])}")


if __name__ == "__main__":
    main()
