"""Generate scenario-v2 instances: skewed size mix, outbound shipments, deadlines.

Every parcel is released at t = 0; the temporal structure is carried entirely by
shipment departure deadlines, staggered through the horizon:

    D_g = gamma * C_ref * (i + 1) / K

where C_ref is the collision-free makespan of a fixed reference rule on that
instance, i indexes shipments at the same outbound station, and K is how many
there are. Deriving deadlines from a reference schedule is mildly circular, so the
reference rule is fixed, recorded in the file header, and conclusions should be
confirmed against a second reference (--reference).

Usage
-----
    python generate_instances_v2.py --jobs 60 --count 100 --gamma 1.0 \
        --out ../test_case/v2/instances_60_g100.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

STATIC_DIR = os.path.abspath(os.path.dirname(__file__))
if STATIC_DIR not in sys.path:
    sys.path.insert(0, STATIC_DIR)

import GA.GA as GA  # noqa: E402
import scenario_v2 as sc  # noqa: E402
from GA.GA import Job  # noqa: E402

DEFAULT_REFERENCE = "material_match+earliest_completion"


def build_parcels(n: int, rng: random.Random) -> list:
    """Size class from the skewed mix; origin and destination drawn independently."""
    docks = list(GA.INBOUND_DOCK_LOCATIONS.keys())
    stations = list(GA.STATIONS.keys())
    parcels = []
    for idx in range(n):
        size = sc.sample_size_class(rng)
        parcels.append(
            Job(
                idx=idx,
                type_=size,
                duration=float(GA.TYPE_DURATION[size]),
                station=rng.choice(stations),
                inbound_dock=rng.choice(docks),
                arrival_time=0.0,
            )
        )
    return parcels


def reference_makespan(parcels, rule: str) -> float:
    """Collision-free makespan of a fixed rule. Deadlines are scaled from this."""
    from reinforce_baseline import complete_with_dispatch_rule, evaluate_makespan

    individual = complete_with_dispatch_rule(
        parcels, prefix_operations=[], prefix_assignment={}, baseline_rule=rule, seed=42
    )
    makespan, _ = evaluate_makespan(individual, parcels, check_collision=False)
    return float(makespan)


def attach_shipments(parcels, rng: random.Random, gamma: float, rule: str):
    shipments = sc.partition_shipments(parcels, rng)
    c_ref = reference_makespan(parcels, rule)
    deadlines = sc.assign_deadlines(shipments, parcels, c_ref, gamma=gamma)

    shipment_of = {}
    for sid, members in shipments.items():
        for job_idx in members:
            shipment_of[job_idx] = sid

    out = []
    for job in parcels:
        sid = shipment_of[job.idx]
        out.append(
            Job(
                idx=job.idx,
                type_=job.type_,
                duration=job.duration,
                station=job.station,
                inbound_dock=job.inbound_dock,
                arrival_time=0.0,
                shipment_id=sid,
                deadline=float(deadlines[sid]),
            )
        )
    return out, c_ref, len(shipments)


def to_record(parcels, index: int, c_ref: float, gamma: float) -> dict:
    return {
        "index": index,
        "dispatch_time": 0.0,
        "scenario": "v2",
        "reference_makespan": round(c_ref, 3),
        "gamma": gamma,
        "jobs": [
            {
                "jid": j.idx,
                "type": j.type_,
                "proc_time": j.duration,
                "station": j.station,
                "inbound_dock": j.inbound_dock,
                "arrival_time": 0.0,
                "shipment_id": j.shipment_id,
                "deadline": round(j.deadline, 3),
            }
            for j in parcels
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=60)
    ap.add_argument("--count", type=int, default=100, help="number of instances")
    ap.add_argument("--gamma", type=float, default=sc.DEFAULT_GAMMA,
                    help="deadline tightness; sweep 0.8 / 1.0 / 1.2")
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--reference", type=str, default=DEFAULT_REFERENCE)
    ap.add_argument("--amrs", type=int, default=sc.NUM_AMRS)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    sc.apply_layout(num_amrs=args.amrs)
    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ship, refs = [], []
    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(args.count):
            parcels = build_parcels(args.jobs, rng)
            parcels, c_ref, k = attach_shipments(parcels, rng, args.gamma, args.reference)
            n_ship.append(k)
            refs.append(c_ref)
            fh.write(json.dumps(to_record(parcels, i, c_ref, args.gamma)) + "\n")

    mix = {}
    for j in parcels:
        mix[j.type_] = mix.get(j.type_, 0) + 1
    print(f"wrote {args.count} instances of {args.jobs} parcels -> {out_path}")
    print(f"  fleet {args.amrs} AMRs, eta = {sc.contention_ratio(args.amrs):.2f} robots/door")
    print(f"  gamma {args.gamma}, reference rule '{args.reference}'")
    print(f"  shipments/instance  mean {sum(n_ship)/len(n_ship):.1f}")
    print(f"  reference makespan  mean {sum(refs)/len(refs):.1f}")
    print(f"  last-instance size mix {mix} (target {sc.SIZE_MIX})")


if __name__ == "__main__":
    main()
