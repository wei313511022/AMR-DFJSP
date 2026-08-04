"""Generate scenario-v3 instances: uniform size mix, all releases at t = 0.

Replaces generate_instances_v2.py. What went away, and why:

  shipments / deadlines   The objective is pure makespan (eq. 8). v2 partitioned
                          parcels into trailer-sized shipments and assigned
                          staggered departures D_g = gamma * C_ref * (i+1) / K.
                          That construction was both degenerate (typically two
                          distinct deadline values per instance) and mildly
                          circular -- C_ref came from running a reference rule
                          on the instance being generated.
  --gamma / --reference   arguments for the above; both removed.

What remains is the whole instance: n parcels, each with a size class, an
inbound door, an outbound station, and r_j = 0. All time structure comes from
contention for the doors, which is why the layout and the fleet ratio
eta = m / |D_in| carry the experiment rather than the parcel data.

Usage
-----
    python generate_instances_v3.py --jobs 60  --count 100 --out ../test_case/v3/instances_60.jsonl
    python generate_instances_v3.py --jobs 120 --count 30  --out ../test_case/v3/instances_120.jsonl
    python generate_instances_v3.py --jobs 240 --count 30  --out ../test_case/v3/instances_240.jsonl
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
import scenario_v3 as sc  # noqa: E402
from GA.GA import Job  # noqa: E402


def build_parcels(n: int, rng: random.Random) -> list:
    """Size class from the uniform mix; origin and destination drawn independently."""
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


def to_record(parcels, index: int) -> dict:
    return {
        "index": index,
        "dispatch_time": 0.0,
        "scenario": "v3",
        "jobs": [
            {
                "jid": j.idx,
                "type": j.type_,
                "proc_time": j.duration,
                "station": j.station,
                "inbound_dock": j.inbound_dock,
                "arrival_time": 0.0,
            }
            for j in parcels
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=60,
                    help="parcels per instance; train at 60, evaluate zero-shot at 60/120/240")
    ap.add_argument("--count", type=int, default=100, help="number of instances")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--amrs", type=int, default=sc.NUM_AMRS)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    sc.apply_layout(num_amrs=args.amrs)
    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mix = {}
    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(args.count):
            parcels = build_parcels(args.jobs, rng)
            for j in parcels:
                mix[j.type_] = mix.get(j.type_, 0) + 1
            fh.write(json.dumps(to_record(parcels, i)) + "\n")

    total = sum(mix.values())
    shares = {k: round(v / total, 3) for k, v in sorted(mix.items())}
    print(f"wrote {args.count} instances of {args.jobs} parcels -> {out_path}")
    print(f"  fleet {args.amrs} AMRs, eta = {sc.contention_ratio(args.amrs):.2f} robots/door")
    print(f"  size mix {shares} (target {sc.SIZE_MIX})")
    print(f"  releases all at t = 0, no deadlines")


if __name__ == "__main__":
    main()
