#!/usr/bin/env python3
# usage: python random_job_gen.py --batches 100 --min-size 1 --max-size 10 --arrival-mean 10 --out dispatch_batches.jsonl
# batches: number of dispatch events to generate
# size: fixed number of jobs per dispatch (optional)
# min-size/max-size: random job count per dispatch (used when size not provided)
# arrival-mean: mean inter-arrival time between dispatches (exponential)
import json
import random
import time
import argparse
import math

# --------------------------- Config ---------------------------
DEFAULT_STATION_COUNT = 5
DEFAULT_DOCK_COUNT = 5

# Job definitions (Phase III contract: material A/B/C -> duration 5/10/15)
JOB_TYPES = {
    "A": {"time": 5},
    "B": {"time": 10},
    "C": {"time": 15},
}
JOB_TYPE_KEYS = list(JOB_TYPES.keys())

# --------------------------- Logic ---------------------------

def _sample_exponential(mean: float) -> float:
    if mean <= 0:
        return 0.0
    return -math.log(1.0 - random.random()) * mean


def generate_data(
    num_batches: int,
    fixed_size: int | None,
    min_size: int,
    max_size: int,
    arrival_mean: float,
    output_file: str,
    station_count: int,
    seed: int | None,
    dock_count: int = DEFAULT_DOCK_COUNT,
):
    """
    Generates N dispatch events and writes to JSONL.
    Each record has a dispatch_time and a list of jobs that arrive at that time.
    """
    if seed is not None:
        random.seed(seed)

    # Clear the file first
    with open(output_file, "w", encoding="utf-8") as f:
        pass

    if fixed_size is not None:
        size_desc = f"fixed size={fixed_size}"
    else:
        size_desc = f"size range=[{min_size}, {max_size}]"

    print(f"Generating {num_batches} dispatch events ({size_desc}) to {output_file}...")

    current_job_id = 0
    simulated_time = 0.0

    with open(output_file, "a", encoding="utf-8") as f:
        for i in range(num_batches):
            batch_jobs = []

            # Advance time to next dispatch event
            simulated_time += _sample_exponential(arrival_mean)

            # Determine job count for this dispatch
            if fixed_size is not None:
                batch_size = fixed_size
            else:
                batch_size = random.randint(min_size, max_size)

            for _ in range(batch_size):
                jtype = random.choice(JOB_TYPE_KEYS)
                proc_time = JOB_TYPES[jtype]["time"]
                station = random.randint(1, station_count)
                dock = random.randint(1, dock_count)

                job_data = {
                    "jid": current_job_id,
                    "type": jtype,
                    "proc_time": float(proc_time),
                    "station": station,
                    "dock": dock
                }
                batch_jobs.append(job_data)
                current_job_id += 1
            
            # Construct the JSON record for this batch
            record = {
                "generated_at": time.time(),
                "dispatch_time": float(simulated_time),
                "jobs": batch_jobs
            }
            
            # Write line
            f.write(json.dumps(record) + "\n")
            
    print(f"Done! Generated {num_batches} lines. Total jobs: {current_job_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dispatch inbox JSONL data.")
    
    # Argument for how many lines (batches) to generate
    parser.add_argument(
        "--batches", "-b",
        type=int, 
        default=5, 
        help="Number of lines (batches) to generate. Default is 5."
    )

    # Fixed jobs per dispatch (optional)
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=None,
        help="Fixed number of jobs per dispatch. If omitted, use min/max size."
    )

    # Random job count range per dispatch
    parser.add_argument(
        "--min-size",
        type=int,
        default=1,
        help="Minimum jobs per dispatch when --size not set. Default is 1."
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=10,
        help="Maximum jobs per dispatch when --size not set. Default is 10."
    )

    # Inter-arrival time (exponential mean)
    parser.add_argument(
        "--arrival-mean",
        type=float,
        default=10.0,
        help="Mean inter-arrival time between dispatch events. Default is 10.0."
    )

    # Output file
    parser.add_argument(
        "--out",
        type=str,
        default="dispatch_batches.jsonl",
        help="Output JSONL path. Default is dispatch_batches.jsonl."
    )

    # Station count (match environment)
    parser.add_argument(
        "--stations",
        type=int,
        default=DEFAULT_STATION_COUNT,
        help="Number of stations. Default is 3."
    )

    # Random seed for reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Default is None."
    )
    
    args = parser.parse_args()
    if args.size is not None and args.size <= 0:
        raise ValueError("--size must be > 0")
    if args.min_size <= 0 or args.max_size <= 0:
        raise ValueError("--min-size/--max-size must be > 0")
    if args.min_size > args.max_size:
        raise ValueError("--min-size must be <= --max-size")

    generate_data(
        num_batches=args.batches,
        fixed_size=args.size,
        min_size=args.min_size,
        max_size=args.max_size,
        arrival_mean=args.arrival_mean,
        output_file=args.out,
        station_count=args.stations,
        seed=args.seed
    )
