#!/usr/bin/env python3
# usage: python3 Random_Job_Generator.py --batches 10 --size 15
# batch: number of lines (batches) to generate
# size: number of jobs inside each batch
import json
import random
import time
import argparse
import math

# --------------------------- Config ---------------------------
STATION_COUNT = 5

# Job definitions
JOB_TYPES = {
    "A": {"time": 5},
    "B": {"time": 10},
    "C": {"time": 15},
}
JOB_TYPE_KEYS = list(JOB_TYPES.keys())

# --------------------------- Logic ---------------------------

def generate_data(num_batches, batch_size):
    """
    Generates N batches of jobs instantly and writes to JSONL.
    Each batch contains 'batch_size' jobs.
    """
    import os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_FILE = os.path.join(SCRIPT_DIR, f"dispatch_inbox_{batch_size}.jsonl")
    # Clear the file first
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass
    
    print(f"Generating {num_batches} batches (each with {batch_size} jobs) to {OUTPUT_FILE}...")
    
    current_job_id = 0
    simulated_time = 0.0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i in range(num_batches):
            batch_jobs = []
            
            # Generate exactly 'batch_size' jobs for this line
            for _ in range(batch_size):
                # Simulate a small time gap between job arrivals
                interarrival = -math.log(1.0 - random.random()) * 2.0 
                simulated_time += interarrival
                
                jtype = random.choice(JOB_TYPE_KEYS)
                proc_time = float(JOB_TYPES[jtype]["time"])
                station = random.randint(1, STATION_COUNT)
                
                job_data = {
                    "jid": current_job_id,
                    "type": jtype,
                    "proc_time": proc_time,
                    "station": station
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
            
    print(f"Done! Generated {num_batches} lines. Total jobs: {num_batches * batch_size}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dispatch inbox JSONL data.")
    
    # Argument for how many lines (batches) to generate
    parser.add_argument(
        "--batches", "-b",
        type=int, 
        default=5, 
        help="Number of lines (batches) to generate. Default is 5."
    )

    # Argument for how many jobs per line
    parser.add_argument(
        "--size", "-s",
        type=int, 
        default=10, 
        help="Number of jobs inside each batch. Default is 10."
    )
    
    args = parser.parse_args()
    generate_data(args.batches, args.size)