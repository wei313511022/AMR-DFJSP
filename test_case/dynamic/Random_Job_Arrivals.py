import json
import random
import math

# --- Configuration ---
NUM_EPISODES = 1         # Total training episodes
EPISODE_DURATION = 150.0    # Duration of one episode (sim-seconds)
MEAN_ARRIVAL_TIME = 2.0    # Average seconds between jobs (Poisson Lambda)
OUTPUT_FILE = "test_dataset_demo_v2.jsonl"

# --- Domain Constants ---
JOB_TYPES = ["A", "B", "C"]
# Station IDs for Work Stations (Where the job needs to go)
# Assuming 5 stations labeled 1 to 5 as per your visualizer
DESTINATION_STATION_IDS = [1, 2, 3, 4, 5] 

def generate_exponential_time(mean):
    """Generates time intervals for Poisson Arrival Process"""
    return -math.log(1.0 - random.random()) * mean

def generate_episode(episode_id):
    current_time = 0.0
    jobs = []
    job_counter = 0

    while current_time < EPISODE_DURATION:
        # 1. Next Arrival Time
        dt = generate_exponential_time(MEAN_ARRIVAL_TIME)
        current_time += dt
        
        if current_time > EPISODE_DURATION:
            break

        # 2. Random Attributes
        j_type = random.choice(JOB_TYPES)
        dest_id = random.choice(DESTINATION_STATION_IDS)

        # 3. Create Job Record (Minimal Data)
        job = {
            "id": job_counter,
            "type": j_type,
            "arrival_time": round(current_time, 2),
            "dest_station_id": dest_id
        }
        
        jobs.append(job)
        job_counter += 1

    return {
        "episode_id": episode_id,
        "jobs": jobs
    }

def main():
    print(f"Generating {NUM_EPISODES} episodes...")
    
    # Use 'w' to overwrite/create the file
    with open(OUTPUT_FILE, "w") as f: 
        for i in range(NUM_EPISODES):
            scenario = generate_episode(i)
            
            # Write one line at a time
            f.write(json.dumps(scenario) + "\n")
            
            if (i+1) % 100 == 0:
                print(f"Generated {i+1} episodes...")

    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()