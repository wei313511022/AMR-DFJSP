import json
import os
from collections import Counter

# Configuration
INPUT_FILE = "test_dataset.jsonl"

def analyze_job_types(file_path):
    """Reads a jsonl file and calculates the percentage of each job type."""
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found in the current directory.")
        return

    job_counts = Counter()
    total_jobs = 0
    episodes_count = 0

    print(f"Reading from {file_path}...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    episode_data = json.loads(line)
                    episodes_count += 1
                    jobs = episode_data.get("jobs", [])
                    
                    for job in jobs:
                        j_type = job.get("type")
                        if j_type:
                            job_counts[j_type] += 1
                            total_jobs += 1
                            
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipped invalid JSON line: {e}")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Output Results
    print("\n" + "="*30)
    print(f"   JOB TYPE ANALYSIS")
    print("="*30)
    print(f"Episodes Processed : {episodes_count}")
    print(f"Total Jobs         : {total_jobs}")
    print("-" * 30)
    
    if total_jobs == 0:
        print("No jobs found to analyze.")
        return

    # Calculate and print percentages for each type found (sorted alphabetically)
    for j_type in sorted(job_counts.keys()):
        count = job_counts[j_type]
        percentage = (count / total_jobs) * 100
        print(f"Type {j_type}: {count:5d} jobs  ({percentage:6.2f}%)")
    print("="*30)

if __name__ == "__main__":
    analyze_job_types(INPUT_FILE)
