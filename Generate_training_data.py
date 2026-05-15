import json
import random

def generate_fjssp_instance():
    # Rule 1: The number of machines is fixed at 6
    num_machines = 6
    
    # Rule 2: The number of jobs can be 10, 15, 20, 25, or 30
    num_jobs = random.choice([10, 15, 20, 25, 30])
    # num_jobs = random.choice([10])
    
    jobs_list = []
    
    for _ in range(num_jobs):
        job = []
        # Rule 3: The number of operations is sampled uniformly from 4 to 8
        num_operations = random.randint(4, 8)
        
        for _ in range(num_operations):
            # Rule 5: Processing time is sampled uniformly from 10 to 99
            # (Note: Kept consistent for the same operation across different machines 
            # to match the structure of your example)
            processing_time = random.randint(10, 99)
            
            # Rule 4: Feasible machine set sampled uniformly from the 6 machines
            # First, decide how many machines are feasible for this operation (1 to 6)
            num_feasible_machines = random.randint(1, num_machines)
            
            # Randomly select which specific machines are feasible
            feasible_machines = random.sample(range(num_machines), num_feasible_machines)
            
            # Create the operation list with machine ID and processing time
            operation = [
                {"machine": machine_id, "processing": processing_time}
                for machine_id in feasible_machines
            ]
            
            job.append(operation)
            
        # Assign a material to the job
        material = random.choice(['A', 'B', 'C'])
        
        # Create job dictionary
        job_dict = {
            "operations": job,
            "material": material
        }
        
        jobs_list.append(job_dict)
        
    # Construct the final dictionary for this instance
    instance = {
        "machines": num_machines,
        "jobs": jobs_list
    }
    
    return instance

def create_jsonl_dataset(filename, num_instances):
    """Generates a specific number of instances and saves them in JSONL format."""
    with open(filename, 'w') as f:
        for _ in range(num_instances):
            instance = generate_fjssp_instance()
            # json.dumps converts the dictionary to a JSON string on a single line
            f.write(json.dumps(instance) + '\n')
            
    print(f"Successfully generated {num_instances} instances and saved to {filename}")

# --- Execute the generation ---
if __name__ == "__main__":
    # Specify how many training instances you want to generate
    TOTAL_INSTANCES_TO_GENERATE = 100
    OUTPUT_FILENAME = "fjssp_training_dataset.jsonl"
    
    create_jsonl_dataset(OUTPUT_FILENAME, TOTAL_INSTANCES_TO_GENERATE)