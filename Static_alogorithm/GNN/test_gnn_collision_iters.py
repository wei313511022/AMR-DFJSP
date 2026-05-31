import subprocess
import csv
import matplotlib.pyplot as plt
import os

def run_tests():
    # Defined values of collision_iters to test
    iters_to_test = [0, 100, 500, 1000, 2000]
    
    makespans = []
    compute_times = []
    
    # Path to the GNN script
    gnn_script = "GNN.py"
    
    for iters in iters_to_test:
        print("==================================================")
        print(f"Testing with collision_iters = {iters}...")
        
        output_csv = f"results_iters_{iters}.csv"
        
        # Build command
        cmd = [
            "python", gnn_script,
            "--collision_iters", str(iters),
            "--output_csv", output_csv
        ]
        
        # Run the GNN script. We use check=True to raise an exception if it fails
        # but we also inherit the environment variables in case DISPATCH_EVENT_INDEX is set
        subprocess.run(cmd, check=True)
        
        # Read the output CSV to extract the makespan and computation time
        total_makespan = 0.0
        total_time = 0.0
        count = 0
        
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_makespan += float(row['Makespan'])
                total_time += float(row['Computation_Time'])
                count += 1
                
        if count > 0:
            avg_makespan = total_makespan / count
            avg_time = total_time / count
            makespans.append(avg_makespan)
            compute_times.append(avg_time)
            print(f"Result for iters={iters}: Avg Makespan={avg_makespan:.2f}, Avg Time={avg_time:.4f}")
        else:
            makespans.append(0)
            compute_times.append(0)
            print(f"No results found for iters={iters}")
            
    # Plotting the results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Collision Routing Iters')
    ax1.set_ylabel('Makespan (s)', color=color)
    ax1.plot(iters_to_test, makespans, marker='o', color=color, label='Makespan')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Computation Time (s)', color=color)
    ax2.plot(iters_to_test, compute_times, marker='s', color=color, linestyle='--', label='Computation Time')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('GNN Performance vs Collision Routing Iters')
    plt.grid(True)
    
    plot_file = 'collision_iters_performance.png'
    plt.savefig(plot_file)
    print(f"\nSaved performance plot to {plot_file}")
    
    # Print summary table
    print("\nSummary Table:")
    print(f"{'Iters':<10} | {'Avg Makespan':<15} | {'Avg Comp Time':<15}")
    print("-" * 45)
    for iters, ms, ct in zip(iters_to_test, makespans, compute_times):
        print(f"{iters:<10} | {ms:<15.2f} | {ct:<15.4f}")

if __name__ == "__main__":
    run_tests()
