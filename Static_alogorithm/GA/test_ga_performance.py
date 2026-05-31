import csv
import random
import time
import matplotlib.pyplot as plt
import GA as ga  # Import your GA module

def run_test_suite():
    """
    Runs the GA with different configurations to test performance.
    """
    # --- Test Configurations ---
    # You can adjust these lists to test different values
    ROUTING_ITERS_LIST = [1000]
    COLLISION_ROUTING_ITERS_LIST = [20] # Fixed for Pop/Gen test. Original: [0, 1, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    POPULATION_SIZE_LIST = [20, 50, 100, 150, 200]
    GENERATIONS_LIST = [20, 50, 100, 150, 200, 250, 300]
    
    # --- Load a consistent job set for fair comparison ---
    dispatch_events = ga.load_dispatch_events()
    events_to_test = []
    if not dispatch_events:
        print("No dispatch file found or file is empty. Generating 10 random job sets for test.")
        for i in range(10):
            events_to_test.append({'index': f"random_{i}", 'jobs': ga.make_jobs()})
    else:
        # Use up to 10 events from the file for a consistent benchmark
        count = min(10, len(dispatch_events))
        events_to_test = dispatch_events[:count]
        print(f"Using first {count} events from dispatch file for testing.")

    # --- Results Storage ---
    results = []
    
    # --- Run the tests ---
    for r_iters in ROUTING_ITERS_LIST:
        for c_iters in COLLISION_ROUTING_ITERS_LIST:
            for pop_size in POPULATION_SIZE_LIST:
                for gens in GENERATIONS_LIST:
                    print(f"\n--- Testing: routing={r_iters}, collision={c_iters}, pop={pop_size}, gen={gens} ---")
                    
                    # Reset random seed for fair comparison between configurations
                    random.seed(42)
                    
                    # Temporarily set the global variables in the GA module for this run
                    ga.routing_iters = r_iters
                    ga.collision_routing_iters = c_iters
                    ga.POPULATION_SIZE = pop_size
                    ga.GENERATIONS = gens
                    
                    total_makespan = 0.0
                    total_solve_time = 0.0
                    
                    for event in events_to_test:
                        jobs = event['jobs']
                        start_time = time.perf_counter()
                        
                        # Run the main evolution function from your GA module
                        best_individual, _ = ga.evolve(jobs)
                        
                        solve_dur = time.perf_counter() - start_time
                        
                        # Calculate the final makespan using the collision-aware decoder
                        availability, _, _, _, _ = ga.decode_schedule(best_individual, jobs, need_log=False, check_collision=True)
                        makespan = max(availability.values()) if availability else 0.0

                        total_makespan += makespan
                        total_solve_time += solve_dur

                    avg_makespan = total_makespan / len(events_to_test)
                    avg_solve_time = total_solve_time / len(events_to_test)
                    
                    print(f"  -> Avg Makespan: {avg_makespan:.2f}s, Avg Solve Time: {avg_solve_time:.4f}s")
                    
                    results.append({
                        "routing_iters": r_iters,
                        "collision_iters": c_iters,
                        "pop_size": pop_size,
                        "generations": gens,
                        "makespan": avg_makespan,
                        "solve_time": avg_solve_time
                    })

    # --- Save results to CSV ---
    output_filename = "ga_performance_summary.csv"
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["routing_iters", "collision_iters", "pop_size", "generations", "makespan", "solve_time"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPerformance results saved to {output_filename}")
    
    # --- Plotting ---
    # Determine what to plot on X-axis based on what varies
    x_var = "collision_iters"
    if len(GENERATIONS_LIST) > 1:
        x_var = "generations"
    elif len(POPULATION_SIZE_LIST) > 1:
        x_var = "pop_size"
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Makespan
    for r_iters in ROUTING_ITERS_LIST:
        # Filter results for this routing_iters
        subset = [r for r in results if r['routing_iters'] == r_iters]
        
        # We need to group by other fixed parameters to draw lines
        # For simplicity, let's group by (collision_iters, pop_size) if x is generations
        # or (collision_iters, generations) if x is pop_size
        
        # Create a key for grouping lines
        def get_group_key(r):
            if x_var == "generations":
                return (r['collision_iters'], r['pop_size'])
            elif x_var == "pop_size":
                return (r['collision_iters'], r['generations'])
            else:
                return (r['pop_size'], r['generations'])

        groups = {}
        for r in subset:
            key = get_group_key(r)
            if key not in groups: groups[key] = []
            groups[key].append(r)
            
        for key, group_data in groups.items():
            # Sort by x_var
            group_data.sort(key=lambda x: x[x_var])
            x_vals = [r[x_var] for r in group_data]
            makespan_vals = [r['makespan'] for r in group_data]
            time_vals = [r['solve_time'] for r in group_data]
            
            # Label generation
            if x_var == "generations":
                lbl = f"C={key[0]}, Pop={key[1]}"
            elif x_var == "pop_size":
                lbl = f"C={key[0]}, Gen={key[1]}"
            else:
                lbl = f"Pop={key[0]}, Gen={key[1]}"
            
            ax1.plot(x_vals, makespan_vals, marker='o', linestyle='-', label=lbl)
            ax2.plot(x_vals, time_vals, marker='s', linestyle='--', label=lbl)

    ax1.set_title(f'GA Performance: Avg Makespan vs. {x_var}')
    ax1.set_ylabel('Avg Makespan (s)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.set_title(f'GA Performance: Avg Solve Time vs. {x_var}')
    ax2.set_xlabel(x_var)
    ax2.set_ylabel('Avg Solve Time (s)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_filename = "ga_performance_plot.png"
    plt.savefig(plot_filename)
    print(f"Performance plot saved to {plot_filename}")
    plt.show()

if __name__ == "__main__":
    run_test_suite()
