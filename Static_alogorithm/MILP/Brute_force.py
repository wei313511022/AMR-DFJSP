import json
import itertools
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
#calculate distance 
def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def check_temporal_conflict(agv_id, candidate_station, completion_time, agv_states):
    """Check temporal conflicts (currently disabled - always returns False)."""
    return False
def load_dispatch_data(filename, batch_index=0):
    """Load job data from dispatch_inbox.jsonl file."""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if batch_index >= len(lines):
        raise IndexError(f"Batch index {batch_index} exceeds available batches ({len(lines)})")
    
    batch_data = json.loads(lines[batch_index])
    print(f"=== Batch {batch_index}: {batch_data['dispatch_time']:.1f}s, {len(batch_data['jobs'])} jobs ===")
    
    job_types, job_proc, job_machines = {}, {}, {}
    type_proc_time = {'A': 10, 'B': 15, 'C': 20}
    
    for job in batch_data['jobs']:
        jid = job['jid']
        job_type = job['type']
        proc_time = type_proc_time.get(job_type, job['proc_time'])
        machine_idx = job['station'] - 1
        
        job_types[jid] = job_type
        job_proc[jid] = proc_time
        job_machines[jid] = machine_idx
        print(f"  Job{jid}: {job_type}({proc_time}s) -> M{job['station']}")
    
    return job_types, job_proc, job_machines, batch_data['dispatch_time']
def dispatch_scheduler_flexible(job_types_dict, job_proc_dict, job_machines_dict, agv_pos, mach_pos, start_time=0):
    """Brute force scheduler: enumerate all AGV-job assignments to find optimal makespan."""
    jobs = list(job_types_dict.keys())
    agvs = list(range(len(agv_pos)))
    
    print(">>> Starting scheduler (brute force)")
    start_calc = time.time()
    
    best_makespan = float('inf')
    best_assignment = best_schedule = None
    all_results = []
    
    for combo_num, agv_assignment in enumerate(itertools.product(agvs, repeat=len(jobs)), 1):
        job_assignments = {j: (agv_assignment[jobs.index(j)], job_machines_dict[j]) for j in jobs}
        makespan, schedule = calculate_makespan_flexible_dispatch(
            jobs, job_assignments, job_proc_dict, agv_pos, mach_pos, start_time)
        
        all_results.append({
            'combination': combo_num,
            'assignment': agv_assignment,
            'makespan': makespan,
            'job_assignments': job_assignments.copy()
        })
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_assignment = job_assignments.copy()
            best_schedule = schedule.copy()
    
    print(f">>> Completed: {len(all_results)} combinations in {time.time() - start_calc:.3f}s")
    plot_dispatch_makespan_chart(all_results, best_makespan)
    
    return best_makespan, best_assignment, best_schedule, None

def calculate_makespan_flexible_dispatch(jobs, job_assignments, job_proc, agv_pos, mach_pos, start_time=0):
    """Simulate execution and calculate makespan for a given assignment."""
    agvs = list(range(len(agv_pos)))
    agv_states = {a: {'current_location': agv_pos[a], 'current_time': start_time, 'schedule': []} 
                  for a in agvs}
    
    # Group jobs by AGV
    agv_jobs = {a: [] for a in agvs}
    for j in jobs:
        agv_jobs[job_assignments[j][0]].append(j)
    
    station_schedule = {pos: [] for pos in agv_pos}
    
    # Create task list with priority: distance + processing time
    all_tasks = []
    for a in agvs:
        for idx, j in enumerate(agv_jobs[a]):
            all_tasks.append({
                'job': j, 'agv': a, 'machine': job_assignments[j][1],
                'task_index': idx, 'is_last_task': idx == len(agv_jobs[a]) - 1,
                'proc_time': job_proc[j]
            })
    all_tasks.sort(key=lambda t: manhattan(agv_pos[t['agv']], mach_pos[t['machine']]) + t['proc_time'])
    # Execute tasks in priority order
    for task in all_tasks:
        j, a, m = task['job'], task['agv'], task['machine']
        current_pos = agv_states[a]['current_location']
        current_time = agv_states[a]['current_time']
        
        # Go to machine and process job
        machine_pos = mach_pos[m]
        travel_to_machine = manhattan(current_pos, machine_pos)
        arrival_at_machine = current_time + travel_to_machine
        job_completion = arrival_at_machine + job_proc[j]
        
        # Select return station
        best_station = find_optimal_station_dispatch(machine_pos, job_completion, a, task['is_last_task'],
                                           agv_states, station_schedule, agv_pos, all_tasks, job_assignments, mach_pos, task)
        travel_to_station = manhattan(machine_pos, best_station)
        station_arrival = job_completion + travel_to_station
        
        # Record task
        agv_states[a]['schedule'].append({
            'job': j, 'start_location': current_pos, 'machine_location': machine_pos,
            'end_location': best_station, 'start_time': current_time,
            'arrival_at_machine': arrival_at_machine, 'actual_start_time': arrival_at_machine,
            'job_completion': job_completion, 'total_completion': station_arrival,
            'travel_to_machine': travel_to_machine, 'travel_to_station': travel_to_station,
            'is_last_task': task['is_last_task']
        })
        
        # Update AGV state
        agv_states[a]['current_location'] = best_station
        agv_states[a]['current_time'] = station_arrival

    # After all tasks are scheduled, re-mark the true last task
    # for each AGV based on time order in its schedule. Only the
    # final return of this last task should be treated as "final".
    for a in agvs:
        if agv_states[a]['schedule']:
            last_idx = len(agv_states[a]['schedule']) - 1
            for idx, task in enumerate(agv_states[a]['schedule']):
                task['is_last_task'] = (idx == last_idx)

    # Calculate final makespan:
    # use the latest processing completion time across all jobs,
    # ignoring the final return legs to stations, and measure it
    # relative to the batch dispatch time (start_time).
    latest_processing_completion = max(
        task['job_completion']
        for a in agvs
        for task in agv_states[a]['schedule']
    )
    makespan = latest_processing_completion - start_time
    
    return makespan, agv_states

def get_machine_available_time(machine_schedule, request_time):
    """
    Machine availability function.
    In the current problem setting, a single machine is allowed to
    process multiple jobs at the same time, so the machine never
    delays a job because of capacity. We keep this function for
    compatibility, but simply return the requested time.
    """
    return request_time

def find_optimal_station_dispatch(machine_pos, completion_time, current_agv, is_last_task,
                                agv_states, station_schedule, agv_pos, all_tasks, job_assignments, mach_pos,
                                current_task):
    """
    Select the best AGV station after finishing a job.

    Original logic only looked at the return-to-station time and
    station distribution. Here we additionally look ahead to the
    *next* job of this AGV (if any) and include the distance from
    candidate station to the next job's machine, so that the chosen
    station also shortens the next trip.
    """

    # Determine the next job for this AGV (if any)
    next_machine_pos = None
    if not is_last_task:
        current_index = current_task.get('task_index', None)
        if current_index is not None:
            # Among all tasks of this AGV, find the one with the
            # smallest task_index greater than current_index.
            next_candidates = [
                t for t in all_tasks
                if t['agv'] == current_agv and t['task_index'] > current_index
            ]
            if next_candidates:
                next_task = min(next_candidates, key=lambda t: t['task_index'])
                next_job = next_task['job']
                _, next_machine_idx = job_assignments[next_job]
                next_machine_pos = mach_pos[next_machine_idx]

    candidates = []
    
    # Evaluate all possible stations
    for pos in agv_pos:
        try:
            # Return from current machine to this station
            travel_back = manhattan(machine_pos, pos)
            earliest_arrival = completion_time + travel_back
            
            station_available_time = get_station_available_time(station_schedule[pos], earliest_arrival)
            actual_arrival = max(earliest_arrival, station_available_time)
            
            # Forward-looking distance from this station to the next machine (if any)
            next_travel_time = 0
            if next_machine_pos is not None:
                next_travel_time = manhattan(pos, next_machine_pos)
            
            # Calculate various scoring factors
            makespan_impact = calculate_makespan_impact_dispatch(pos, actual_arrival, agv_states, all_tasks,
                                                               job_assignments, mach_pos, agv_pos)
            
            distance_score = travel_back  # Return distance
            distribution_score = calculate_distribution_score_dispatch(pos, actual_arrival, agv_states, agv_pos)
            
            # Check temporal conflicts (currently disabled, but kept for compatibility)
            has_temporal_conflict = check_temporal_conflict(current_agv, pos, completion_time, agv_states)
            conflict_penalty = 1000 if has_temporal_conflict else 0
            
            # Station wait time
            wait_time = actual_arrival - earliest_arrival
            
            candidates.append({
                'position': pos,
                'arrival_time': actual_arrival,
                'makespan_impact': makespan_impact,
                'distance_score': distance_score,
                'next_travel_time': next_travel_time,
                'distribution_score': distribution_score,
                'wait_time': wait_time,
                'conflict_penalty': conflict_penalty,
                # Heavier weight on next_travel_time so that stations
                # closer to the next machine are preferred.
                'total_score': (
                    makespan_impact * 1000 +
                    distance_score * 10 +
                    next_travel_time * 10 +
                    distribution_score +
                    wait_time +
                    conflict_penalty
                )
            })
        except Exception as e:
            print(f"Error processing station {pos}: {e}")
            print(f"machine_pos: {machine_pos}, completion_time: {completion_time}")
            print(f"agv_pos: {agv_pos}")
            print(f"station_schedule keys: {list(station_schedule.keys())}")
            raise e
    
    # Select best candidate
    candidates.sort(key=lambda x: x['total_score'])
    
    # Prioritize conflict-free options
    no_conflict_candidates = [c for c in candidates if c['conflict_penalty'] == 0]
    if no_conflict_candidates:
        return no_conflict_candidates[0]['position']
    else:
        return candidates[0]['position']

def calculate_makespan_impact_dispatch(station_pos, arrival_time, agv_states, all_tasks,
                                     job_assignments, mach_pos, agv_pos):
    return arrival_time  # Simplified: directly use arrival time as impact factor

def calculate_distribution_score_dispatch(station_pos, arrival_time, agv_states, agv_pos):
    base_score = 50  # Base score
    
    # Count usage statistics for each station
    station_occupancy = {pos: 0 for pos in agv_pos}
    
    for agv_id, agv_state in agv_states.items():
        if agv_state['schedule']:
            # Get endpoint station of last task
            last_task = agv_state['schedule'][-1]
            end_pos = last_task['end_location']
            # Only count endpoints that are actual stations
            if end_pos in station_occupancy:
                station_occupancy[end_pos] += 1
    
    # If target station is used less, give better score
    current_usage = station_occupancy.get(station_pos, 0)
    distribution_bonus = max(0, 3 - current_usage) * 10
    
    return base_score - distribution_bonus

def get_station_available_time(station_schedule, request_time):
    """
    Station availability function.
    Multiple AGVs are allowed to use the same station simultaneously,
    so we do not delay arrivals because of station occupancy.
    """
    return request_time

def plot_dispatch_makespan_chart(all_results, best_makespan):
    """Plot makespan trend chart for all combinations."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    combinations = [r['combination'] for r in all_results]
    makespans = [r['makespan'] for r in all_results]
    mean_ms, std_ms = np.mean(makespans), np.std(makespans)
    best_indices = [i for i, ms in enumerate(makespans) if ms == best_makespan]
    
    # Plot trend line and optimal points
    ax.plot(combinations, makespans, 'o-', color='steelblue', linewidth=1.2, 
           markersize=4, alpha=0.8, label='Makespan')
    for i in best_indices:
        ax.plot(combinations[i], makespans[i], '*', markersize=12, color='red',
               markeredgecolor='darkred', label='Optimal' if i == best_indices[0] else "", zorder=5)
    
    # Reference lines
    ax.axhline(best_makespan, color='red', linestyle='-', alpha=0.7, linewidth=2,
              label=f'Best: {best_makespan:.1f}s')
    ax.axhline(mean_ms, color='orange', linestyle='--', alpha=0.7, linewidth=2,
              label=f'Avg: {mean_ms:.1f}s')
    
    # Labels and stats
    ax.set_xlabel('Combination', fontsize=12)
    ax.set_ylabel('Makespan (s)', fontsize=12)
    ax.set_title('AGV Assignment Combinations - Makespan Trend', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    stats_text = f'Combinations: {len(all_results)}\nBest: {best_makespan:.1f}s\n' \
                f'Avg: {mean_ms:.1f}s\nStd: {std_ms:.1f}s\nOptimal: {len(best_indices)}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, va='top', ha='right',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9), fontsize=9)
    
    plt.tight_layout()
    plt.savefig('dispatch_makespan_trend.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Chart saved: Best={best_makespan:.1f}s, Avg={mean_ms:.1f}s (±{std_ms:.1f}), "
          f"Optimal={len(best_indices)}/{len(all_results)}")




def generate_gantt_chart(agv_states, assignment, job_types_dict, job_proc_dict,job_machines_dict, batch_info, output_prefix='gantt_chart'):
    batch_idx, dispatch_time, num_jobs = batch_info 
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False    
    fig, ax = plt.subplots(figsize=(16, 8))
    # Define color scheme
    colors = {
        'transport': '#3498db',    # Blue - Transport time
        'processing': '#e74c3c',   # Red - Processing time
        'return': '#f39c12',       # Orange - Return time
        'A': '#2ecc71',            # Green - Type A tasks
        'B': '#9b59b6',            # Purple - Type B tasks
        'C': '#e67e22'             # Orange - Type C tasks
    }
    # AGV numbers and Y-axis positions
    agv_positions = {}
    y_positions = []
    agv_labels = []
    active_agvs = sorted([agv_id for agv_id in agv_states.keys() if agv_states[agv_id]['schedule']])
    for i, agv_id in enumerate(active_agvs):
        agv_positions[agv_id] = i
        y_positions.append(i)
        agv_labels.append(f'AGV{agv_id+1}')
    time_offset = dispatch_time
    max_total_time = 0.0   
    max_proc_time = 0.0    
    for agv_id in active_agvs:
        y_pos = agv_positions[agv_id]
        for task in agv_states[agv_id]['schedule']:
            job_id = task['job']
            job_type = job_types_dict[job_id]
            machine_id = job_machines_dict[job_id]
            # Task time segments (shifted so that dispatch_time -> 0)
            start_time = task['start_time'] - time_offset
            transport_end = task['arrival_at_machine'] - time_offset
            processing_end = task['job_completion'] - time_offset
            return_end = task['total_completion'] - time_offset
            # 1. Transport to machine time (blue)
            transport_duration = transport_end - start_time
            if transport_duration > 0:
                rect1 = Rectangle((start_time, y_pos-0.4), transport_duration, 0.8, 
                                facecolor=colors['transport'], alpha=0.7, 
                                edgecolor='black', linewidth=0.5)
                ax.add_patch(rect1)
                # Annotate transport time
                if transport_duration >= 2:  # Only annotate in sufficiently wide areas
                    ax.text(start_time + transport_duration/2, y_pos, f'Trans\n{transport_duration:.0f}s', 
                           ha='center', va='center', fontsize=8, fontweight='bold')
            # 2. Processing time (task type color)
            processing_start = task['actual_start_time'] - time_offset
            processing_duration = processing_end - processing_start
            if processing_duration > 0:
                rect2 = Rectangle((processing_start, y_pos-0.4), processing_duration, 0.8, 
                                facecolor=colors[job_type], alpha=0.8, 
                                edgecolor='black', linewidth=0.5)
                ax.add_patch(rect2)
                # Annotate task information
                ax.text(processing_start + processing_duration/2, y_pos, 
                       f'Job{job_id}\n{job_type}({processing_duration:.0f}s)\nM{machine_id+1}', 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            # 3. Return time (orange) - only for non-last tasks
            is_last_task = task.get('is_last_task', False)
            return_duration = return_end - processing_end
            if return_duration > 0 and not is_last_task:
                rect3 = Rectangle((processing_end, y_pos-0.4), return_duration, 0.8, 
                                facecolor=colors['return'], alpha=0.7, 
                                edgecolor='black', linewidth=0.5)
                ax.add_patch(rect3)
                # Annotate return time
                if return_duration >= 2:  # Only annotate in sufficiently wide areas
                    ax.text(processing_end + return_duration/2, y_pos, f'Return\n{return_duration:.0f}s', 
                           ha='center', va='center', fontsize=8, fontweight='bold')
            max_total_time = max(max_total_time, return_end)
            max_proc_time = max(max_proc_time, processing_end)
    
    ax.set_ylim(-0.5, len(active_agvs) - 0.5)
    ax.set_xlim(0, max_total_time + 10)
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('AGV', fontsize=12, fontweight='bold')
    ax.set_title(f'AGV Task Allocation Gantt Chart (Final Return Not Counted in Makespan) - Batch {batch_idx+1}\n'
                f'Jobs: {num_jobs}, Makespan: {max_proc_time:.1f}s', 
                fontsize=14, fontweight='bold')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(agv_labels)
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=colors['transport'], alpha=0.7, label='Transport Time'),
        Rectangle((0, 0), 1, 1, facecolor=colors['A'], alpha=0.8, label='Type A Processing'),
        Rectangle((0, 0), 1, 1, facecolor=colors['B'], alpha=0.8, label='Type B Processing'),
        Rectangle((0, 0), 1, 1, facecolor=colors['C'], alpha=0.8, label='Type C Processing'),
        Rectangle((0, 0), 1, 1, facecolor=colors['return'], alpha=0.7, label='Return Time (Non-last Tasks)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    # Save chart
    output_filename = f'{output_prefix}_batch_{batch_idx+1}.png'
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Gantt chart saved as '{output_filename}'")
    # Return statistics information
    total_transport_time = sum(
        task['arrival_at_machine'] - task['start_time'] 
        for agv_id in active_agvs 
        for task in agv_states[agv_id]['schedule']
    )
    total_processing_time = sum(
        task['job_completion'] - task['actual_start_time'] 
        for agv_id in active_agvs 
        for task in agv_states[agv_id]['schedule']
    )
    total_return_time = sum(
        task['total_completion'] - task['job_completion'] 
        for agv_id in active_agvs 
        for task in agv_states[agv_id]['schedule']
        if not task.get('is_last_task', False)  # Exclude final returns
    )
    
    return {
        'batch_idx': batch_idx,
        'makespan': max_proc_time,
        'total_transport_time': total_transport_time,
        'total_processing_time': total_processing_time,
        'total_return_time': total_return_time,
        'chart_filename': output_filename
    }

def output_schedule_results(assignment, job_types_dict, job_proc_dict, job_machines_dict, 
                           output_filename='schedule_outbox.jsonl'):

    import time
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        for job_id in sorted(assignment.keys()):
            agv_id, machine_id = assignment[job_id]
            
            record = {
                "generated_at": time.time(),  # Current timestamp
                "amr": agv_id + 1,  # AGV number (starting from 1)
                "jid": job_id,      # Job ID
                "type": job_types_dict[job_id],  # Job type
                "proc_time": job_proc_dict[job_id],  # Processing time
                "station": str(machine_id + 1)  # Workstation number (string format, starting from 1)
            }
            
            # Write one line in JSON format
            f.write(json.dumps(record) + '\n')
    
    print(f"\n📋 Schedule results saved to '{output_filename}'")
    print(f"📊 Total jobs processed: {len(assignment)}")

def process_all_dispatch_batches(filename, agv_pos, mach_pos):

    print("=== Processing All Scheduling Batches and Generating Gantt Charts ===")
    
    # Read all batches
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_assignments = {}
    all_job_types = {}
    all_job_proc = {}
    all_job_machines = {}
    total_jobs = 0
    max_makespan = 0
    batch_statistics = []
    
    print(f"Found {len(lines)} scheduling batches")
    
    # Process each batch
    for batch_idx, line in enumerate(lines):
        batch_data = json.loads(line)
        dispatch_time = batch_data['dispatch_time']
        jobs = batch_data['jobs']
        
        print(f"\n--- Processing Batch {batch_idx + 1}: {len(jobs)} tasks (time: {dispatch_time:.1f}s) ---")
        
        # Convert batch data to scheduler format
        job_types_dict = {}
        job_proc_dict = {}
        job_machines_dict = {}
        
        for job in jobs:
            jid = job['jid']
            job_type = job['type']
            
            # Set processing time based on type
            if job_type == 'A':
                proc_time = 10
            elif job_type == 'B':
                proc_time = 15
            elif job_type == 'C':
                proc_time = 20
            else:
                proc_time = job['proc_time']
            
            machine_index = job['station'] - 1
            
            job_types_dict[jid] = job_type
            job_proc_dict[jid] = proc_time
            job_machines_dict[jid] = machine_index
        
        # Execute scheduling
        if job_types_dict:  # Only execute when there are tasks in the batch
            result = dispatch_scheduler_flexible(job_types_dict, job_proc_dict, job_machines_dict, 
                                               agv_pos, mach_pos, start_time=dispatch_time)
            
            makespan, assignment, agv_states, travel_time = result
            
            print(f"   Batch {batch_idx + 1} makespan: {makespan:.1f}s")
            
            # Generate Gantt chart for this batch
            batch_info = (batch_idx, dispatch_time, len(jobs))
            gantt_stats = generate_gantt_chart(agv_states, assignment, job_types_dict, 
                                             job_proc_dict, job_machines_dict, batch_info)
            batch_statistics.append(gantt_stats)
            
            # Merge results
            all_assignments.update(assignment)
            all_job_types.update(job_types_dict)
            all_job_proc.update(job_proc_dict)
            all_job_machines.update(job_machines_dict)
            total_jobs += len(job_types_dict)
            max_makespan = max(max_makespan, makespan)
    
    print(f"\n=== All Batches Processing Completed ===")
    print(f"Total tasks: {total_jobs}")
    print(f"Maximum makespan: {max_makespan:.1f}s")
    print(f"Generated {len(batch_statistics)} Gantt charts")
    
    return max_makespan, all_assignments, (all_job_types, all_job_proc, all_job_machines), batch_statistics

# ===============================
#  Main Program
# ===============================

if __name__ == "__main__":
    # Set AGV and machine positions
    agv_pos = [(0, 0), (0, 5), (0, 9)]  # AGV1, AGV2, AGV3
    mach_pos = [(9, 2), (9, 5), (9, 7)]  # M1, M2, M3
    
    print("=== Dispatch Data Processor - Complete Batch Processing ===")
    print("AGV scheduling system based on dispatch_inbox.jsonl file")
    
    try:
        print(f"\n=== AGV Positions ===")
        for a, pos in enumerate(agv_pos):
            print(f"AGV{a+1}: {pos}")
        
        print(f"\n=== Machine Positions ===")
        for m, pos in enumerate(mach_pos):
            print(f"M{m+1}: {pos}")
        
        print(f"\n=== Scheduling Constraints Description ===")
        print("- Each AGV can transport only one job at a time")
        print("- Machines can process multiple jobs in parallel (no capacity limit)")
        print("- After processing, AGVs return to any AGV station to pick up the next job")
        print("- Multiple AGVs are allowed to use the same station simultaneously")
        print("- Objective: Minimize makespan (ignoring final return legs of all AGVs)")
        print("- Process all batches in dispatch_inbox.jsonl")
        
        # Process all dispatch batches
        total_makespan, all_assignments, job_info, batch_stats = process_all_dispatch_batches(
            'dispatch_inbox.jsonl', agv_pos, mach_pos)
        
        all_job_types, all_job_proc, all_job_machines = job_info
        
        # Output scheduling results to schedule_outbox.jsonl format
        output_schedule_results(all_assignments, all_job_types, all_job_proc, all_job_machines)
        
        # Display final statistics
        print(f"\n=== Final Scheduling Statistics ===")
        print(f"Total processed tasks: {len(all_assignments)}")
        print(f"Overall maximum makespan: {total_makespan:.1f}s")
        
        # Display Gantt chart statistics summary
        if batch_stats:
            print(f"\n=== Gantt Chart Statistics Summary ===")
            total_transport = sum(stat['total_transport_time'] for stat in batch_stats)
            total_processing = sum(stat['total_processing_time'] for stat in batch_stats)
            total_return = sum(stat['total_return_time'] for stat in batch_stats)
            avg_makespan = sum(stat['makespan'] for stat in batch_stats) / len(batch_stats)
            
            print(f"Average batch makespan: {avg_makespan:.1f}s")
            print(f"Total transport time: {total_transport:.1f}s")
            print(f"Total processing time: {total_processing:.1f}s")
            print(f"Total return time: {total_return:.1f}s")
            print(f"Generated Gantt chart files: {[stat['chart_filename'] for stat in batch_stats]}")
        
        # Analyze AGV work assignment
        agv_job_count = {0: 0, 1: 0, 2: 0}
        for job_id, (agv_id, machine_id) in all_assignments.items():
            agv_job_count[agv_id] += 1
        
        print(f"\n=== AGV Task Assignment Statistics ===")
        for agv_id, count in agv_job_count.items():
            percentage = (count / len(all_assignments)) * 100
            print(f"AGV{agv_id+1}: {count} tasks ({percentage:.1f}%)")
        
        # Analyze machine usage statistics
        machine_job_count = {0: 0, 1: 0, 2: 0}
        for job_id, (agv_id, machine_id) in all_assignments.items():
            machine_job_count[machine_id] += 1
        
        print(f"\n=== Machine Usage Statistics ===")
        for machine_id, count in machine_job_count.items():
            percentage = (count / len(all_assignments)) * 100
            print(f"M{machine_id+1}: {count} tasks ({percentage:.1f}%)")
        
    except FileNotFoundError:
        print("❌ Error: dispatch_inbox.jsonl file not found")
    except IndexError as e:
        print(f"❌ Error: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Error: JSON parsing failed - {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
