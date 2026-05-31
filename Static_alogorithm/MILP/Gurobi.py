# ...existing code...
import json
import time
import gurobipy as gp
from gurobipy import GRB

# try import plotting
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# Print Gurobi version
try:
    print("Gurobi version:", gp.gurobi.version())
except Exception:
    print("Gurobi version: (unknown)")

# GLOBAL
GRID_SIZE = 10
M_BIG = 100
P_NODES = {1, 6, 10}                   # pickup candidate nodes (grid indices)
# station id -> delivery node (grid idx) (station 1:(9,2)->93, 2:(9,5)->96, 3:(9,7)->98)
JSON_STATION_MAPPING = {1: 93, 2: 96, 3: 98}
M_SET = range(1, 4)                    # 3 AGVs
S_m = {1: 1, 2: 6, 3: 10}              # AGV start nodes (grid indices)

def calculate_distance(node1, node2, grid_size=GRID_SIZE):
    """Manhattan distance on GRID_SIZE x GRID_SIZE (1..100 indices)."""
    r1, c1 = (node1 - 1) // grid_size, (node1 - 1) % grid_size
    r2, c2 = (node2 - 1) // grid_size, (node2 - 1) % grid_size
    return abs(r1 - r2) + abs(c1 - c2)

def load_dispatch_events(filename: str):
    """Load JSONL dispatch file. Return list of events: {'time': float, 'jobs': [job,...]}.
    Each job will include 'arrival_time' == dispatch_time."""
    events = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                dispatch_time = float(data.get('dispatch_time', 0.0))
                jobs_raw = data.get('jobs', [])
                jobs = []
                for j in jobs_raw:
                    station = j.get('station')
                    g_l = JSON_STATION_MAPPING.get(station)
                    if g_l is None:
                        continue
                    jobs.append({
                        'jid': int(j.get('jid')),
                        'type': j.get('type'),
                        'proc_time': float(j.get('proc_time', 0.0)),
                        'station': int(station),
                        'g_l': int(g_l),
                        # 每個 event 獨立處理，從時間 0 開始（忽略 dispatch_time）
                        'arrival_time': 0.0
                    })
                events.append({'time': dispatch_time, 'jobs': jobs})
    except FileNotFoundError:
        print(f"Error: file not found: {filename}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: JSON decode error: {e}")
        return []
    events.sort(key=lambda x: x['time'])
    return events

def solve_vrp_from_jobs(jobs, time_limit=360):
    """Build & solve model for one dispatch event (jobs list). Returns dict or None."""
    if not jobs:
        return None

    L_REAL = [j['jid'] for j in jobs]
    n_tasks = len(L_REAL)
    L_SET = range(1, n_tasks + 1)
    VIRTUAL_END = n_tasks + 1
    L_PRIME = range(0, VIRTUAL_END + 1)

    # map index -> job data
    L_REAL_MAP = {idx: L_REAL[idx - 1] for idx in L_SET}
    TASK_DATA = {}
    for idx in L_SET:
        job = next((j for j in jobs if j['jid'] == L_REAL_MAP[idx]), None)
        TASK_DATA[idx] = {
            'E_l': float(job['proc_time']),
            'g_l': int(job['g_l']),
            'type': job.get('type', '?'),
            'arrival_time': float(job.get('arrival_time', 0.0))
        }

    # Compute a tighter, event-specific Big-M to improve numerical behaviour
    delivery_nodes = [TASK_DATA[i]['g_l'] for i in L_SET]
    D_max = 0
    for p in P_NODES:
        for g in delivery_nodes:
            D_max = max(D_max, calculate_distance(p, g))
    S_max = 0
    for s_node in S_m.values():
        for p in P_NODES:
            S_max = max(S_max, calculate_distance(s_node, p))
    E_sum = sum(TASK_DATA[i]['E_l'] for i in L_SET)
    margin = 5.0
    M_local = float(S_max + (n_tasks * D_max) + E_sum + margin)
    if M_local < 1.0:
        M_local = 1.0
    try:
        print(f"Using local Big-M = {M_local:.2f} (S_max={S_max}, D_max={D_max}, E_sum={E_sum:.2f})")
    except Exception:
        pass

    model = gp.Model(f"Reschedule_event")
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', 0.0)
    model.setParam('OutputFlag', 0)

    # Vars
    Y = model.addVars(L_SET, M_SET, vtype=GRB.BINARY, name="Y_assign")
    W = model.addVars(L_PRIME, L_PRIME, M_SET, vtype=GRB.BINARY, name="W_seq")
    A_P = model.addVars(L_SET, P_NODES, vtype=GRB.BINARY, name="A_pick")
    T_Pick = model.addVars(L_SET, vtype=GRB.CONTINUOUS, lb=0.0, name="T_pick")
    T_Del = model.addVars(L_SET, vtype=GRB.CONTINUOUS, lb=0.0, name="T_del")
    T_End = model.addVars(L_SET, vtype=GRB.CONTINUOUS, lb=0.0, name="T_end")

    # Makespan variable and objective: minimize the maximum completion time (makespan)
    T_makespan = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="T_makespan")
    # link makespan to all end times
    model.addConstrs((T_makespan >= T_End[l] for l in L_SET), name="C_makespan_link")
    model.setObjective(T_makespan, GRB.MINIMIZE)

    # Constraints
    model.addConstrs((gp.quicksum(A_P[l, p] for p in P_NODES) == 1 for l in L_SET), name="C1_one_pick")
    model.addConstrs((gp.quicksum(Y[l, m] for m in M_SET) == 1 for l in L_SET), name="C2_assign")

    # Predecessor/successor and link to Y
    for l in L_SET:
        model.addConstr(gp.quicksum(W[lp, l, m] for lp in L_PRIME if lp != l for m in M_SET) == 1, name=f"C3_pred_{l}")
        model.addConstr(gp.quicksum(W[l, ln, m] for ln in L_PRIME if ln != l for m in M_SET) == 1, name=f"C3_succ_{l}")
        model.addConstrs((W[lp, l, m] <= Y[l, m] for lp in L_PRIME if lp != l for m in M_SET), name=f"C3_link_in_{l}")
        model.addConstrs((W[l, ln, m] <= Y[l, m] for ln in L_PRIME if ln != l for m in M_SET), name=f"C3_link_out_{l}")

    # start/end per AGV (keeps same structure as test.py)
    for m in M_SET:
        model.addConstr(gp.quicksum(W[0, l, m] for l in L_SET) == 1, name=f"C4_start_{m}")
        model.addConstr(gp.quicksum(W[l, VIRTUAL_END, m] for l in L_SET) == 1, name=f"C4_end_{m}")

    # no return to start
    model.addConstrs((W[l, 0, m] == 0 for l in L_SET for m in M_SET), name="C5_no_return_start")

    # execution time
    model.addConstrs((T_End[l] == T_Del[l] + TASK_DATA[l]['E_l'] for l in L_SET), name="C6_exec")

    # arrival time constraint: cannot pick before job arrival
    model.addConstrs((T_Pick[l] >= TASK_DATA[l]['arrival_time'] for l in L_SET), name="C_arrival")

    # transport time to delivery - CORRECTED to use exact pickup choice
    for l in L_SET:
        g_l = TASK_DATA[l]['g_l']
        for p in P_NODES:
            d_pg = calculate_distance(p, g_l)
            # When A_P[l,p]=1, T_Del[l] >= T_Pick[l] + d_pg
            # When A_P[l,p]=0, constraint is relaxed by M_BIG
            model.addConstr(T_Del[l] >= T_Pick[l] + d_pg - M_BIG * (1 - A_P[l, p]), 
                           name=f"C7_trans_{l}_{p}")
            # 額外保險：確保 T_Del >= T_Pick + min_distance（所有 pickup 中的最小距離）
            min_d = min(calculate_distance(pp, g_l) for pp in P_NODES)
            model.addConstr(T_Del[l] >= T_Pick[l] + min_d, name=f"C7_trans_lower_{l}")

    # sequencing gating
    for l in L_SET:
        # between tasks
        for lp in L_SET:
            if lp == l:
                continue
            g_lp = TASK_DATA[lp]['g_l']
            for p in P_NODES:
                d_dp = calculate_distance(g_lp, p)
                model.addConstrs((T_Pick[l] >= T_End[lp] + d_dp - M_local * (2 - W[lp, l, m] - A_P[l, p]) for m in M_SET),
                                 name=f"C8_between_{lp}_{l}_{p}")
        # from starts
        for m in M_SET:
            Snode = S_m[m]
            for p in P_NODES:
                d_sp = calculate_distance(Snode, p)
                model.addConstr(T_Pick[l] >= d_sp - M_local * (2 - W[0, l, m] - A_P[l, p]), name=f"C8_start_{l}_{m}_{p}")

    model.update()
    model.optimize()

    if model.status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} or getattr(model, "SolCount", 0) == 0:
        try:
            model.write(f"debug_readjson_{int(time.time())}.lp")
            print("No feasible solution or no solution count. Model written for inspection.")
        except Exception:
            print("No feasible solution and failed to write model file.")
        return None

    W_vals = model.getAttr('X', W)
    A_vals = model.getAttr('X', A_P)
    T_pick_vals = model.getAttr('X', T_Pick)
    T_del_vals = model.getAttr('X', T_Del)
    T_end_vals = model.getAttr('X', T_End)

    # reconstruct sequences and transport totals, now including pickup node & return distances
    seq_map = {m: [] for m in M_SET}
    total_transport = 0.0
    total_transport_distance = 0.0
    total_return_distance = 0.0

    # Also prepare per-job summary (by real jid)
    job_summaries = {}

    for m in M_SET:
        curr = 0
        steps = 0
        seq = []
        prev_delivery_node = None
        while True:
            if steps > n_tasks + 5:
                break
            found = False
            for j in L_PRIME:
                val = W_vals.get((curr, j, m), 0.0)
                if val >= 0.5:
                    # reached virtual end
                    if j == VIRTUAL_END:
                        curr = j
                        found = True
                        break
                    # j is a task index
                    pick_t = T_pick_vals.get(j, 0.0)
                    del_t = T_del_vals.get(j, 0.0)
                    end_t = T_end_vals.get(j, 0.0)
                    transport = del_t - pick_t
                    proc_time = TASK_DATA[j]['E_l']
                    # determine chosen pickup node (opt var A_vals)
                    chosen_pick = None
                    for p in P_NODES:
                        if A_vals.get((j, p), 0.0) >= 0.5:
                            chosen_pick = p
                            break
                    delivery_node = TASK_DATA[j]['g_l']
                    # compute transport distance (Manhattan) from chosen pickup to delivery node
                    grid_dist = None
                    if chosen_pick is not None and delivery_node is not None:
                        try:
                            grid_dist = float(calculate_distance(chosen_pick, delivery_node))
                        except Exception:
                            grid_dist = None
                    # transport (time) from model values (del - pick)
                    transport_time_val = float(max(0.0, transport))
                    # transport distance (grid) fallback
                    transport_dist = float(grid_dist) if (grid_dist is not None) else 0.0

                    # compute return distance: from previous delivery (or start) to this pickup
                    if prev_delivery_node is None:
                        # from AGV start to pickup
                        start_node = S_m[m]
                        return_dist = calculate_distance(start_node, chosen_pick) if chosen_pick is not None else 0.0
                    else:
                        return_dist = calculate_distance(prev_delivery_node, chosen_pick) if chosen_pick is not None else 0.0
                    total_return_distance += float(return_dist)
                    seq_entry = {
                        'idx': j,
                        'jid': L_REAL_MAP[j],
                        'assigned_agv': int(m),
                        'type': TASK_DATA[j].get('type', '?'),
                        'pickup_node': int(chosen_pick) if chosen_pick is not None else None,
                        'delivery_node': int(delivery_node),
                        'pick_time': float(pick_t),
                        'del_time': float(del_t),
                        'end_time': float(end_t),
                        'transport_time': float(transport_time_val),
                        'transport_distance': float(transport_dist),
                        'proc_time': float(proc_time),
                        'return_dist_to_pickup': float(return_dist)
                    }
                    seq.append(seq_entry)
                    # update job_summaries keyed by real jid
                    job_summaries[seq_entry['jid']] = seq_entry.copy()
                    # update for next
                    prev_delivery_node = delivery_node
                    total_transport += float(transport_time_val)
                    total_transport_distance += float(transport_dist)
                    curr = j
                    found = True
                    break
            if not found:
                break
            if curr == VIRTUAL_END:
                break
            steps += 1
        seq_map[m] = seq

    try:
        makespan = max(float(v) for v in T_end_vals.values()) if T_end_vals else 0.0
    except Exception:
        makespan = 0.0

    return {
        'model': model,
        'sequence_map': seq_map,
        'job_summaries': job_summaries,
        'total_transport_reconstructed': total_transport,
        'total_transport_distance': total_transport_distance,
        'total_return_distance': total_return_distance,
        'makespan': float(makespan),
        'T_end_vals': T_end_vals
    }

def write_results_and_gantt(all_event_results, out_dir='results'):
    """Write each event to separate JSON file and create Gantt charts."""
    import os
    
    # 建立 results 目錄
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # ...existing gantt chart code...
    
    # Write separate JSON file for each event (JSONL format - one job per line)
    for ev_idx, ev in enumerate(all_event_results, start=1):
        event_file = os.path.join(out_dir, f'event_{ev_idx}.jsonl')
        
        try:
            with open(event_file, 'w', encoding='utf-8') as f:
                seq_map = ev.get('sequence_map', {})
                job_summaries = ev.get('job_summaries', {})
                
                for jid in sorted(job_summaries.keys()):
                    job = job_summaries[jid]
                    amr = job.get('assigned_agv', None)
                    if amr is None:
                        continue
                    
                    # Find original station from job data
                    delivery_node = job.get('delivery_node')
                    station = None
                    for st, del_node in JSON_STATION_MAPPING.items():
                        if del_node == delivery_node:
                            station = st
                            break
                    
                    # Output one line per job
                    output_record = {
                        'generated_at': float(ev.get('event_time', 0.0)),
                        'amr': int(amr),
                        'jid': int(jid),
                        'type': job.get('type', '?'),
                        'proc_time': float(job.get('proc_time', 0.0)),
                        'station': str(station) if station else '?',
                        'pickup_node': int(job.get('pickup_node')) if job.get('pickup_node') else None,
                        'pick_time': float(job.get('pick_time', 0.0)),
                        'del_time': float(job.get('del_time', 0.0)),
                        'end_time': float(job.get('end_time', 0.0))
                    }
                    f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            
            print(f"✓ Wrote {event_file}")
        except Exception as e:
            print(f"✗ Failed to write {event_file}: {e}")
    
    # Also write summary file (all events in one JSON)
    summary_file = os.path.join(out_dir, 'summary.json')
    try:
        simple = []
        for ev_idx, ev in enumerate(all_event_results, start=1):
            ev_out = {
                'event_index': ev_idx,
                'event_time': ev['event_time'],
                'n_jobs': ev['n_jobs'],
                'makespan': ev.get('makespan', 0.0),
                'sequences': {str(m): ev.get('sequence_map', {}).get(m, []) for m in M_SET}
            }
            simple.append(ev_out)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(simple, f, indent=2, ensure_ascii=False)
        print(f"✓ Wrote {summary_file}")
    except Exception as e:
        print(f"✗ Failed to write summary: {e}")

def main():
    FILENAME = 'dispatch_inbox_5Jobs_generation.jsonl'
    events = load_dispatch_events(FILENAME)
    if not events:
        print("No dispatch events loaded.")
        return

    all_event_results = []
    print(f"Loaded {len(events)} dispatch events. Processing each event independently...\n")
    for idx, ev in enumerate(events, start=1):
        print(f"=== Event {idx} @ t={ev['time']:.2f}s | jobs={len(ev['jobs'])} ===")
        res = solve_vrp_from_jobs(ev['jobs'])
        if res is None:
            print("  Solver returned no result for this event.\n")
            all_event_results.append({
                'event_time': ev['time'],
                'n_jobs': len(ev['jobs']),
                'error': 'no_solution'
            })
            continue
        # flatten results into simple serializable dict
        event_result = {
            'event_time': ev['time'],
            'n_jobs': len(ev['jobs']),
            'sequence_map': res['sequence_map'],
            'job_summaries': res.get('job_summaries', {}),
            'total_transport_reconstructed': res.get('total_transport_reconstructed', 0.0),
            'total_return_distance': res.get('total_return_distance', 0.0),
            'makespan': res.get('makespan', 0.0)
        }
        all_event_results.append(event_result)

        # print summary to console (concise)
        for m in M_SET:
            seq = res['sequence_map'].get(m, [])
            if not seq:
                print(f"  AGV {m}: idle")
                continue
            print(f"  AGV {m}: assigned {len(seq)} tasks")
            for t in seq:
                print(f"    jid={t['jid']}, pickup={t['pickup_node']}, del={t['delivery_node']}, return_dist={t['return_dist_to_pickup']:.2f}, trans={t['transport_time']:.2f}, proc={t['proc_time']:.2f}, end={t['end_time']:.2f}")
        print(f"  Total transport(sum del-pick): {res['total_transport_reconstructed']:.2f}")
        print(f"  Total return distance (sum): {res['total_return_distance']:.2f}")
        print(f"  Makespan (max T_end): {res['makespan']:.2f}\n")

    # write results + gantt charts
    write_results_and_gantt(all_event_results, out_dir='results')

if __name__ == "__main__":
    main()
# ...existing code...