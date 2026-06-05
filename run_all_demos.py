import subprocess
import time
import os
import signal
import sys

def launch():
    print("Launching all demos...")
    
    # Base paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(root_dir, "Random_Job_Arrivals", "models")
    routing_dir = os.path.join(root_dir, "AMRs_Routing")
    python_executable = sys.executable
    
    processes = []
    
    sync_file = os.path.join(root_dir, "sync.txt")
    with open(sync_file, "w") as f:
        f.write("0")
        
    # Clean old amr_state.json files
    dyn_json = os.path.join(models_dir, "dynamic_amr_state.json")
    per_json = os.path.join(models_dir, "periodic_amr_state.json")
    if os.path.exists(dyn_json): os.remove(dyn_json)
    if os.path.exists(per_json): os.remove(per_json)
    
    try:
        # 1. Dynamic Pairing
        print("Starting Dynamic Pairing Demo...")
        p1 = subprocess.Popen(
            [python_executable, "Dynamic_Pairing_Demo.py", "--window_pos", "+0+0", "--state_file", "dynamic_amr_state.json", "--sync_file", sync_file],
            cwd=models_dir
        )
        processes.append(p1)
        
        # 2. Routing Demo for Dynamic
        print("Starting Routing Demo (Dynamic)...")
        p2 = subprocess.Popen(
            [python_executable, "Routing_Demo.py", "--state_file", "../Random_Job_Arrivals/models/dynamic_amr_state.json", "--window_pos", "+1300+0", "--title", "Dynamic Route Map", "--sync_file", sync_file],
            cwd=routing_dir
        )
        processes.append(p2)
        
        # 3. Periodic Pairing
        print("Starting Periodic Pairing Demo...")
        p3 = subprocess.Popen(
            [python_executable, "Periodic_Pairing_Demo.py", "--window_pos", "+0+550", "--state_file", "periodic_amr_state.json", "--sync_file", sync_file],
            cwd=models_dir
        )
        processes.append(p3)
        
        # 4. Routing Demo for Periodic
        print("Starting Routing Demo (Periodic)...")
        p4 = subprocess.Popen(
            [python_executable, "Routing_Demo.py", "--state_file", "../Random_Job_Arrivals/models/periodic_amr_state.json", "--window_pos", "+1300+550", "--title", "Periodic Route Map", "--sync_file", sync_file],
            cwd=routing_dir
        )
        processes.append(p4)

        print("\nAll processes launched successfully!")
        print("Press SPACE in any of the windows to start the simulation.")
        print("Press Ctrl+C in this terminal to close all windows and exit.")
        
        # Wait for all processes
        for p in processes:
            p.wait()
            
    except KeyboardInterrupt:
        print("\nTerminating all processes...")
        for p in processes:
            p.send_signal(signal.SIGINT)
            p.terminate()

if __name__ == "__main__":
    launch()
