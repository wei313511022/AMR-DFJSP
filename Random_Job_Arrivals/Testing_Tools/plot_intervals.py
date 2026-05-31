import os
import re
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# FALLBACK LOG DATA (from previous execution)
# -------------------------------------------------------------
FALLBACK_LOG = """
[AI-DECIDE] t=  21.0 step=  20 choose RESCHEDULE | active=11 unstarted=8 | mask=[1.0, 1.0]
[AI-AFTER ] t=  22.0 step=  20 GA= 481.52 ms | active=11 unstarted=8 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  23.0 step=  22 choose RESCHEDULE | active=13 unstarted=10 | mask=[1.0, 1.0]
[AI-AFTER ] t=  24.0 step=  22 GA= 535.44 ms | active=13 unstarted=10 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  26.0 step=  25 choose RESCHEDULE | active=12 unstarted=9 | mask=[1.0, 1.0]
[AI-AFTER ] t=  28.0 step=  25 GA=1166.20 ms | active=12 unstarted=9 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  31.0 step=  29 choose RESCHEDULE | active=13 unstarted=10 | mask=[1.0, 1.0]
[AI-AFTER ] t=  32.0 step=  29 GA= 762.89 ms | active=13 unstarted=10 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  33.0 step=  31 choose RESCHEDULE | active=14 unstarted=11 | mask=[1.0, 1.0]
[AI-AFTER ] t=  35.0 step=  31 GA=1220.50 ms | active=14 unstarted=11 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  36.0 step=  33 choose RESCHEDULE | active=17 unstarted=14 | mask=[1.0, 1.0]
[AI-AFTER ] t=  37.0 step=  33 GA= 655.39 ms | active=17 unstarted=14 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  38.0 step=  35 choose RESCHEDULE | active=19 unstarted=16 | mask=[1.0, 1.0]
[AI-AFTER ] t=  40.0 step=  35 GA=1711.81 ms | active=20 unstarted=17 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  41.0 step=  37 choose RESCHEDULE | active=24 unstarted=21 | mask=[1.0, 1.0]
[AI-AFTER ] t=  44.0 step=  37 GA=2816.63 ms | active=24 unstarted=21 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  45.0 step=  39 choose RESCHEDULE | active=26 unstarted=23 | mask=[1.0, 1.0]
[AI-AFTER ] t=  48.0 step=  39 GA=2381.30 ms | active=27 unstarted=24 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  49.0 step=  41 choose RESCHEDULE | active=28 unstarted=25 | mask=[1.0, 1.0]
[AI-AFTER ] t=  53.0 step=  41 GA=3003.40 ms | active=28 unstarted=25 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  58.0 step=  47 choose RESCHEDULE | active=30 unstarted=27 | mask=[1.0, 1.0]
[AI-AFTER ] t=  62.0 step=  47 GA=3569.69 ms | active=30 unstarted=27 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  67.0 step=  53 choose RESCHEDULE | active=37 unstarted=34 | mask=[1.0, 1.0]
[AI-AFTER ] t=  72.0 step=  53 GA=4218.88 ms | active=37 unstarted=34 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  79.0 step=  61 choose RESCHEDULE | active=37 unstarted=34 | mask=[1.0, 1.0]
[AI-AFTER ] t=  84.0 step=  61 GA=4159.30 ms | active=38 unstarted=35 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  89.0 step=  67 choose RESCHEDULE | active=43 unstarted=40 | mask=[1.0, 1.0]
[AI-AFTER ] t=  94.0 step=  67 GA=4644.98 ms | active=43 unstarted=40 | mask_after=[1.0, 0.0]
[AI-DECIDE] t=  95.0 step=  69 choose RESCHEDULE | active=45 unstarted=42 | mask=[1.0, 1.0]
[AI-AFTER ] t= 101.0 step=  69 GA=5073.20 ms | active=45 unstarted=42 | mask_after=[1.0, 0.0]
[AI-DECIDE] t= 104.0 step=  73 choose RESCHEDULE | active=47 unstarted=44 | mask=[1.0, 1.0]
[AI-AFTER ] t= 109.0 step=  73 GA=4977.28 ms | active=47 unstarted=44 | mask_after=[1.0, 0.0]
[AI-DECIDE] t= 110.0 step=  75 choose RESCHEDULE | active=50 unstarted=47 | mask=[1.0, 1.0]
[AI-AFTER ] t= 116.0 step=  75 GA=5522.58 ms | active=50 unstarted=47 | mask_after=[1.0, 0.0]
[AI-DECIDE] t= 125.0 step=  85 choose RESCHEDULE | active=53 unstarted=50 | mask=[1.0, 1.0]
[AI-AFTER ] t= 132.0 step=  85 GA=6148.65 ms | active=52 unstarted=50 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 136.0 step=  90 choose RESCHEDULE | active=58 unstarted=56 | mask=[1.0, 1.0]
[AI-AFTER ] t= 143.0 step=  90 GA=6859.33 ms | active=60 unstarted=58 | mask_after=[1.0, 0.0]
[AI-DECIDE] t= 144.0 step=  92 choose RESCHEDULE | active=65 unstarted=62 | mask=[1.0, 1.0]
[AI-AFTER ] t= 152.0 step=  92 GA=7287.11 ms | active=64 unstarted=62 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 153.0 step=  94 choose RESCHEDULE | active=67 unstarted=65 | mask=[1.0, 1.0]
[AI-AFTER ] t= 162.0 step=  94 GA=8155.07 ms | active=67 unstarted=65 | mask_after=[1.0, 0.0]
[AI-DECIDE] t= 174.0 step= 107 choose RESCHEDULE | active=71 unstarted=68 | mask=[1.0, 1.0]
[AI-AFTER ] t= 184.0 step= 107 GA=9182.47 ms | active=70 unstarted=68 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 187.0 step= 111 choose RESCHEDULE | active=78 unstarted=76 | mask=[1.0, 1.0]
[AI-AFTER ] t= 197.0 step= 111 GA=9530.13 ms | active=77 unstarted=76 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 199.0 step= 114 choose RESCHEDULE | active=81 unstarted=79 | mask=[1.0, 1.0]
[AI-AFTER ] t= 210.0 step= 114 GA=10862.30 ms | active=80 unstarted=79 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 215.0 step= 120 choose RESCHEDULE | active=94 unstarted=92 | mask=[1.0, 1.0]
[AI-AFTER ] t= 227.0 step= 120 GA=11542.18 ms | active=94 unstarted=92 | mask_after=[1.0, 0.0]
[AI-DECIDE] t= 231.0 step= 125 choose RESCHEDULE | active=97 unstarted=94 | mask=[1.0, 1.0]
[AI-AFTER ] t= 243.0 step= 125 GA=11140.07 ms | active=98 unstarted=95 | mask_after=[1.0, 0.0]
[AI-DECIDE] t= 279.0 step= 162 choose RESCHEDULE | active=111 unstarted=108 | mask=[1.0, 1.0]
[AI-AFTER ] t= 294.0 step= 162 GA=14198.34 ms | active=110 unstarted=108 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 299.0 step= 168 choose RESCHEDULE | active=109 unstarted=106 | mask=[1.0, 1.0]
[AI-AFTER ] t= 313.0 step= 168 GA=13359.73 ms | active=108 unstarted=106 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 313.0 step= 169 choose RESCHEDULE | active=108 unstarted=106 | mask=[1.0, 1.0]
[AI-AFTER ] t= 327.0 step= 169 GA=13768.43 ms | active=107 unstarted=106 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 327.0 step= 170 choose RESCHEDULE | active=107 unstarted=106 | mask=[1.0, 1.0]
[AI-AFTER ] t= 342.0 step= 170 GA=14546.59 ms | active=107 unstarted=106 | mask_after=[1.0, 0.0]
[AI-DECIDE] t= 354.0 step= 183 choose RESCHEDULE | active=105 unstarted=102 | mask=[1.0, 1.0]
[AI-AFTER ] t= 368.0 step= 183 GA=13836.32 ms | active=104 unstarted=102 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 368.0 step= 184 choose RESCHEDULE | active=104 unstarted=102 | mask=[1.0, 1.0]
[AI-AFTER ] t= 381.0 step= 184 GA=12990.69 ms | active=103 unstarted=102 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 385.0 step= 189 choose RESCHEDULE | active=102 unstarted=100 | mask=[1.0, 1.0]
[AI-AFTER ] t= 399.0 step= 189 GA=13777.19 ms | active=100 unstarted=100 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 406.0 step= 197 choose RESCHEDULE | active=100 unstarted=98 | mask=[1.0, 1.0]
[AI-AFTER ] t= 420.0 step= 197 GA=13890.03 ms | active=99 unstarted=98 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 420.0 step= 198 choose RESCHEDULE | active=99 unstarted=98 | mask=[1.0, 1.0]
[AI-AFTER ] t= 434.0 step= 198 GA=13005.40 ms | active=98 unstarted=98 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 439.0 step= 204 choose RESCHEDULE | active=98 unstarted=95 | mask=[1.0, 1.0]
[AI-AFTER ] t= 452.0 step= 204 GA=12180.21 ms | active=98 unstarted=95 | mask_after=[1.0, 0.0]
[AI-DECIDE] t= 457.0 step= 210 choose RESCHEDULE | active=96 unstarted=93 | mask=[1.0, 1.0]
[AI-AFTER ] t= 469.0 step= 210 GA=11639.12 ms | active=95 unstarted=93 | mask_after=[1.0, 1.0]
[AI-DECIDE] t= 469.0 step= 211 choose RESCHEDULE | active=95 unstarted=93 | mask=[1.0, 1.0]
[AI-AFTER ] t= 482.0 step= 211 GA=12691.75 ms | active=95 unstarted=93 | mask_after=[1.0, 0.0]
[AI-DECIDE] t= 500.0 step= 230 choose RESCHEDULE | active=92 unstarted=89 | mask=[1.0, 1.0]
[AI-AFTER ] t= 512.0 step= 230 GA=11302.60 ms | active=92 unstarted=89 | mask_after=[1.0, 0.0]
"""

def parse_logs(log_text):
    decide_pattern = re.compile(
        r"\[AI-DECIDE\]\s+t=\s*([\d\.]+)\s+step=\s*(\d+)\s+choose\s+(\w+)\s+\|\s+active=(\d+)\s+unstarted=(\d+)"
    )
    after_pattern = re.compile(
        r"\[AI-AFTER\s*\]\s+t=\s*([\d\.]+)\s+step=\s*(\d+)\s+GA=\s*([\d\.]+)\s+ms\s+\|\s+active=(\d+)\s+unstarted=(\d+)"
    )
    
    decides = []
    afters = []
    
    for line in log_text.strip().split("\n"):
        dec_match = decide_pattern.search(line)
        aft_match = after_pattern.search(line)
        
        if dec_match:
            decides.append({
                "t": float(dec_match.group(1)),
                "step": int(dec_match.group(2)),
                "action": dec_match.group(3),
                "active": int(dec_match.group(4)),
                "unstarted": int(dec_match.group(5))
            })
        elif aft_match:
            afters.append({
                "t": float(aft_match.group(1)),
                "step": int(aft_match.group(2)),
                "ga_ms": float(aft_match.group(3)),
                "active": int(aft_match.group(4)),
                "unstarted": int(aft_match.group(5))
            })
            
    return decides, afters

def generate_analysis():
    # Attempt to read from task-31.log first
    log_path = "/home/wei/.gemini/antigravity-ide/brain/4223ea06-ea25-4e11-ae9f-524b77fd58b1/.system_generated/tasks/task-31.log"
    log_content = ""
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_content = f.read()
            
    decides_new, afters_new = parse_logs(log_content)
    decides_fallback, afters_fallback = parse_logs(FALLBACK_LOG)
    
    # Use the richer one
    if len(decides_new) >= len(decides_fallback):
        decides, afters = decides_new, afters_new
        print(f"Loaded {len(decides)} events from running log file.")
    else:
        decides, afters = decides_fallback, afters_fallback
        print(f"Loaded {len(decides)} events from fallback terminal buffer.")
        
    # Align decides and afters by step number
    events = []
    after_by_step = {a["step"]: a for a in afters}
    
    for d in decides:
        step = d["step"]
        if step in after_by_step:
            a = after_by_step[step]
            events.append({
                "step": step,
                "t_decide": d["t"],
                "t_after": a["t"],
                "ga_ms": a["ga_ms"],
                "active_decide": d["active"],
                "unstarted_decide": d["unstarted"],
                "active_after": a["active"],
                "unstarted_after": a["unstarted"]
            })
            
    print(f"Aligned {len(events)} rescheduling events successfully.")
    
    # Compute intervals
    # 1. Rescheduling Phase Duration (Simulation Time) for same step: t_after - t_decide
    # 2. Simulation Time between subsequent reschedule events: t_decide(i) - t_after(i-1)
    resched_durations = []
    intervals_between = []
    steps_between = []
    
    for i, ev in enumerate(events):
        resched_durations.append(ev["t_after"] - ev["t_decide"])
        if i > 0:
            intervals_between.append(ev["t_decide"] - events[i-1]["t_after"])
            steps_between.append(ev["step"] - events[i-1]["step"])
            
    # Premium layout styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Elegant color palette
    primary_color = "#3A86FF"    # Radiant Blue
    secondary_color = "#FF006E"  # Electric Pink
    tertiary_color = "#8338EC"   # Violet Purple
    accent_color = "#FFBE0B"     # Amber Gold
    text_color = "#2D3142"
    
    # Adjust global plot parameters
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.color': text_color,
        'axes.labelcolor': text_color
    })
    
    steps_x = [ev["step"] for ev in events]
    
    # Panel 1: Rescheduling Delay (Simulation Time & CPU Time)
    ax1 = axes[0, 0]
    lns1 = ax1.plot(steps_x, resched_durations, color=primary_color, marker='o', linewidth=2, label="Sim Time Delay (dt)")
    ax1.set_xlabel("Simulation Step")
    ax1.set_ylabel("Simulation Time Units (seconds)", color=primary_color)
    ax1.tick_params(axis='y', labelcolor=primary_color)
    ax1.set_title("1. Rescheduling Phase Duration (AI-DECIDE to AI-AFTER)", fontweight='bold')
    
    # Secondary Y axis for CPU compute time (GA ms)
    ax1_twin = ax1.twinx()
    ga_sec = [ev["ga_ms"] / 1000.0 for ev in events]
    lns2 = ax1_twin.plot(steps_x, ga_sec, color=secondary_color, marker='s', linestyle='--', linewidth=1.5, label="CPU Compute Time (s)")
    ax1_twin.set_ylabel("Real CPU Compute Time (seconds)", color=secondary_color)
    ax1_twin.tick_params(axis='y', labelcolor=secondary_color)
    ax1_twin.grid(False) # avoid overlapping grids
    
    # Combine legends
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')
    
    # Panel 2: Simulation Time Intervals Between Reschedules
    ax2 = axes[0, 1]
    steps_between_x = steps_x[1:]
    ax2.bar(steps_between_x, intervals_between, color=tertiary_color, width=4.0, alpha=0.85, edgecolor='black', label="Interval (AI-AFTER to AI-DECIDE)")
    ax2.set_xlabel("Simulation Step (Initiation of next Reschedule)")
    ax2.set_ylabel("Interval: AI-AFTER(i-1) to AI-DECIDE(i) (seconds)")
    ax2.set_title("2. Sim Time from Last Computation Finish to Next Reschedule", fontweight='bold')
    ax2.legend(loc='upper right')
    
    # Panel 3: Step Intervals Between Reschedules
    ax3 = axes[1, 0]
    ax3.plot(steps_between_x, steps_between, color=accent_color, marker='^', linewidth=2, markersize=7, markeredgecolor='black', label="Step Interval")
    ax3.set_xlabel("Simulation Step")
    ax3.set_ylabel("Steps Elapsed")
    ax3.set_title("3. Number of Environment Steps Between Rescheduling Events", fontweight='bold')
    ax3.legend(loc='upper right')
    
    # Panel 4: GA Compute Time Scaling with Active Backlog Size
    ax4 = axes[1, 1]
    active_jobs = [ev["active_decide"] for ev in events]
    ga_ms = [ev["ga_ms"] for ev in events]
    
    # Fit a trendline
    z = np.polyfit(active_jobs, ga_ms, 1)
    p = np.poly1d(z)
    
    ax4.scatter(active_jobs, ga_ms, color="#38B000", s=60, edgecolor='black', alpha=0.85, label="Data Point")
    x_range = np.linspace(min(active_jobs), max(active_jobs), 100)
    ax4.plot(x_range, p(x_range), color="#70E000", linestyle=":", linewidth=2, label=f"Trendline (slope={z[0]:.2f} ms/job)")
    ax4.set_xlabel("Number of Active Jobs at Decision Time")
    ax4.set_ylabel("GA Compute Overhead (ms)")
    ax4.set_title("4. GA Compute Overhead vs. Active Backlog Size", fontweight='bold')
    ax4.legend(loc='upper left')
    
    plt.suptitle("Analysis of AI Rescheduling Decision & Execution Intervals (GNN-DDQN-GA)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save the artifact image
    plot_filename = "/home/wei/.gemini/antigravity-ide/brain/4223ea06-ea25-4e11-ae9f-524b77fd58b1/rescheduling_interval_analysis.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSuccessfully generated premium plot at {plot_filename}")
    
    # Print rich text statistical summary
    print("\n--- STATISTICAL SUMMARY ---")
    print(f"Total Rescheduling Events Logged: {len(events)}")
    print(f"Average Simulation Delay (dt) per Reschedule: {np.mean(resched_durations):.2f} s")
    print(f"Average CPU GA Compute Overhead: {np.mean(ga_ms):.2f} ms (Min: {np.min(ga_ms):.2f} ms, Max: {np.max(ga_ms):.2f} ms)")
    if intervals_between:
        print(f"Average Sim Time Between Reschedules: {np.mean(intervals_between):.2f} s")
        print(f"Average Steps Between Reschedules: {np.mean(steps_between):.2f} steps")
    print(f"GA Compute Scale Factor: Slope is {z[0]:.2f} ms per active job in the system.")

if __name__ == "__main__":
    generate_analysis()
