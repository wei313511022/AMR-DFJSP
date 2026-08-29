"""Clean per-instance compute time for the F / D search-budget comparison.

One process, nothing else on the box. The parallel sweep's solve_s is inflated by
18 workers contending for memory bandwidth and is not a compute cost.
"""
import json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/full_benchmark"))
import sweep_all as S

N = 3   # instances per (model, mode, size)
JOBS = [("v10_F_s44", "greedy"), ("v10_F_s44", "best8"), ("v10_F_s44", "best16"),
        ("v9_only60_s44", "greedy"), ("v9_only60_s44", "best8"), ("v9_only60_s44", "best16")]

S.worker_init()
out = REPO / "experiments/full_benchmark/raw/timing_k.jsonl"
t0 = time.perf_counter()
with out.open("w", encoding="utf-8") as fh:
    for name, mode in JOBS:
        for n in S.SIZES:
            for r in S.run_task(("model", (name, mode), n, 0, N)):
                r["timing_run"] = True
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"  {name:<16}{mode:>8}  n={n:<4} done  ({time.perf_counter()-t0:.0f}s elapsed)", flush=True)
print(f"timings -> {out}")
