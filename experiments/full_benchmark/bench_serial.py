"""Clean compute-time measurement: ONE process, no contention.

sweep_all.py runs 22 workers, so its solve_s is inflated by memory-bandwidth contention
and cannot be reported as a compute cost. This re-runs a subsample of the same work
strictly serially. Quality metrics come from the parallel sweep; timings come from here.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/full_benchmark"))
import sweep_all as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rules", type=int, default=5)
    ap.add_argument("--n_greedy", type=int, default=10)
    ap.add_argument("--n_best8", type=int, default=5)
    ap.add_argument("--n_ga", type=int, default=5)
    ap.add_argument("--out", default=str(REPO / "experiments/full_benchmark/raw/timing.jsonl"))
    args = ap.parse_args()

    S.worker_init()
    from dispatching_rules.dispatching_rules import JOB_RULES, AMR_RULES
    tasks = []
    for n in S.SIZES:
        for jr in JOB_RULES:
            for ar in AMR_RULES:
                tasks.append(("rule", f"{jr}+{ar}", n, 0, args.n_rules))
        for name in S.MODELS:
            tasks.append(("model", (name, "greedy"), n, 0, args.n_greedy))
            tasks.append(("model", (name, "best8"), n, 0, args.n_best8))
        tasks.append(("ga", None, n, 0, args.n_ga))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with out.open("w", encoding="utf-8") as fh:
        for i, t in enumerate(tasks, 1):
            for r in S.run_task(t):
                r["timing_run"] = True
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 20 == 0 or i == len(tasks):
                el = time.perf_counter() - t0
                print(f"  {i:>4}/{len(tasks)} | {el/60:6.1f} min | ETA {el/i*(len(tasks)-i)/60:6.1f} min",
                      flush=True)
    print(f"timings -> {out}")


if __name__ == "__main__":
    main()
