"""Wall-clock cost of each scheduler, measured on identical instances and one machine.

The dispatching-rule sweeps never recorded compute time, so there was no way to state what
the policy's makespan advantage costs. This measures both sides the same way:

  rule    complete_with_dispatch_rule(...)              -- CPU
  policy  solve_with_extend_gnn(..., deterministic)     -- GPU forward passes, 2*|J| steps
  policy  best-of-K                                     -- K independent rollouts

Reported separately from `evaluate`, which both sides pay identically and which is a property
of the executor rather than of the scheduler.

Run this with nothing else on the GPU. Contention inflates the policy numbers and the point
of the table is a fair per-instance cost.

    python bench_compute.py --instances 20
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATIC = REPO / "Static_alogorithm"
for p in (str(STATIC), str(STATIC / "extend_GNN")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

import GA.GA as GA  # noqa: E402
import ideal_evaluator as ie  # noqa: E402
import scenario_v3 as sc  # noqa: E402
from GA.GA import load_dispatch_events  # noqa: E402
from extend_GNN import ExtendSchedulerGNN, solve_with_extend_gnn  # noqa: E402
from operation_policy import load_required_operation_checkpoint  # noqa: E402
from reinforce_baseline import complete_with_dispatch_rule  # noqa: E402

HERE = Path(__file__).resolve().parent
RULES = ("milk_run+earliest_completion", "milk_run+earliest_available", "lpt+earliest_completion")


def time_rule(jobs, rule_name: str) -> tuple:
    t0 = time.perf_counter()
    ind = complete_with_dispatch_rule(jobs, [], {}, baseline_rule=rule_name, seed=42)
    solve = time.perf_counter() - t0
    t0 = time.perf_counter()
    metrics = ie.evaluate(ind, jobs)
    return solve, time.perf_counter() - t0, metrics


def time_policy(jobs, model, k: int, instance: int) -> tuple:
    solve = 0.0
    best = None
    evals = 0.0
    for i in range(max(k, 1)):
        if k > 1:
            torch.manual_seed(1000 + 1_000_003 * instance + i)
        t0 = time.perf_counter()
        with torch.no_grad():
            ind, _, _ = solve_with_extend_gnn(jobs, model, deterministic=(k <= 1))
        if torch.cuda.is_available():
            torch.cuda.synchronize()      # forward passes are async; sync or the timing lies
        solve += time.perf_counter() - t0
        model.eval()
        # Best-of-K has to SCORE all K candidates to pick one, so every evaluation is part
        # of its cost. Keeping only the first rollout (an earlier bug here) made K=8 and
        # K=16 report identical makespans and understated eval by a factor of K.
        t0 = time.perf_counter()
        metrics = ie.evaluate(ind, jobs)
        evals += time.perf_counter() - t0
        if best is None or (metrics["nu"], metrics["executed"]) < (best[1]["nu"], best[1]["executed"]):
            best = (ind, metrics)
    return solve, evals, best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--num_amrs", type=int, default=16)
    ap.add_argument("--ks", type=str, default="1,8,16")
    ap.add_argument("--weights", type=str,
                    default=str(REPO / "checkpoints_v8" / "ppo_gc1_s42_best.pth"))
    ap.add_argument("--out", type=str, default=str(HERE / "compute_time.csv"))
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    sc.apply_layout(num_amrs=args.num_amrs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExtendSchedulerGNN(hidden_dim=128, gin_layers=3)
    load_required_operation_checkpoint(model, Path(args.weights), torch,
                                       required_keys=("op_emb.weight", "operation_actor.0.weight"))
    model.to(device).eval()

    rows = []
    for n in (60, 120, 240):
        events = load_dispatch_events(REPO / "test_case" / "v3" / f"test_{n}.jsonl")[: args.instances]

        for rule_name in RULES:
            solves, evals, mks = [], [], []
            for ev in events:
                s, e, m = time_rule(list(ev["jobs"]), rule_name)
                solves.append(s); evals.append(e)
                if m["nu"] == 0:
                    mks.append(m["executed"])
            rows.append({
                "n_jobs": n, "scheduler": rule_name, "kind": "rule", "k": 1,
                "solve_s": round(statistics.mean(solves), 4),
                "eval_s": round(statistics.mean(evals), 4),
                "total_s": round(statistics.mean(solves) + statistics.mean(evals), 4),
                "executed": round(statistics.mean(mks), 1) if mks else float("nan"),
                "instances": len(events),
            })
            print(f"  n={n:>3} {rule_name:>30} solve {statistics.mean(solves):7.4f}s")

        # warm up CUDA graphs/allocator so the first timed rollout is not the outlier
        with torch.no_grad():
            solve_with_extend_gnn(list(events[0]["jobs"]), model, deterministic=True)
        model.eval()

        for k in ks:
            solves, evals, mks = [], [], []
            for ev in events:
                s, e, m = time_policy(list(ev["jobs"]), model, k, ev["index"])
                solves.append(s); evals.append(e)
                if m["nu"] == 0:
                    mks.append(m["executed"])
            label = "extend_gnn_ppo greedy" if k <= 1 else f"extend_gnn_ppo best-of-{k}"
            rows.append({
                "n_jobs": n, "scheduler": label, "kind": "policy", "k": k,
                "solve_s": round(statistics.mean(solves), 4),
                "eval_s": round(statistics.mean(evals), 4),
                "total_s": round(statistics.mean(solves) + statistics.mean(evals), 4),
                "executed": round(statistics.mean(mks), 1) if mks else float("nan"),
                "instances": len(events),
            })
            print(f"  n={n:>3} {label:>30} solve {statistics.mean(solves):7.4f}s")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    meta = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "num_amrs": args.num_amrs,
        "instances_per_point": args.instances, "weights": args.weights,
        "note": "solve_s excludes ie.evaluate, which both sides pay identically; "
                "CUDA synchronised before each stop-clock",
    }
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
