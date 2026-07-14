"""Fit the fast-model calibration artifact from real decoder rollouts.

Compares, per operation, the raw fast estimate (Manhattan travel, base dock
wait) against the collision-aware decoder (space-time A*, wait lines,
hold_upstream), then fits:

  - travel:  real_travel ~= scale * manhattan + offset   (affine, clamped >= manhattan at use)
  - dock_wait_penalty[d]: mean extra wait beyond the base estimate, indexed by
    how many committed services are still unfinished when the AMR is ready
    (d in {0, 1, 2, 3+}).

Writes Static_alogorithm/calibration/fast_model_calibration.json, which
operation_policy loads at import. Schedules are sourced from dispatching
rules and (when checkpoints load) the current neural models.

Usage:
  python calibrate_fast_model.py [--instances 5] [--sizes 10,20,40,60] [--no_neural]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

STATIC_DIR = Path(__file__).resolve().parent
ROOT_DIR = STATIC_DIR.parent
if str(STATIC_DIR) not in sys.path:
    sys.path.insert(0, str(STATIC_DIR))

from GA.GA import (  # noqa: E402
    AMR_KEYS,
    AMR_STARTS,
    INBOUND_DOCK_KEYS,
    INBOUND_DOCK_LOCATIONS,
    JOB_TYPE_KEYS,
    PICKUP,
    STATIONS,
    TYPE_DURATION,
    UNLOAD,
    Individual,
    Job,
    decode_schedule_tick_by_tick,
    dock_key_from_value,
    heuristic,
    repair_operation_order,
)
from operation_policy import set_calibration  # noqa: E402
from dispatching_rules import dispatching_rules as dr  # noqa: E402

TRAVEL_KINDS = {"travel", "return"}
JOB_LABEL_RE = re.compile(r"Job(\d+)\b")

RULE_SOURCES = (
    ("earliest_completion_job", "earliest_completion"),
    ("fifo", "earliest_available"),
    ("nearest_station", "nearest_amr"),
    ("spt", "least_loaded"),
    ("material_match", "material_match"),
)


def make_jobs_sized(num_jobs: int, rng: random.Random) -> List[Job]:
    jobs = []
    for idx in range(num_jobs):
        type_ = rng.choice(JOB_TYPE_KEYS)
        jobs.append(
            Job(
                idx=idx,
                type_=type_,
                duration=float(TYPE_DURATION[type_]),
                station=rng.choice(list(STATIONS.keys())),
                inbound_dock=rng.choice(INBOUND_DOCK_KEYS),
                arrival_time=0.0,
            )
        )
    return jobs


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_checkpoint(candidates: Sequence[str]) -> Optional[Path]:
    for directory in (ROOT_DIR, STATIC_DIR):
        for name in candidates:
            path = directory / name
            if path.exists():
                return path
    return None


def load_neural_sources() -> List[Tuple[str, callable]]:
    """Best-effort loading of the current checkpoints as schedule sources."""
    import torch
    from operation_policy import load_required_operation_checkpoint

    sources = []

    def try_source(label, module_path, build):
        try:
            module = _load_module(f"calib_{label}", module_path)
            solver = build(module, torch, load_required_operation_checkpoint)
            sources.append((label, solver))
            print(f"  neural source ready: {label}")
        except Exception as exc:  # noqa: BLE001 — data sources are optional
            print(f"  skipping neural source {label}: {exc}")

    def build_attention(module, torch_mod, loader):
        model = module.SchedulerAttention()
        checkpoint = _find_checkpoint(["attention_scheduler_best.pth"])
        loader(model, checkpoint, torch_mod, required_keys=("op_emb.weight", "policy_head.0.weight"))
        return lambda jobs: module.solve_with_attention(list(jobs), model, deterministic=True)[0]

    def build_gnn(module, torch_mod, loader):
        model = module.SchedulerGNN()
        checkpoint = _find_checkpoint(["gnn_mpn_scheduler_best.pth", "gnn_scheduler_best.pth"])
        loader(model, checkpoint, torch_mod, required_keys=("op_emb.weight", "operation_actor.0.weight"))
        return lambda jobs: module.solve_with_gnn(list(jobs), model, deterministic=True)[0]

    def build_extend(module, torch_mod, loader):
        model = module.ExtendSchedulerGNN()
        checkpoint = _find_checkpoint(["extend_gnn_scheduler_best.pth"])
        loader(model, checkpoint, torch_mod, required_keys=("op_emb.weight", "operation_actor.0.weight"))
        return lambda jobs: module.solve_with_extend_gnn(list(jobs), model, deterministic=True)[0]

    try_source("attention", STATIC_DIR / "Attention" / "Attention.py", build_attention)
    try_source("gnn", STATIC_DIR / "GNN" / "GNN.py", build_gnn)
    try_source("extend_gnn", STATIC_DIR / "extend_GNN" / "extend_GNN.py", build_extend)
    return sources


def raw_fast_replay(individual: Individual, jobs: Sequence[Job]) -> Dict[Tuple[str, int, str], dict]:
    """Replay the schedule through the uncalibrated fast model.

    Deliberately re-implements the raw kinematics (Manhattan travel, base dock
    wait) so the measurement is independent of whatever calibration is active.
    """
    ops = repair_operation_order(list(individual.order), list(jobs))
    positions = {amr: AMR_STARTS[amr] for amr in AMR_KEYS}
    avail = {amr: 0.0 for amr in AMR_KEYS}
    station_avail = {key: 0.0 for key in list(INBOUND_DOCK_LOCATIONS) + list(STATIONS)}
    committed: Dict[str, List[Tuple[float, float]]] = {key: [] for key in station_avail}
    job_map = {job.idx: job for job in jobs}
    records: Dict[Tuple[str, int, str], dict] = {}

    for op in ops:
        job = job_map[op.job_idx]
        amr = individual.amr_assignment[op.job_idx]
        if op.kind == PICKUP:
            dock = dock_key_from_value(job.inbound_dock)
            dock_pos = INBOUND_DOCK_LOCATIONS[dock]
        else:
            dock = job.station
            dock_pos = STATIONS[dock]

        distance = heuristic(positions[amr], dock_pos)
        ready = avail[amr] + distance
        if op.kind == PICKUP:
            ready = max(ready, float(job.arrival_time))
        queue_depth = sum(1 for _, end in committed[dock] if end > ready)
        base_wait = max(0.0, station_avail[dock] - ready)
        service_start = ready + base_wait
        service_end = service_start + job.duration

        records[(amr, op.job_idx, op.kind)] = {
            "manhattan": float(distance),
            "fast_wait": base_wait,
            "queue_depth": queue_depth,
        }
        committed[dock].append((service_start, service_end))
        positions[amr] = dock_pos
        avail[amr] = service_end
        station_avail[dock] = service_end

    return records


def parse_real_legs(timeline: Sequence[tuple]) -> Dict[Tuple[str, int, str], dict]:
    """Split the decoder timeline into per-service legs.

    A leg spans from the previous service end (or t=0) to the next service
    start on the same AMR: its travel is the summed travel/return entries and
    everything else (wait-line, hold_upstream, unlogged idling) counts as wait.
    """
    per_amr: Dict[str, List[tuple]] = defaultdict(list)
    for entry in timeline:
        per_amr[entry[0]].append(entry)

    legs: Dict[Tuple[str, int, str], dict] = {}
    for amr, entries in per_amr.items():
        entries.sort(key=lambda e: (e[1], e[2]))
        boundary = 0.0
        travel_acc = 0.0
        for entry in entries:
            _, start, end, kind, label = entry
            if kind == "load_inbound" or kind.startswith("process_"):
                match = JOB_LABEL_RE.search(str(label))
                if match:
                    op_kind = PICKUP if kind == "load_inbound" else UNLOAD
                    real_travel = travel_acc
                    real_wait = max(0.0, (start - boundary) - real_travel)
                    legs[(amr, int(match.group(1)), op_kind)] = {
                        "real_travel": real_travel,
                        "real_wait": real_wait,
                    }
                boundary = end
                travel_acc = 0.0
            elif kind in TRAVEL_KINDS:
                travel_acc += max(0.0, end - start)
    return legs


def fit_affine(samples: List[Tuple[float, float]]) -> Tuple[float, float]:
    n = len(samples)
    sum_x = sum(x for x, _ in samples)
    sum_y = sum(y for _, y in samples)
    sum_xx = sum(x * x for x, _ in samples)
    sum_xy = sum(x * y for x, y in samples)
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-9:
        return 1.0, 0.0
    scale = (n * sum_xy - sum_x * sum_y) / denom
    offset = (sum_y - scale * sum_x) / n
    return scale, offset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=5, help="Instances per job-count size")
    parser.add_argument("--sizes", type=str, default="10,20,40,60")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no_neural", action="store_true", help="Use dispatch-rule schedules only")
    parser.add_argument(
        "--output",
        type=str,
        default=str(STATIC_DIR / "calibration" / "fast_model_calibration.json"),
    )
    args = parser.parse_args()

    # Measure against the raw fast model regardless of any active artifact.
    set_calibration(None)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    rng = random.Random(args.seed)

    sources: List[Tuple[str, callable]] = [
        (f"{job_rule}+{amr_rule}", (lambda jr, ar: lambda jobs: dr.solve_with_dispatching_rules(list(jobs), jr, ar, seed=args.seed)[0])(job_rule, amr_rule))
        for job_rule, amr_rule in RULE_SOURCES
    ]
    if not args.no_neural:
        print("Loading neural schedule sources...")
        sources.extend(load_neural_sources())

    travel_samples: List[Tuple[float, float]] = []
    wait_samples: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    used = skipped_invalid = 0

    for size in sizes:
        for instance in range(args.instances):
            jobs = make_jobs_sized(size, rng)
            for label, solver in sources:
                individual = solver(jobs)
                availability, timeline, _, _, invalid = decode_schedule_tick_by_tick(
                    individual, list(jobs), need_log=True, check_collision=True
                )
                if invalid > 0:
                    skipped_invalid += 1
                    continue
                fast = raw_fast_replay(individual, jobs)
                real = parse_real_legs(timeline)
                matched = 0
                for key, fast_record in fast.items():
                    real_record = real.get(key)
                    if real_record is None:
                        continue
                    matched += 1
                    travel_samples.append((fast_record["manhattan"], real_record["real_travel"]))
                    depth = min(fast_record["queue_depth"], 3)
                    wait_samples[depth].append((fast_record["fast_wait"], real_record["real_wait"]))
                used += 1
                if matched < 2 * len(jobs) * 0.9:
                    print(f"  warning: only matched {matched}/{2 * len(jobs)} ops ({label}, size {size})")
        print(f"size {size}: cumulative samples travel={len(travel_samples)}")

    if not travel_samples:
        raise SystemExit("No calibration samples collected; aborting without writing an artifact.")

    scale, offset = fit_affine(travel_samples)
    mean_real = sum(y for _, y in travel_samples) / len(travel_samples)
    ss_tot = sum((y - mean_real) ** 2 for _, y in travel_samples)
    ss_res = sum((y - (scale * x + offset)) ** 2 for x, y in travel_samples)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae_before = sum(abs(y - x) for x, y in travel_samples) / len(travel_samples)
    mae_after = sum(abs(y - max(x, scale * x + offset)) for x, y in travel_samples) / len(travel_samples)
    # Mean bias is what accumulates into rollout clock drift; MAE is per-leg noise.
    bias_before = sum(y - x for x, y in travel_samples) / len(travel_samples)
    bias_after = sum(y - max(x, scale * x + offset) for x, y in travel_samples) / len(travel_samples)

    penalties = []
    penalty_stats = {}
    for depth in range(4):
        samples = wait_samples.get(depth, [])
        if samples:
            penalty = sum(real - fast for fast, real in samples) / len(samples)
        else:
            penalty = penalties[-1] if penalties else 0.0
        penalty = max(0.0, penalty)
        # Physically, more queue can't mean less overhead; enforce monotonicity
        # so sparse deep-queue bins don't dip below shallower ones.
        if penalties:
            penalty = max(penalty, penalties[-1])
        penalties.append(round(penalty, 4))
        penalty_stats[str(depth)] = {"samples": len(samples), "mean_extra_wait": round(penalty, 4)}

    artifact = {
        "fitted_on": str(date.today()),
        "schedules_used": used,
        "schedules_skipped_invalid": skipped_invalid,
        "travel": {
            "scale": round(scale, 6),
            "offset": round(offset, 6),
            "samples": len(travel_samples),
            "r2": round(r2, 4),
            "mae_before": round(mae_before, 4),
            "mae_after": round(mae_after, 4),
            "bias_before": round(bias_before, 4),
            "bias_after": round(bias_after, 4),
        },
        "dock_wait_penalty": penalties,
        "dock_wait_stats": penalty_stats,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n=== Calibration fitted ===")
    print(f"travel: real ~= {scale:.4f} * manhattan + {offset:.4f}  (R2={r2:.4f}, n={len(travel_samples)})")
    print(f"travel MAE: {mae_before:.3f} -> {mae_after:.3f} | mean bias: {bias_before:+.3f} -> {bias_after:+.3f}")
    print(f"dock wait penalty by queue depth: {penalties}  (samples: {[penalty_stats[str(d)]['samples'] for d in range(4)]})")
    print(f"schedules used: {used}, skipped for invalid jobs: {skipped_invalid}")
    print(f"written to {output}")


if __name__ == "__main__":
    main()
