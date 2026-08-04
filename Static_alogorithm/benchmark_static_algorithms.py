import argparse
import csv
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent
GA_DIR = STATIC_DIR / "GA"
ATTENTION_DIR = STATIC_DIR / "Attention"
GNN_DIR = STATIC_DIR / "GNN"
EXTEND_GNN_DIR = STATIC_DIR / "extend_GNN"
DEFAULT_CASE_DIR = STATIC_DIR / "benchmark_cases"

for path in (STATIC_DIR, ATTENTION_DIR, GNN_DIR, EXTEND_GNN_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from GA.GA import (  # noqa: E402
    DISPATCH_EVENT_INDEX_ENV,
    DOCK_ALIASES,
    INBOUND_DOCK_KEYS,
    Individual,
    Job,
    STATIONS,
    TYPE_DURATION,
    collision_routing_iters,
    decode_schedule_tick_by_tick,
    routing_iters,
    station_key_from_value,
)

from GA import GA as ga_normal  # noqa: E402
from GA.GA_precise import evolve_precise  # noqa: E402
from dispatching_rules import dispatching_rules as dr  # noqa: E402
from neural_local_improvement import apply_neural_local_improvement  # noqa: E402
import ideal_evaluator as ie  # noqa: E402


DEFAULT_JOB_COUNTS = (20, 40, 60, 80, 100)
NN_ALGORITHMS = {"attention", "attention_precise", "gnn", "gnn_precise", "extend_gnn"}
SEARCH_ALGORITHMS = {"ga", "ga_precise"}
SPECIAL_ALGORITHMS = SEARCH_ALGORITHMS | NN_ALGORITHMS | {"dispatching_rules"}
BENCHMARK_SCHEMA = "static_benchmark_v1"


class BenchmarkCaseError(ValueError):
    pass


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_csv_list(raw: str, cast=str):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def case_file_path(case_dir: Path, job_count: int) -> Path:
    return case_dir / f"dispatch_inbox_{job_count}.jsonl"


def stable_case_seed(seed: int, job_count: int, sample_index: int) -> int:
    return seed + job_count * 1_000_003 + sample_index * 9_176


def generate_case_record(job_count: int, sample_index: int, seed: int) -> Dict[str, object]:
    """Deterministic benchmark instance drawn against the LIVE GA facility.

    Stations and docks come from GA's globals, so this follows whatever layout
    is installed -- v3 by default. Any case file generated before the v3 layout
    landed names the same docks and stations but at the old coordinates, so
    regenerate with --generate rather than reusing stale files.

    Uniform over size classes, all releases at t = 0: matches scenario_v3.
    """
    rng = random.Random(stable_case_seed(seed, job_count, sample_index))
    job_types = sorted(TYPE_DURATION.keys())
    station_ids = list(range(1, len(STATIONS) + 1))
    dock_keys = list(INBOUND_DOCK_KEYS)

    jobs = []
    for job_idx in range(job_count):
        job_type = rng.choice(job_types)
        jobs.append(
            {
                "jid": job_idx,
                "type": job_type,
                "proc_time": float(TYPE_DURATION[job_type]),
                "station": rng.choice(station_ids),
                "inbound_dock": rng.choice(dock_keys),
                "arrival_time": 0.0,
            }
        )

    return {
        "schema": BENCHMARK_SCHEMA,
        "sample_index": sample_index,
        "dispatch_time": 0.0,
        "jobs": jobs,
    }


def generate_samples(case_dir: Path, job_counts: Sequence[int], samples: int, seed: int) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    for job_count in job_counts:
        path = case_file_path(case_dir, job_count)
        with path.open("w", encoding="utf-8") as f:
            for sample_index in range(samples):
                record = generate_case_record(job_count, sample_index, seed)
                f.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"Generated {samples} deterministic sample(s) for {job_count} jobs at {path}")


def require_field(raw_job: Dict[str, object], field: str, context: str):
    if field not in raw_job:
        raise BenchmarkCaseError(f"{context}: missing required job field '{field}'")
    return raw_job[field]


def normalize_station(raw_station, context: str) -> str:
    station = station_key_from_value(raw_station)
    if station is None:
        valid = ", ".join(STATIONS.keys())
        raise BenchmarkCaseError(f"{context}: unsupported station '{raw_station}'. Valid: {valid} or numeric ids.")
    return station


def normalize_dock(raw_dock, context: str) -> str:
    dock_text = str(raw_dock).strip()
    dock = DOCK_ALIASES.get(dock_text, DOCK_ALIASES.get(dock_text.upper()))
    if dock not in INBOUND_DOCK_KEYS:
        valid = ", ".join(INBOUND_DOCK_KEYS)
        raise BenchmarkCaseError(f"{context}: unsupported inbound_dock '{raw_dock}'. Valid: {valid}.")
    return dock


def normalize_job_type(raw_type, proc_time: float, context: str) -> str:
    job_type = str(raw_type).strip().upper()
    if job_type not in TYPE_DURATION:
        valid = ", ".join(sorted(TYPE_DURATION.keys()))
        raise BenchmarkCaseError(f"{context}: unsupported job type '{raw_type}'. Valid: {valid}.")
    expected = float(TYPE_DURATION[job_type])
    if not math.isclose(float(proc_time), expected):
        raise BenchmarkCaseError(
            f"{context}: proc_time {proc_time} does not match type {job_type} duration {expected}."
        )
    return job_type


def parse_float(value, field: str, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkCaseError(f"{context}: field '{field}' must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise BenchmarkCaseError(f"{context}: field '{field}' must be finite, got {value!r}")
    return parsed


def validate_and_build_event(data: Dict[str, object], event_index: int, path: Path) -> Dict[str, object]:
    context = f"{path} line {event_index + 1}"
    if not isinstance(data, dict):
        raise BenchmarkCaseError(f"{context}: each JSONL line must be an object")

    if "dispatch_time" not in data:
        raise BenchmarkCaseError(f"{context}: missing required event field 'dispatch_time'")
    dispatch_time = parse_float(data["dispatch_time"], "dispatch_time", context)

    raw_jobs = data.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise BenchmarkCaseError(f"{context}: 'jobs' must be a non-empty list")

    jobs = []
    for job_idx, raw_job in enumerate(raw_jobs):
        job_context = f"{context} job {job_idx}"
        if not isinstance(raw_job, dict):
            raise BenchmarkCaseError(f"{job_context}: job must be an object")

        raw_type = require_field(raw_job, "type", job_context)
        proc_time = parse_float(require_field(raw_job, "proc_time", job_context), "proc_time", job_context)
        station = normalize_station(require_field(raw_job, "station", job_context), job_context)
        inbound_dock = normalize_dock(require_field(raw_job, "inbound_dock", job_context), job_context)
        arrival_time = parse_float(require_field(raw_job, "arrival_time", job_context), "arrival_time", job_context)
        if arrival_time < 0.0:
            raise BenchmarkCaseError(f"{job_context}: arrival_time must be >= 0.0")

        job_type = normalize_job_type(raw_type, proc_time, job_context)
        jobs.append(
            Job(
                idx=job_idx,
                type_=job_type,
                duration=float(TYPE_DURATION[job_type]),
                station=station,
                inbound_dock=inbound_dock,
                arrival_time=arrival_time,
            )
        )

    sample_index = data.get("sample_index", event_index)
    return {
        "index": int(sample_index),
        "dispatch_time": dispatch_time,
        "jobs": jobs,
    }


def load_cases(case_dir: Path, job_count: int, limit: Optional[int] = None):
    path = case_file_path(case_dir, job_count)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing benchmark case file: {path}. Run with --generate to create deterministic cases."
        )

    events = []
    with path.open(encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            payload = line.strip()
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise BenchmarkCaseError(f"{path} line {line_index + 1}: invalid JSON: {exc}") from exc
            events.append(validate_and_build_event(data, line_index, path))

    if not events:
        raise BenchmarkCaseError(f"{path}: no benchmark events found. Run with --generate to regenerate.")

    target_index = os.environ.get(DISPATCH_EVENT_INDEX_ENV)
    if target_index is not None:
        events = [event for event in events if str(event["index"]) == str(target_index)]
        if not events:
            raise BenchmarkCaseError(f"{path}: no event matched {DISPATCH_EVENT_INDEX_ENV}={target_index!r}")

    if limit is not None:
        events = events[:limit]
    if not events:
        raise BenchmarkCaseError(f"{path}: no events selected for benchmarking")
    return events


def evaluate_individual(individual: Individual, jobs: Sequence[Job]) -> Dict[str, float]:
    """Executed makespan, idealised makespan, Lambda, Omega_q/Omega_r, and nu.

    Returns the full metric dict from ideal_evaluator so every algorithm in the
    grid is measured against the same idealised reference the paper uses. The
    caller must filter on `routable` before averaging `executed` or `penalty`:
    on a failed leg the decoder charges MAX_DEPTH (=100) per failure, so those
    numbers are penalty constants, not durations.
    """
    return ie.evaluate(individual, list(jobs))


def run_dispatch_rule(jobs: Sequence[Job], rule_name: str, seed: int) -> Tuple[Individual, float]:
    job_rule, amr_rule = rule_name.split("+", 1)
    return dr.solve_with_dispatching_rules(jobs, job_rule, amr_rule, seed=seed)


def run_ga(jobs: Sequence[Job], args: argparse.Namespace) -> Tuple[Individual, float]:
    original = (
        ga_normal.POPULATION_SIZE,
        ga_normal.GENERATIONS,
        ga_normal.routing_iters,
        ga_normal.collision_routing_iters,
    )
    ga_normal.POPULATION_SIZE = args.ga_population
    ga_normal.GENERATIONS = args.ga_generations
    ga_normal.routing_iters = args.ga_local_iters
    ga_normal.collision_routing_iters = args.ga_collision_iters
    try:
        start_time = time.perf_counter()
        individual, _ = ga_normal.evolve(list(jobs))
        solve_time = time.perf_counter() - start_time
    finally:
        (
            ga_normal.POPULATION_SIZE,
            ga_normal.GENERATIONS,
            ga_normal.routing_iters,
            ga_normal.collision_routing_iters,
        ) = original
    return individual, solve_time


def run_ga_precise(jobs: Sequence[Job], args: argparse.Namespace) -> Tuple[Individual, float]:
    start_time = time.perf_counter()
    individual, _ = evolve_precise(
        list(jobs),
        population_size=args.ga_population,
        generations=args.ga_generations,
        local_iters=args.ga_local_iters,
        verbose=args.verbose_ga,
    )
    solve_time = time.perf_counter() - start_time
    return individual, solve_time


def find_checkpoint(names: Sequence[str], search_dirs: Sequence[Path]) -> Path:
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return Path()


def load_compatible_state_dict(model, checkpoint: Path, torch_module, required_keys: Sequence[str]) -> str:
    if not checkpoint:
        raise RuntimeError("no operation-policy checkpoint found; retrain this model before benchmark inference")

    try:
        state_dict = torch_module.load(checkpoint, map_location="cpu")
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model_state = model.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        missing_required = [key for key in required_keys if key not in compatible]
        if missing_required:
            raise RuntimeError(
                "checkpoint is not an operation-policy checkpoint; retrain after the pickup/unload "
                f"conversion. Missing/incompatible tensors: {', '.join(missing_required)}"
            )
        model.load_state_dict(compatible, strict=False)
        return f"loaded {len(compatible)}/{len(model_state)} tensors from {checkpoint}"
    except Exception as exc:
        raise RuntimeError(f"failed to load {checkpoint}: {exc}") from exc


def build_attention_model(module, precise: bool, device):
    import torch

    model = module.SchedulerAttention(amr_in_dim=8, job_in_dim=16, hidden_dim=128, attention_layers=2).to(device)
    checkpoint_names = (
        ["attention_precise_scheduler_best.pth", "attention_scheduler_best.pth"]
        if precise
        else ["attention_scheduler_best.pth"]
    )
    checkpoint = find_checkpoint(checkpoint_names, [ROOT_DIR, ATTENTION_DIR])
    status = load_compatible_state_dict(
        model,
        checkpoint,
        torch,
        required_keys=("op_emb.weight", "policy_head.0.weight"),
    )
    model.eval()
    print(f"Attention{' precise' if precise else ''}: {status}")
    return model


def build_gnn_model(module, precise: bool, device):
    import torch

    model = module.SchedulerGNN(job_in_dim=16, amr_in_dim=8, hidden_dim=128, gin_layers=3).to(device)
    checkpoint_names = (
        ["gnn_precise_mpn_scheduler_best.pth", "gnn_mpn_scheduler_best.pth", "gnn_scheduler_best.pth"]
        if precise
        else ["gnn_mpn_scheduler_best.pth", "gnn_scheduler_best.pth"]
    )
    checkpoint = find_checkpoint(checkpoint_names, [ROOT_DIR, GNN_DIR])
    status = load_compatible_state_dict(
        model,
        checkpoint,
        torch,
        required_keys=("op_emb.weight", "operation_actor.0.weight"),
    )
    model.eval()
    print(f"GNN{' precise' if precise else ''}: {status}")
    return model


def build_extend_gnn_model(module, device):
    import torch

    model = module.ExtendSchedulerGNN(hidden_dim=128, gin_layers=3).to(device)
    checkpoint = find_checkpoint(["extend_gnn_scheduler_best.pth"], [ROOT_DIR, EXTEND_GNN_DIR])
    status = load_compatible_state_dict(
        model,
        checkpoint,
        torch,
        required_keys=("op_emb.weight", "operation_actor.0.weight"),
    )
    model.eval()
    print(f"extend_GNN: {status}")
    return model


def load_nn_context(selected_algorithms: Sequence[str], device_name: str):
    selected_nn_algorithms = set(selected_algorithms) & NN_ALGORITHMS
    if not selected_nn_algorithms:
        return {}

    context = {}
    try:
        import torch
    except Exception as exc:
        for algorithm in selected_nn_algorithms:
            context[f"{algorithm}_load_error"] = exc
        return context

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    context["device"] = device

    if "attention" in selected_algorithms:
        try:
            module = import_module_from_path("benchmark_attention", ATTENTION_DIR / "Attention.py")
            context["attention_module"] = module
            context["attention_model"] = build_attention_model(module, precise=False, device=device)
        except Exception as exc:
            context["attention_load_error"] = exc

    if "attention_precise" in selected_algorithms:
        try:
            module = import_module_from_path("benchmark_attention_precise", ATTENTION_DIR / "Attention_precise.py")
            context["attention_precise_module"] = module
            context["attention_precise_model"] = build_attention_model(module, precise=True, device=device)
        except Exception as exc:
            context["attention_precise_load_error"] = exc

    if "gnn" in selected_algorithms:
        try:
            module = import_module_from_path("benchmark_gnn", GNN_DIR / "GNN.py")
            context["gnn_module"] = module
            context["gnn_model"] = build_gnn_model(module, precise=False, device=device)
        except Exception as exc:
            context["gnn_load_error"] = exc

    if "gnn_precise" in selected_algorithms:
        try:
            module = import_module_from_path("benchmark_gnn_precise", GNN_DIR / "GNN_precise.py")
            context["gnn_precise_module"] = module
            context["gnn_precise_model"] = build_gnn_model(module, precise=True, device=device)
        except Exception as exc:
            context["gnn_precise_load_error"] = exc

    if "extend_gnn" in selected_algorithms:
        try:
            module = import_module_from_path("benchmark_extend_gnn", EXTEND_GNN_DIR / "extend_GNN.py")
            context["extend_gnn_module"] = module
            context["extend_gnn_model"] = build_extend_gnn_model(module, device=device)
        except Exception as exc:
            context["extend_gnn_load_error"] = exc

    return context


def require_nn_context(context: Dict, algorithm: str, key: str):
    load_error = context.get(f"{algorithm}_load_error")
    if load_error is not None:
        raise RuntimeError(f"{algorithm} unavailable: {load_error}") from load_error
    if key not in context:
        raise RuntimeError(f"{algorithm} unavailable: model context was not initialized")
    return context[key]


def run_attention(jobs: Sequence[Job], args: argparse.Namespace, context: Dict) -> Tuple[Individual, float]:
    module = require_nn_context(context, "attention", "attention_module")
    model = require_nn_context(context, "attention", "attention_model")

    start_time = time.perf_counter()
    individual, _, _ = module.solve_with_attention(list(jobs), model, deterministic=True)
    improvement = apply_neural_local_improvement(
        individual,
        list(jobs),
        simplified_iters=args.neural_local_iters,
        collision_iters=args.neural_collision_iters,
    )
    individual = improvement.individual
    solve_time = time.perf_counter() - start_time
    return individual, solve_time


def run_attention_precise(jobs: Sequence[Job], args: argparse.Namespace, context: Dict) -> Tuple[Individual, float]:
    module = require_nn_context(context, "attention_precise", "attention_precise_module")
    model = require_nn_context(context, "attention_precise", "attention_precise_model")

    start_time = time.perf_counter()
    individual, _, _ = module.solve_with_attention(list(jobs), model, deterministic=True)
    improvement = apply_neural_local_improvement(
        individual,
        list(jobs),
        simplified_iters=args.precise_neural_local_iters,
        collision_iters=args.precise_neural_collision_iters,
    )
    individual = improvement.individual
    solve_time = time.perf_counter() - start_time
    return individual, solve_time


def run_gnn(jobs: Sequence[Job], args: argparse.Namespace, context: Dict) -> Tuple[Individual, float]:
    module = require_nn_context(context, "gnn", "gnn_module")
    model = require_nn_context(context, "gnn", "gnn_model")

    start_time = time.perf_counter()
    individual, _, _ = module.solve_with_gnn(list(jobs), model, deterministic=True)
    improvement = apply_neural_local_improvement(
        individual,
        list(jobs),
        simplified_iters=args.neural_local_iters,
        collision_iters=args.neural_collision_iters,
    )
    individual = improvement.individual
    solve_time = time.perf_counter() - start_time
    return individual, solve_time


def run_gnn_precise(jobs: Sequence[Job], args: argparse.Namespace, context: Dict) -> Tuple[Individual, float]:
    module = require_nn_context(context, "gnn_precise", "gnn_precise_module")
    model = require_nn_context(context, "gnn_precise", "gnn_precise_model")

    start_time = time.perf_counter()
    individual, _, _ = module.solve_with_gnn(list(jobs), model, deterministic=True)
    improvement = apply_neural_local_improvement(
        individual,
        list(jobs),
        simplified_iters=args.precise_neural_local_iters,
        collision_iters=args.precise_neural_collision_iters,
    )
    individual = improvement.individual
    solve_time = time.perf_counter() - start_time
    return individual, solve_time


def run_extend_gnn(jobs: Sequence[Job], args: argparse.Namespace, context: Dict) -> Tuple[Individual, float]:
    module = require_nn_context(context, "extend_gnn", "extend_gnn_module")
    model = require_nn_context(context, "extend_gnn", "extend_gnn_model")

    start_time = time.perf_counter()
    individual, _, _ = module.solve_with_extend_gnn(list(jobs), model, deterministic=True)
    improvement = apply_neural_local_improvement(
        individual,
        list(jobs),
        simplified_iters=args.neural_local_iters,
        collision_iters=args.neural_collision_iters,
    )
    individual = improvement.individual
    solve_time = time.perf_counter() - start_time
    return individual, solve_time


def selected_dispatch_rules(args: argparse.Namespace) -> List[str]:
    job_rules = dr.parse_rule_list(args.job_rules, dr.JOB_RULES, "job")
    amr_rules = dr.parse_rule_list(args.amr_rules, dr.AMR_RULES, "AMR")
    return [f"{job_rule}+{amr_rule}" for job_rule, amr_rule in dr.make_rule_pairs(job_rules, amr_rules)]


def expand_algorithms(raw_algorithms: Sequence[str], dispatch_rule_names: Sequence[str]) -> List[str]:
    selected = []
    for algorithm in raw_algorithms:
        if algorithm == "all":
            selected.extend(list(dispatch_rule_names))
            selected.extend(["ga", "attention", "attention_precise", "gnn", "gnn_precise", "extend_gnn"])
        elif algorithm == "dispatching_rules":
            selected.extend(list(dispatch_rule_names))
        else:
            selected.append(algorithm)

    unknown = [name for name in selected if name not in SPECIAL_ALGORITHMS and "+" not in name]
    if unknown:
        raise ValueError(f"Unknown algorithm(s): {', '.join(unknown)}")

    deduped = []
    seen = set()
    for algorithm in selected:
        if algorithm not in seen:
            deduped.append(algorithm)
            seen.add(algorithm)
    return deduped


def run_algorithm(
    algorithm: str,
    jobs: Sequence[Job],
    args: argparse.Namespace,
    context: Dict,
    seed: int,
) -> Tuple[Individual, float]:
    random.seed(seed)
    if "+" in algorithm:
        return run_dispatch_rule(jobs, algorithm, seed)
    if algorithm == "ga":
        return run_ga(jobs, args)
    if algorithm == "ga_precise":
        return run_ga_precise(jobs, args)
    if algorithm == "attention":
        return run_attention(jobs, args, context)
    if algorithm == "attention_precise":
        return run_attention_precise(jobs, args, context)
    if algorithm == "gnn":
        return run_gnn(jobs, args, context)
    if algorithm == "gnn_precise":
        return run_gnn_precise(jobs, args, context)
    if algorithm == "extend_gnn":
        return run_extend_gnn(jobs, args, context)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def safe_float(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean_or_nan(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def pstdev_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[int, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["job_count"]), str(row["algorithm"])), []).append(row)

    summary_rows = []
    for (job_count, algorithm), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        successful = [row for row in group if row["status"] != "failed"]
        # "valid" == the executor routed every leg (nu = 0). EVERY duration-valued
        # aggregate is computed over these rows only. A run with nu > 0 has had
        # MAX_DEPTH (=100) added to an AMR's availability per failed leg, so its
        # "makespan" is a penalty constant, not elapsed time -- averaging it in
        # produces a number that looks like congestion and is not. This project
        # has already been burnt by that once: a 33.9% "congestion tax" collapsed
        # to 16.3% when the failed runs were dropped.
        valid = [row for row in successful if str(row["valid"]).lower() == "true"]
        feasible_makespans = [float(row["makespan"]) for row in valid]
        ideals = [
            parsed for parsed in (safe_float(str(row.get("ideal_makespan", "nan"))) for row in valid)
            if parsed is not None
        ]
        penalties = [
            parsed for parsed in (safe_float(str(row.get("penalty", "nan"))) for row in valid)
            if parsed is not None
        ]
        omega_qs = [
            parsed for parsed in (safe_float(str(row.get("omega_q", "nan"))) for row in valid)
            if parsed is not None
        ]
        omega_rs = [
            parsed for parsed in (safe_float(str(row.get("omega_r", "nan"))) for row in valid)
            if parsed is not None
        ]
        solve_times = [
            parsed
            for parsed in (safe_float(str(row["computation_time"])) for row in successful)
            if parsed is not None
        ]
        invalids = [int(row["invalid_jobs"]) for row in successful]
        runs = len(group)
        valid_runs = len(valid)
        summary_rows.append(
            {
                "job_count": job_count,
                "algorithm": algorithm,
                "samples": len({row["sample_index"] for row in group}),
                "runs": runs,
                "failures": sum(1 for row in group if row["status"] == "failed"),
                "valid_runs": valid_runs,
                "valid_rate": valid_runs / runs if runs else 0.0,
                "mean_feasible_makespan": mean_or_nan(feasible_makespans),
                "std_feasible_makespan": pstdev_or_nan(feasible_makespans),
                "mean_feasible_ideal": mean_or_nan(ideals),
                "mean_feasible_penalty": mean_or_nan(penalties),
                "mean_feasible_omega_q": mean_or_nan(omega_qs),
                "mean_feasible_omega_r": mean_or_nan(omega_rs),
                "mean_computation_time": mean_or_nan(solve_times),
                "mean_invalid_jobs": mean_or_nan(invalids),
            }
        )
    return summary_rows


def format_float(value) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{parsed:.6f}" if math.isfinite(parsed) else "nan"


def write_detail_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "job_count",
        "sample_index",
        "algorithm",
        "status",
        "makespan",
        "ideal_makespan",
        "penalty",
        "omega_q",
        "omega_r",
        "computation_time",
        "invalid_jobs",
        "valid",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "job_count",
        "algorithm",
        "samples",
        "runs",
        "failures",
        "valid_runs",
        "valid_rate",
        "mean_feasible_makespan",
        "std_feasible_makespan",
        "mean_feasible_ideal",
        "mean_feasible_penalty",
        "mean_feasible_omega_q",
        "mean_feasible_omega_r",
        "mean_computation_time",
        "mean_invalid_jobs",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in (
                "valid_rate",
                "mean_feasible_makespan",
                "std_feasible_makespan",
                "mean_feasible_ideal",
                "mean_feasible_penalty",
                "mean_feasible_omega_q",
                "mean_feasible_omega_r",
                "mean_computation_time",
                "mean_invalid_jobs",
            ):
                formatted[key] = format_float(formatted[key])
            writer.writerow(formatted)


def make_result_row(
    job_count: int,
    sample_index,
    algorithm: str,
    status: str,
    makespan,
    solve_time,
    invalid_jobs: int,
    valid: bool,
    ideal=float("nan"),
    penalty=float("nan"),
    omega_q=float("nan"),
    omega_r=float("nan"),
) -> Dict[str, object]:
    return {
        "job_count": job_count,
        "sample_index": sample_index,
        "algorithm": algorithm,
        "status": status,
        "makespan": format_float(makespan),
        "ideal_makespan": format_float(ideal),
        "penalty": format_float(penalty),
        "omega_q": format_float(omega_q),
        "omega_r": format_float(omega_r),
        "computation_time": format_float(solve_time),
        "invalid_jobs": invalid_jobs,
        "valid": "true" if valid else "false",
    }


def run(args: argparse.Namespace) -> None:
    job_counts = parse_csv_list(args.job_counts, int)
    case_dir = Path(args.case_dir)
    if args.generate:
        generate_samples(case_dir, job_counts, args.samples, args.seed)

    dispatch_rule_names = selected_dispatch_rules(args)
    algorithms = expand_algorithms(parse_csv_list(args.algorithms), dispatch_rule_names)
    context = load_nn_context(algorithms, args.device)

    detail_rows = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Case directory: {case_dir}")
    print(f"Job counts: {job_counts}")
    print(f"Samples per job count: {args.samples}")
    print(f"Algorithms: {len(algorithms)}")

    for job_count in job_counts:
        events = load_cases(case_dir, job_count, limit=args.samples)
        print(f"\n=== Job count {job_count} | Loaded {len(events)} validated sample(s) ===")

        for event_pos, event in enumerate(events, start=1):
            jobs = event["jobs"]
            sample_index = event["index"]
            print(f"Sample {event_pos}/{len(events)} (event {sample_index})")

            for algorithm in algorithms:
                seed = args.seed + job_count * 1000 + int(sample_index) * 37 + event_pos
                try:
                    individual, solve_time = run_algorithm(algorithm, jobs, args, context, seed)
                    metrics = evaluate_individual(individual, jobs)
                    makespan = metrics["executed"]
                    invalid_count = int(metrics["nu"])
                    valid = invalid_count == 0
                    status = "ok" if valid else "invalid"
                    detail_rows.append(
                        make_result_row(
                            job_count,
                            sample_index,
                            algorithm,
                            status,
                            makespan,
                            solve_time,
                            invalid_count,
                            valid,
                            ideal=metrics["ideal"],
                            penalty=metrics["penalty"],
                            omega_q=metrics["omega_q"],
                            omega_r=metrics["omega_r"],
                        )
                    )
                    print(
                        f"  {algorithm:45s} status={status:7s} "
                        f"makespan={makespan:8.2f} ideal={metrics['ideal']:8.2f} "
                        f"Lambda={100 * metrics['penalty']:5.1f}% "
                        f"time={solve_time:.4f}s invalid={invalid_count}"
                    )
                except Exception as exc:
                    detail_rows.append(
                        make_result_row(
                            job_count,
                            sample_index,
                            algorithm,
                            "failed",
                            float("nan"),
                            float("nan"),
                            -1,
                            False,
                        )
                    )
                    print(f"  {algorithm:45s} FAILED: {exc}")

    detail_path = output_dir / args.detail_csv
    summary_path = output_dir / args.summary_csv
    write_detail_csv(detail_path, detail_rows)
    write_summary_csv(summary_path, summarize(detail_rows))

    print(f"\nDetailed benchmark rows saved to {detail_path}")
    print(f"Feasibility-aware benchmark summary saved to {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic static AMR-DFJSP samples and benchmark static algorithms."
    )
    parser.add_argument("--generate", action="store_true", help="Generate deterministic JSONL samples before benchmarking")
    parser.add_argument("--samples", type=int, default=100, help="Samples per job count")
    parser.add_argument("--job_counts", type=str, default="20,40,60,80,100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case_dir", type=str, default=str(DEFAULT_CASE_DIR), help="Directory containing dispatch_inbox_<N>.jsonl cases")
    parser.add_argument(
        "--algorithms",
        type=str,
        default="all",
        help="Comma list: all, dispatching_rules, ga, ga_precise, attention, attention_precise, gnn, gnn_precise, extend_gnn.",
    )
    parser.add_argument("--job_rules", type=str, default="all", help=f"Dispatch job rules: all or comma list from {', '.join(dr.JOB_RULES)}")
    parser.add_argument("--amr_rules", type=str, default="all", help=f"Dispatch AMR rules: all or comma list from {', '.join(dr.AMR_RULES)}")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--output_dir", type=str, default=str(STATIC_DIR / "benchmark_results"))
    parser.add_argument("--detail_csv", type=str, default="static_benchmark_details.csv")
    parser.add_argument("--summary_csv", type=str, default="static_benchmark_summary.csv")
    parser.add_argument("--ga_population", type=int, default=ga_normal.POPULATION_SIZE)
    parser.add_argument("--ga_generations", type=int, default=ga_normal.GENERATIONS)
    parser.add_argument("--ga_local_iters", type=int, default=routing_iters)
    parser.add_argument("--ga_collision_iters", type=int, default=collision_routing_iters)
    parser.add_argument("--neural_local_iters", type=int, default=routing_iters)
    parser.add_argument("--neural_collision_iters", type=int, default=collision_routing_iters)
    parser.add_argument("--precise_neural_local_iters", type=int, default=0)
    parser.add_argument("--precise_neural_collision_iters", type=int, default=0)
    parser.add_argument("--verbose_ga", action="store_true", help="Print GA precise generation logs")
    return parser


if __name__ == "__main__":
    try:
        run(build_parser().parse_args())
    except (BenchmarkCaseError, FileNotFoundError) as exc:
        print(f"Benchmark case validation failed: {exc}")
        print("Regenerate deterministic benchmark cases with --generate, or point --case_dir at valid v1 cases.")
        raise SystemExit(2)
