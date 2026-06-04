import argparse
import csv
import importlib.util
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent
GA_DIR = STATIC_DIR / "GA"
ATTENTION_DIR = STATIC_DIR / "Attention"
GNN_DIR = STATIC_DIR / "GNN"
DISPATCH_RULES_DIR = STATIC_DIR / "dispatching_rules"
STATIC_TESTCASE_DIR = ROOT_DIR / "test_case" / "static"
RANDOM_GENERATOR_PATH = STATIC_TESTCASE_DIR / "Random_Job_Generator.py"

for path in (STATIC_DIR, ATTENTION_DIR, GNN_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from GA.GA import (  # noqa: E402
    DISPATCH_EVENT_INDEX_ENV,
    Individual,
    Job,
    collision_routing_iters,
    decode_schedule_tick_by_tick,
    load_dispatch_events,
    local_improve,
    routing_iters,
)

from GA import GA as ga_normal  # noqa: E402
from GA.GA_precise import evolve_precise  # noqa: E402
from dispatching_rules import dispatching_rules as dr  # noqa: E402


DEFAULT_JOB_COUNTS = (20, 40, 60, 80, 100)
NN_ALGORITHMS = {"attention", "attention_precise", "gnn", "gnn_precise"}
SEARCH_ALGORITHMS = {"ga", "ga_precise"}
SPECIAL_ALGORITHMS = SEARCH_ALGORITHMS | NN_ALGORITHMS | {"dispatching_rules"}


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


def generate_samples(job_counts: Sequence[int], samples: int, seed: int) -> None:
    generator = import_module_from_path("static_random_job_generator", RANDOM_GENERATOR_PATH)
    for job_count in job_counts:
        random.seed(seed + job_count)
        generator.generate_data(samples, job_count)


def load_cases(job_count: int, limit: int = None):
    path = STATIC_TESTCASE_DIR / f"dispatch_inbox_{job_count}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing sample file: {path}. Run with --generate first.")

    events = load_dispatch_events(path)
    target_index = os.environ.get(DISPATCH_EVENT_INDEX_ENV)
    if target_index is not None:
        events = [event for event in events if str(event["index"]) == str(target_index)]
    if limit is not None:
        events = events[:limit]
    return events


def evaluate_individual(individual: Individual, jobs: Sequence[Job]) -> Tuple[float, int]:
    availability, _, _, _, invalid_count = decode_schedule_tick_by_tick(
        individual,
        list(jobs),
        need_log=False,
        check_collision=True,
    )
    return max(availability.values()), invalid_count


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


def load_compatible_state_dict(model, checkpoint: Path, torch_module) -> str:
    if not checkpoint:
        return "missing"

    try:
        state_dict = torch_module.load(checkpoint, map_location="cpu")
        model_state = model.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        model.load_state_dict(compatible, strict=False)
        return f"loaded {len(compatible)}/{len(model_state)} tensors from {checkpoint}"
    except Exception as exc:
        return f"failed to load {checkpoint}: {exc}"


def build_attention_model(module, precise: bool, device):
    import torch

    model = module.SchedulerAttention(amr_in_dim=8, job_in_dim=11, hidden_dim=128, attention_layers=2).to(device)
    checkpoint_names = (
        ["attention_precise_scheduler_best.pth", "attention_scheduler_best.pth"]
        if precise
        else ["attention_scheduler_best.pth"]
    )
    checkpoint = find_checkpoint(checkpoint_names, [ROOT_DIR, ATTENTION_DIR])
    status = load_compatible_state_dict(model, checkpoint, torch)
    model.eval()
    print(f"Attention{' precise' if precise else ''}: {status}")
    return model


def build_gnn_model(module, precise: bool, device):
    import torch

    model = module.SchedulerGNN(job_in_dim=12, amr_in_dim=8, hidden_dim=128, gin_layers=3).to(device)
    checkpoint_names = (
        ["gnn_precise_mpn_scheduler_best.pth", "gnn_mpn_scheduler_best.pth", "gnn_scheduler_best.pth"]
        if precise
        else ["gnn_mpn_scheduler_best.pth", "gnn_scheduler_best.pth"]
    )
    checkpoint = find_checkpoint(checkpoint_names, [ROOT_DIR, GNN_DIR])
    status = load_compatible_state_dict(model, checkpoint, torch)
    model.eval()
    print(f"GNN{' precise' if precise else ''}: {status}")
    return model


def load_nn_context(selected_algorithms: Sequence[str], device_name: str):
    if not (set(selected_algorithms) & NN_ALGORITHMS):
        return {}

    import torch

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    context = {"device": device}

    if "attention" in selected_algorithms:
        module = import_module_from_path("benchmark_attention", ATTENTION_DIR / "Attention.py")
        context["attention_module"] = module
        context["attention_model"] = build_attention_model(module, precise=False, device=device)

    if "attention_precise" in selected_algorithms:
        module = import_module_from_path("benchmark_attention_precise", ATTENTION_DIR / "Attention_precise.py")
        context["attention_precise_module"] = module
        context["attention_precise_model"] = build_attention_model(module, precise=True, device=device)

    if "gnn" in selected_algorithms:
        module = import_module_from_path("benchmark_gnn", GNN_DIR / "GNN.py")
        context["gnn_module"] = module
        context["gnn_model"] = build_gnn_model(module, precise=False, device=device)

    if "gnn_precise" in selected_algorithms:
        module = import_module_from_path("benchmark_gnn_precise", GNN_DIR / "GNN_precise.py")
        context["gnn_precise_module"] = module
        context["gnn_precise_model"] = build_gnn_model(module, precise=True, device=device)

    return context


def run_attention(jobs: Sequence[Job], args: argparse.Namespace, context: Dict) -> Tuple[Individual, float]:
    module = context["attention_module"]
    model = context["attention_model"]

    start_time = time.perf_counter()
    individual, _, _ = module.solve_with_attention(list(jobs), model, deterministic=True)
    if args.neural_local_iters > 0:
        individual = local_improve(individual, list(jobs), max_iters=args.neural_local_iters)
    if args.neural_collision_iters > 0:
        individual = local_improve(
            individual,
            list(jobs),
            max_iters=args.neural_collision_iters,
            check_collision=True,
        )
    solve_time = time.perf_counter() - start_time
    return individual, solve_time


def run_attention_precise(jobs: Sequence[Job], context: Dict) -> Tuple[Individual, float]:
    module = context["attention_precise_module"]
    model = context["attention_precise_model"]

    start_time = time.perf_counter()
    individual, _, _ = module.solve_with_attention(list(jobs), model, deterministic=True)
    solve_time = time.perf_counter() - start_time
    return individual, solve_time


def run_gnn(jobs: Sequence[Job], args: argparse.Namespace, context: Dict) -> Tuple[Individual, float]:
    module = context["gnn_module"]
    model = context["gnn_model"]

    start_time = time.perf_counter()
    individual, _, _ = module.solve_with_gnn(list(jobs), model, deterministic=True)
    if args.neural_local_iters > 0:
        individual = local_improve(individual, list(jobs), max_iters=args.neural_local_iters)
    if args.neural_collision_iters > 0:
        individual = local_improve(
            individual,
            list(jobs),
            max_iters=args.neural_collision_iters,
            check_collision=True,
        )
    solve_time = time.perf_counter() - start_time
    return individual, solve_time


def run_gnn_precise(jobs: Sequence[Job], context: Dict) -> Tuple[Individual, float]:
    module = context["gnn_precise_module"]
    model = context["gnn_precise_model"]

    start_time = time.perf_counter()
    individual, _, _ = module.solve_with_gnn(list(jobs), model, deterministic=True)
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
            selected.extend(["ga", "attention", "attention_precise", "gnn", "gnn_precise"])
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
    if "+" in algorithm:
        return run_dispatch_rule(jobs, algorithm, seed)
    if algorithm == "ga":
        return run_ga(jobs, args)
    if algorithm == "ga_precise":
        return run_ga_precise(jobs, args)
    if algorithm == "attention":
        return run_attention(jobs, args, context)
    if algorithm == "attention_precise":
        return run_attention_precise(jobs, context)
    if algorithm == "gnn":
        return run_gnn(jobs, args, context)
    if algorithm == "gnn_precise":
        return run_gnn_precise(jobs, context)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[int, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["job_count"]), str(row["algorithm"])), []).append(row)

    summary_rows = []
    for (job_count, algorithm), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        makespans = [float(row["makespan"]) for row in group]
        solve_times = [float(row["computation_time"]) for row in group]
        invalids = [int(row["invalid_jobs"]) for row in group]
        summary_rows.append(
            {
                "job_count": job_count,
                "algorithm": algorithm,
                "samples": len(group),
                "mean_makespan": statistics.fmean(makespans),
                "std_makespan": statistics.pstdev(makespans) if len(makespans) > 1 else 0.0,
                "mean_computation_time": statistics.fmean(solve_times),
                "std_computation_time": statistics.pstdev(solve_times) if len(solve_times) > 1 else 0.0,
                "mean_invalid_jobs": statistics.fmean(invalids),
            }
        )
    return summary_rows


def write_detail_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "job_count",
        "sample_index",
        "algorithm",
        "makespan",
        "computation_time",
        "invalid_jobs",
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
        "mean_makespan",
        "std_makespan",
        "mean_computation_time",
        "std_computation_time",
        "mean_invalid_jobs",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ("mean_makespan", "std_makespan", "mean_computation_time", "std_computation_time", "mean_invalid_jobs"):
                formatted[key] = f"{float(formatted[key]):.6f}"
            writer.writerow(formatted)


def run(args: argparse.Namespace) -> None:
    job_counts = parse_csv_list(args.job_counts, int)
    if args.generate:
        generate_samples(job_counts, args.samples, args.seed)

    dispatch_rule_names = selected_dispatch_rules(args)
    algorithms = expand_algorithms(parse_csv_list(args.algorithms), dispatch_rule_names)
    context = load_nn_context(algorithms, args.device)

    detail_rows = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Job counts: {job_counts}")
    print(f"Samples per job count: {args.samples}")
    print(f"Algorithms: {len(algorithms)}")

    for job_count in job_counts:
        events = load_cases(job_count, limit=args.samples)
        print(f"\n=== Job count {job_count} | Loaded {len(events)} sample(s) ===")

        for event_pos, event in enumerate(events, start=1):
            jobs = event["jobs"]
            sample_index = event["index"]
            print(f"Sample {event_pos}/{len(events)} (event {sample_index})")

            for algorithm in algorithms:
                seed = args.seed + job_count * 1000 + int(event_pos)
                try:
                    individual, solve_time = run_algorithm(algorithm, jobs, args, context, seed)
                    makespan, invalid_count = evaluate_individual(individual, jobs)
                    detail_rows.append(
                        {
                            "job_count": job_count,
                            "sample_index": sample_index,
                            "algorithm": algorithm,
                            "makespan": f"{makespan:.6f}",
                            "computation_time": f"{solve_time:.6f}",
                            "invalid_jobs": invalid_count,
                        }
                    )
                    print(f"  {algorithm:45s} makespan={makespan:8.2f} time={solve_time:.4f}s invalid={invalid_count}")
                except Exception as exc:
                    detail_rows.append(
                        {
                            "job_count": job_count,
                            "sample_index": sample_index,
                            "algorithm": algorithm,
                            "makespan": "nan",
                            "computation_time": "nan",
                            "invalid_jobs": -1,
                        }
                    )
                    print(f"  {algorithm:45s} FAILED: {exc}")

    detail_path = output_dir / args.detail_csv
    summary_path = output_dir / args.summary_csv
    write_detail_csv(detail_path, detail_rows)

    valid_rows = [row for row in detail_rows if row["makespan"] != "nan" and row["computation_time"] != "nan"]
    summary_rows = summarize(valid_rows)
    write_summary_csv(summary_path, summary_rows)

    print(f"\nDetailed benchmark rows saved to {detail_path}")
    print(f"Mean benchmark summary saved to {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate static AMR-DFJSP samples and benchmark dispatching rules, GA, Attention, and GNN."
    )
    parser.add_argument("--generate", action="store_true", help="Generate JSONL samples before benchmarking")
    parser.add_argument("--samples", type=int, default=100, help="Samples per job count")
    parser.add_argument("--job_counts", type=str, default="20,40,60,80,100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--algorithms",
        type=str,
        default="all",
        help="Comma list: all, dispatching_rules, ga, ga_precise, attention, attention_precise, gnn, gnn_precise.",
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
    parser.add_argument("--verbose_ga", action="store_true", help="Print GA precise generation logs")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
