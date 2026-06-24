"""Two-phase local-improvement study for neural AMR-DFJSP schedulers.

The tuning phase evaluates the complete simplified/collision-aware grid on a
small deterministic case subset.  The validation phase evaluates only the
baseline and three Pareto representatives on held-out cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


STATIC_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = STATIC_DIR / "benchmark_results" / "local_improvement"
DEFAULT_MODELS = ("attention", "gnn", "attention_precise", "gnn_precise")
DETAIL_FIELDS = (
    "phase",
    "model",
    "job_count",
    "sample_index",
    "repeat",
    "simplified_iters",
    "collision_iters",
    "status",
    "valid",
    "makespan",
    "invalid_jobs",
    "inference_time",
    "simplified_time",
    "collision_time",
    "postprocess_time",
    "total_time",
    "simplified_seed",
    "collision_seed",
    "error",
)
SUMMARY_FIELDS = (
    "phase",
    "model",
    "job_count",
    "simplified_iters",
    "collision_iters",
    "runs",
    "failures",
    "valid_runs",
    "valid_rate",
    "mean_makespan",
    "mean_feasible_makespan",
    "std_feasible_makespan",
    "mean_paired_makespan_ratio",
    "mean_inference_time",
    "mean_simplified_time",
    "mean_collision_time",
    "mean_postprocess_time",
    "mean_total_time",
)
RECOMMENDATION_FIELDS = (
    "phase",
    "model",
    "scope",
    "simplified_iters",
    "collision_iters",
    "valid_rate",
    "baseline_valid_rate",
    "mean_paired_makespan_ratio",
    "mean_total_time",
    "pareto_size",
    "selection",
)


def parse_nonnegative_grid(raw: str) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise argparse.ArgumentTypeError("Iteration grids cannot contain negative values")
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("Iteration grid cannot be empty")
    return values


def parse_csv_values(raw: str, cast=str) -> List:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def expand_grid(simplified: Sequence[int], collision: Sequence[int]) -> List[Tuple[int, int]]:
    return [(local_iters, collision_iters) for local_iters in simplified for collision_iters in collision]


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def detail_key(row: Mapping[str, object]) -> Tuple[str, ...]:
    return tuple(
        str(row[name])
        for name in (
            "phase",
            "model",
            "job_count",
            "sample_index",
            "repeat",
            "simplified_iters",
            "collision_iters",
        )
    )


def safe_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed


def mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float("nan")


def pstdev(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return statistics.pstdev(finite) if len(finite) > 1 else 0.0


def format_float(value: object) -> str:
    parsed = safe_float(value)
    return f"{parsed:.9f}" if math.isfinite(parsed) else "nan"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


class DetailWriter:
    def __init__(self, path: Path, resume: bool):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.existing_rows = read_csv_rows(path) if resume else []
        self.completed = {detail_key(row) for row in self.existing_rows}
        mode = "a" if resume and path.exists() else "w"
        self.handle = path.open(mode, newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=list(DETAIL_FIELDS))
        if mode == "w" or path.stat().st_size == 0:
            self.writer.writeheader()
            self.handle.flush()

    def append(self, row: Mapping[str, object]) -> bool:
        key = detail_key(row)
        if key in self.completed:
            return False
        self.writer.writerow({field: row.get(field, "") for field in DETAIL_FIELDS})
        self.handle.flush()
        self.completed.add(key)
        return True

    def close(self) -> None:
        self.handle.close()


def _group_key(row: Mapping[str, object]) -> Tuple[str, str, int, int, int]:
    return (
        str(row["phase"]),
        str(row["model"]),
        int(row["job_count"]),
        int(row["simplified_iters"]),
        int(row["collision_iters"]),
    )


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    baseline = {}
    for row in rows:
        if int(row["simplified_iters"]) == 0 and int(row["collision_iters"]) == 0:
            baseline[(row["phase"], row["model"], row["job_count"], row["sample_index"], row["repeat"])] = row

    grouped: MutableMapping[Tuple[str, str, int, int, int], List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    summaries: List[Dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        phase, model, job_count, simplified_iters, collision_iters = key
        successful = [row for row in group if row["status"] != "failed"]
        valid = [row for row in successful if str(row["valid"]).lower() == "true"]
        ratios = []
        for row in successful:
            base = baseline.get((phase, model, row["job_count"], row["sample_index"], row["repeat"]))
            if not base or base["status"] == "failed":
                continue
            base_makespan = safe_float(base["makespan"])
            current_makespan = safe_float(row["makespan"])
            if math.isfinite(base_makespan) and base_makespan > 0 and math.isfinite(current_makespan):
                ratios.append(current_makespan / base_makespan)

        summaries.append(
            {
                "phase": phase,
                "model": model,
                "job_count": job_count,
                "simplified_iters": simplified_iters,
                "collision_iters": collision_iters,
                "runs": len(group),
                "failures": len(group) - len(successful),
                "valid_runs": len(valid),
                "valid_rate": len(valid) / len(group) if group else 0.0,
                "mean_makespan": mean(safe_float(row["makespan"]) for row in successful),
                "mean_feasible_makespan": mean(safe_float(row["makespan"]) for row in valid),
                "std_feasible_makespan": pstdev(safe_float(row["makespan"]) for row in valid),
                "mean_paired_makespan_ratio": mean(ratios),
                "mean_inference_time": mean(safe_float(row["inference_time"]) for row in successful),
                "mean_simplified_time": mean(safe_float(row["simplified_time"]) for row in successful),
                "mean_collision_time": mean(safe_float(row["collision_time"]) for row in successful),
                "mean_postprocess_time": mean(safe_float(row["postprocess_time"]) for row in successful),
                "mean_total_time": mean(safe_float(row["total_time"]) for row in successful),
            }
        )

    overall_groups: MutableMapping[Tuple[str, str, int, int], List[Mapping[str, object]]] = {}
    for row in summaries:
        overall_groups.setdefault(
            (str(row["phase"]), str(row["model"]), int(row["simplified_iters"]), int(row["collision_iters"])),
            [],
        ).append(row)

    for (phase, model, simplified_iters, collision_iters), group in sorted(overall_groups.items()):
        overall = {
            "phase": phase,
            "model": model,
            "job_count": "overall",
            "simplified_iters": simplified_iters,
            "collision_iters": collision_iters,
            "runs": sum(int(row["runs"]) for row in group),
            "failures": sum(int(row["failures"]) for row in group),
            "valid_runs": sum(int(row["valid_runs"]) for row in group),
        }
        for metric in (
            "valid_rate",
            "mean_makespan",
            "mean_feasible_makespan",
            "std_feasible_makespan",
            "mean_paired_makespan_ratio",
            "mean_inference_time",
            "mean_simplified_time",
            "mean_collision_time",
            "mean_postprocess_time",
            "mean_total_time",
        ):
            overall[metric] = mean(safe_float(row[metric]) for row in group)
        summaries.append(overall)
    return summaries


def config_tuple(row: Mapping[str, object]) -> Tuple[int, int]:
    return int(row["simplified_iters"]), int(row["collision_iters"])


def pareto_front(rows: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    finite = [
        row
        for row in rows
        if math.isfinite(safe_float(row["mean_paired_makespan_ratio"]))
        and math.isfinite(safe_float(row["mean_total_time"]))
    ]
    frontier = []
    for candidate in finite:
        quality = safe_float(candidate["mean_paired_makespan_ratio"])
        runtime = safe_float(candidate["mean_total_time"])
        dominated = False
        for other in finite:
            if other is candidate:
                continue
            other_quality = safe_float(other["mean_paired_makespan_ratio"])
            other_runtime = safe_float(other["mean_total_time"])
            if (
                other_quality <= quality
                and other_runtime <= runtime
                and (other_quality < quality or other_runtime < runtime)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: (safe_float(row["mean_total_time"]), safe_float(row["mean_paired_makespan_ratio"]), config_tuple(row)))


def eligible_and_frontier(rows: Sequence[Mapping[str, object]]) -> Tuple[List[Mapping[str, object]], List[Mapping[str, object]], float]:
    baseline = next((row for row in rows if config_tuple(row) == (0, 0)), None)
    if baseline is None:
        raise ValueError("Cannot select a recommendation without the (0, 0) baseline")
    baseline_valid_rate = safe_float(baseline["valid_rate"])
    eligible = [row for row in rows if safe_float(row["valid_rate"]) + 1e-12 >= baseline_valid_rate]
    return eligible, pareto_front(eligible), baseline_valid_rate


def select_knee(frontier: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not frontier:
        raise ValueError("Cannot select a knee from an empty Pareto frontier")
    qualities = [safe_float(row["mean_paired_makespan_ratio"]) for row in frontier]
    runtimes = [safe_float(row["mean_total_time"]) for row in frontier]
    quality_min, quality_max = min(qualities), max(qualities)
    runtime_min, runtime_max = min(runtimes), max(runtimes)

    def normalized(value: float, low: float, high: float) -> float:
        return 0.0 if math.isclose(low, high) else (value - low) / (high - low)

    def rank(row: Mapping[str, object]):
        quality = safe_float(row["mean_paired_makespan_ratio"])
        runtime = safe_float(row["mean_total_time"])
        distance = math.hypot(
            normalized(quality, quality_min, quality_max),
            normalized(runtime, runtime_min, runtime_max),
        )
        simplified_iters, collision_iters = config_tuple(row)
        return distance, quality, runtime, simplified_iters + collision_iters, simplified_iters, collision_iters

    return min(frontier, key=rank)


def select_shortlist(rows: Sequence[Mapping[str, object]]) -> List[Tuple[int, int]]:
    eligible, frontier, _ = eligible_and_frontier(rows)
    if not frontier:
        return [(0, 0)]
    knee = select_knee(frontier)
    quality_best = min(
        eligible,
        key=lambda row: (
            safe_float(row["mean_paired_makespan_ratio"]),
            safe_float(row["mean_total_time"]),
            sum(config_tuple(row)),
            config_tuple(row),
        ),
    )
    nonzero = [row for row in frontier if config_tuple(row) != (0, 0)]
    fastest_nonzero = min(
        nonzero,
        key=lambda row: (
            safe_float(row["mean_total_time"]),
            safe_float(row["mean_paired_makespan_ratio"]),
            sum(config_tuple(row)),
            config_tuple(row),
        ),
    ) if nonzero else knee
    selected = []
    for config in ((0, 0), config_tuple(knee), config_tuple(quality_best), config_tuple(fastest_nonzero)):
        if config not in selected:
            selected.append(config)
    return selected


def build_recommendations(summary: Sequence[Mapping[str, object]], phase: str) -> List[Dict[str, object]]:
    scoped: MutableMapping[Tuple[str, str], List[Mapping[str, object]]] = {}
    for row in summary:
        if row["phase"] == phase:
            scoped.setdefault((str(row["model"]), str(row["job_count"])), []).append(row)

    recommendations = []
    for (model, scope), rows in sorted(scoped.items(), key=lambda item: (item[0][0], item[0][1] != "overall", item[0][1])):
        _, frontier, baseline_valid_rate = eligible_and_frontier(rows)
        if not frontier:
            continue
        knee = select_knee(frontier)
        recommendations.append(
            {
                "phase": phase,
                "model": model,
                "scope": scope,
                "simplified_iters": knee["simplified_iters"],
                "collision_iters": knee["collision_iters"],
                "valid_rate": format_float(knee["valid_rate"]),
                "baseline_valid_rate": format_float(baseline_valid_rate),
                "mean_paired_makespan_ratio": format_float(knee["mean_paired_makespan_ratio"]),
                "mean_total_time": format_float(knee["mean_total_time"]),
                "pareto_size": len(frontier),
                "selection": "equal-weight normalized Pareto knee",
            }
        )
    return recommendations


def overall_rows(summary: Sequence[Mapping[str, object]], phase: str, model: str) -> List[Mapping[str, object]]:
    return [row for row in summary if row["phase"] == phase and row["model"] == model and row["job_count"] == "overall"]


def write_summary(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    formatted = []
    numeric_metrics = set(SUMMARY_FIELDS) - {
        "phase", "model", "job_count", "simplified_iters", "collision_iters", "runs", "failures", "valid_runs"
    }
    for row in rows:
        output = dict(row)
        for metric in numeric_metrics:
            output[metric] = format_float(output.get(metric))
        formatted.append(output)
    write_csv(path, SUMMARY_FIELDS, formatted)


def write_pareto_plots(summary: Sequence[Mapping[str, object]], phase: str, output_dir: Path, models: Sequence[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for model in models:
        rows = overall_rows(summary, phase, model)
        if not rows:
            continue
        eligible, frontier, _ = eligible_and_frontier(rows)
        if not frontier:
            continue
        knee = select_knee(frontier)
        plt.figure(figsize=(9, 6))
        plt.scatter(
            [safe_float(row["mean_total_time"]) for row in eligible],
            [safe_float(row["mean_paired_makespan_ratio"]) for row in eligible],
            color="steelblue",
            alpha=0.65,
            label="validity-eligible",
        )
        ordered_frontier = sorted(frontier, key=lambda row: safe_float(row["mean_total_time"]))
        plt.plot(
            [safe_float(row["mean_total_time"]) for row in ordered_frontier],
            [safe_float(row["mean_paired_makespan_ratio"]) for row in ordered_frontier],
            color="darkorange",
            marker="o",
            label="Pareto frontier",
        )
        plt.scatter(
            [safe_float(knee["mean_total_time"])],
            [safe_float(knee["mean_paired_makespan_ratio"])],
            marker="*",
            s=220,
            color="crimson",
            label=f"knee {config_tuple(knee)}",
            zorder=5,
        )
        plt.xlabel("Mean total computation time (s)")
        plt.ylabel("Paired makespan ratio vs. (0, 0)")
        plt.title(f"{model}: {phase} local-improvement trade-off")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"pareto_{model}_{phase}.png", dpi=160)
        plt.close()


def model_solver(context: Mapping[str, object], model_name: str):
    import benchmark_static_algorithms as benchmark

    module = benchmark.require_nn_context(context, model_name, f"{model_name}_module")
    model = benchmark.require_nn_context(context, model_name, f"{model_name}_model")
    solver = module.solve_with_attention if model_name.startswith("attention") else module.solve_with_gnn
    return solver, model


def failure_row(
    phase: str,
    model: str,
    job_count: int,
    sample_index: object,
    repeat: int,
    config: Tuple[int, int],
    simplified_seed: int,
    collision_seed: int,
    error: Exception,
) -> Dict[str, object]:
    return {
        "phase": phase,
        "model": model,
        "job_count": job_count,
        "sample_index": sample_index,
        "repeat": repeat,
        "simplified_iters": config[0],
        "collision_iters": config[1],
        "status": "failed",
        "valid": "false",
        "makespan": "nan",
        "invalid_jobs": -1,
        "inference_time": "nan",
        "simplified_time": "nan",
        "collision_time": "nan",
        "postprocess_time": "nan",
        "total_time": "nan",
        "simplified_seed": simplified_seed,
        "collision_seed": collision_seed,
        "error": str(error).replace("\n", " ")[:1000],
    }


def run_phase(
    *,
    phase: str,
    cases: Mapping[int, Sequence[Mapping[str, object]]],
    models: Sequence[str],
    configurations: Mapping[str, Sequence[Tuple[int, int]]],
    repeats: int,
    base_seed: int,
    context: Mapping[str, object],
    writer: DetailWriter,
) -> None:
    import benchmark_static_algorithms as benchmark
    from neural_local_improvement import apply_neural_local_improvement

    for job_count, events in cases.items():
        print(f"\n=== {phase.title()} | {job_count} jobs | {len(events)} case(s) ===")
        for event_position, event in enumerate(events, start=1):
            jobs = list(event["jobs"])
            sample_index = event["index"]
            print(f"Case {event_position}/{len(events)} (sample {sample_index})")
            for model_name in models:
                configs = list(configurations[model_name])
                pending = []
                for repeat in range(repeats):
                    for config in configs:
                        key = tuple(
                            str(value)
                            for value in (phase, model_name, job_count, sample_index, repeat, config[0], config[1])
                        )
                        if key not in writer.completed:
                            pending.append((repeat, config))
                if not pending:
                    print(f"  {model_name}: already complete")
                    continue

                try:
                    solver, model = model_solver(context, model_name)
                    inference_started = time.perf_counter()
                    baseline_individual, _, _ = solver(jobs, model, deterministic=True)
                    inference_time = time.perf_counter() - inference_started
                except Exception as exc:
                    for repeat, config in pending:
                        simplified_seed = stable_seed(base_seed, phase, model_name, job_count, sample_index, repeat, "simplified")
                        collision_seed = stable_seed(base_seed, phase, model_name, job_count, sample_index, repeat, "collision")
                        writer.append(
                            failure_row(
                                phase, model_name, job_count, sample_index, repeat, config,
                                simplified_seed, collision_seed, exc,
                            )
                        )
                    print(f"  {model_name}: inference failed: {exc}")
                    continue

                for repeat, config in pending:
                    simplified_seed = stable_seed(base_seed, phase, model_name, job_count, sample_index, repeat, "simplified")
                    collision_seed = stable_seed(base_seed, phase, model_name, job_count, sample_index, repeat, "collision")
                    try:
                        result = apply_neural_local_improvement(
                            baseline_individual,
                            jobs,
                            simplified_iters=config[0],
                            collision_iters=config[1],
                            simplified_seed=simplified_seed,
                            collision_seed=collision_seed,
                        )
                        makespan, invalid_jobs = benchmark.evaluate_individual(result.individual, jobs)
                        valid = invalid_jobs == 0
                        writer.append(
                            {
                                "phase": phase,
                                "model": model_name,
                                "job_count": job_count,
                                "sample_index": sample_index,
                                "repeat": repeat,
                                "simplified_iters": config[0],
                                "collision_iters": config[1],
                                "status": "ok" if valid else "invalid",
                                "valid": "true" if valid else "false",
                                "makespan": format_float(makespan),
                                "invalid_jobs": invalid_jobs,
                                "inference_time": format_float(inference_time),
                                "simplified_time": format_float(result.simplified_time),
                                "collision_time": format_float(result.collision_time),
                                "postprocess_time": format_float(result.postprocess_time),
                                "total_time": format_float(inference_time + result.postprocess_time),
                                "simplified_seed": simplified_seed,
                                "collision_seed": collision_seed,
                                "error": "",
                            }
                        )
                    except Exception as exc:
                        writer.append(
                            failure_row(
                                phase, model_name, job_count, sample_index, repeat, config,
                                simplified_seed, collision_seed, exc,
                            )
                        )
                print(f"  {model_name}: completed {len(pending)} run(s)")


def load_case_slices(args: argparse.Namespace) -> Tuple[Dict[int, Sequence[Mapping[str, object]]], Dict[int, Sequence[Mapping[str, object]]]]:
    import benchmark_static_algorithms as benchmark

    tune_cases = {}
    validation_cases = {}
    required = args.tuning_samples + args.validation_samples
    for job_count in args.job_counts_values:
        events = benchmark.load_cases(Path(args.case_dir), job_count, limit=required)
        if len(events) < required:
            raise benchmark.BenchmarkCaseError(
                f"{job_count}-job case file contains {len(events)} cases; {required} are required"
            )
        tune_cases[job_count] = events[: args.tuning_samples]
        validation_cases[job_count] = events[args.tuning_samples : required]
    return tune_cases, validation_cases


def runtime_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema": "neural_local_improvement_study_v1",
        "models": args.models_values,
        "job_counts": args.job_counts_values,
        "simplified_grid": args.simplified_grid_values,
        "collision_grid": args.collision_grid_values,
        "tuning_samples": args.tuning_samples,
        "validation_samples": args.validation_samples,
        "search_repeats": args.search_repeats,
        "seed": args.seed,
        "case_dir": str(Path(args.case_dir).resolve()),
        "device": args.device,
    }


def validate_resume_config(path: Path, config: Mapping[str, object]) -> None:
    if not path.exists():
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    for key, value in config.items():
        if key == "device":
            continue
        if saved.get(key) != value:
            raise ValueError(f"Cannot resume: run configuration changed for {key!r}")


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    args.simplified_grid_values = parse_nonnegative_grid(args.simplified_grid)
    args.collision_grid_values = parse_nonnegative_grid(args.collision_grid)
    args.job_counts_values = parse_csv_values(args.job_counts, int)
    args.models_values = parse_csv_values(args.models)
    unknown = sorted(set(args.models_values) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unsupported model(s): {', '.join(unknown)}")
    if args.tuning_samples < 1 or args.validation_samples < 1 or args.search_repeats < 1:
        raise ValueError("Sample counts and search repeats must be positive")
    return args


def print_dry_run(args: argparse.Namespace) -> None:
    tune_configs = len(expand_grid(args.simplified_grid_values, args.collision_grid_values))
    tune_runs = tune_configs * len(args.models_values) * len(args.job_counts_values) * args.tuning_samples * args.search_repeats
    validation_configs = 4  # baseline plus at most three distinct representatives
    validation_runs = validation_configs * len(args.models_values) * len(args.job_counts_values) * args.validation_samples * args.search_repeats
    if args.phase == "tune":
        validation_runs = 0
    elif args.phase == "validate":
        tune_runs = 0
    print(f"Models: {', '.join(args.models_values)}")
    print(f"Job counts: {args.job_counts_values}")
    print(f"Tuning grid combinations: {tune_configs}")
    print(f"Planned tuning runs: {tune_runs}")
    print(f"Maximum planned validation runs: {validation_runs}")
    print(f"Maximum total runs: {tune_runs + validation_runs}")


def run(args: argparse.Namespace) -> None:
    import benchmark_static_algorithms as benchmark

    args = prepare_args(args)
    if args.dry_run:
        print_dry_run(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "local_improvement_runs.csv"
    summary_path = output_dir / "local_improvement_summary.csv"
    recommendation_path = output_dir / "local_improvement_recommendations.csv"
    config_path = output_dir / "run_config.json"
    config = runtime_config(args)
    if args.resume:
        validate_resume_config(config_path, config)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    tune_cases, validation_cases = load_case_slices(args)
    context = benchmark.load_nn_context(args.models_values, args.device)
    writer_resume = args.resume or args.phase == "validate"
    writer = DetailWriter(detail_path, resume=writer_resume)

    try:
        full_grid = expand_grid(args.simplified_grid_values, args.collision_grid_values)
        if args.phase in ("all", "tune"):
            run_phase(
                phase="tune",
                cases=tune_cases,
                models=args.models_values,
                configurations={model: full_grid for model in args.models_values},
                repeats=args.search_repeats,
                base_seed=args.seed,
                context=context,
                writer=writer,
            )

        current_rows = read_csv_rows(detail_path)
        current_summary = aggregate_rows(current_rows)
        tuning_configurations = {}
        for model in args.models_values:
            rows = overall_rows(current_summary, "tune", model)
            if not rows:
                raise ValueError(f"No tuning results found for {model}; run the tuning phase first")
            tuning_configurations[model] = select_shortlist(rows)
            print(f"{model} validation shortlist: {tuning_configurations[model]}")

        if args.phase in ("all", "validate"):
            run_phase(
                phase="validation",
                cases=validation_cases,
                models=args.models_values,
                configurations=tuning_configurations,
                repeats=args.search_repeats,
                base_seed=args.seed,
                context=context,
                writer=writer,
            )
    finally:
        writer.close()

    rows = read_csv_rows(detail_path)
    summary = aggregate_rows(rows)
    write_summary(summary_path, summary)
    report_phase = "validation" if any(row["phase"] == "validation" for row in rows) else "tune"
    recommendations = build_recommendations(summary, report_phase)
    write_csv(recommendation_path, RECOMMENDATION_FIELDS, recommendations)
    write_pareto_plots(summary, report_phase, output_dir, args.models_values)
    print(f"\nRun details: {detail_path}")
    print(f"Aggregated results: {summary_path}")
    print(f"Recommendations: {recommendation_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune simplified and collision-aware local improvement for neural schedulers")
    parser.add_argument("--phase", choices=("all", "tune", "validate"), default="all")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--job_counts", default="20,40,60,80,100")
    parser.add_argument("--simplified_grid", default="0,100,250,500,1000,2000")
    parser.add_argument("--collision_grid", default="0,10,25,50,100,200")
    parser.add_argument("--tuning_samples", type=int, default=10)
    parser.add_argument("--validation_samples", type=int, default=90)
    parser.add_argument("--search_repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--case_dir", default=str(benchmark_case_dir()))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", action="store_true", help="Skip completed rows after validating run settings")
    parser.add_argument("--dry_run", action="store_true", help="Print the planned run count without loading models")
    return parser


def benchmark_case_dir() -> Path:
    return STATIC_DIR / "benchmark_cases"


if __name__ == "__main__":
    try:
        run(build_parser().parse_args())
    except (ValueError, argparse.ArgumentError, FileNotFoundError) as exc:
        print(f"Local-improvement study failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
