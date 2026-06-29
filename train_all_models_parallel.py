#!/usr/bin/env python3
"""
Launch Attention, Attention_precise, GNN, GNN_precise, and extend_GNN training in parallel.

Run from the AMR-DFJSP root:
    python train_all_models_parallel.py

Useful options:
    python train_all_models_parallel.py --inbox test_case/static/dispatch_inbox_60.jsonl
    python train_all_models_parallel.py --inboxes test_case/static/dispatch_inbox_20.jsonl,test_case/static/dispatch_inbox_40.jsonl,test_case/static/dispatch_inbox_60.jsonl,test_case/static/dispatch_inbox_80.jsonl,test_case/static/dispatch_inbox_100.jsonl
    python train_all_models_parallel.py --epochs 2000
    python train_all_models_parallel.py --rl_method reinforce --baseline_rule earliest_completion_job+earliest_completion
    python train_all_models_parallel.py --load_balance_coef 0
    python train_all_models_parallel.py --threads-per-process 2
    python train_all_models_parallel.py --models attention gnn_precise extend_gnn
    python train_all_models_parallel.py   --inboxes "test_case/static/dispatch_inbox_60.jsonl"   --validation_inboxes "test_case/static/dispatch_validation_60.jsonl"   --epochs 1000   --batch_size 16   --max-concurrent 4   --threads-per-process 2 --load_balance_coef 0 --baseline_rule "fifo+earliest_available"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainTarget:
    key: str
    script: Path
    description: str
    legacy_best_model: str


TARGETS = {
    "attention": TrainTarget(
        key="attention",
        script=ROOT / "Static_alogorithm" / "Attention" / "train.py",
        description="Attention with fast heuristic rollout",
        legacy_best_model="attention_scheduler_best.pth",
    ),
    "attention_precise": TrainTarget(
        key="attention_precise",
        script=ROOT / "Static_alogorithm" / "Attention" / "train_precise.py",
        description="Attention with dynamic pathfinding and reservations",
        legacy_best_model="attention_precise_scheduler_best.pth",
    ),
    "gnn": TrainTarget(
        key="gnn",
        script=ROOT / "Static_alogorithm" / "GNN" / "train_gnn.py",
        description="GNN with fast heuristic rollout",
        legacy_best_model="gnn_mpn_scheduler_best.pth",
    ),
    "gnn_precise": TrainTarget(
        key="gnn_precise",
        script=ROOT / "Static_alogorithm" / "GNN" / "train_gnn_precise.py",
        description="GNN with dynamic pathfinding and reservations",
        legacy_best_model="gnn_precise_mpn_scheduler_best.pth",
    ),
    "extend_gnn": TrainTarget(
        key="extend_gnn",
        script=ROOT / "Static_alogorithm" / "extend_GNN" / "train_extend_gnn.py",
        description="Hybrid dock-aware GNN with fast heuristic rollout",
        legacy_best_model="extend_gnn_scheduler_best.pth",
    ),
}

BASELINE_CURRICULA = {
    "gradual": [
        "fifo+earliest_available",
        "fifo+least_loaded",
        "earliest_completion_job+least_loaded",
        "earliest_completion_job+earliest_completion",
    ],
}

ALIASES = {
    "attension": "attention",
    "attension_precise": "attention_precise",
    "attention-precise": "attention_precise",
    "gnn-precise": "gnn_precise",
    "extend-gnn": "extend_gnn",
    "extended_gnn": "extend_gnn",
}


@dataclass
class RunningJob:
    target: TrainTarget
    process: subprocess.Popen
    log_path: Path
    log_handle: object
    started_at: float


def parse_args() -> argparse.Namespace:
    default_parallelism = min(4, max(1, os.cpu_count() or 1))
    default_threads = max(1, (os.cpu_count() or 1) // default_parallelism)

    parser = argparse.ArgumentParser(
        description="Train Attention, Attention_precise, GNN, and GNN_precise in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(TARGETS.keys()),
        help="Models to train. Choices: attention, attention_precise, gnn, gnn_precise, extend_gnn.",
    )
    parser.add_argument(
        "--inbox",
        type=Path,
        default=None,
        help="Optional dispatch JSONL passed to every training script.",
    )
    parser.add_argument(
        "--inboxes",
        type=str,
        default="",
        help="Comma-separated dispatch JSONL files passed to every training script.",
    )
    parser.add_argument(
        "--validation_inbox",
        "--validation-inbox",
        type=Path,
        default=None,
        help="Optional fixed validation dispatch JSONL passed to every training script.",
    )
    parser.add_argument(
        "--validation_inboxes",
        "--validation-inboxes",
        type=str,
        default="",
        help="Comma-separated fixed validation dispatch JSONL files passed to every training script.",
    )
    parser.add_argument(
        "--validation_interval",
        "--validation-interval",
        type=int,
        default=50,
        help="Epoch interval for fixed validation scoring in every training script.",
    )
    parser.add_argument(
        "--validation_invalid_penalty",
        "--validation-invalid-penalty",
        type=float,
        default=1000.0,
        help="Penalty added per average invalid validation job when selecting best checkpoints.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch child training processes.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of epochs passed to every selected training script.",
    )
    parser.add_argument(
        "--rl_method",
        "--rl-method",
        choices=["reinforce", "ppo"],
        default="reinforce",
        help="Training method passed to every selected training script.",
    )
    parser.add_argument(
        "--baseline_rule",
        "--baseline-rule",
        default="earliest_completion_job+earliest_completion",
        help="Dispatching-rule baseline used by REINFORCE training.",
    )
    parser.add_argument(
        "--baseline_curriculum",
        "--baseline-curriculum",
        choices=sorted(BASELINE_CURRICULA),
        default="",
        help="Optional weak-to-strong baseline curriculum. When set, --epochs is total epochs split across phases.",
    )
    parser.add_argument(
        "--baseline_mode",
        "--baseline-mode",
        choices=["stepwise", "episode"],
        default="stepwise",
        help="REINFORCE baseline comparison mode.",
    )
    parser.add_argument(
        "--entropy_coef",
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy regularization coefficient passed to REINFORCE trainers.",
    )
    parser.add_argument(
        "--load_balance_coef",
        "--load-balance-coef",
        type=float,
        default=0.1,
        help="Load-balance shaping coefficient passed to REINFORCE trainers; use 0 to disable.",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size passed to every selected training script.",
    )
    parser.add_argument(
        "--no_normalize_advantage",
        "--no-normalize-advantage",
        dest="normalize_advantage",
        action="store_false",
        default=True,
        help="Disable batch advantage normalization in REINFORCE trainers.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=default_parallelism,
        help="Maximum number of training processes running at the same time.",
    )
    parser.add_argument(
        "--threads-per-process",
        type=int,
        default=default_threads,
        help="OpenMP/MKL/OpenBLAS thread limit per training process.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=ROOT / "training_logs",
        help="Directory for one log file per training process.",
    )
    parser.add_argument(
        "--stagger-seconds",
        type=float,
        default=2.0,
        help="Delay between process launches to reduce startup contention.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Hide CUDA devices from child processes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training.",
    )
    return parser.parse_args()


def normalize_models(raw_models: Iterable[str]) -> list[TrainTarget]:
    selected = []
    seen = set()
    for raw in raw_models:
        key = ALIASES.get(raw.lower(), raw.lower())
        if key not in TARGETS:
            valid = ", ".join(TARGETS)
            raise SystemExit(f"Unknown model '{raw}'. Valid choices: {valid}")
        if key not in seen:
            selected.append(TARGETS[key])
            seen.add(key)
    return selected


def make_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    if args.threads_per_process > 0:
        threads = str(args.threads_per_process)
        env["OMP_NUM_THREADS"] = threads
        env["MKL_NUM_THREADS"] = threads
        env["OPENBLAS_NUM_THREADS"] = threads
        env["NUMEXPR_NUM_THREADS"] = threads

    if args.cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""

    return env


def command_for(
    target: TrainTarget,
    args: argparse.Namespace,
    *,
    epochs: int | None = None,
    baseline_rule: str | None = None,
    init_checkpoint: str = "",
    latest_checkpoint_path: str = "",
    best_model_path: str = "",
) -> list[str]:
    cmd = [
        args.python,
        str(target.script),
        "--epochs",
        str(args.epochs if epochs is None else epochs),
        "--rl_method",
        args.rl_method,
        "--baseline_rule",
        args.baseline_rule if baseline_rule is None else baseline_rule,
        "--baseline_mode",
        args.baseline_mode,
        "--entropy_coef",
        str(args.entropy_coef),
        "--load_balance_coef",
        str(args.load_balance_coef),
    ]
    if init_checkpoint:
        cmd.extend(["--init_checkpoint", init_checkpoint])
    if latest_checkpoint_path:
        cmd.extend(["--latest_checkpoint_path", latest_checkpoint_path])
    if best_model_path:
        cmd.extend(["--best_model_path", best_model_path])
    if args.inbox is not None:
        cmd.extend(["--inbox", str(args.inbox)])
    if args.inboxes:
        cmd.extend(["--inboxes", args.inboxes])
    if args.validation_inbox is not None:
        cmd.extend(["--validation_inbox", str(args.validation_inbox)])
    if args.validation_inboxes:
        cmd.extend(["--validation_inboxes", args.validation_inboxes])
    cmd.extend(["--validation_interval", str(args.validation_interval)])
    cmd.extend(["--validation_invalid_penalty", str(args.validation_invalid_penalty)])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if not args.normalize_advantage:
        cmd.append("--no_normalize_advantage")
    return cmd


def launch(
    target: TrainTarget,
    args: argparse.Namespace,
    env: dict[str, str],
    cmd: list[str] | None = None,
    log_path: Path | None = None,
) -> RunningJob:
    if log_path is None:
        args.logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = args.logs_dir / f"{timestamp}_{target.key}.log"
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    if cmd is None:
        cmd = command_for(target, args)

    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"[start] {target.key:<17} pid={process.pid:<7} log={log_path}")
    return RunningJob(target, process, log_path, log_handle, time.time())


def run_training_wave(
    targets: list[TrainTarget],
    args: argparse.Namespace,
    env: dict[str, str],
    commands: dict[str, list[str]],
    log_paths: dict[str, Path],
) -> tuple[int, list[RunningJob]]:
    pending = list(targets)
    running: list[RunningJob] = []
    failures: list[RunningJob] = []

    try:
        while pending or running:
            while pending and len(running) < args.max_concurrent:
                target = pending.pop(0)
                running.append(
                    launch(
                        target,
                        args,
                        env,
                        commands[target.key],
                        log_paths.get(target.key),
                    )
                )
                if args.stagger_seconds > 0 and (pending or len(running) < len(targets)):
                    time.sleep(args.stagger_seconds)

            time.sleep(5)
            still_running = []
            for job in running:
                code = job.process.poll()
                if code is None:
                    still_running.append(job)
                    continue

                elapsed = time.time() - job.started_at
                job.log_handle.close()
                status = "done" if code == 0 else "fail"
                print(f"[{status}] {job.target.key:<17} exit={code:<4} elapsed={elapsed:,.1f}s log={job.log_path}")
                if code != 0:
                    failures.append(job)

            running = still_running

    except KeyboardInterrupt:
        print("\nInterrupted. Stopping child training processes...")
        terminate_all(running)
        return 130, failures

    return (1 if failures else 0), failures


def terminate_all(running: list[RunningJob]) -> None:
    for job in running:
        if job.process.poll() is None:
            print(f"[stop]  terminating {job.target.key} pid={job.process.pid}")
            job.process.terminate()
    for job in running:
        try:
            job.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"[kill]  killing {job.target.key} pid={job.process.pid}")
            job.process.kill()
        finally:
            job.log_handle.close()


def main() -> int:
    args = parse_args()
    if args.logs_dir is not None and not args.logs_dir.is_absolute():
        args.logs_dir = ROOT / args.logs_dir
    if args.inbox is not None and not args.inbox.is_absolute():
        args.inbox = ROOT / args.inbox
    if args.validation_inbox is not None and not args.validation_inbox.is_absolute():
        args.validation_inbox = ROOT / args.validation_inbox
    if args.inboxes:
        normalized_inboxes = []
        for raw_path in args.inboxes.split(","):
            raw_path = raw_path.strip()
            if not raw_path:
                continue
            inbox_path = Path(raw_path)
            if not inbox_path.is_absolute():
                inbox_path = ROOT / inbox_path
            normalized_inboxes.append(str(inbox_path))
        args.inboxes = ",".join(normalized_inboxes)
    if args.validation_inboxes:
        normalized_validation_inboxes = []
        for raw_path in args.validation_inboxes.split(","):
            raw_path = raw_path.strip()
            if not raw_path:
                continue
            inbox_path = Path(raw_path)
            if not inbox_path.is_absolute():
                inbox_path = ROOT / inbox_path
            normalized_validation_inboxes.append(str(inbox_path))
        args.validation_inboxes = ",".join(normalized_validation_inboxes)

    targets = normalize_models(args.models)

    if args.max_concurrent < 1:
        raise SystemExit("--max-concurrent must be at least 1")
    if args.threads_per_process < 1:
        raise SystemExit("--threads-per-process must be at least 1")
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")
    if args.validation_interval < 1:
        raise SystemExit("--validation_interval must be at least 1")
    if args.validation_invalid_penalty < 0:
        raise SystemExit("--validation_invalid_penalty must be non-negative")

    missing = [target.script for target in targets if not target.script.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise SystemExit(f"Missing training script(s):\n{joined}")

    env = make_env(args)

    if args.baseline_curriculum:
        if args.rl_method != "reinforce":
            raise SystemExit("--baseline_curriculum is only supported with --rl_method reinforce")
        baselines = BASELINE_CURRICULA[args.baseline_curriculum]
        if args.epochs % len(baselines) != 0:
            raise SystemExit(
                f"--epochs must be divisible by {len(baselines)} for curriculum '{args.baseline_curriculum}'"
            )

        phase_epochs = args.epochs // len(baselines)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.logs_dir / f"curriculum_{run_id}"
        previous_latest: dict[str, Path] = {}
        final_best: dict[str, Path] = {}

        print(f"Project root: {ROOT}")
        print(f"CPU cores detected: {os.cpu_count() or 1}")
        print(f"Max concurrent jobs: {args.max_concurrent}")
        print(f"Threads per process: {args.threads_per_process}")
        print(f"Total epochs per model: {args.epochs}")
        print(f"Epochs per phase: {phase_epochs}")
        print(f"RL method: {args.rl_method}")
        print(f"Baseline curriculum: {args.baseline_curriculum}")
        print(f"Baseline mode: {args.baseline_mode}")
        print(f"Load balance coef: {args.load_balance_coef}")
        print(f"Validation interval: {args.validation_interval}")
        print(f"Validation invalid penalty: {args.validation_invalid_penalty}")
        if args.validation_inbox is not None:
            print(f"Validation inbox: {args.validation_inbox}")
        if args.validation_inboxes:
            print(f"Validation inboxes: {args.validation_inboxes}")
        print(f"Curriculum run directory: {run_dir}")
        print()

        for phase_idx, baseline_rule in enumerate(baselines, start=1):
            phase_name = f"phase_{phase_idx:02d}"
            phase_logs_dir = run_dir / "logs" / phase_name
            phase_checkpoint_dir = run_dir / "checkpoints" / phase_name
            next_latest: dict[str, Path] = {}
            commands: dict[str, list[str]] = {}
            log_paths: dict[str, Path] = {}

            print(f"=== {phase_name}: {baseline_rule} ({phase_epochs} epochs) ===")
            for target in targets:
                latest_checkpoint = phase_checkpoint_dir / f"{target.key}_latest.pth"
                best_model = phase_checkpoint_dir / f"{target.key}_best.pth"
                init_checkpoint = previous_latest.get(target.key)
                commands[target.key] = command_for(
                    target,
                    args,
                    epochs=phase_epochs,
                    baseline_rule=baseline_rule,
                    init_checkpoint=str(init_checkpoint) if init_checkpoint else "",
                    latest_checkpoint_path=str(latest_checkpoint),
                    best_model_path=str(best_model),
                )
                log_paths[target.key] = phase_logs_dir / f"{target.key}.log"
                next_latest[target.key] = latest_checkpoint
                final_best[target.key] = best_model
                print(f"{target.key:<17} {' '.join(commands[target.key])}")
            print()

            if not args.dry_run:
                code, failures = run_training_wave(targets, args, env, commands, log_paths)
                if failures:
                    print("\nOne or more training jobs failed:")
                    for job in failures:
                        print(f"  {job.target.key}: {job.log_path}")
                    return code
                if code != 0:
                    return code

            previous_latest = next_latest

        if args.dry_run:
            return 0

        print("Promoting final-phase best models to legacy inference filenames...")
        for target in targets:
            src = final_best[target.key]
            dst = ROOT / target.legacy_best_model
            if not src.exists():
                print(f"[warn] missing final best for {target.key}: {src}")
                continue
            shutil.copy2(src, dst)
            print(f"[promote] {target.key:<17} {src} -> {dst}")

        print("All requested curriculum training jobs completed successfully.")
        return 0

    print(f"Project root: {ROOT}")
    print(f"CPU cores detected: {os.cpu_count() or 1}")
    print(f"Max concurrent jobs: {args.max_concurrent}")
    print(f"Threads per process: {args.threads_per_process}")
    print(f"Epochs per model: {args.epochs}")
    print(f"RL method: {args.rl_method}")
    print(f"Baseline rule: {args.baseline_rule}")
    print(f"Baseline mode: {args.baseline_mode}")
    print(f"Load balance coef: {args.load_balance_coef}")
    print(f"Validation interval: {args.validation_interval}")
    print(f"Validation invalid penalty: {args.validation_invalid_penalty}")
    if args.validation_inbox is not None:
        print(f"Validation inbox: {args.validation_inbox}")
    if args.validation_inboxes:
        print(f"Validation inboxes: {args.validation_inboxes}")
    print(f"Logs directory: {args.logs_dir}")
    print()

    commands = [(target, command_for(target, args)) for target in targets]
    for target, cmd in commands:
        print(f"{target.key:<17} {' '.join(cmd)}")
    print()

    if args.dry_run:
        return 0

    command_map = {target.key: command_for(target, args) for target in targets}
    code, failures = run_training_wave(targets, args, env, command_map, {})

    if failures:
        print("\nOne or more training jobs failed:")
        for job in failures:
            print(f"  {job.target.key}: {job.log_path}")
        return 1
    if code != 0:
        return code

    print("All requested training jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
