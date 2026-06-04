#!/usr/bin/env python3
"""
Launch Attention, Attention_precise, GNN, and GNN_precise training in parallel.

Run from the AMR-DFJSP root:
    python train_all_models_parallel.py

Useful options:
    python train_all_models_parallel.py --inbox test_case/static/dispatch_inbox_60.jsonl
    python train_all_models_parallel.py --inboxes test_case/static/dispatch_inbox_20.jsonl,test_case/static/dispatch_inbox_40.jsonl
    python train_all_models_parallel.py --epochs 2000
    python train_all_models_parallel.py --rl_method reinforce --baseline_rule earliest_completion_job+earliest_completion
    python train_all_models_parallel.py --threads-per-process 2
    python train_all_models_parallel.py --models attention gnn_precise
"""

from __future__ import annotations

import argparse
import os
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


TARGETS = {
    "attention": TrainTarget(
        key="attention",
        script=ROOT / "Static_alogorithm" / "Attention" / "train.py",
        description="Attention with fast heuristic rollout",
    ),
    "attention_precise": TrainTarget(
        key="attention_precise",
        script=ROOT / "Static_alogorithm" / "Attention" / "train_precise.py",
        description="Attention with dynamic pathfinding and reservations",
    ),
    "gnn": TrainTarget(
        key="gnn",
        script=ROOT / "Static_alogorithm" / "GNN" / "train_gnn.py",
        description="GNN with fast heuristic rollout",
    ),
    "gnn_precise": TrainTarget(
        key="gnn_precise",
        script=ROOT / "Static_alogorithm" / "GNN" / "train_gnn_precise.py",
        description="GNN with dynamic pathfinding and reservations",
    ),
}

ALIASES = {
    "attension": "attention",
    "attension_precise": "attention_precise",
    "attention-precise": "attention_precise",
    "gnn-precise": "gnn_precise",
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
        help="Models to train. Choices: attention, attention_precise, gnn, gnn_precise.",
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


def command_for(target: TrainTarget, args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python,
        str(target.script),
        "--epochs",
        str(args.epochs),
        "--rl_method",
        args.rl_method,
        "--baseline_rule",
        args.baseline_rule,
        "--baseline_mode",
        args.baseline_mode,
        "--entropy_coef",
        str(args.entropy_coef),
    ]
    if args.inbox is not None:
        cmd.extend(["--inbox", str(args.inbox)])
    if args.inboxes:
        cmd.extend(["--inboxes", args.inboxes])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if not args.normalize_advantage:
        cmd.append("--no_normalize_advantage")
    return cmd


def launch(target: TrainTarget, args: argparse.Namespace, env: dict[str, str]) -> RunningJob:
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.logs_dir / f"{timestamp}_{target.key}.log"
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
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

    targets = normalize_models(args.models)

    if args.max_concurrent < 1:
        raise SystemExit("--max-concurrent must be at least 1")
    if args.threads_per_process < 1:
        raise SystemExit("--threads-per-process must be at least 1")
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")

    missing = [target.script for target in targets if not target.script.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise SystemExit(f"Missing training script(s):\n{joined}")

    env = make_env(args)

    print(f"Project root: {ROOT}")
    print(f"CPU cores detected: {os.cpu_count() or 1}")
    print(f"Max concurrent jobs: {args.max_concurrent}")
    print(f"Threads per process: {args.threads_per_process}")
    print(f"Epochs per model: {args.epochs}")
    print(f"RL method: {args.rl_method}")
    print(f"Baseline rule: {args.baseline_rule}")
    print(f"Baseline mode: {args.baseline_mode}")
    print(f"Logs directory: {args.logs_dir}")
    print()

    commands = [(target, command_for(target, args)) for target in targets]
    for target, cmd in commands:
        print(f"{target.key:<17} {' '.join(cmd)}")
    print()

    if args.dry_run:
        return 0

    pending = list(targets)
    running: list[RunningJob] = []
    failures: list[RunningJob] = []

    try:
        while pending or running:
            while pending and len(running) < args.max_concurrent:
                running.append(launch(pending.pop(0), args, env))
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
        return 130

    if failures:
        print("\nOne or more training jobs failed:")
        for job in failures:
            print(f"  {job.target.key}: {job.log_path}")
        return 1

    print("All requested training jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
