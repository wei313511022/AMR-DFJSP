"""Evaluate a trained extend_GNN checkpoint on a jsonl instance file.

Fills the gap between the two existing entry points: `benchmark_static_algorithms.py`
hardcodes the checkpoint filename and can only read `benchmark_cases/dispatch_inbox_<N>.jsonl`,
while `eval_fleet_size.py` takes an arbitrary `--weights`/`--inbox` but is wired to
`SchedulerGNN`. This does the latter for `ExtendSchedulerGNN`, and persists rows instead of
printing them.

Rows are the unmodified `ideal_evaluator.evaluate` dict plus join keys that match the
dispatching-rule sweeps written by `sweep_fleet.py`, so policy and rule rows can be paired on
`(amrs, instance)` and both go through `ie.aggregate` unchanged. Policy rows carry
`job_rule="policy"` and `rule=<run_key>`; keep them in their own file rather than appending
into a rule sweep.

    python eval_extend_gnn.py \
        --weights ../checkpoints_v8/ppo_gc1_s42_best.pth \
        --inbox ../test_case/v3/test_60.jsonl \
        --num_amrs 16 --out raw/policy_test60_m16.jsonl

    python eval_extend_gnn.py --summarise --out raw/policy_test60_m16.jsonl

Guard rails, each of which exists because the corresponding failure is silent:

  * `load_required_operation_checkpoint` loads with `strict=False`, so a renamed or resized
    layer becomes a randomly-initialised layer and the eval still prints a plausible number.
    We parse its "loaded X/Y tensors" status and refuse unless X == Y.
  * `scenario_v3.apply_layout` patches `operation_policy` inside a bare `except Exception:
    pass`. If that ever fails the queue features stay in m=16 units and every other fleet
    size is quietly wrong, so DOCK_QUEUE_SCALE is asserted after the layout call.
  * `solve_with_extend_gnn` calls `model.train()` for sampled rollouts and never restores
    eval mode, so a sampled row would change the regime of every greedy row after it in the
    same process. We restore explicitly.
  * `test_case/v3/instances_60.jsonl` is entirely contained in `train_60.jsonl`; evaluating a
    policy on it measures training performance. Refused outright.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

STATIC_DIR = os.path.abspath(os.path.dirname(__file__))
# None of extend_GNN/, GNN/, Attention/ is a package, and each shadows the module of the
# same name inside it, so the SUBDIRECTORY has to precede STATIC_DIR on sys.path or
# `from GNN import SchedulerGNN` resolves to the empty namespace package. Inserting at 0 in
# this order leaves the subdirectories ahead of STATIC_DIR, matching eval_fleet_size.py.
for _path in (STATIC_DIR,
              os.path.join(STATIC_DIR, "extend_GNN"),
              os.path.join(STATIC_DIR, "GNN"),
              os.path.join(STATIC_DIR, "Attention")):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

import GA.GA as GA  # noqa: E402
import ideal_evaluator as ie  # noqa: E402
import operation_policy  # noqa: E402
import scenario_v3 as sc  # noqa: E402
from GA.GA import load_dispatch_events  # noqa: E402
from extend_GNN import ExtendSchedulerGNN, solve_with_extend_gnn  # noqa: E402
from neural_local_improvement import apply_neural_local_improvement  # noqa: E402
from operation_policy import load_required_operation_checkpoint  # noqa: E402

REQUIRED_KEYS = ("op_emb.weight", "operation_actor.0.weight")

# Architectures sharing the operation-policy interface. `gnn` is the station-feature
# ablation of `extend_gnn`: SchedulerGNN has no dock encoder and no dock/AMR attention at
# all (job_in_dim=16, amr_in_dim=8), whereas ExtendSchedulerGNN adds dock_in_dim=9 plus the
# cross-attention block. checkpoints_v6 holds all three trained under one recipe
# (REINFORCE/multisample, 2000 epochs, seeds 42/43/44), which is the matched comparison.
def _build_extend():
    return ExtendSchedulerGNN(hidden_dim=128, gin_layers=3)


def _build_gnn():
    from GNN import SchedulerGNN
    return SchedulerGNN(job_in_dim=16, amr_in_dim=8, hidden_dim=128, gin_layers=3)


def _build_attention():
    from Attention import SchedulerAttention
    return SchedulerAttention(hidden_dim=128)


def _solve_gnn(jobs, model, deterministic=True):
    from GNN import solve_with_gnn
    return solve_with_gnn(jobs, model, deterministic=deterministic)


def _solve_attention(jobs, model, deterministic=True):
    from Attention import solve_with_attention
    return solve_with_attention(jobs, model, deterministic=deterministic)


MODELS = {
    "extend_gnn": (_build_extend, solve_with_extend_gnn),
    "gnn": (_build_gnn, _solve_gnn),
    "attention": (_build_attention, _solve_attention),
}
# Congestion-blind inference: zero named feature slices AFTER the state extractor returns,
# so the trained forward path is untouched and the mask is auditable in one place.
#
# This is NOT an observability ablation -- the network was trained WITH these inputs, so a
# drop mixes loss of congestion signal with distribution shift. What it does measure is the
# question "what if the policy plans believing the facility is uncongested, then we execute
# it for real": zero is the least-congested value of every one of these features, so masking
# installs a specific false belief rather than hiding information.
#
# Levels are nested so the ladder localises where congestion actually enters the policy.
#   amr[9]    local_density        neighbours within Manhattan 2
#   dock[2]   available_delay      max(0, station_availabilities - t)
#   dock[3]   queue_count / m
#   dock[4]   queue_count / valid waiting slots, clipped at 1
#   dock[5]   service_remaining    remaining time of the in-progress service
#   dock[6]   committed_workload   sum over all unfinished events
#   act[2]    estimated_dock_wait  per-action predicted wait -- same source as dock[2]
MASKS = {
    "none": {},
    "L1": {"amr": (9,), "dock": (2, 3, 4)},
    "L2": {"amr": (9,), "dock": (2, 3, 4, 5, 6)},
    "L3": {"amr": (9,), "dock": (2, 3, 4, 5, 6), "action": (2,)},
    # Control: four non-congestion slices of comparable magnitude. If this degrades as much
    # as L1 then the effect is distribution shift, not congestion blindness.
    "control": {"amr": (10, 11), "dock": (7, 8)},
}

CONTAMINATED = "instances_60.jsonl"
MAX_FLEET = 24          # aisle_bays() offers 6 columns x 4 aisle rows
EXPECTED_DOCK_KEYS = 10  # 5 inbound + 5 outbound


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=STATIC_DIR, text=True
        ).strip()
    except Exception:
        return "unknown"


def checkpoint_provenance(weights: Path) -> dict:
    """Training args for this checkpoint, from its sibling `_latest.pth` and runs/.

    `_best.pth` is a bare state_dict and carries no args, but the `_latest.pth` written by
    the same run holds the full payload including the resolved argparse namespace.
    """
    out = {"train_args": None, "run_dir": None}
    stem = weights.stem
    latest = weights.with_name(stem.replace("_best", "_latest") + weights.suffix)
    if latest.exists():
        try:
            payload = torch.load(latest, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "args" in payload:
                args = payload["args"]
                out["train_args"] = vars(args) if hasattr(args, "__dict__") else dict(args)
                out["train_epoch"] = payload.get("epoch")
                out["train_best_metric"] = payload.get("best_metric")
        except Exception as exc:                      # pragma: no cover - provenance only
            out["train_args_error"] = repr(exc)

    run_name = (out.get("train_args") or {}).get("run_name")
    if run_name:
        runs = Path(STATIC_DIR).parent / "runs"
        for cand in sorted(runs.glob(f"*_{run_name}")):
            if cand.name.endswith(f"_{run_name}"):
                out["run_dir"] = str(cand)
                break
    return out


def check_training_provenance(prov: dict, weights: Path, strict: bool) -> list:
    """Warn (or fail) if this checkpoint was not trained on the expected datasets.

    The geometry change (inward queues, aisle depot, dock clearance) landed in d2f014f on
    2026-08-08 23:59. A checkpoint trained before it was trained against a different
    facility and its numbers are not comparable to anything measured today.
    """
    problems = []
    args = prov.get("train_args") or {}
    if not args:
        problems.append(f"{weights.name}: no training args found; provenance unverified")
    else:
        inbox = str(args.get("inbox", ""))
        val = str(args.get("validation_inbox", ""))
        if not inbox.endswith("train_60.jsonl"):
            problems.append(f"{weights.name}: trained on {inbox!r}, expected train_60.jsonl")
        if not val.endswith("val_60.jsonl"):
            problems.append(f"{weights.name}: validated on {val!r}, expected val_60.jsonl")

    run_dir = prov.get("run_dir")
    if run_dir:
        stamp = Path(run_dir).name.split("_")[0]
        if stamp < "20260809":
            problems.append(
                f"{weights.name}: run {Path(run_dir).name} predates the d2f014f geometry "
                f"change (2026-08-08); its facility differs from the current one"
            )
    if problems and strict:
        raise SystemExit("provenance check failed:\n  " + "\n  ".join(problems))
    return problems


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

def install_fleet(num_amrs: int) -> None:
    """Rebuild the facility for `num_amrs`, then verify it actually took effect."""
    if num_amrs > MAX_FLEET:
        raise SystemExit(
            f"--num_amrs {num_amrs} exceeds the depot's {MAX_FLEET} bays; "
            f"build_pod_depot would raise. Widen aisle_bays() first."
        )
    sc.apply_layout(num_amrs=num_amrs)

    if len(GA.AMR_KEYS) != num_amrs:
        raise SystemExit(f"apply_layout produced {len(GA.AMR_KEYS)} AMRs, expected {num_amrs}")
    # scenario_v3.apply_layout patches operation_policy inside `except Exception: pass`.
    # A swallowed failure leaves the dock-queue features in the previous fleet's units,
    # which is invisible in the output but wrong for every fleet size except the last one used.
    if operation_policy.DOCK_QUEUE_SCALE != float(num_amrs):
        raise SystemExit(
            f"operation_policy.DOCK_QUEUE_SCALE is {operation_policy.DOCK_QUEUE_SCALE}, "
            f"expected {float(num_amrs)} -- apply_layout's operation_policy patch did not run"
        )
    if len(operation_policy.DOCK_KEYS) != EXPECTED_DOCK_KEYS:
        raise SystemExit(
            f"operation_policy.DOCK_KEYS has {len(operation_policy.DOCK_KEYS)} entries, "
            f"expected {EXPECTED_DOCK_KEYS}"
        )


def load_model(weights: Path, device: torch.device, arch: str = "extend_gnn"):
    build, _ = MODELS[arch]
    model = build()
    status = load_required_operation_checkpoint(model, weights, torch, required_keys=REQUIRED_KEYS)

    match = re.match(r"loaded (\d+)/(\d+) tensors", status)
    if not match:
        raise SystemExit(f"unexpected checkpoint status string: {status!r}")
    n_loaded, n_total = int(match.group(1)), int(match.group(2))
    if n_loaded != n_total:
        # strict=False silently drops shape-mismatched tensors, leaving them randomly
        # initialised. The eval would still produce a number, just not a meaningful one.
        raise SystemExit(
            f"{weights}: only {n_loaded}/{n_total} tensors loaded. The checkpoint does not "
            f"match the {arch} architecture; refusing to evaluate."
        )
    return model.to(device).eval()


def install_feature_mask(level: str):
    """Wrap extract_state_extend_gnn so the named slices are zeroed on every call."""
    import extend_GNN as _xg
    spec = MASKS[level]
    if not spec:
        return
    original = _xg.extract_state_extend_gnn

    def masked(*a, **kw):
        t = list(original(*a, **kw))
        for idx in spec.get("amr", ()):       # t[0] = (1, num_amrs, AMR_IN_DIM)
            t[0][..., idx] = 0.0
        for idx in spec.get("dock", ()):      # t[2] = (1, num_docks, DOCK_IN_DIM)
            t[2][..., idx] = 0.0
        for idx in spec.get("action", ()):    # t[3] = (1, 2, amrs, jobs, ACTION_IN_DIM)
            t[3][..., idx] = 0.0
        return tuple(t)

    _xg.extract_state_extend_gnn = masked
    # solve_with_extend_gnn resolved the symbol at import time, so rebind there too.
    if getattr(_xg.solve_with_extend_gnn, "__globals__", {}).get("extract_state_extend_gnn"):
        _xg.solve_with_extend_gnn.__globals__["extract_state_extend_gnn"] = masked


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def parse_run_meta(run_key: str) -> dict:
    """Pull seed / arm / checkpoint-kind out of a run key like `ppo_gc1.5_s43_best`."""
    seed = re.search(r"_s(\d+)", run_key)
    arm = re.search(r"_(gc[\d.]+)_", run_key)
    if run_key.endswith("_latest"):
        kind = "latest"
    elif run_key.endswith("_best"):
        kind = "best"
    else:
        kind = "unknown"
    return {
        "train_seed": int(seed.group(1)) if seed else None,
        "arm": arm.group(1) if arm else None,
        "ckpt_kind": kind,
    }


# --------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------

def _sync() -> None:
    """CUDA kernels are launched asynchronously, so a perf_counter delta taken right after
    the last forward pass measures launch time, not compute time. `bench_compute.py`
    synchronises before every stop-clock for exactly this reason; without the same call
    here `solve_s` on the policy rows is not comparable to the rule rows, which are pure
    CPU and therefore always "synchronous"."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def rollout(model, jobs, args, instance_index: int) -> tuple:
    """One evaluated schedule. Returns (individual, solve_s, select_s, samples_clean).

    `select_s` is the executor time best-of-K spends SCORING its K candidates in order to
    choose one. It is zero for greedy, which has nothing to choose between. Charging it is
    not optional: without a score there is no "best" of K, so those K evaluations are part
    of the scheduler's cost rather than of the measurement. The `eval_s` recorded by the
    caller stays what it has always been -- one final evaluation of the returned schedule,
    which greedy and every dispatching rule pay identically.
    """
    solve_fn = MODELS[args.model][1]
    if args.samples <= 1:
        t0 = time.perf_counter()
        with torch.no_grad():
            individual, _, _ = solve_fn(jobs, model, deterministic=True)
        _sync()
        solve_s = time.perf_counter() - t0
        model.eval()
        return individual, solve_s, 0.0, 1

    # Best-of-K: a search budget the single-shot dispatching rules do not have. Reported
    # separately from the greedy headline, never mixed into it.
    best, best_metrics, solve_s, select_s, clean = None, None, 0.0, 0.0, 0
    for k in range(args.samples):
        torch.manual_seed(args.sample_seed + 1_000_003 * instance_index + k)
        t0 = time.perf_counter()
        with torch.no_grad():
            candidate, _, _ = solve_fn(jobs, model, deterministic=False)
        _sync()
        solve_s += time.perf_counter() - t0
        # solve_with_extend_gnn flips the module into train() for sampling and never
        # restores it; without this the next greedy rollout would run in the wrong regime.
        model.eval()

        t0 = time.perf_counter()
        metrics = ie.evaluate(candidate, jobs)
        select_s += time.perf_counter() - t0
        if metrics["nu"] == 0:
            clean += 1
        if best_metrics is None or _better(metrics, best_metrics):
            best, best_metrics = candidate, metrics
    return best, solve_s, select_s, clean


def _better(a: dict, b: dict) -> bool:
    """Prefer routable schedules, then shorter ones."""
    return (a["nu"], a["executed"]) < (b["nu"], b["executed"])


# --------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------

def summarise(path: Path) -> None:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        print(f"no rows in {path}")
        return
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("dataset", "?"), r.get("amrs"), r.get("rule", "?"), r.get("mode", "?"))].append(r)

    print(f"{'dataset':>10} {'m':>3} {'run_key':>26} {'mode':>9} "
          f"{'executed':>9} {'ideal':>8} {'Lambda':>8} {'nu/ep':>6} {'clean':>9} {'s/inst':>7}")
    for key in sorted(groups):
        rs = groups[key]
        agg = ie.aggregate(rs)
        secs = statistics.mean([r.get("solve_s", 0.0) for r in rs])
        print(f"{key[0]:>10} {key[1]:>3} {key[2]:>26} {key[3]:>9} "
              f"{agg['executed']:>9.1f} {agg['ideal']:>8.1f} {100 * agg['penalty']:>7.2f}% "
              f"{agg['nu_per_episode']:>6.2f} "
              f"{str(int(agg['clean_instances'])) + '/' + str(int(agg['instances'])):>9} {secs:>7.2f}")


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", action="append", default=[],
                    help="checkpoint path; repeat or comma-separate for several")
    ap.add_argument("--run_key", action="append", default=[],
                    help="label per --weights; defaults to the checkpoint stem")
    ap.add_argument("--inbox", type=str,
                    default=os.path.join(STATIC_DIR, "..", "test_case", "v3", "test_60.jsonl"))
    ap.add_argument("--mask", choices=tuple(MASKS), default="none",
                    help="congestion-blind inference level; see MASKS")
    ap.add_argument("--model", choices=tuple(MODELS), default="extend_gnn",
                    help="architecture; `gnn` is the no-station-features ablation")
    ap.add_argument("--num_amrs", type=int, default=16)
    ap.add_argument("--events", type=int, default=0, help="0 = all")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1,
                    help="1 = deterministic greedy; K > 1 = best-of-K sampling")
    ap.add_argument("--sample_seed", type=int, default=1000)
    ap.add_argument("--local_improve", action="store_true")
    ap.add_argument("--li_simplified", type=int, default=1000)
    ap.add_argument("--li_collision", type=int, default=100)
    ap.add_argument("--allow_stacked_budget", action="store_true",
                    help="permit --local_improve together with --samples K")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--allow_stale_geometry", action="store_true",
                    help="downgrade provenance failures to warnings")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--manifest", type=str, default="")
    ap.add_argument("--summarise", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out)

    if args.summarise:
        summarise(out)
        return

    weights = [Path(w).resolve() for spec in args.weights for w in spec.split(",") if w.strip()]
    if not weights:
        raise SystemExit("--weights is required")
    keys = [k for spec in args.run_key for k in spec.split(",") if k.strip()]
    if keys and len(keys) != len(weights):
        raise SystemExit(f"got {len(keys)} --run_key values for {len(weights)} --weights")
    if not keys:
        keys = [w.stem for w in weights]

    inbox = Path(args.inbox).resolve()
    if inbox.name == CONTAMINATED:
        raise SystemExit(
            f"{CONTAMINATED} is entirely contained in train_60.jsonl; evaluating a policy on "
            f"it measures training performance. Use test_60.jsonl or val_60.jsonl."
        )
    if not inbox.exists():
        raise SystemExit(f"inbox not found: {inbox}")
    if args.samples > 1 and args.local_improve and not args.allow_stacked_budget:
        raise SystemExit(
            "--samples K with --local_improve stacks two search budgets and makes the compute "
            "comparison unreadable. Pass --allow_stacked_budget if that is really intended."
        )

    install_fleet(args.num_amrs)
    if args.mask != "none":
        if args.model != "extend_gnn":
            raise SystemExit("--mask is defined against extend_gnn feature indices only")
        install_feature_mask(args.mask)
    device = resolve_device(args.device)
    events = load_dispatch_events(inbox)
    if args.start:
        events = events[args.start:]
    if args.events:
        events = events[: args.events]
    if not events:
        raise SystemExit("no events selected")

    n_jobs = len(events[0]["jobs"])
    dataset = inbox.stem
    dataset_sha = sha256_file(inbox)
    mode = "greedy" if args.samples <= 1 else f"sample_k{args.samples}"

    manifest = {
        "inbox": str(inbox), "dataset": dataset, "dataset_sha256": dataset_sha,
        "n_jobs": n_jobs, "events": len(events), "start": args.start,
        "num_amrs": args.num_amrs, "mode": mode, "samples": args.samples,
        "mask": args.mask, "mask_spec": MASKS[args.mask],
        "local_improve": bool(args.local_improve),
        "li_simplified": args.li_simplified if args.local_improve else 0,
        "li_collision": args.li_collision if args.local_improve else 0,
        "geometry": {
            "queue_geometry": GA.QUEUE_GEOMETRY, "depot_layout": GA.DEPOT_LAYOUT,
            "dock_clearance_ticks": GA.DOCK_CLEARANCE_TICKS,
            "grid_w": GA.GRID_W, "grid_h": GA.GRID_H,
            "door_rows": list(GA.DOOR_ROWS), "slot_capacity": dict(GA.SLOT_CAPACITY),
        },
        "git_head": git_head(), "torch": torch.__version__, "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "checkpoints": [],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(events)} events x {n_jobs} jobs | m={args.num_amrs} "
          f"(eta={sc.contention_ratio(args.num_amrs):.1f}) | {mode} | device {device}")

    with out.open("a", encoding="utf-8") as fh:
        for weight, run_key in zip(weights, keys):
            if not weight.exists():
                raise SystemExit(f"checkpoint not found: {weight}")
            prov = checkpoint_provenance(weight)
            warnings = check_training_provenance(prov, weight, strict=not args.allow_stale_geometry)
            for w in warnings:
                print(f"  WARNING {w}")

            model = load_model(weight, device, args.model)
            ckpt_sha = sha256_file(weight)
            meta = parse_run_meta(run_key)
            manifest["checkpoints"].append({
                "run_key": run_key, "path": str(weight), "sha256": ckpt_sha,
                **meta, "train_args": prov.get("train_args"), "run_dir": prov.get("run_dir"),
                "train_epoch": prov.get("train_epoch"),
                "train_best_metric": prov.get("train_best_metric"),
                "provenance_warnings": warnings,
            })

            t_run = time.perf_counter()
            executed = []
            for event in events:
                jobs = list(event["jobs"])
                individual, solve_s, select_s, samples_clean = rollout(
                    model, jobs, args, event["index"])

                li_s = 0.0
                if args.local_improve:
                    t0 = time.perf_counter()
                    result = apply_neural_local_improvement(
                        individual, jobs,
                        simplified_iters=args.li_simplified,
                        collision_iters=args.li_collision,
                        simplified_seed=42, collision_seed=42,
                    )
                    individual = result.individual
                    li_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                row = ie.evaluate(individual, jobs)
                eval_s = time.perf_counter() - t0

                row.update({
                    "amrs": args.num_amrs, "instance": event["index"],
                    "job_rule": "policy", "rule": run_key,
                    "family": "policy", "model": args.model, "run_key": run_key,
                    "ckpt_sha8": ckpt_sha[:8], "dataset": dataset,
                    "dataset_sha8": dataset_sha[:8], "n_jobs": n_jobs,
                    "mode": mode, "samples": args.samples, "samples_clean": samples_clean,
                    "mask": args.mask,
                    "local_improve": bool(args.local_improve),
                    "li_simplified": args.li_simplified if args.local_improve else 0,
                    "li_collision": args.li_collision if args.local_improve else 0,
                    "solve_s": round(solve_s, 4), "li_s": round(li_s, 4),
                    "select_s": round(select_s, 4), "eval_s": round(eval_s, 4),
                    **meta,
                })
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if row["nu"] == 0:
                    executed.append(row["executed"])

            mean_ex = statistics.mean(executed) if executed else float("nan")
            print(f"  {run_key:>28}: executed {mean_ex:8.2f} "
                  f"| clean {len(executed)}/{len(events)} "
                  f"| {time.perf_counter() - t_run:6.1f}s total")

    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        print(f"manifest -> {args.manifest}")
    print(f"rows -> {out}")


if __name__ == "__main__":
    main()
