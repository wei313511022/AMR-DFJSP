#!/usr/bin/env python3
"""Phase III contract acceptance script (docs/Phase3_Model_IO_Contract.md, section 9).

Checks, given a contract checkpoint:
  1. load_model() restores the model from the checkpoint alone
  2. predict(scene) returns a plan passing every section-4 hard constraint
     (cold-start scene AND mid-state scene)
  3. predict is deterministic (same scene -> same plan)
  4. single-call latency < 1 second
It also writes example scene/plan JSON files to docs/examples/.

Usage:
  python scripts/validate_contract.py                        # uses checkpoints/my_scheduler_v1.pth
  python scripts/validate_contract.py --checkpoint path.pth
  python scripts/validate_contract.py --init-random          # smoke-test with untrained weights
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.env import TaskSchedulingEnv, load_env_spec  # noqa: E402
from core.model import QNetwork  # noqa: E402
from inference.checkpoint_io import export_contract_checkpoint  # noqa: E402
from inference.scheduler import load_model, validate_plan  # noqa: E402


def build_scene(env_spec: dict, num_jobs: int, mid_state: bool, seed: int) -> dict:
    """Create a contract section-3 scene from the env spec."""
    rng = random.Random(seed)
    stations = list(env_spec["stations"].values())
    docks = env_spec["docks"]
    materials = env_spec["materials"]
    material_dock_map = env_spec.get("material_dock_map", {})
    start_positions = env_spec["amrs"]["start_positions"]

    grid = env_spec.get("grid", {})
    width = int(grid.get("width", 10))
    height = int(grid.get("height", 10))
    blocked = {(int(x), int(y)) for x, y in grid.get("obstacles", [])}
    free_cells = [
        [x, y] for x in range(width) for y in range(height) if (x, y) not in blocked
    ]

    now = 1234.0 if mid_state else 0.0
    amrs = []
    for pos in start_positions:
        if mid_state:
            position = list(rng.choice(free_cells))
            available_at = now + rng.uniform(0.0, 30.0)
            inventory = {m: rng.randint(0, 1) for m in materials}
        else:
            position = list(pos)
            available_at = now
            inventory = {m: 0 for m in materials}
        amrs.append(
            {"position": position, "available_at": available_at, "inventory": inventory}
        )

    jobs = []
    for _ in range(num_jobs):
        material = rng.choice(sorted(materials.keys()))
        if material in material_dock_map:
            dock_xy = list(docks[material_dock_map[material]])
        else:
            dock_xy = list(rng.choice(list(docks.values())))
        jobs.append(
            {
                "material": material,
                "duration": float(materials[material]),
                "station_xy": list(rng.choice(stations)),
                "dock_xy": dock_xy,
                "arrival_time": now - rng.uniform(0.0, 50.0),
            }
        )

    return {"schema_version": "1.0", "time": now, "amrs": amrs, "jobs": jobs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "checkpoints" / "my_scheduler_v1.pth"),
        help="Contract checkpoint to validate.",
    )
    parser.add_argument(
        "--init-random",
        action="store_true",
        help="Export an untrained checkpoint first (pipeline smoke test). "
        "Always writes to its own file so a trained my_scheduler_v1.pth is "
        "never overwritten.",
    )
    parser.add_argument("--jobs", type=int, default=12, help="Jobs per test scene.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)

    if args.init_random:
        # Never touch the deliverable checkpoint: smoke tests get their own file.
        smoke_path = REPO_ROOT / "checkpoints" / "contract_smoke.pth"
        if Path(args.checkpoint).resolve() != smoke_path.resolve():
            print(f"[init-random] using {smoke_path.name} (ignoring --checkpoint)")
        ckpt_path = smoke_path
        env = TaskSchedulingEnv()
        state_dim = len(env.reset([]))
        action_dim = 4
        net = QNetwork(state_dim + action_dim, state_dim=state_dim, action_dim=action_dim)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        export_contract_checkpoint(net, str(ckpt_path), env_spec=env.env_spec)
        print(f"[init-random] exported untrained checkpoint -> {ckpt_path}")

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        print("Train first (python main.py) or run with --init-random for a smoke test.")
        return 1

    print(f"[1/4] load_model({ckpt_path.name}) ...")
    scheduler = load_model(str(ckpt_path))
    env_spec = scheduler.env_spec
    num_amrs = int(env_spec["amrs"]["count"])
    print(f"      ok (M={num_amrs} AMRs, state_dim={scheduler.policy_net.state_dim})")

    failures = 0
    example_scene = None
    example_plan = None

    for name, mid_state in (("cold-start", False), ("mid-state", True)):
        scene = build_scene(env_spec, args.jobs, mid_state, args.seed)
        n = len(scene["jobs"])

        t0 = time.perf_counter()
        plan = scheduler.predict(scene)
        latency = time.perf_counter() - t0

        print(f"[2/4] {name}: predict N={n} jobs in {latency*1000:.1f} ms")
        try:
            validate_plan(plan, n, num_amrs)
            print(f"      section-4 constraints: PASS")
        except ValueError as e:
            print(f"      section-4 constraints: FAIL ({e})")
            failures += 1

        plan2 = scheduler.predict(scene)
        if plan == plan2:
            print(f"      determinism: PASS")
        else:
            print(f"      determinism: FAIL (two calls disagree)")
            failures += 1

        if latency < 1.0:
            print(f"      latency < 1s: PASS")
        else:
            print(f"      latency < 1s: FAIL ({latency:.2f}s)")
            failures += 1

        if mid_state:
            example_scene, example_plan = scene, plan

    print("[3/4] writing example scene/plan JSON ...")
    examples_dir = REPO_ROOT / "docs" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    with open(examples_dir / "scene_example.json", "w", encoding="utf-8") as f:
        json.dump(example_scene, f, indent=2)
    with open(examples_dir / "plan_example.json", "w", encoding="utf-8") as f:
        json.dump(example_plan, f, indent=2)
    print(f"      -> {examples_dir}")

    print(f"[4/4] result: {'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
