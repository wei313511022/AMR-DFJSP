"""Phase III inference entry point (contract sections 3-6).

Public API:
    scheduler = load_model("checkpoints/my_scheduler_v1.pth")
    plan = scheduler.predict(scene)   # scene: contract section 3, plan: section 4

Properties guaranteed by this module:
  - deterministic: eval mode, noisy layers disabled, pure argmax
  - stateless: every predict() call rebuilds the episode from the scene only
  - constraint-safe: the plan is validated against contract section 4 before
    being returned (the autoregressive rollout satisfies them by construction)

Runtime dependencies: torch, numpy (plus the pure-python core simulator in
this repository — no training code involved).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from core.env import TaskSchedulingEnv
from core.features import build_actions_for_tasks, q_values_batch, select_action_index
from core.model import QNetwork

IO_SCHEMA_VERSION = "1.0"


def _check_schema_version(version: str, what: str) -> None:
    major = str(version).split(".")[0]
    expected_major = IO_SCHEMA_VERSION.split(".")[0]
    if major != expected_major:
        raise ValueError(
            f"Incompatible {what} schema version {version!r}; "
            f"this package implements {IO_SCHEMA_VERSION!r}"
        )


def validate_plan(plan: dict, num_jobs: int, num_amrs: int) -> None:
    """Enforce contract section 4 hard constraints; raise ValueError on violation."""
    assignment = plan.get("assignment")
    order = plan.get("order")
    if not isinstance(assignment, list) or len(assignment) != num_jobs:
        raise ValueError(f"assignment must be a list of length N={num_jobs}")
    for j, k in enumerate(assignment):
        if not isinstance(k, int) or not (0 <= k < num_amrs):
            raise ValueError(f"assignment[{j}]={k!r} out of range [0, {num_amrs})")

    if not isinstance(order, list) or len(order) != 2 * num_jobs:
        raise ValueError(f"order must have length 2N={2 * num_jobs}")
    seen: Dict[int, List[str]] = {j: [] for j in range(num_jobs)}
    for item in order:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"order item {item!r} must be [job_index, op]")
        j, op = item
        if not isinstance(j, int) or not (0 <= j < num_jobs):
            raise ValueError(f"order job index {j!r} out of range [0, {num_jobs})")
        if op not in ("pickup", "unload"):
            raise ValueError(f"order op {op!r} must be 'pickup' or 'unload'")
        seen[j].append(op)
    for j, ops in seen.items():
        if ops != ["pickup", "unload"]:
            raise ValueError(
                f"job {j} ops must be exactly ['pickup', 'unload'] in order, got {ops}"
            )


class Scheduler:
    """Restores a trained DDQN dispatcher and serves contract-format predictions."""

    def __init__(
        self,
        policy_net: QNetwork,
        env_spec: dict,
        feature_config: Optional[dict] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.policy_net = policy_net.to(self.device)
        self.policy_net.eval()
        if hasattr(self.policy_net, "set_noise_enabled"):
            self.policy_net.set_noise_enabled(False)
        self.env_spec = env_spec
        self.feature_config = feature_config or {}

        # Internal simulator used to roll the policy forward from the scene.
        # Collision avoidance is disabled: predict() only has to produce
        # assignment + order; the integrating system re-simulates the plan
        # with its own A*/collision engine. This keeps latency well below 1s.
        self.env = TaskSchedulingEnv(env_spec)
        self.env.enable_collision_avoidance = False
        # Conservative: stock reported in the scene is not consumed — the
        # integrating system replays every job's own dock pickup, so only
        # stock acquired inside this rollout may serve later jobs.
        self.env.consume_initial_inventory = False

        # Action-selection bias weights (Score = Q + cover/load/wait bonuses)
        # must match the values used during training; they travel with the
        # checkpoint in feature_config.selection_bias.
        bias = (feature_config or {}).get("selection_bias", {})
        self._bias_cover = float(bias.get("cover", 0.0))
        self._bias_load = float(bias.get("load", 0.0))
        self._bias_wait = float(bias.get("wait", 0.0))

        state_dim = len(self.env.reset([]))
        if int(self.policy_net.state_dim) != int(state_dim):
            raise ValueError(
                f"checkpoint state_dim={self.policy_net.state_dim} does not match "
                f"env_spec-derived state_dim={state_dim}; the weights were trained "
                "against a different environment layout"
            )

    # ------------------------------------------------------------------ scene

    def _scene_to_episode(self, scene: dict) -> Tuple[List[dict], dict, bool]:
        t0 = float(scene.get("time", 0.0))
        amrs = scene.get("amrs", [])
        if len(amrs) != self.env.num_robots:
            raise ValueError(
                f"scene has {len(amrs)} AMRs but the model was trained with "
                f"{self.env.num_robots}"
            )

        jobs: List[dict] = []
        docks_by_material: Dict[str, set] = {}
        for idx, job in enumerate(scene.get("jobs", [])):
            material = str(job["material"]).upper()
            station_xy = (int(job["station_xy"][0]), int(job["station_xy"][1]))
            station_key = self.env._station_by_coord.get(station_xy)
            if station_key is None:
                raise ValueError(f"job {idx}: unknown station_xy {job['station_xy']}")
            docks_by_material.setdefault(material, set()).add(tuple(job["dock_xy"]))
            jobs.append(
                {
                    "jid": idx,
                    "type": material,
                    "proc_time": float(job.get("duration", 0.0))
                    or self.env.material_durations[material],
                    "station": station_key,
                    "dock_xy": list(job["dock_xy"]),
                    # Contract guarantees arrival_time <= scene.time; keep the
                    # relative value anyway so future-dated jobs (defensive)
                    # are released at the right moment instead of early.
                    "_release": max(0.0, float(job.get("arrival_time", t0)) - t0),
                }
            )

        # Batched pickups are only faithful when a material always comes from
        # one dock (a stocked unit is interchangeable with the job's own dock
        # pickup). Otherwise fall back to one-unit-per-visit pickups.
        batching_ok = all(len(d) <= 1 for d in docks_by_material.values())

        init_state = {
            "positions": [list(a["position"]) for a in amrs],
            # absolute -> relative time, clamped at 0
            "free_times": [max(0.0, float(a.get("available_at", t0)) - t0) for a in amrs],
            "inventory": [dict(a.get("inventory", {})) for a in amrs],
        }
        return jobs, init_state, batching_ok

    # ---------------------------------------------------------------- predict

    def predict(self, scene: dict) -> dict:
        """scene (contract section 3) -> plan (contract section 4)."""
        _check_schema_version(scene.get("schema_version", "1.0"), "scene")

        jobs, init_state, batching_ok = self._scene_to_episode(scene)
        num_jobs = len(jobs)
        num_amrs = self.env.num_robots
        if num_jobs == 0:
            return {
                "schema_version": IO_SCHEMA_VERSION,
                "assignment": [],
                "order": [],
            }

        # Group jobs by (relative) release time; contract scenes normally
        # collapse to a single release at t=0.
        releases: Dict[float, List[dict]] = {}
        for j in jobs:
            releases.setdefault(float(j.pop("_release", 0.0)), []).append(j)
        scenario = [
            {"dispatch_time": t, "jobs": batch} for t, batch in sorted(releases.items())
        ]

        env = self.env
        env.reset(scenario, init_state=init_state)
        max_add = None if batching_ok else 1

        with torch.no_grad():
            while not env.done():
                rid = env.current_robot
                actions = build_actions_for_tasks(
                    env.available_tasks,
                    env.robot_inventory[rid],
                    env.capacity_per_type,
                    allow_proactive_replenish=batching_ok,
                    max_add=max_add,
                )
                if not actions:
                    raise RuntimeError("no valid actions during rollout")
                state = np.array(env._get_state(), dtype=np.float32)
                feats = np.zeros((len(actions), 4), dtype=np.float32)
                for i, (task_idx, replenish) in enumerate(actions):
                    task = env.available_tasks[task_idx]
                    feats[i] = env.action_features(rid, task, replenish)
                q_all = q_values_batch(self.policy_net, state, feats, self.device)
                best = select_action_index(
                    q_values=q_all,
                    actions=actions,
                    tasks=env.available_tasks,
                    inventory=env.robot_inventory[rid],
                    capacity_per_type=env.capacity_per_type,
                    proactive_replenish_bias_weight=self._bias_cover,
                    action_feats=feats,
                    full_load_bias_weight=self._bias_load,
                    waiting_replenish_bias_weight=self._bias_wait,
                )
                env.step(actions[best])

        assignment: List[int] = [-1] * num_jobs
        ops: List[Tuple[float, int, int, str]] = []
        for item in env.trace:
            jid = int(item["jid"])
            if not (0 <= jid < num_jobs):
                continue
            assignment[jid] = int(item["robot"])
            seq = int(item["seq"])
            pickup_t = float(item.get("pickup_start", item["segments"][0]["start"]))
            unload_t = float(item["segments"][2]["start"])
            ops.append((pickup_t, seq, jid, "pickup"))
            ops.append((unload_t, seq, jid, "unload"))

        # Global execution order: by simulated event time; ties resolved by
        # dispatch sequence, with pickup before unload for stability.
        ops.sort(key=lambda x: (x[0], x[1], 0 if x[3] == "pickup" else 1))
        order = [[jid, op] for (_t, _seq, jid, op) in ops]

        plan = {
            "schema_version": IO_SCHEMA_VERSION,
            "assignment": assignment,
            "order": order,
        }
        validate_plan(plan, num_jobs, num_amrs)
        return plan


def load_model(checkpoint_path: str, device: str = "cpu") -> Scheduler:
    """Restore a Scheduler purely from a self-describing checkpoint (section 6)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(
            f"{checkpoint_path} is not a Phase III contract checkpoint "
            "(expected keys: state_dict / arch_config / feature_config / env_spec)"
        )
    _check_schema_version(ckpt.get("io_schema_version", "?"), "checkpoint")

    arch = dict(ckpt["arch_config"])
    model_class = arch.pop("model_class", "QNetwork")
    if model_class != "QNetwork":
        raise ValueError(f"Unsupported model_class {model_class!r}")

    policy_net = QNetwork(
        input_dim=int(arch["input_dim"]),
        state_dim=int(arch["state_dim"]),
        action_dim=int(arch["action_dim"]),
        use_rainbow=str(arch.get("model_kind", "rainbow")) == "rainbow",
        num_atoms=int(arch.get("num_atoms", 51)),
        v_min=float(arch.get("v_min", -10000.0)),
        v_max=float(arch.get("v_max", 0.0)),
        noisy_std=float(arch.get("noisy_std", 0.5)),
    )
    policy_net.load_state_dict(ckpt["state_dict"])

    env_spec = ckpt.get("env_spec")
    if not env_spec:
        raise ValueError("checkpoint is missing env_spec (contract section 6)")

    return Scheduler(
        policy_net,
        env_spec=env_spec,
        feature_config=ckpt.get("feature_config", {}),
        device=device,
    )
