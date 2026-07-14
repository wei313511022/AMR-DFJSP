"""Self-describing checkpoint export (Phase III contract, section 6).

The exported .pth file carries everything needed to rebuild the model and its
feature pre-processing at inference time — no training code required:
  - format_version / io_schema_version
  - state_dict
  - arch_config: every QNetwork.__init__ hyper-parameter
  - feature_config: feature layout + normalization constants used at inference
  - env_spec: snapshot of the environment constants assumed during training
  - metrics: optional traceability info
"""
from __future__ import annotations

from typing import Optional

import torch

from core.env import load_env_spec

FORMAT_VERSION = "1.1"
IO_SCHEMA_VERSION = "1.0"
# Environment/feature semantics generation. Bump whenever the meaning of the
# state vector or action features changes (checkpoints keep loading dimension-
# wise, so the loader uses this marker to warn about stale weights).
ENV_SEMANTICS = "fjssp-v2-single-buffer-delivery"


def build_feature_config(env_spec: dict, selection_bias: Optional[dict] = None) -> dict:
    """Describe the exact feature pipeline used by core.env / core.features."""
    materials = sorted(str(k).upper() for k in env_spec.get("materials", {}).keys())
    bias = selection_bias or {}
    return {
        "materials": materials,
        "env_semantics": ENV_SEMANTICS,
        # All times fed to the network are relative to the scene's `time`
        # (episodes always start at t=0 inside the simulator).
        "relative_time": True,
        # Raw features are fed to the network without scaling (must match training).
        "normalize": {"enabled": False},
        "state_layout": (
            "onehot(current_robot, M) | n_available_tasks | t | "
            "per-robot [free_time, x, y, invA, invB, invC] | "
            "per-station accepts-new-part-from time (occupant's scheduled "
            "pickup if known, else its process end; sorted station keys) | "
            "dock_busy_until (sorted dock keys)"
        ),
        "action_features": [
            "travel(move+dock_queue+loading)",
            "machine_buffer_wait",
            "proc",
            "replenish_add",
        ],
        # Action-selection shaping (Score = Q + cover/load/wait bonuses); the
        # inference rollout must use the same weights as training.
        "selection_bias": {
            "cover": float(bias.get("cover", 0.0)),
            "load": float(bias.get("load", 0.0)),
            "wait": float(bias.get("wait", 0.0)),
        },
    }


def export_contract_checkpoint(
    policy_net,
    path: str,
    env_spec: Optional[dict] = None,
    feature_config: Optional[dict] = None,
    metrics: Optional[dict] = None,
    selection_bias: Optional[dict] = None,
) -> str:
    """Save `policy_net` as a self-describing Phase III checkpoint."""
    spec = load_env_spec(env_spec)
    arch_config = {"model_class": "QNetwork"}
    arch_config.update(policy_net.rainbow_config())

    ckpt = {
        "format_version": FORMAT_VERSION,
        "io_schema_version": IO_SCHEMA_VERSION,
        "state_dict": policy_net.state_dict(),
        "arch_config": arch_config,
        "feature_config": feature_config or build_feature_config(spec, selection_bias),
        "env_spec": spec,
        "metrics": metrics or {},
    }
    torch.save(ckpt, path)
    return path
