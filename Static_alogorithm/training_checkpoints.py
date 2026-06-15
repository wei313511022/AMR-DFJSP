from __future__ import annotations

import os
from typing import Mapping

import torch


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_training_checkpoint(model, optimizers: Mapping[str, object], checkpoint_path: str, device) -> None:
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        loaded_optimizers = []
        optimizer_map = checkpoint.get("optimizers", {})
        for name, optimizer in optimizers.items():
            state = optimizer_map.get(name)
            if state is None:
                state = checkpoint.get(f"{name}_state_dict")
            if state is None and len(optimizers) == 1:
                state = checkpoint.get("optimizer_state_dict")
            if state is not None:
                optimizer.load_state_dict(state)
                loaded_optimizers.append(name)
        print(
            f"Loaded training checkpoint from {checkpoint_path}"
            + (f" (optimizers: {', '.join(loaded_optimizers)})" if loaded_optimizers else "")
        )
        return

    model.load_state_dict(checkpoint)
    print(f"Loaded legacy model weights from {checkpoint_path}")


def save_training_checkpoint(
    checkpoint_path: str,
    model,
    optimizers: Mapping[str, object],
    epoch: int,
    best_makespan: float,
    args,
) -> None:
    if not checkpoint_path:
        return
    _ensure_parent(checkpoint_path)
    optimizer_states = {name: optimizer.state_dict() for name, optimizer in optimizers.items()}
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizers": optimizer_states,
        "epoch": epoch,
        "best_makespan": best_makespan,
        "args": vars(args),
    }
    if len(optimizer_states) == 1:
        payload["optimizer_state_dict"] = next(iter(optimizer_states.values()))
    for name, state in optimizer_states.items():
        payload[f"{name}_state_dict"] = state
    torch.save(payload, checkpoint_path)


def save_model_weights(model, model_path: str) -> None:
    _ensure_parent(model_path)
    torch.save(model.state_dict(), model_path)
