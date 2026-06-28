from __future__ import annotations

import os
from typing import Mapping

import torch


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_partial_state_dict(model, state_dict, checkpoint_path: str) -> bool:
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and getattr(value, "shape", None) == current[key].shape
    }
    skipped = sorted(key for key in state_dict if key not in compatible)
    current.update(compatible)
    model.load_state_dict(current)
    print(
        f"Loaded {len(compatible)} compatible tensors from {checkpoint_path}"
        + (f"; skipped {len(skipped)} incompatible tensors" if skipped else "")
    )
    return not skipped


def load_training_checkpoint(model, optimizers: Mapping[str, object], checkpoint_path: str, device) -> None:
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        fully_compatible = _load_partial_state_dict(model, checkpoint["model_state_dict"], checkpoint_path)
        loaded_optimizers = []
        if fully_compatible:
            optimizer_payload = checkpoint.get("optimizers")
            if isinstance(optimizer_payload, dict):
                for name, optimizer in optimizers.items():
                    if name in optimizer_payload:
                        optimizer.load_state_dict(optimizer_payload[name])
                        loaded_optimizers.append(name)
            elif "optimizer_state_dict" in checkpoint and len(optimizers) == 1:
                name, optimizer = next(iter(optimizers.items()))
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                loaded_optimizers.append(name)
        elif optimizers:
            print("Skipped optimizer state because model weights were only partially compatible.")
        print(
            f"Loaded training checkpoint from {checkpoint_path}"
            + (f" (optimizers: {', '.join(loaded_optimizers)})" if loaded_optimizers else "")
        )
        return

    _load_partial_state_dict(model, checkpoint, checkpoint_path)
    print(f"Loaded legacy model weights from {checkpoint_path}")


def save_training_checkpoint(
    checkpoint_path: str,
    model,
    optimizers: Mapping[str, object],
    epoch: int,
    best_metric: float,
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
        "best_metric": best_metric,
        "best_makespan": best_metric,
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


def evaluate_validation_events(events, model, solve_fn, evaluate_makespan_fn) -> dict[str, float]:
    if not events:
        return {
            "makespan": float("nan"),
            "invalid_jobs": float("nan"),
            "samples": 0,
        }

    was_training = model.training
    model.eval()
    makespans = []
    invalid_counts = []
    try:
        with torch.no_grad():
            for event in events:
                individual, _, _ = solve_fn(list(event["jobs"]), model, deterministic=True)
                makespan, invalid_count = evaluate_makespan_fn(individual, event["jobs"])
                makespans.append(float(makespan))
                invalid_counts.append(float(invalid_count))
    finally:
        model.train(was_training)

    return {
        "makespan": sum(makespans) / len(makespans),
        "invalid_jobs": sum(invalid_counts) / len(invalid_counts),
        "samples": len(makespans),
    }


def validation_checkpoint_score(validation: Mapping[str, float], invalid_penalty: float) -> float:
    return float(validation["makespan"]) + float(invalid_penalty) * float(validation["invalid_jobs"])


def maybe_save_best_model(
    *,
    model,
    best_model_path: str,
    fallback_model_path: str,
    current_metric: float,
    best_metric: float,
    metric_label: str,
) -> float:
    if current_metric < best_metric:
        save_model_weights(model, best_model_path or fallback_model_path)
        print(f"   -> Saved new best model ({metric_label}: {current_metric:.2f})")
        return current_metric
    return best_metric
