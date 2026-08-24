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


def evaluate_validation_events(events, model, solve_fn, evaluate_makespan_fn,
                               secondary_fn=None, extra_fns=None) -> dict:
    """Greedy-decode every validation instance once and score it.

    The PRIMARY score is always `evaluate_makespan_fn`, i.e. the executor, so every arm of
    an experiment is selected and reported on the deployment objective.

    `secondary_fn(individual, jobs) -> float` and `extra_fns` -- a `{name: fn}` mapping --
    score the SAME decoded schedules under further evaluators, returned as "secondary" and
    under `extras[name]`. Decoding dominates the cost (~1 s per instance) while the cheap
    evaluators cost 0.1-0.5 ms, so each extra criterion is effectively free and does not
    require a second training run. `secondary_fn` is the one-evaluator shorthand and is
    reported in `extras` as well.
    """
    scorers = dict(extra_fns or {})
    if secondary_fn is not None:
        scorers["secondary"] = secondary_fn

    if not events:
        return {
            "makespan": float("nan"),
            "invalid_jobs": float("nan"),
            "secondary": float("nan"),
            "extras": {name: float("nan") for name in scorers},
            "samples": 0,
        }

    was_training = model.training
    model.eval()
    makespans = []
    invalid_counts = []
    extras: dict[str, list] = {name: [] for name in scorers}
    try:
        with torch.no_grad():
            for event in events:
                individual, _, _ = solve_fn(list(event["jobs"]), model, deterministic=True)
                makespan, invalid_count = evaluate_makespan_fn(individual, event["jobs"])
                makespans.append(float(makespan))
                invalid_counts.append(float(invalid_count))
                for name, fn in scorers.items():
                    extras[name].append(float(fn(individual, event["jobs"])))
    finally:
        model.train(was_training)

    means = {name: (sum(vals) / len(vals) if vals else float("nan"))
             for name, vals in extras.items()}
    return {
        "makespan": sum(makespans) / len(makespans),
        "invalid_jobs": sum(invalid_counts) / len(invalid_counts),
        "secondary": means.get("secondary", float("nan")),
        "extras": means,
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


class SmoothedBestSelector:
    """Checkpoint selection on a trailing-window mean instead of a running argmin.

    WHY THIS EXISTS
    ---------------
    `maybe_save_best_model` keeps a checkpoint whenever its validation score beats the
    running minimum. Over a long run that is an argmin across many noisy draws, and the
    expected minimum of N noisy samples sits BELOW the true value by an amount that grows
    with N. The selected score is therefore optimistic by construction, and increasing
    `epochs` or decreasing `validation_interval` makes it more so -- both buy more draws,
    not a better model.

    Measured on `runs/20260812_231751_ppo_gc1_s42` (4000 epochs, validation every 19):
      * per-evaluation noise, after detrending, is ~2.85 makespan units, which is what
        averaging 50 val instances with an instance sd near 20 predicts (20/sqrt(50)=2.83)
      * argmin over its 201 draws is therefore ~7.8 units optimistic
      * its argmin landed at epoch 3860 scoring 344.38 while the local trend was 348.51,
        so 4.13 units of the saved score were a single lucky evaluation
      * the same recipe at 2000 vs 4000 epochs moved val 357.7 -> 344.4 but test only
        344.29 -> 344.72: the extra draws bought a better NUMBER, not a better model

    That bias is larger than several quantities the project decides on, including the
    2.0-unit tie band used to pick between the gc1 and gc1.5 arms.

    WHAT THIS DOES
    --------------
    Scores each checkpoint by the mean of the last `window` validation scores. Averaging
    `window` independent draws divides the noise sd by sqrt(window) and shrinks the
    selection bias in proportion; at window=5 a simulation at the measured noise level
    cuts it from -7.84 to -3.49.

    The weights saved are the ones at the END of the winning window, i.e. a point drawn
    from the best REGION of training rather than the single luckiest evaluation. That is
    the intended trade: it does not try to identify the exact best epoch, because at this
    noise level the exact best epoch is not identifiable.

    Selection is withheld until the window is full, so a partial early window built from
    one or two noisy draws cannot win.

    `window=1` reproduces the previous argmin behaviour exactly, for reproducing old runs.
    """

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = int(window)
        self._recent: list = []
        self.best_metric = float("inf")
        self.best_raw = float("inf")

    def update(self, *, model, current_metric: float, best_model_path: str,
               fallback_model_path: str, metric_label: str = "Val Score") -> float:
        """Record one validation score; save weights if this window is the best so far.

        Returns the current best smoothed metric.
        """
        self._recent.append(float(current_metric))
        if len(self._recent) > self.window:
            self._recent.pop(0)
        self.best_raw = min(self.best_raw, float(current_metric))

        if len(self._recent) < self.window:
            print(f"   -> {metric_label} {current_metric:.2f} "
                  f"(warming up {len(self._recent)}/{self.window}, no selection yet)")
            return self.best_metric

        smoothed = sum(self._recent) / len(self._recent)
        if smoothed < self.best_metric:
            save_model_weights(model, best_model_path or fallback_model_path)
            print(f"   -> Saved new best model ({metric_label} mean of last "
                  f"{self.window}: {smoothed:.2f}; this eval {current_metric:.2f})")
            self.best_metric = smoothed
        return self.best_metric
