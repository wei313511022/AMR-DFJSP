from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GNN_DIR = os.path.join(STATIC_DIR, "GNN")
for path in (STATIC_DIR, GNN_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from GA.GA import (  # noqa: E402
    Operation,
    collision_routing_iters,
    local_improve,
    make_jobs,
    routing_iters,
)
from ideal_evaluator import per_robot_ideal as ideal_per_robot  # noqa: E402
from surrogate_evaluator import surrogate_makespan  # noqa: E402
from operation_policy import (  # noqa: E402
    action_mask,
    apply_fast_action as apply_operation_action,
    decode_action_id,
    initial_operation_state,
)
from reinforce_baseline import (  # noqa: E402
    cosine_actor_lr,
    DEFAULT_BASELINE_RULE,
    compute_dispatch_baseline_comparison,
    evaluate_makespan,
    load_training_events,
    normalize_advantage_batches,
    score_instance_group,
    validate_sampling_args,
)
from train_gnn import (  # noqa: E402
    _load_validation_events,
    _save_ppo_chart,
    _save_reinforce_chart,
    _select_jobs,
    _should_validate,
    action_sequences_from_individual,
    assignment_load_stats,
    finite_log_probs_and_entropy,
    load_balance_step_advantages,
)
from training_checkpoints import (  # noqa: E402
    SmoothedBestSelector,
    evaluate_validation_events,
    load_training_checkpoint,
    maybe_save_best_model,
    save_training_checkpoint,
    validation_checkpoint_score,
)
from extend_GNN.extend_GNN import (  # noqa: E402
    ExtendSchedulerGNN,
    _encode_state_tensors,
    _empty_dock_service_events,
    extract_state_extend_gnn,
    solve_with_extend_gnn,
)


LEGACY_BEST_MODEL_PATH = "extend_gnn_scheduler_best.pth"

REPO_ROOT = os.path.abspath(os.path.join(STATIC_DIR, ".."))

# Columns written to metrics.csv. The prose in run.log is for reading; these
# are for plotting. Validation columns stay empty on non-validation epochs, so
# the validation curve is `df.dropna(subset=["val_score"])`.
REINFORCE_FIELDS = (
    "epoch", "elapsed_s", "lr", "sampled", "baseline", "improvement", "win_rate",
    "actor_loss", "entropy", "grad_norm", "invalid_sampled", "invalid_baseline",
    "max_load", "load_gap", "group_cv", "val_makespan", "val_invalid", "val_score",
)
PPO_FIELDS = (
    "epoch", "elapsed_s", "lr", "makespan", "actor_loss", "critic_loss",
    "grad_norm", "clipped_updates", "val_makespan", "val_invalid", "val_score",
    "val_ideal", "val_surrogate",
)


class RunLogger:
    """Per-run artifact directory: args.json, run.log, metrics.csv, chart.png.

    Both files are opened line-buffered (`buffering=1`) so that a run which is
    killed, or whose stdout was redirected into a block-buffered pipe, still
    leaves a complete record on disk. Four v7 variants were lost to a forgotten
    shell redirect and three more lagged hours behind in an 8 KB stdout buffer;
    the trainer now writes its own artifacts regardless of how it was launched.

    Checkpoint paths are deliberately NOT moved in here -- they stay wherever
    --best_model_path points, so existing checkpoints_v*/ layouts and the
    evaluation harness keep working. Only the chart moves, because a path fixed
    to this file's own directory meant every concurrent run overwrote one PNG.
    """

    def __init__(self, args, fields):
        stem = args.run_name or os.path.splitext(os.path.basename(args.best_model_path))[0]
        self.dir = args.run_dir or os.path.join(
            REPO_ROOT, "runs", f"{time.strftime('%Y%m%d_%H%M%S')}_{stem}"
        )
        os.makedirs(self.dir, exist_ok=True)
        self.chart_path = os.path.join(self.dir, "chart.png")
        self._start = time.time()

        with open(os.path.join(self.dir, "args.json"), "w") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)

        self._log = open(os.path.join(self.dir, "run.log"), "a", buffering=1)

        csv_path = os.path.join(self.dir, "metrics.csv")
        fresh = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        self._csv_handle = open(csv_path, "a", buffering=1, newline="")
        self._csv = csv.DictWriter(self._csv_handle, fieldnames=list(fields), extrasaction="ignore")
        if fresh:
            self._csv.writeheader()

    def log(self, message):
        print(message, flush=True)
        self._log.write(message + "\n")

    def row(self, **values):
        values.setdefault("elapsed_s", round(time.time() - self._start, 1))
        self._csv.writerow(values)

    def close(self):
        for handle in (self._log, self._csv_handle):
            try:
                handle.close()
            except OSError:
                pass


def _actor_params(model):
    return list(model.actor_parameters())


def evaluate_action_steps_extend(
    jobs,
    model,
    action_seq,
    init_state=None,
    include_values: bool = False,
):
    amr_positions, amr_availabilities, station_availabilities, amr_inventory = initial_operation_state(init_state)
    picked_jobs_set = set()
    completed_jobs_set = set()
    carrier_map = {}
    order_seq = []
    dock_service_events = _empty_dock_service_events()

    step_log_probs = []
    step_entropies = []
    values = []

    model.train()
    device = next(model.parameters()).device

    for chosen_action in action_seq:
        tensors = extract_state_extend_gnn(
            jobs,
            picked_jobs_set,
            completed_jobs_set,
            carrier_map,
            amr_positions,
            amr_availabilities,
            amr_inventory,
            station_availabilities,
            order_seq,
            dock_service_events,
        )
        (
            job_embeddings,
            amr_embeddings,
            dock_embeddings,
            action_feat,
            job_mask,
            inbound_idx,
            outbound_idx,
        ) = _encode_state_tensors(model, tensors, device)
        op_mask = torch.tensor(
            [action_mask(jobs, picked_jobs_set, completed_jobs_set, carrier_map, amr_inventory)],
            dtype=torch.bool,
            device=device,
        )

        if include_values:
            values.append(model.forward_critic(job_embeddings, amr_embeddings, dock_embeddings, job_mask).squeeze())

        logits = model.forward_operation_actor(
            job_embeddings,
            amr_embeddings,
            dock_embeddings,
            action_feat,
            inbound_idx,
            outbound_idx,
            op_mask,
        ).view(-1)
        if not torch.isfinite(logits[chosen_action]):
            raise RuntimeError("Chosen extend_GNN operation action has a non-finite logit during replay.")
        action_log_probs, entropy = finite_log_probs_and_entropy(logits, "extend_GNN operation replay")
        step_log_probs.append(action_log_probs[chosen_action])
        step_entropies.append(entropy)

        action = decode_action_id(chosen_action, jobs)
        order_seq.append(Operation(action.job_id, action.kind))
        apply_operation_action(
            action,
            jobs,
            picked_jobs_set,
            completed_jobs_set,
            carrier_map,
            amr_positions,
            amr_availabilities,
            station_availabilities,
            amr_inventory,
            dock_service_events=dock_service_events,
        )

    value_tensor = torch.stack(values) if values else None
    return torch.stack(step_log_probs), torch.stack(step_entropies), value_tensor


def evaluate_actions_extend_ppo(jobs, model, action_seq, init_state=None):
    """PER-STEP log probs, not their sum.

    A trajectory-level importance ratio is the product of ~120 per-step ratios,
    so it leaves the PPO clip window after a single ordinary update: measured
    ratio 4.97e+02 against a window of [0.8, 1.2]. Past that point `min(surr1,
    surr2)` returns the clipped branch for positive advantages (zero gradient)
    and the unclipped branch for negative ones (497x amplified) -- no trust
    region in either direction. Clipping each step separately is what PPO
    actually specifies.
    """
    step_log_probs, _, values = evaluate_action_steps_extend(
        jobs,
        model,
        action_seq,
        init_state=init_state,
        include_values=True,
    )
    return step_log_probs, values


# Congestion-observability levels for the A/B/C/D factorial. Indices match
# eval_extend_gnn.MASKS so a model trained under a level can be evaluated under the same
# one; keep the two tables in step.
#   amr[9]    local_density        dock[5]  service_remaining
#   dock[2]   available_delay      dock[6]  committed_workload
#   dock[3]   queue_count / m      act[2]   estimated_dock_wait
#   dock[4]   queue_count / slots
TRAIN_MASKS = {
    "none": {},
    "L1": {"amr": (9,), "dock": (2, 3, 4)},
    "L2": {"amr": (9,), "dock": (2, 3, 4, 5, 6)},
    "L3": {"amr": (9,), "dock": (2, 3, 4, 5, 6), "action": (2,)},
    "control": {"amr": (10, 11), "dock": (7, 8)},
}


def ideal_validation_score(individual, jobs) -> float:
    """C~ of eq. (5) for one schedule -- the idealised paradigm's own yardstick."""
    per_robot = ideal_per_robot(individual, jobs)
    return max(per_robot.values()) if per_robot else 0.0


# Every validation decode is scored under all three evaluators. The executor score is the
# primary one and selects `_best.pth` in EVERY cell, so the factorial changes one thing per
# axis; these two are free riders on the same decoded schedules (0.1-0.5 ms against ~1 s to
# decode) and cost no extra training run.
VALIDATION_EXTRAS = {
    "ideal": ideal_validation_score,
    "surrogate": surrogate_makespan,
}

# SECOND CHECKPOINT PROTOCOL. An arm trained on a cheap evaluator is also selected inside
# its own paradigm: a practitioner who never calls the executor cannot select on it either,
# and cannot even SEE a routing failure -- hence the native selection carries no invalid
# penalty. Saving both from one run lets the factorial be reported either with selection
# held fixed (executor everywhere, one factor changed) or with each cheap arm run end to
# end inside its own paradigm. The executor arm has no entry: for it the two protocols are
# the same checkpoint.
#   --train_evaluator  ->  (validation extra it selects on, checkpoint suffix, log label)
NATIVE_SELECTION = {
    "ideal": ("ideal", "_bestideal.pth", "Val C~"),
    "surrogate": ("surrogate", "_bestsurr.pth", "Val Psi-hat"),
}


def install_train_feature_mask(level: str):
    """Zero the named feature slices on EVERY state extraction, for the whole run.

    Unlike the inference-time mask in eval_extend_gnn.py, this one is installed before
    training starts, so the network never sees the channels at all and learns to
    compensate without them. That is the difference between an observability ablation
    and feeding a trained network zeros, and the two give very different answers.

    The name is resolved from two different namespaces and BOTH must be rebound:
      solve_with_extend_gnn         -> extend_GNN.extend_GNN globals   (rollout)
      evaluate_action_steps_extend  -> this module's globals           (PPO re-evaluation)
    Rebinding only one leaves the rollout and the log-prob re-evaluation disagreeing
    about the state, which corrupts the PPO ratio silently -- the loss still goes down.
    """
    if level not in TRAIN_MASKS:
        raise ValueError(f"unknown --train_mask {level!r}; valid: {sorted(TRAIN_MASKS)}")
    spec = TRAIN_MASKS[level]
    if not spec:
        return
    import extend_GNN.extend_GNN as _xg
    original = _xg.extract_state_extend_gnn

    def masked(*a, **kw):
        t = list(original(*a, **kw))
        for idx in spec.get("amr", ()):        # t[0] (1, m, AMR_IN_DIM)
            t[0][..., idx] = 0.0
        for idx in spec.get("dock", ()):       # t[2] (1, |D|, DOCK_IN_DIM)
            t[2][..., idx] = 0.0
        for idx in spec.get("action", ()):     # t[3] (1, 2, m, n, ACTION_IN_DIM)
            t[3][..., idx] = 0.0
        return tuple(t)

    _xg.extract_state_extend_gnn = masked
    globals()["extract_state_extend_gnn"] = masked
    if getattr(_xg.solve_with_extend_gnn, "__globals__", {}).get("extract_state_extend_gnn"):
        _xg.solve_with_extend_gnn.__globals__["extract_state_extend_gnn"] = masked
    print(f"   -> training feature mask {level} installed: {spec}")


def _load_validation_events_extend(args):
    return _load_validation_events(args)


def train_reinforce(args):
    validate_sampling_args(args.batch_size, args.samples_per_instance, args.baseline_mode)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = RunLogger(args, REINFORCE_FIELDS)
    run.log(f"Training on: {device}")
    run.log(f"Run directory: {run.dir}")

    dispatch_events = load_training_events(args.inbox, args.inboxes)
    validation_events = _load_validation_events_extend(args)
    model = ExtendSchedulerGNN(hidden_dim=128, gin_layers=3).to(device)
    optimizer_actor = optim.Adam(_actor_params(model), lr=args.lr_actor)
    load_training_checkpoint(
        model,
        {"optimizer_actor": optimizer_actor},
        args.init_checkpoint,
        device,
    )

    run.log(
        "Starting hybrid dock-aware extend_GNN REINFORCE training "
        f"with dispatch baseline '{args.baseline_rule}' ({args.baseline_mode})."
    )

    selector = SmoothedBestSelector(window=args.val_window)
    best_makespan = float("inf")
    actor_losses = []
    sampled_makespans = []
    baseline_makespans = []
    improvements = []
    win_rates = []
    chart_path = run.chart_path

    def _actor_lr(epoch: int) -> float:
        return cosine_actor_lr(epoch, args.epochs, args.lr_actor, args.lr_min)

    for epoch in range(1, args.epochs + 1):
        current_lr = _actor_lr(epoch)
        for group in optimizer_actor.param_groups:
            group["lr"] = current_lr
        trajectories = []
        batch_advantages = []
        batch_sampled = []
        batch_baseline = []
        batch_improvement = []
        batch_wins = []
        batch_sampled_invalid = []
        batch_baseline_invalid = []
        batch_max_load = []
        batch_load_gap = []

        batch_group_cvs = []

        for group_idx in range(args.batch_size // args.samples_per_instance):
            jobs = _select_jobs(dispatch_events)
            group_individuals = []
            group_action_seqs = []
            for _ in range(args.samples_per_instance):
                individual, _, _ = solve_with_extend_gnn(jobs, model, deterministic=False)
                group_individuals.append(individual)
                group_action_seqs.append(action_sequences_from_individual(individual, jobs))

            group_samples, group_cv = score_instance_group(
                jobs,
                group_individuals,
                baseline_rule=args.baseline_rule,
                baseline_mode=args.baseline_mode,
                seed=args.seed + group_idx,
                invalid_penalty=args.train_invalid_penalty,
            )
            batch_group_cvs.append(group_cv)

            for individual, action_seq, sample in zip(
                group_individuals, group_action_seqs, group_samples
            ):
                trajectories.append(
                    (
                        jobs,
                        action_seq,
                        sample.step_advantages,
                        load_balance_step_advantages(action_seq, jobs),
                    )
                )
                batch_advantages.append(sample.step_advantages)
                batch_sampled.append(sample.sampled_makespan)
                batch_baseline.append(sample.baseline_makespan)
                batch_improvement.append(sample.improvement)
                batch_wins.append(1.0 if sample.win else 0.0)
                batch_sampled_invalid.append(sample.sampled_invalid_jobs)
                batch_baseline_invalid.append(sample.baseline_invalid_jobs)
                max_load, _, load_gap = assignment_load_stats(individual)
                batch_max_load.append(max_load)
                batch_load_gap.append(load_gap)

        normalized_advantages = normalize_advantage_batches(
            batch_advantages,
            enabled=args.normalize_advantage,
        )

        optimizer_actor.zero_grad()
        epoch_actor_loss = 0.0
        epoch_entropy = 0.0

        for (jobs, action_seq, _, load_advantages), advantages in zip(trajectories, normalized_advantages):
            step_log_probs, step_entropies, _ = evaluate_action_steps_extend(jobs, model, action_seq)
            advantage_tensor = torch.tensor(advantages, dtype=torch.float32, device=step_log_probs.device)
            load_advantage_tensor = torch.tensor(load_advantages, dtype=torch.float32, device=step_log_probs.device)
            action_advantage_tensor = advantage_tensor + args.load_balance_coef * load_advantage_tensor
            actor_loss = -(step_log_probs * action_advantage_tensor).sum()
            entropy_bonus = step_entropies.sum()
            total_loss = (actor_loss - args.entropy_coef * entropy_bonus) / args.batch_size
            if not torch.isfinite(total_loss).item():
                raise RuntimeError("Non-finite extend_GNN REINFORCE loss before backward.")
            total_loss.backward()
            epoch_actor_loss += actor_loss.item()
            epoch_entropy += step_entropies.mean().item()

        # clip_grad_norm_ returns the total norm BEFORE clipping, which is the
        # only way to tell whether --grad_clip is binding at all. Passing inf
        # when clipping is disabled measures without altering the gradient.
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            _actor_params(model), args.grad_clip if args.grad_clip > 0 else float("inf")
        ))
        optimizer_actor.step()

        avg_actor_loss = epoch_actor_loss / args.batch_size
        avg_sampled = sum(batch_sampled) / args.batch_size
        avg_baseline = sum(batch_baseline) / args.batch_size
        avg_improvement = sum(batch_improvement) / args.batch_size
        win_rate = sum(batch_wins) / args.batch_size
        avg_entropy = epoch_entropy / args.batch_size
        avg_sampled_invalid = sum(batch_sampled_invalid) / args.batch_size
        avg_baseline_invalid = sum(batch_baseline_invalid) / args.batch_size
        avg_max_load = sum(batch_max_load) / args.batch_size
        avg_load_gap = sum(batch_load_gap) / args.batch_size

        actor_losses.append(avg_actor_loss)
        sampled_makespans.append(avg_sampled)
        baseline_makespans.append(avg_baseline)
        improvements.append(avg_improvement)
        win_rates.append(win_rate)

        # Group CV is the saturation monitor for --baseline_mode multisample:
        # the group mean IS the baseline, so a collapsing group means vanishing
        # advantages. Healthy is ~6-8%; under ~1-2% the signal is dying.
        avg_group_cv = (
            sum(batch_group_cvs) / len(batch_group_cvs)
            if args.samples_per_instance > 1 and batch_group_cvs
            else None
        )
        group_cv_note = "" if avg_group_cv is None else f" | Group CV: {avg_group_cv:.4f}"

        run.log(
            f"Epoch [{epoch}/{args.epochs}] | Sampled: {avg_sampled:.2f} "
            f"| Baseline: {avg_baseline:.2f} | Improvement: {avg_improvement:.2f} "
            f"| Win Rate: {win_rate:.2%} | Actor Loss: {avg_actor_loss:.4f} "
            f"| Entropy: {avg_entropy:.4f} "
            f"| Invalid S/B: {avg_sampled_invalid:.2f}/{avg_baseline_invalid:.2f} "
            f"| Max Load: {avg_max_load:.1f} | Load Gap: {avg_load_gap:.1f} "
            f"| LR: {current_lr:.2e}"
            f"{group_cv_note}"
        )
        metrics_row = {
            "epoch": epoch,
            "lr": f"{current_lr:.6e}",
            "sampled": round(avg_sampled, 4),
            "baseline": round(avg_baseline, 4),
            "improvement": round(avg_improvement, 4),
            "win_rate": round(win_rate, 6),
            "actor_loss": round(avg_actor_loss, 6),
            "entropy": round(avg_entropy, 6),
            "grad_norm": round(grad_norm, 6),
            "invalid_sampled": round(avg_sampled_invalid, 4),
            "invalid_baseline": round(avg_baseline_invalid, 4),
            "max_load": round(avg_max_load, 4),
            "load_gap": round(avg_load_gap, 4),
            "group_cv": "" if avg_group_cv is None else round(avg_group_cv, 6),
        }

        if _should_validate(epoch, args, validation_events):
            validation = evaluate_validation_events(
                validation_events,
                model,
                solve_with_extend_gnn,
                evaluate_makespan,
            )
            validation_score = validation_checkpoint_score(validation, args.validation_invalid_penalty)
            run.log(
                f"   -> Validation | Samples: {validation['samples']} "
                f"| Makespan: {validation['makespan']:.2f} "
                f"| Invalid Jobs: {validation['invalid_jobs']:.2f} "
                f"| Score: {validation_score:.2f}"
            )
            metrics_row.update(
                val_makespan=round(float(validation["makespan"]), 4),
                val_invalid=round(float(validation["invalid_jobs"]), 4),
                val_score=round(float(validation_score), 4),
            )
            best_makespan = selector.update(
                model=model,
                current_metric=validation_score,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                metric_label="Val Score",
            )

            # Rollback-on-divergence was REMOVED so that all three trainers run
            # an identical loop and an architecture comparison is not confounded
            # by one model getting a recovery mechanism the others lack. It used
            # to reload the best checkpoint and halve the LR after
            # `rollback_patience` worsening validations. If a run diverges, that
            # is now a result to report, not something the loop hides -- the
            # best-by-validation checkpoint is still saved, so a late collapse
            # costs the run's tail but not its best model.
        elif not validation_events:
            best_makespan = selector.update(
                model=model,
                current_metric=avg_sampled,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                metric_label="Makespan",
            )

        run.row(**metrics_row)

        save_training_checkpoint(
            args.latest_checkpoint_path,
            model,
            {"optimizer_actor": optimizer_actor},
            epoch,
            best_makespan,
            args,
        )

        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            _save_reinforce_chart(
                chart_path,
                "extend_GNN",
                actor_losses,
                sampled_makespans,
                baseline_makespans,
                improvements,
                win_rates,
            )
            run.log(f"   -> Saved updated training chart to {chart_path}")

    run.close()


def train_ppo(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = RunLogger(args, PPO_FIELDS)
    run.log(f"Training on: {device}")
    run.log(f"Run directory: {run.dir}")

    dispatch_events = load_training_events(args.inbox, args.inboxes)
    validation_events = _load_validation_events_extend(args)
    model = ExtendSchedulerGNN(hidden_dim=128, gin_layers=3).to(device)
    optimizer_actor = optim.Adam(_actor_params(model), lr=args.lr_actor)
    optimizer_critic = optim.Adam(model.critic.parameters(), lr=args.lr_critic)
    load_training_checkpoint(
        model,
        {"optimizer_actor": optimizer_actor, "optimizer_critic": optimizer_critic},
        args.init_checkpoint,
        device,
    )

    run.log("Starting hybrid dock-aware extend_GNN PPO training with critic-based advantage.")

    selector = SmoothedBestSelector(window=args.val_window)
    native = NATIVE_SELECTION.get(args.train_evaluator)
    selector_native = SmoothedBestSelector(window=args.val_window) if native else None
    native_best_path = ""
    if native:
        native_key, native_suffix, native_label = native
        native_best_path = (args.best_model_path or LEGACY_BEST_MODEL_PATH).replace(
            "_best.pth", native_suffix)
        run.log(f"Second checkpoint protocol: {native_label} selects {native_best_path}")
    best_makespan = float("inf")
    losses_actor = []
    losses_critic = []
    makespans = []
    chart_path = run.chart_path

    def _actor_lr(epoch: int) -> float:
        return cosine_actor_lr(epoch, args.epochs, args.lr_actor, args.lr_min)

    # Running return normaliser. Raw targets are -makespan, around -650, while a
    # freshly initialised critic emits ~0 -- an MSE near 420000 whose gradient is
    # then clipped to norm `grad_clip`, so the critic would need on the order of
    # a million steps just to reach the right ORDER OF MAGNITUDE. It never gets
    # there, V stays flat, and PPO silently degenerates into a batch-mean
    # baseline forever. Predicting a normalised return puts the critic on a unit
    # scale where it can actually learn the per-step structure.
    ret_mean, ret_var, ret_count = 0.0, 1.0, 1e-4

    for epoch in range(1, args.epochs + 1):
        current_lr = _actor_lr(epoch)
        for group in optimizer_actor.param_groups:
            group["lr"] = current_lr
        batch_makespans = []
        trajectories = []

        for _ in range(args.batch_size):
            jobs = _select_jobs(dispatch_events)
            individual, _, _ = solve_with_extend_gnn(jobs, model, deterministic=False)
            action_seq = action_sequences_from_individual(individual, jobs)

            # Score the schedule the POLICY produced. Previously `individual`
            # was rebound by local_improve after `action_seq` had been taken
            # from it, so the value target measured a schedule the recorded
            # actions did not build -- the policy was credited for gains the
            # local search made. The REINFORCE path never applied local_improve
            # at all; dropping it here makes the two paths comparable.
            if args.train_evaluator == "ideal":
                # Train against C~ of eq. (5): free-space travel, service starting on
                # arrival, no queueing and no inter-robot interference. This is the
                # transport model of the fixed-travel-time literature. It cannot fail to
                # route, so there is no invalid count to penalise -- which is itself part
                # of what the arm measures. Validation stays executor-relative, so every
                # cell is still SELECTED and REPORTED on the deployment objective.
                per_robot = ideal_per_robot(individual, jobs)
                stochastic_makespan = max(per_robot.values()) if per_robot else 0.0
                invalid = 0
            elif args.train_evaluator == "surrogate":
                # Train against Psi-hat: the calibrated fast model the rollout already
                # advances, replayed over the finished schedule. C~ is the WEAK cheap
                # opponent -- 15.3% value error and tau 0.454 against the executor at n=60,
                # against Psi-hat's 7.0% and 0.736 (experiments/surrogate_fidelity). An arm
                # that loses to C~ may only be losing to a strawman, so this is the cheap
                # pipeline a deployment would really build. Like C~ it does no routing, so
                # again there is no invalid count.
                stochastic_makespan = surrogate_makespan(individual, jobs)
                invalid = 0
            else:
                stochastic_makespan, invalid = evaluate_makespan(individual, jobs)
            score = float(stochastic_makespan) + args.train_invalid_penalty * float(invalid)
            batch_makespans.append(float(stochastic_makespan))

            # old_log_probs AND old values, per step, under the pre-update
            # policy. PPO computes advantages ONCE per batch from the old
            # policy and holds them fixed across the inner epochs; recomputing
            # them each inner epoch would make the "old" reference drift.
            with torch.no_grad():
                old_log_probs, _, old_values = evaluate_action_steps_extend(
                    jobs, model, action_seq, include_values=True
                )
            trajectories.append(
                (jobs, action_seq, old_log_probs.detach(), old_values.detach(), -score)
            )

        # Advantages once per batch, from the old policy, normalised ACROSS the
        # batch rather than within a trajectory. Within-trajectory normalisation
        # is degenerate here: an untrained critic emits near-constant values, so
        # dividing by their tiny spread amplifies pure critic noise into the
        # whole gradient. Pooling keeps early training sane -- while V is flat,
        # each trajectory gets a near-constant advantage set by its score
        # relative to the batch, which is exactly a batch-mean baseline, and
        # per-step structure emerges only as the critic actually learns.
        # Update the running return statistics, then normalise the targets.
        batch_returns = [vt for _, _, _, _, vt in trajectories]
        b_mean = sum(batch_returns) / len(batch_returns)
        b_var = sum((r - b_mean) ** 2 for r in batch_returns) / max(1, len(batch_returns) - 1)
        delta = b_mean - ret_mean
        tot = ret_count + len(batch_returns)
        ret_mean += delta * len(batch_returns) / tot
        ret_var = (ret_var * ret_count + b_var * len(batch_returns)
                   + delta * delta * ret_count * len(batch_returns) / tot) / tot
        ret_count = tot
        ret_std = max(math.sqrt(max(ret_var, 0.0)), 1e-6)
        trajectories = [(j, s, olp, ov, (vt - ret_mean) / ret_std)
                        for j, s, olp, ov, vt in trajectories]

        adv_list = [torch.full_like(ov, vt) - ov for _, _, _, ov, vt in trajectories]
        flat = torch.cat(adv_list)
        adv_mean, adv_std = flat.mean(), flat.std()
        adv_list = [(a - adv_mean) / (adv_std + 1e-8) for a in adv_list]
        trajectories = [
            (j, s, olp, adv, vt)
            for (j, s, olp, _, vt), adv in zip(trajectories, adv_list)
        ]

        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0
        epoch_grad_norm = 0.0
        clipped_updates = 0

        for _ in range(args.ppo_epochs):
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            batch_actor_loss = 0.0
            batch_critic_loss = 0.0

            for jobs, action_seq, old_log_probs, advantages, value_target in trajectories:
                new_log_probs, values = evaluate_actions_extend_ppo(jobs, model, action_seq)
                value_target_tensor = torch.tensor(value_target, dtype=torch.float32, device=values.device)
                value_targets = value_target_tensor.expand_as(values)

                # `advantages` is PER STEP: R - V(s_t), computed once above.
                # With a terminal-only reward and no discounting the return-to-go
                # is the same R from every state, so `value_targets` is correctly
                # one scalar -- but V(s_t) varies with s_t, and that variation IS
                # the per-step credit a critic exists to provide. The previous
                # `values.detach().mean()` collapsed 111 distinct values into a
                # single number, leaving PPO with the same credit granularity as
                # --baseline_mode episode, plus the clip instability.
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, value_targets)
                total_loss = (actor_loss + args.value_loss_coef * critic_loss) / args.batch_size
                total_loss.backward()
                batch_actor_loss += actor_loss.item()
                batch_critic_loss += critic_loss.item()

            cap = args.grad_clip if args.grad_clip > 0 else float("inf")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(_actor_params(model), cap))
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), cap)
            epoch_grad_norm += grad_norm
            clipped_updates += 1 if grad_norm > cap else 0
            optimizer_actor.step()
            optimizer_critic.step()
            epoch_actor_loss += batch_actor_loss / args.batch_size
            epoch_critic_loss += batch_critic_loss / args.batch_size

        epoch_actor_loss /= args.ppo_epochs
        epoch_critic_loss /= args.ppo_epochs
        avg_batch_makespan = sum(batch_makespans) / args.batch_size
        losses_actor.append(epoch_actor_loss)
        losses_critic.append(epoch_critic_loss)
        makespans.append(avg_batch_makespan)

        avg_grad_norm = epoch_grad_norm / args.ppo_epochs
        run.log(
            f"Epoch [{epoch}/{args.epochs}] | Avg Makespan: {avg_batch_makespan:.2f} "
            f"| Actor Loss: {epoch_actor_loss:.4f} | Critic Loss: {epoch_critic_loss:.4f} "
            f"| Grad Norm: {avg_grad_norm:.4f} | Clipped: {clipped_updates}/{args.ppo_epochs}"
        )
        metrics_row = {
            "epoch": epoch,
            "lr": f"{current_lr:.6e}",
            "makespan": round(avg_batch_makespan, 4),
            "actor_loss": round(epoch_actor_loss, 6),
            "critic_loss": round(epoch_critic_loss, 6),
            "grad_norm": round(avg_grad_norm, 6),
            "clipped_updates": clipped_updates,
        }

        if _should_validate(epoch, args, validation_events):
            validation = evaluate_validation_events(
                validation_events,
                model,
                solve_with_extend_gnn,
                evaluate_makespan,
                extra_fns=VALIDATION_EXTRAS,
            )
            validation_score = validation_checkpoint_score(validation, args.validation_invalid_penalty)
            run.log(
                f"   -> Validation | Samples: {validation['samples']} "
                f"| Makespan: {validation['makespan']:.2f} "
                f"| Invalid Jobs: {validation['invalid_jobs']:.2f} "
                f"| Score: {validation_score:.2f}"
            )
            metrics_row.update(
                val_makespan=round(float(validation["makespan"]), 4),
                val_invalid=round(float(validation["invalid_jobs"]), 4),
                val_score=round(float(validation_score), 4),
                val_ideal=round(float(validation["extras"]["ideal"]), 4),
                val_surrogate=round(float(validation["extras"]["surrogate"]), 4),
            )
            best_makespan = selector.update(
                model=model,
                current_metric=validation_score,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                metric_label="Val Score",
            )
            # Second checkpoint, selected on the arm's own cheap evaluator alone and
            # with no invalid penalty -- see NATIVE_SELECTION.
            if selector_native is not None:
                selector_native.update(
                    model=model,
                    current_metric=float(validation["extras"][native_key]),
                    best_model_path=native_best_path,
                    fallback_model_path=native_best_path,
                    metric_label=native_label,
                )
        elif not validation_events:
            best_makespan = selector.update(
                model=model,
                current_metric=avg_batch_makespan,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                metric_label="Makespan",
            )

        run.row(**metrics_row)

        save_training_checkpoint(
            args.latest_checkpoint_path,
            model,
            {"optimizer_actor": optimizer_actor, "optimizer_critic": optimizer_critic},
            epoch,
            best_makespan,
            args,
        )

        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            _save_ppo_chart(chart_path, losses_actor, losses_critic, makespans)
            run.log(f"   -> Saved updated training chart to {chart_path}")

    run.close()


def train(args):
    if args.validation_interval < 1:
        raise ValueError("--validation_interval must be at least 1")
    if args.validation_invalid_penalty < 0:
        raise ValueError("--validation_invalid_penalty must be non-negative")
    if args.lr_min <= 0 or args.lr_min > args.lr_actor:
        raise ValueError("--lr_min must be positive and at most --lr_actor")
    if args.train_invalid_penalty < 0:
        raise ValueError("--train_invalid_penalty must be non-negative")
    if args.train_mask not in TRAIN_MASKS:
        raise ValueError(f"--train_mask must be one of {sorted(TRAIN_MASKS)}")
    if args.train_evaluator != "executor" and args.rl_method != "ppo":
        # score_instance_group in the REINFORCE path computes its dispatch baseline with
        # the executor too; swapping only the policy's return there would compare a
        # cheaply scored rollout against a Phi-scored baseline. Refuse rather than train a
        # meaningless advantage for four thousand epochs.
        raise ValueError(
            f"--train_evaluator {args.train_evaluator} is implemented for --rl_method ppo only")
    # Installed before any rollout so the network never sees the masked channels.
    install_train_feature_mask(args.train_mask)
    if args.rl_method == "ppo":
        train_ppo(args)
    else:
        train_reinforce(args)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file")
    parser.add_argument("--inboxes", type=str, default="", help="Comma-separated dispatch JSONL files used as one combined training pool")
    parser.add_argument("--validation_inbox", type=str, default="", help="Optional fixed validation dispatch JSONL file")
    parser.add_argument("--validation_inboxes", type=str, default="", help="Comma-separated fixed validation dispatch JSONL files")
    parser.add_argument("--validation_interval", type=int, default=50, help="Epoch interval for fixed validation scoring")
    # Checkpoint selection. `argmin` over N noisy validation scores is optimistic by an
    # amount that GROWS with N, so more frequent validation buys a better number rather
    # than a better model. Averaging the last `--val_window` scores divides the noise by
    # sqrt(window). Pass 1 to restore the old running-argmin exactly.
    #
    # The other half of the fix is the validation set itself: prefer FEWER, BIGGER
    # evaluations at equal cost. 201 evals x 50 instances (what checkpoints_v8 used) is
    # strictly worse than 40 x 250 -- the latter has ~1/2 the per-eval noise AND ~1/5 the
    # draws to be lucky in. See test_case/v3/val_250.jsonl and SmoothedBestSelector.
    parser.add_argument("--val_window", type=int, default=5,
                        help="select on the mean of the last N validation scores (1 = old argmin)")
    parser.add_argument("--validation_invalid_penalty", type=float, default=1000.0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5, help="Floor of the cosine actor LR decay; set equal to the actor LR to disable the schedule")
    parser.add_argument("--train_invalid_penalty", type=float, default=0.0, help="Per-invalid-job penalty folded into training advantages (validation uses --validation_invalid_penalty)")
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="", help="Label for this run's artifact directory; defaults to the --best_model_path stem")
    parser.add_argument("--run_dir", type=str, default="", help="Write args.json/run.log/metrics.csv/chart.png here instead of runs/<timestamp>_<run_name>/. Reusing a directory appends to its log and CSV")
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--latest_checkpoint_path", type=str, default="extend_gnn_training_checkpoint.pth")
    parser.add_argument("--best_model_path", type=str, default="extend_gnn_scheduler_best.pth")
    parser.add_argument("--rl_method", choices=["reinforce", "ppo"], default="reinforce")
    parser.add_argument("--baseline_rule", type=str, default=DEFAULT_BASELINE_RULE)
    parser.add_argument(
        "--baseline_mode",
        choices=["stepwise", "episode", "multisample"],
        default="stepwise",
        help="How the REINFORCE baseline is built. 'multisample' centres each "
             "sample on the mean score of --samples_per_instance samples of the "
             "SAME instance instead of on the dispatch rule",
    )
    parser.add_argument(
        "--samples_per_instance",
        type=int,
        default=1,
        help="Rollouts drawn per training instance (K). Must divide --batch_size. "
             "K>=2 is required by --baseline_mode multisample; K>1 with the other "
             "modes is the ablation that changes instance sampling but keeps the "
             "dispatch-rule baseline",
    )
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--load_balance_coef", type=float, default=0.1)
    parser.add_argument("--normalize_advantage", dest="normalize_advantage", action="store_true", default=True)
    parser.add_argument("--no_normalize_advantage", dest="normalize_advantage", action="store_false")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    # --- A/B/C/D/E/F factorial axes -----------------------------------------------------
    # observability x training evaluator, holding architecture, action space and
    # parameter count fixed. Cell D is (none, executor), which is the default, so an
    # existing run is reproduced by omitting both flags.
    parser.add_argument("--train_mask", type=str, default="none",
                        help="congestion channels zeroed for the whole run: "
                             "none|L1|L2|L3|control (see TRAIN_MASKS)")
    parser.add_argument("--train_evaluator", type=str, default="executor",
                        choices=("executor", "ideal", "surrogate"),
                        help="what prices the training return: the collision-aware "
                             "executor Phi, the idealised decode C~ of eq. (5), or the "
                             "calibrated surrogate Psi-hat. Validation stays "
                             "executor-relative in every case; the two cheap arms "
                             "additionally save a checkpoint selected inside their own "
                             "paradigm (see NATIVE_SELECTION).")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
