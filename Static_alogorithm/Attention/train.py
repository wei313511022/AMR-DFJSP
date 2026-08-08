from __future__ import annotations

import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from Attention import (
    AMR_KEYS,
    AMR_STARTS,
    STATIONS,
    TYPE_DURATION,
    SchedulerAttention,
    extract_state,
    heuristic,
    solve_with_attention,
)
from GA.GA import (
    PICKUP,
    collision_routing_iters,
    empty_count_inventory,
    job_pickup_location,
    local_improve,
    make_jobs,
    normalize_count_inventory,
    repair_operation_order,
    routing_iters,
)
from reinforce_baseline import (
    cosine_actor_lr,
    DEFAULT_BASELINE_RULE,
    compute_dispatch_baseline_comparison,
    evaluate_makespan,
    load_training_events,
    normalize_advantage_batches,
    score_instance_group,
    validate_sampling_args,
)
from operation_policy import (
    action_mask,
    apply_fast_action as apply_operation_action,
    decode_action_id,
    empty_dock_service_events,
    initial_operation_state,
    load_balance_step_advantages_from_actions,
    operation_sequence_from_individual,
)
from training_checkpoints import (
    evaluate_validation_events,
    load_training_checkpoint,
    maybe_save_best_model,
    save_training_checkpoint,
    validation_checkpoint_score,
)


LEGACY_BEST_MODEL_PATH = "attention_scheduler_best.pth"


def _initial_fast_state(init_state=None):
    if init_state:
        amr_positions = {amr: init_state["positions"].get(amr, AMR_STARTS[amr]) for amr in AMR_KEYS}
        amr_availabilities = {amr: float(init_state["availability"].get(amr, 0.0)) for amr in AMR_KEYS}
        station_availabilities = {s: float(init_state["time"]) for s in STATIONS.keys()}
        amr_inventory = normalize_count_inventory(init_state.get("inventory", {}))
    else:
        amr_positions = {amr: AMR_STARTS[amr] for amr in AMR_KEYS}
        amr_availabilities = {amr: 0.0 for amr in AMR_KEYS}
        station_availabilities = {s: 0.0 for s in STATIONS.keys()}
        amr_inventory = empty_count_inventory()
    return amr_positions, amr_availabilities, station_availabilities, amr_inventory


def _apply_fast_action(
    chosen_job,
    chosen_amr: str,
    amr_positions,
    amr_availabilities,
    station_availabilities,
    amr_inventory,
) -> None:
    material = chosen_job.type_
    curr_pos = amr_positions[chosen_amr]
    avail = amr_availabilities[chosen_amr]

    pickup_location = job_pickup_location(chosen_job)
    avail = max(avail + heuristic(curr_pos, pickup_location), float(chosen_job.arrival_time))
    curr_pos = pickup_location
    amr_inventory[chosen_amr][material] = min(amr_inventory[chosen_amr][material] + 1, 3)

    target_station = STATIONS[chosen_job.station]
    avail += heuristic(curr_pos, target_station)
    process_start = max(avail, station_availabilities[chosen_job.station])
    process_end = process_start + chosen_job.duration
    amr_inventory[chosen_amr][material] -= 1
    station_availabilities[chosen_job.station] = process_end

    amr_availabilities[chosen_amr] = process_end
    amr_positions[chosen_amr] = target_station


def action_sequence_from_individual(individual, jobs):
    return operation_sequence_from_individual(individual, jobs)


def finite_log_probs_and_entropy(logits, context: str):
    finite_mask = torch.isfinite(logits)
    if not finite_mask.any():
        raise RuntimeError(f"No finite policy logits in {context}; model parameters may contain NaN.")

    valid_logits = logits[finite_mask]
    valid_log_probs = F.log_softmax(valid_logits, dim=0)
    valid_probs = torch.exp(valid_log_probs)
    entropy = -(valid_probs * valid_log_probs).sum()

    log_probs = torch.full_like(logits, float("-inf"))
    log_probs[finite_mask] = valid_log_probs
    return log_probs, entropy


def evaluate_action_steps(jobs, model, action_seq, init_state=None, include_values: bool = False):
    amr_positions, amr_availabilities, station_availabilities, amr_inventory = initial_operation_state(init_state)
    picked_jobs_set = set()
    completed_jobs_set = set()
    carrier_map = {}
    dock_service_events = empty_dock_service_events()
    step_log_probs = []
    step_entropies = []
    values = []

    model.train()
    device = next(model.parameters()).device

    for chosen_action in action_seq:
        amr_feat, job_feat, job_mask = extract_state(
            jobs,
            picked_jobs_set,
            completed_jobs_set,
            carrier_map,
            amr_positions,
            amr_availabilities,
            amr_inventory,
            station_availabilities,
        )
        amr_feat = amr_feat.to(device)
        job_feat = job_feat.to(device)
        job_mask = job_mask.to(device)
        op_mask = torch.tensor(
            [action_mask(jobs, picked_jobs_set, completed_jobs_set, carrier_map, amr_inventory)],
            dtype=torch.bool,
            device=device,
        )

        if include_values:
            values.append(model.forward_critic(amr_feat, job_feat, job_mask).squeeze())

        logits = model(amr_feat, job_feat, job_mask, op_mask)
        flat_logits = logits.view(-1)
        if not torch.isfinite(flat_logits[chosen_action]):
            raise RuntimeError("Chosen Attention action has a non-finite logit during replay.")
        log_probs, entropy = finite_log_probs_and_entropy(flat_logits, "Attention replay")
        step_log_probs.append(log_probs[chosen_action])
        step_entropies.append(entropy)

        action = decode_action_id(chosen_action, jobs)
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


def evaluate_actions(jobs, model, action_seq, init_state=None):
    step_log_probs, _, values = evaluate_action_steps(
        jobs, model, action_seq, init_state=init_state, include_values=True
    )
    return step_log_probs.sum(), values


def _select_jobs(dispatch_events):
    if dispatch_events:
        return random.choice(dispatch_events)["jobs"]
    return make_jobs()


def assignment_load_stats(individual) -> tuple[int, int, float]:
    counts = [sum(1 for amr in individual.amr_assignment if amr == key) for key in AMR_KEYS]
    return max(counts), min(counts), max(counts) - min(counts)


def load_balance_step_advantages(action_seq, jobs):
    return load_balance_step_advantages_from_actions(action_seq, jobs)


def _load_validation_events(args):
    events = load_training_events(args.validation_inbox, args.validation_inboxes)
    if (args.validation_inbox or args.validation_inboxes) and not events:
        raise ValueError("Validation inbox path(s) were provided, but no validation events were loaded.")
    if events:
        print(f"Loaded {len(events)} validation event(s).")
    return events


def _should_validate(epoch: int, args, validation_events) -> bool:
    return bool(validation_events) and (
        epoch == 1 or epoch == args.epochs or epoch % args.validation_interval == 0
    )


def _save_reinforce_chart(
    chart_path: str,
    title_prefix: str,
    actor_losses,
    sampled_makespans,
    baseline_makespans,
    improvements,
    win_rates,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(actor_losses, color="#e74c3c", linewidth=1.5, label="Actor Loss")
    axes[0, 0].set_title(f"{title_prefix} REINFORCE Actor Loss", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()

    axes[0, 1].plot(sampled_makespans, color="#ff7f0e", linewidth=1.5, label="Sampled Makespan")
    axes[0, 1].plot(baseline_makespans, color="#34495e", linewidth=1.5, label="Dispatch Baseline")
    axes[0, 1].set_title("Average Makespan", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Makespan (s)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()

    axes[1, 0].plot(improvements, color="#2ecc71", linewidth=1.5, label="Baseline - Sampled")
    axes[1, 0].axhline(0.0, color="#7f8c8d", linewidth=1.0)
    axes[1, 0].set_title("Average Improvement", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()

    axes[1, 1].plot(win_rates, color="#3498db", linewidth=1.5, label="Win Rate")
    axes[1, 1].set_title("Win Rate vs Dispatch Baseline", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Win Rate")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_ppo_chart(chart_path, losses_actor, losses_critic, losses_total, makespans, title_prefix):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(losses_actor, color="#e74c3c", linewidth=1.5, label="Actor Loss")
    axes[0, 0].set_title(f"{title_prefix} Actor Loss", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()

    axes[0, 1].plot(losses_critic, color="#2ecc71", linewidth=1.5, label="Critic Loss")
    axes[0, 1].set_title(f"{title_prefix} Critic Loss", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()

    axes[1, 0].plot(losses_total, color="#3498db", linewidth=1.5, label="Total PPO Loss")
    axes[1, 0].set_title("Total PPO Loss", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()

    axes[1, 1].plot(makespans, color="#ff7f0e", linewidth=1.5, label="Avg Makespan (s)")
    axes[1, 1].set_title("Average Batch Makespan Trend", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Makespan (s)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()


def train_reinforce(args):
    validate_sampling_args(args.batch_size, args.samples_per_instance, args.baseline_mode)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dispatch_events = load_training_events(args.inbox, args.inboxes)
    validation_events = _load_validation_events(args)
    attention_model = SchedulerAttention(
        amr_in_dim=8, job_in_dim=16, hidden_dim=128, attention_layers=2
    ).to(device)
    optimizer = optim.Adam(attention_model.parameters(), lr=args.lr)
    load_training_checkpoint(
        attention_model,
        {"optimizer": optimizer},
        args.init_checkpoint,
        device,
    )

    print(
        "Starting Attention REINFORCE training "
        f"with dispatch baseline '{args.baseline_rule}' ({args.baseline_mode})."
    )

    best_makespan = float("inf")
    actor_losses = []
    sampled_makespans = []
    baseline_makespans = []
    improvements = []
    win_rates = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(script_dir, "attention_training_metrics.png")

    def _actor_lr(epoch: int) -> float:
        return cosine_actor_lr(epoch, args.epochs, args.lr, args.lr_min)

    for epoch in range(1, args.epochs + 1):
        current_lr = _actor_lr(epoch)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        trajectories = []
        batch_sampled = []
        batch_baseline = []
        batch_improvement = []
        batch_wins = []
        batch_advantages = []
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
                individual, _, _ = solve_with_attention(jobs, attention_model, deterministic=False)
                group_individuals.append(individual)
                group_action_seqs.append(action_sequence_from_individual(individual, jobs))

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
            batch_advantages, enabled=args.normalize_advantage
        )

        optimizer.zero_grad()
        epoch_actor_loss = 0.0
        epoch_entropy = 0.0

        for (jobs, action_seq, _, load_advantages), advantages in zip(trajectories, normalized_advantages):
            step_log_probs, step_entropies, _ = evaluate_action_steps(jobs, attention_model, action_seq)
            advantage_tensor = torch.tensor(
                advantages,
                dtype=torch.float32,
                device=step_log_probs.device,
            )
            load_advantage_tensor = torch.tensor(
                load_advantages,
                dtype=torch.float32,
                device=step_log_probs.device,
            )
            action_advantage_tensor = advantage_tensor + args.load_balance_coef * load_advantage_tensor
            actor_loss = -(step_log_probs * action_advantage_tensor).sum()
            entropy_bonus = step_entropies.sum()
            loss = (actor_loss - args.entropy_coef * entropy_bonus) / args.batch_size
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite Attention REINFORCE loss before backward.")
            loss.backward()
            epoch_actor_loss += actor_loss.item()
            epoch_entropy += step_entropies.mean().item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(attention_model.parameters(), args.grad_clip)
        optimizer.step()

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
        group_cv_note = (
            f" | Group CV: {sum(batch_group_cvs) / len(batch_group_cvs):.4f}"
            if args.samples_per_instance > 1
            else ""
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] | Sampled: {avg_sampled:.2f} "
            f"| Baseline: {avg_baseline:.2f} | Improvement: {avg_improvement:.2f} "
            f"| Win Rate: {win_rate:.2%} | Actor Loss: {avg_actor_loss:.4f} "
            f"| Entropy: {avg_entropy:.4f} "
            f"| Invalid S/B: {avg_sampled_invalid:.2f}/{avg_baseline_invalid:.2f} "
            f"| Max Load: {avg_max_load:.1f} | Load Gap: {avg_load_gap:.1f}"
            f"{group_cv_note}"
        )

        if _should_validate(epoch, args, validation_events):
            validation = evaluate_validation_events(
                validation_events,
                attention_model,
                solve_with_attention,
                evaluate_makespan,
            )
            validation_score = validation_checkpoint_score(validation, args.validation_invalid_penalty)
            print(
                f"   -> Validation | Samples: {validation['samples']} "
                f"| Makespan: {validation['makespan']:.2f} "
                f"| Invalid Jobs: {validation['invalid_jobs']:.2f} "
                f"| Score: {validation_score:.2f}"
            )
            best_makespan = maybe_save_best_model(
                model=attention_model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=validation_score,
                best_metric=best_makespan,
                metric_label="Val Score",
            )
        elif not validation_events:
            best_makespan = maybe_save_best_model(
                model=attention_model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=avg_sampled,
                best_metric=best_makespan,
                metric_label="Makespan",
            )

        save_training_checkpoint(
            args.latest_checkpoint_path,
            attention_model,
            {"optimizer": optimizer},
            epoch,
            best_makespan,
            args,
        )

        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            _save_reinforce_chart(
                chart_path,
                "Attention",
                actor_losses,
                sampled_makespans,
                baseline_makespans,
                improvements,
                win_rates,
            )
            print(f"   -> Saved updated training chart to {chart_path}")


def train_ppo(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dispatch_events = load_training_events(args.inbox, args.inboxes)
    validation_events = _load_validation_events(args)
    attention_model = SchedulerAttention(
        amr_in_dim=8, job_in_dim=16, hidden_dim=128, attention_layers=2
    ).to(device)
    optimizer = optim.Adam(attention_model.parameters(), lr=args.lr)
    load_training_checkpoint(
        attention_model,
        {"optimizer": optimizer},
        args.init_checkpoint,
        device,
    )

    print("Starting Attention PPO training with critic-based advantage.")

    best_makespan = float("inf")
    losses_actor = []
    losses_critic = []
    losses_total = []
    makespans = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(script_dir, "attention_training_metrics.png")

    def _actor_lr(epoch: int) -> float:
        return cosine_actor_lr(epoch, args.epochs, args.lr, args.lr_min)

    for epoch in range(1, args.epochs + 1):
        current_lr = _actor_lr(epoch)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        batch_makespans = []
        trajectories = []

        for _ in range(args.batch_size):
            jobs = _select_jobs(dispatch_events)
            individual, old_log_prob, solve_dur = solve_with_attention(
                jobs, attention_model, deterministic=False
            )
            action_seq = action_sequence_from_individual(individual, jobs)

            improve_start = time.perf_counter()
            individual = local_improve(individual, jobs, max_iters=routing_iters)
            if collision_routing_iters > 0:
                individual = local_improve(
                    individual,
                    jobs,
                    max_iters=collision_routing_iters,
                    check_collision=True,
                )
            solve_dur += time.perf_counter() - improve_start
            _ = solve_dur

            stochastic_makespan, _ = evaluate_makespan(individual, jobs)
            batch_makespans.append(stochastic_makespan)
            trajectories.append((jobs, action_seq, old_log_prob.detach(), -stochastic_makespan))

        epoch_loss = 0.0
        epoch_critic_loss = 0.0
        epoch_total_loss = 0.0

        for _ in range(args.ppo_epochs):
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_critic_loss = 0.0
            batch_total_loss = 0.0

            for jobs, action_seq, old_log_prob, value_target in trajectories:
                new_log_prob, values = evaluate_actions(jobs, attention_model, action_seq)
                value_target_tensor = torch.tensor(value_target, dtype=torch.float32, device=values.device)
                value_targets = value_target_tensor.expand_as(values)
                advantage = value_target_tensor - values.detach().mean()

                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantage
                actor_loss = -torch.min(surr1, surr2)
                critic_loss = F.mse_loss(values, value_targets)
                total_loss = (actor_loss + args.value_loss_coef * critic_loss) / args.batch_size
                total_loss.backward()

                batch_loss += actor_loss.item()
                batch_critic_loss += critic_loss.item()
                batch_total_loss += (actor_loss + args.value_loss_coef * critic_loss).item()

            optimizer.step()
            epoch_loss += batch_loss / args.batch_size
            epoch_critic_loss += batch_critic_loss / args.batch_size
            epoch_total_loss += batch_total_loss / args.batch_size

        epoch_loss /= args.ppo_epochs
        epoch_critic_loss /= args.ppo_epochs
        epoch_total_loss /= args.ppo_epochs
        avg_batch_makespan = sum(batch_makespans) / args.batch_size

        losses_actor.append(epoch_loss)
        losses_critic.append(epoch_critic_loss)
        losses_total.append(epoch_total_loss)
        makespans.append(avg_batch_makespan)

        print(
            f"Epoch [{epoch}/{args.epochs}] | Avg Makespan: {avg_batch_makespan:.2f} "
            f"| Actor Loss: {epoch_loss:.4f} | Critic Loss: {epoch_critic_loss:.4f} "
            f"| Total Loss: {epoch_total_loss:.4f}"
        )

        if _should_validate(epoch, args, validation_events):
            validation = evaluate_validation_events(
                validation_events,
                attention_model,
                solve_with_attention,
                evaluate_makespan,
            )
            validation_score = validation_checkpoint_score(validation, args.validation_invalid_penalty)
            print(
                f"   -> Validation | Samples: {validation['samples']} "
                f"| Makespan: {validation['makespan']:.2f} "
                f"| Invalid Jobs: {validation['invalid_jobs']:.2f} "
                f"| Score: {validation_score:.2f}"
            )
            best_makespan = maybe_save_best_model(
                model=attention_model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=validation_score,
                best_metric=best_makespan,
                metric_label="Val Score",
            )
        elif not validation_events:
            best_makespan = maybe_save_best_model(
                model=attention_model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=avg_batch_makespan,
                best_metric=best_makespan,
                metric_label="Makespan",
            )

        save_training_checkpoint(
            args.latest_checkpoint_path,
            attention_model,
            {"optimizer": optimizer},
            epoch,
            best_makespan,
            args,
        )

        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            _save_ppo_chart(
                chart_path,
                losses_actor,
                losses_critic,
                losses_total,
                makespans,
                "Attention",
            )
            print(f"   -> Saved updated training chart to {chart_path}")


def train(args):
    if args.lr_min <= 0:
        raise ValueError("--lr_min must be positive")
    if args.train_invalid_penalty < 0:
        raise ValueError("--train_invalid_penalty must be non-negative")
    if args.validation_interval < 1:
        raise ValueError("--validation_interval must be at least 1")
    if args.validation_invalid_penalty < 0:
        raise ValueError("--validation_invalid_penalty must be non-negative")
    if args.rl_method == "ppo":
        train_ppo(args)
    else:
        train_reinforce(args)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file")
    parser.add_argument(
        "--inboxes",
        type=str,
        default="",
        help="Comma-separated dispatch JSONL files used as one combined training pool",
    )
    parser.add_argument("--validation_inbox", type=str, default="", help="Optional fixed validation dispatch JSONL file")
    parser.add_argument(
        "--validation_inboxes",
        type=str,
        default="",
        help="Comma-separated fixed validation dispatch JSONL files",
    )
    parser.add_argument("--validation_interval", type=int, default=50, help="Epoch interval for fixed validation scoring")
    parser.add_argument(
        "--validation_invalid_penalty",
        type=float,
        default=1000.0,
        help="Penalty added per average invalid validation job when selecting the best checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Schedules sampled per update")
    parser.add_argument("--lr", "--lr_actor", dest="lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lr_min", type=float, default=3e-5,
                        help="Floor of the cosine actor LR decay; set equal to the "
                             "actor LR to disable the schedule (constant rate)")
    parser.add_argument("--train_invalid_penalty", type=float, default=0.0,
                        help="Seconds charged per unroutable parcel INSIDE the training "
                             "advantage (validation uses --validation_invalid_penalty)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_checkpoint", type=str, default="", help="Optional checkpoint or legacy weights to initialize from")
    parser.add_argument("--latest_checkpoint_path", type=str, default="", help="Optional full training checkpoint path updated every epoch")
    parser.add_argument("--best_model_path", type=str, default="", help="Optional best model weights path")
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
    parser.add_argument(
        "--load_balance_coef",
        type=float,
        default=0.1,
        help="Joint action penalty for assigning jobs to already overloaded AMRs during REINFORCE replay",
    )
    parser.add_argument("--normalize_advantage", dest="normalize_advantage", action="store_true", default=True)
    parser.add_argument("--no_normalize_advantage", dest="normalize_advantage", action="store_false")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
