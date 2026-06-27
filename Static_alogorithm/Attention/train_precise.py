from __future__ import annotations

import argparse
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from Attention_precise import (
    AMR_KEYS,
    AMR_STARTS,
    STATIONS,
    TYPE_DURATION,
    SchedulerAttention,
    extract_state,
    solve_with_attention,
)
from GA.GA import (
    PICKUP,
    MAX_DEPTH,
    dock_key_from_value,
    empty_count_inventory,
    find_dynamic_path,
    job_pickup_location,
    make_jobs,
    normalize_count_inventory,
    repair_operation_order,
)
from reinforce_baseline import (
    DEFAULT_BASELINE_RULE,
    compute_dispatch_baseline_comparison,
    evaluate_makespan,
    load_training_events,
    normalize_advantage_batches,
)
from operation_policy import (
    action_mask,
    decode_action_id,
    initial_operation_state,
    load_balance_step_advantages_from_actions,
    operation_sequence_from_individual,
)
from training_checkpoints import (
    evaluate_validation_events,
    load_training_checkpoint,
    maybe_save_best_model,
    save_training_checkpoint,
)


LEGACY_BEST_MODEL_PATH = "attention_precise_scheduler_best.pth"


def _initial_precise_state(init_state=None):
    return initial_operation_state(init_state, precise=True)


def _apply_precise_action(
    action,
    jobs,
    picked_jobs_set,
    completed_jobs_set,
    carrier_map,
    amr_positions,
    amr_availabilities,
    station_availabilities,
    amr_inventory,
    amr_states,
    reservations,
) -> None:
    chosen_job = jobs[action.job_list_idx]
    chosen_amr = action.amr
    material = chosen_job.type_
    start_time = amr_availabilities[chosen_amr]

    if action.kind == PICKUP:
        pickup_location = job_pickup_location(chosen_job)
        pickup_path = find_dynamic_path(
            amr_positions[chosen_amr],
            pickup_location,
            start_time,
            reservations,
            amr_states,
            chosen_amr,
        )
        pickup_time = int(len(pickup_path) - 1)
        if pickup_time == 0 and amr_positions[chosen_amr] != pickup_location:
            pickup_time = MAX_DEPTH
        inbound_dock = dock_key_from_value(chosen_job.inbound_dock)
        pickup_start = max(
            start_time + pickup_time,
            float(chosen_job.arrival_time),
            station_availabilities.get(inbound_dock, 0.0),
        )
        pickup_end = pickup_start + chosen_job.duration

        for t_offset, point in enumerate(pickup_path):
            reservations[(point, int(start_time) + t_offset)] = chosen_amr
        for t_wait in range(int(start_time) + pickup_time, int(pickup_end) + 1):
            reservations[(pickup_location, t_wait)] = chosen_amr

        amr_states[chosen_amr] = (pickup_location, pickup_end)
        amr_availabilities[chosen_amr] = pickup_end
        amr_positions[chosen_amr] = pickup_location
        station_availabilities[inbound_dock] = pickup_end
        amr_inventory[chosen_amr][material] += 1
        picked_jobs_set.add(chosen_job.idx)
        carrier_map[chosen_job.idx] = chosen_amr
        return

    travel_start = start_time
    target_station = STATIONS[chosen_job.station]
    travel_path = find_dynamic_path(
        amr_positions[chosen_amr],
        target_station,
        travel_start,
        reservations,
        amr_states,
        chosen_amr,
    )
    travel_time = int(len(travel_path) - 1)
    if travel_time == 0 and amr_positions[chosen_amr] != target_station:
        travel_time = MAX_DEPTH
    travel_end = travel_start + travel_time

    for t_offset, point in enumerate(travel_path):
        reservations[(point, int(travel_start) + t_offset)] = chosen_amr

    amr_availabilities[chosen_amr] = travel_end
    amr_positions[chosen_amr] = target_station

    process_start = max(travel_end, station_availabilities[chosen_job.station])
    process_end = process_start + chosen_job.duration

    for t_process in range(int(travel_end), int(process_end) + 1):
        reservations[(target_station, t_process)] = chosen_amr

    amr_states[chosen_amr] = (target_station, process_end)
    amr_inventory[chosen_amr][material] -= 1
    station_availabilities[chosen_job.station] = process_end
    completed_jobs_set.add(chosen_job.idx)

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
    (
        amr_positions,
        amr_availabilities,
        station_availabilities,
        amr_inventory,
        amr_states,
        reservations,
    ) = _initial_precise_state(init_state)
    picked_jobs_set = set()
    completed_jobs_set = set()
    carrier_map = {}
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
            raise RuntimeError("Chosen Attention precise action has a non-finite logit during replay.")
        log_probs, entropy = finite_log_probs_and_entropy(flat_logits, "Attention precise replay")
        step_log_probs.append(log_probs[chosen_action])
        step_entropies.append(entropy)

        action = decode_action_id(chosen_action, jobs)
        _apply_precise_action(
            action,
            jobs,
            picked_jobs_set,
            completed_jobs_set,
            carrier_map,
            amr_positions,
            amr_availabilities,
            station_availabilities,
            amr_inventory,
            amr_states,
            reservations,
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
        "Starting Attention Precise REINFORCE training "
        f"with dispatch baseline '{args.baseline_rule}' ({args.baseline_mode})."
    )

    best_makespan = float("inf")
    actor_losses = []
    sampled_makespans = []
    baseline_makespans = []
    improvements = []
    win_rates = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(script_dir, "attention_precise_training_metrics.png")

    for epoch in range(1, args.epochs + 1):
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

        for batch_idx in range(args.batch_size):
            jobs = _select_jobs(dispatch_events)
            individual, _, _ = solve_with_attention(jobs, attention_model, deterministic=False)
            action_seq = action_sequence_from_individual(individual, jobs)

            comparison = compute_dispatch_baseline_comparison(
                jobs,
                individual,
                baseline_rule=args.baseline_rule,
                baseline_mode=args.baseline_mode,
                seed=args.seed + batch_idx,
            )

            trajectories.append(
                (
                    jobs,
                    action_seq,
                    comparison.step_advantages,
                    load_balance_step_advantages(action_seq, jobs),
                )
            )
            batch_advantages.append(comparison.step_advantages)
            batch_sampled.append(comparison.sampled_makespan)
            batch_baseline.append(comparison.baseline_makespan)
            batch_improvement.append(comparison.improvement)
            batch_wins.append(1.0 if comparison.win else 0.0)
            batch_sampled_invalid.append(comparison.sampled_invalid_jobs)
            batch_baseline_invalid.append(comparison.baseline_invalid_jobs)
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
                raise RuntimeError("Non-finite Attention precise REINFORCE loss before backward.")
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

        print(
            f"Epoch [{epoch}/{args.epochs}] | Sampled: {avg_sampled:.2f} "
            f"| Baseline: {avg_baseline:.2f} | Improvement: {avg_improvement:.2f} "
            f"| Win Rate: {win_rate:.2%} | Actor Loss: {avg_actor_loss:.4f} "
            f"| Entropy: {avg_entropy:.4f} "
            f"| Invalid S/B: {avg_sampled_invalid:.2f}/{avg_baseline_invalid:.2f} "
            f"| Max Load: {avg_max_load:.1f} | Load Gap: {avg_load_gap:.1f}"
        )

        if _should_validate(epoch, args, validation_events):
            validation = evaluate_validation_events(
                validation_events,
                attention_model,
                solve_with_attention,
                evaluate_makespan,
            )
            print(
                f"   -> Validation | Samples: {validation['samples']} "
                f"| Makespan: {validation['makespan']:.2f} "
                f"| Invalid Jobs: {validation['invalid_jobs']:.2f}"
            )
            best_makespan = maybe_save_best_model(
                model=attention_model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=validation["makespan"],
                best_metric=best_makespan,
                metric_label="Val Makespan",
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
                "Attention Precise",
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

    print("Starting Attention Precise PPO training with critic-based advantage.")

    best_makespan = float("inf")
    losses_actor = []
    losses_critic = []
    losses_total = []
    makespans = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(script_dir, "attention_precise_training_metrics.png")

    for epoch in range(1, args.epochs + 1):
        batch_makespans = []
        trajectories = []

        for _ in range(args.batch_size):
            jobs = _select_jobs(dispatch_events)
            individual, old_log_prob, _ = solve_with_attention(
                jobs, attention_model, deterministic=False
            )
            stochastic_makespan, _ = evaluate_makespan(individual, jobs)
            batch_makespans.append(stochastic_makespan)
            action_seq = action_sequence_from_individual(individual, jobs)
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
            print(
                f"   -> Validation | Samples: {validation['samples']} "
                f"| Makespan: {validation['makespan']:.2f} "
                f"| Invalid Jobs: {validation['invalid_jobs']:.2f}"
            )
            best_makespan = maybe_save_best_model(
                model=attention_model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=validation["makespan"],
                best_metric=best_makespan,
                metric_label="Val Makespan",
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
                "Attention Precise",
            )
            print(f"   -> Saved updated training chart to {chart_path}")


def train(args):
    if args.validation_interval < 1:
        raise ValueError("--validation_interval must be at least 1")
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
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Schedules sampled per update")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_checkpoint", type=str, default="", help="Optional checkpoint or legacy weights to initialize from")
    parser.add_argument("--latest_checkpoint_path", type=str, default="", help="Optional full training checkpoint path updated every epoch")
    parser.add_argument("--best_model_path", type=str, default="", help="Optional best model weights path")
    parser.add_argument("--rl_method", choices=["reinforce", "ppo"], default="reinforce")
    parser.add_argument("--baseline_rule", type=str, default=DEFAULT_BASELINE_RULE)
    parser.add_argument("--baseline_mode", choices=["stepwise", "episode"], default="stepwise")
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
