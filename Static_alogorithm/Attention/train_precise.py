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
    SUPPLY_LOCATIONS,
    TYPE_DURATION,
    SchedulerAttention,
    extract_state,
    solve_with_attention,
)
from GA.GA import MAX_DEPTH, find_dynamic_path, make_jobs
from reinforce_baseline import (
    DEFAULT_BASELINE_RULE,
    compute_dispatch_baseline_comparison,
    evaluate_makespan,
    load_training_events,
    normalize_advantage_batches,
)


def _initial_precise_state(init_state=None):
    if init_state:
        amr_positions = {amr: init_state["positions"].get(amr, AMR_STARTS[amr]) for amr in AMR_KEYS}
        amr_availabilities = {amr: float(init_state["availability"].get(amr, 0.0)) for amr in AMR_KEYS}
        station_availabilities = {s: float(init_state["time"]) for s in STATIONS.keys()}
        amr_inventory = {
            amr: init_state["inventory"].get(amr, {mat: 0 for mat in TYPE_DURATION.keys()}).copy()
            for amr in AMR_KEYS
        }
        amr_states = {amr: (amr_positions[amr], amr_availabilities[amr]) for amr in AMR_KEYS}
    else:
        amr_positions = {amr: AMR_STARTS[amr] for amr in AMR_KEYS}
        amr_availabilities = {amr: 0.0 for amr in AMR_KEYS}
        station_availabilities = {s: 0.0 for s in STATIONS.keys()}
        amr_inventory = {amr: {mat: 0 for mat in TYPE_DURATION.keys()} for amr in AMR_KEYS}
        amr_inventory["AMR1"]["A"] = 3
        amr_inventory["AMR2"]["B"] = 3
        amr_inventory["AMR3"]["C"] = 3
        amr_states = {amr: (AMR_STARTS[amr], 0.0) for amr in AMR_KEYS}
    return amr_positions, amr_availabilities, station_availabilities, amr_inventory, amr_states, {}


def _apply_precise_action(
    chosen_job,
    chosen_amr: str,
    amr_positions,
    amr_availabilities,
    station_availabilities,
    amr_inventory,
    amr_states,
    reservations,
) -> None:
    material = chosen_job.type_
    start_time = amr_availabilities[chosen_amr]

    if amr_inventory[chosen_amr][material] == 0:
        supply_location = SUPPLY_LOCATIONS[material]
        supply_path = find_dynamic_path(
            amr_positions[chosen_amr],
            supply_location,
            start_time,
            reservations,
            amr_states,
            chosen_amr,
        )
        supply_time = int(len(supply_path) - 1)
        supply_end = start_time + supply_time + TYPE_DURATION[material]
        if supply_time == 0 and amr_positions[chosen_amr] != supply_location:
            supply_time = MAX_DEPTH
            supply_end = start_time + supply_time

        for t_offset, point in enumerate(supply_path):
            reservations[(point, int(start_time) + t_offset)] = chosen_amr
        for t_refill in range(int(start_time) + supply_time, int(supply_end) + 1):
            reservations[(supply_location, t_refill)] = chosen_amr

        amr_states[chosen_amr] = (supply_location, supply_end)
        amr_availabilities[chosen_amr] = supply_end
        amr_positions[chosen_amr] = supply_location
        amr_inventory[chosen_amr][material] = 3

    travel_start = amr_availabilities[chosen_amr]
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

    return_start = process_end
    home_position = AMR_STARTS[chosen_amr]
    return_path = find_dynamic_path(
        target_station,
        home_position,
        return_start,
        reservations,
        amr_states,
        chosen_amr,
    )
    return_time = int(len(return_path) - 1)
    if return_time == 0 and target_station != home_position:
        return_time = MAX_DEPTH
    return_end = return_start + return_time

    for t_offset, point in enumerate(return_path):
        reservations[(point, int(return_start) + t_offset)] = chosen_amr

    amr_states[chosen_amr] = (home_position, return_end)
    amr_availabilities[chosen_amr] = return_end
    amr_positions[chosen_amr] = home_position


def action_sequence_from_individual(individual, jobs):
    job_id_to_list_idx = {job.idx: idx for idx, job in enumerate(jobs)}
    action_seq = []
    for job_id in individual.order:
        job_idx = job_id_to_list_idx[job_id]
        amr = individual.amr_assignment[job_id]
        amr_idx = AMR_KEYS.index(amr)
        action_seq.append(amr_idx * len(jobs) + job_idx)
    return action_seq


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
    assigned_jobs_set = set()
    amr_assignment_map = {}
    step_log_probs = []
    step_entropies = []
    values = []

    model.train()
    device = next(model.parameters()).device

    for chosen_action in action_seq:
        amr_feat, job_feat, job_mask = extract_state(
            jobs,
            assigned_jobs_set,
            amr_positions,
            amr_availabilities,
            amr_inventory,
            amr_assignment_map,
        )
        amr_feat = amr_feat.to(device)
        job_feat = job_feat.to(device)
        job_mask = job_mask.to(device)

        if include_values:
            values.append(model.forward_critic(amr_feat, job_feat, job_mask).squeeze())

        logits = model(amr_feat, job_feat, job_mask)
        flat_logits = logits.view(-1)
        if not torch.isfinite(flat_logits[chosen_action]):
            raise RuntimeError("Chosen Attention precise action has a non-finite logit during replay.")
        log_probs, entropy = finite_log_probs_and_entropy(flat_logits, "Attention precise replay")
        step_log_probs.append(log_probs[chosen_action])
        step_entropies.append(entropy)

        num_jobs = len(jobs)
        amr_idx = chosen_action // num_jobs
        job_list_idx = chosen_action % num_jobs
        chosen_amr = AMR_KEYS[amr_idx]
        chosen_job = jobs[job_list_idx]

        amr_assignment_map[chosen_job.idx] = chosen_amr
        assigned_jobs_set.add(chosen_job.idx)
        _apply_precise_action(
            chosen_job,
            chosen_amr,
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
    attention_model = SchedulerAttention(
        amr_in_dim=8, job_in_dim=11, hidden_dim=128, attention_layers=2
    ).to(device)
    optimizer = optim.Adam(attention_model.parameters(), lr=args.lr)

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

            trajectories.append((jobs, action_seq, comparison.step_advantages))
            batch_advantages.append(comparison.step_advantages)
            batch_sampled.append(comparison.sampled_makespan)
            batch_baseline.append(comparison.baseline_makespan)
            batch_improvement.append(comparison.improvement)
            batch_wins.append(1.0 if comparison.win else 0.0)

        normalized_advantages = normalize_advantage_batches(
            batch_advantages, enabled=args.normalize_advantage
        )

        optimizer.zero_grad()
        epoch_actor_loss = 0.0
        epoch_entropy = 0.0

        for (jobs, action_seq, _), advantages in zip(trajectories, normalized_advantages):
            step_log_probs, step_entropies, _ = evaluate_action_steps(jobs, attention_model, action_seq)
            advantage_tensor = torch.tensor(
                advantages,
                dtype=torch.float32,
                device=step_log_probs.device,
            )
            actor_loss = -(step_log_probs * advantage_tensor).sum()
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

        actor_losses.append(avg_actor_loss)
        sampled_makespans.append(avg_sampled)
        baseline_makespans.append(avg_baseline)
        improvements.append(avg_improvement)
        win_rates.append(win_rate)

        print(
            f"Epoch [{epoch}/{args.epochs}] | Sampled: {avg_sampled:.2f} "
            f"| Baseline: {avg_baseline:.2f} | Improvement: {avg_improvement:.2f} "
            f"| Win Rate: {win_rate:.2%} | Actor Loss: {avg_actor_loss:.4f} "
            f"| Entropy: {avg_entropy:.4f}"
        )

        if avg_sampled < best_makespan:
            best_makespan = avg_sampled
            torch.save(attention_model.state_dict(), "attention_precise_scheduler_best.pth")
            print(f"   -> Saved new best model (Makespan: {best_makespan:.2f})")

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
    attention_model = SchedulerAttention(
        amr_in_dim=8, job_in_dim=11, hidden_dim=128, attention_layers=2
    ).to(device)
    optimizer = optim.Adam(attention_model.parameters(), lr=args.lr)

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

        if avg_batch_makespan < best_makespan:
            best_makespan = avg_batch_makespan
            torch.save(attention_model.state_dict(), "attention_precise_scheduler_best.pth")
            print(f"   -> Saved new best model (Makespan: {best_makespan:.2f})")

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
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Schedules sampled per update")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rl_method", choices=["reinforce", "ppo"], default="reinforce")
    parser.add_argument("--baseline_rule", type=str, default=DEFAULT_BASELINE_RULE)
    parser.add_argument("--baseline_mode", choices=["stepwise", "episode"], default="stepwise")
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--normalize_advantage", dest="normalize_advantage", action="store_true", default=True)
    parser.add_argument("--no_normalize_advantage", dest="normalize_advantage", action="store_false")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
