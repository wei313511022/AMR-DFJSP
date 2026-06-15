from __future__ import annotations

import argparse
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from GNN_precise import (
    AMR_KEYS,
    AMR_STARTS,
    STATIONS,
    TYPE_DURATION,
    SchedulerGNN,
    extract_state_gnn,
    solve_with_gnn,
)
from GA.GA import (
    PICKUP,
    MAX_DEPTH,
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
from training_checkpoints import (
    evaluate_validation_events,
    load_training_checkpoint,
    maybe_save_best_model,
    save_training_checkpoint,
)


LEGACY_BEST_MODEL_PATH = "gnn_precise_mpn_scheduler_best.pth"


def _initial_precise_state(init_state=None):
    if init_state:
        amr_positions = {amr: init_state["positions"].get(amr, AMR_STARTS[amr]) for amr in AMR_KEYS}
        amr_availabilities = {amr: float(init_state["availability"].get(amr, 0.0)) for amr in AMR_KEYS}
        station_availabilities = {s: float(init_state["time"]) for s in STATIONS.keys()}
        amr_inventory = normalize_count_inventory(init_state.get("inventory", {}))
        amr_states = {amr: (amr_positions[amr], amr_availabilities[amr]) for amr in AMR_KEYS}
    else:
        amr_positions = {amr: AMR_STARTS[amr] for amr in AMR_KEYS}
        amr_availabilities = {amr: 0.0 for amr in AMR_KEYS}
        station_availabilities = {s: 0.0 for s in STATIONS.keys()}
        amr_inventory = empty_count_inventory()
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
    pickup_end = max(start_time + pickup_time, float(chosen_job.arrival_time))

    for t_offset, point in enumerate(pickup_path):
        reservations[(point, int(start_time) + t_offset)] = chosen_amr
    for t_wait in range(int(start_time) + pickup_time, int(pickup_end) + 1):
        reservations[(pickup_location, t_wait)] = chosen_amr

    amr_states[chosen_amr] = (pickup_location, pickup_end)
    amr_availabilities[chosen_amr] = pickup_end
    amr_positions[chosen_amr] = pickup_location
    amr_inventory[chosen_amr][material] = min(amr_inventory[chosen_amr][material] + 1, 3)

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
    base_pos = AMR_STARTS[chosen_amr]
    return_path = find_dynamic_path(
        target_station,
        base_pos,
        return_start,
        reservations,
        amr_states,
        chosen_amr,
    )
    return_time = int(len(return_path) - 1)
    if return_time == 0 and target_station != base_pos:
        return_time = MAX_DEPTH
    return_end = return_start + return_time

    for t_offset, point in enumerate(return_path):
        reservations[(point, int(return_start) + t_offset)] = chosen_amr

    amr_states[chosen_amr] = (base_pos, return_end)
    amr_availabilities[chosen_amr] = return_end
    amr_positions[chosen_amr] = base_pos


def action_sequences_from_individual(individual, jobs):
    job_id_to_list_idx = {job.idx: idx for idx, job in enumerate(jobs)}
    job_action_seq = []
    machine_action_seq = []
    for op in repair_operation_order(list(individual.order), list(jobs)):
        if op.kind != PICKUP:
            continue
        job_id = op.job_idx
        job_action_seq.append(job_id_to_list_idx[job_id])
        amr = individual.amr_assignment[job_id]
        machine_action_seq.append(AMR_KEYS.index(amr))
    return job_action_seq, machine_action_seq


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


def evaluate_action_steps_multi(
    jobs,
    model,
    job_action_seq,
    machine_action_seq,
    init_state=None,
    include_values: bool = False,
):
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
    order_seq = []

    job_log_probs = []
    machine_log_probs = []
    job_entropies = []
    machine_entropies = []
    values = []

    model.train()
    device = next(model.parameters()).device

    for chosen_job_list_idx, chosen_amr_idx in zip(job_action_seq, machine_action_seq):
        amr_feat, job_feat, job_mask, adj = extract_state_gnn(
            jobs,
            assigned_jobs_set,
            amr_positions,
            amr_availabilities,
            amr_inventory,
            amr_assignment_map,
            station_availabilities,
            order_seq,
        )
        amr_feat = amr_feat.to(device)
        job_feat = job_feat.to(device)
        job_mask = job_mask.to(device)
        adj = adj.to(device)

        job_embeddings = model.encode_jobs(job_feat, adj)
        amr_embeddings = model.encode_amrs(amr_feat)

        if include_values:
            values.append(model.forward_critic(job_embeddings, amr_embeddings, job_mask).squeeze())

        job_logits = model.forward_job_actor(job_embeddings, job_mask).view(-1)
        if not torch.isfinite(job_logits[chosen_job_list_idx]):
            raise RuntimeError("Chosen GNN precise job action has a non-finite logit during replay.")
        job_step_log_probs, job_entropy = finite_log_probs_and_entropy(job_logits, "GNN precise job replay")
        job_log_probs.append(job_step_log_probs[chosen_job_list_idx])
        job_entropies.append(job_entropy)

        chosen_job = jobs[chosen_job_list_idx]
        selected_job_emb = job_embeddings[:, chosen_job_list_idx, :]

        machine_logits = model.forward_machine_actor(selected_job_emb, amr_embeddings).view(-1)
        if not torch.isfinite(machine_logits[chosen_amr_idx]):
            raise RuntimeError("Chosen GNN precise AMR action has a non-finite logit during replay.")
        machine_step_log_probs, machine_entropy = finite_log_probs_and_entropy(machine_logits, "GNN precise AMR replay")
        machine_log_probs.append(machine_step_log_probs[chosen_amr_idx])
        machine_entropies.append(machine_entropy)

        chosen_amr = AMR_KEYS[chosen_amr_idx]
        order_seq.append(chosen_job.idx)
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
    return (
        torch.stack(job_log_probs),
        torch.stack(machine_log_probs),
        torch.stack(job_entropies),
        torch.stack(machine_entropies),
        value_tensor,
    )


def evaluate_actions_multi_ppo(jobs, model, job_action_seq, machine_action_seq, init_state=None):
    job_log_probs, machine_log_probs, _, _, values = evaluate_action_steps_multi(
        jobs,
        model,
        job_action_seq,
        machine_action_seq,
        init_state=init_state,
        include_values=True,
    )
    return job_log_probs.sum(), machine_log_probs.sum(), values


def _actor_params(model):
    return (
        list(model.job_emb.parameters())
        + list(model.gin_layers.parameters())
        + list(model.amr_emb.parameters())
        + list(model.job_actor.parameters())
        + list(model.machine_actor.parameters())
    )


def _select_jobs(dispatch_events):
    if dispatch_events:
        return random.choice(dispatch_events)["jobs"]
    return make_jobs()


def assignment_load_stats(individual) -> tuple[int, int, float]:
    counts = [sum(1 for amr in individual.amr_assignment if amr == key) for key in AMR_KEYS]
    return max(counts), min(counts), max(counts) - min(counts)


def load_balance_step_advantages(machine_action_seq):
    counts = [0 for _ in AMR_KEYS]
    advantages = []
    for chosen_amr_idx in machine_action_seq:
        advantages.append(float(min(counts) - counts[chosen_amr_idx]))
        counts[chosen_amr_idx] += 1
    return advantages


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
    chart_path,
    title_prefix,
    job_losses,
    machine_losses,
    sampled_makespans,
    baseline_makespans,
    improvements,
    win_rates,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(job_losses, color="#e74c3c", linewidth=1.5, label="Job Actor Loss")
    axes[0, 0].plot(machine_losses, color="#3498db", linewidth=1.5, label="AMR Actor Loss")
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

    axes[1, 1].plot(win_rates, color="#8e44ad", linewidth=1.5, label="Win Rate")
    axes[1, 1].set_title("Win Rate vs Dispatch Baseline", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Win Rate")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_ppo_chart(chart_path, losses_job, losses_machine, losses_critic, makespans):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(losses_job, color="#e74c3c", linewidth=1.5, label="Job Actor Loss")
    axes[0, 0].set_title("Job Actor Loss", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()

    axes[0, 1].plot(losses_machine, color="#3498db", linewidth=1.5, label="Machine Actor Loss")
    axes[0, 1].set_title("Machine Actor Loss", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend()

    axes[1, 0].plot(losses_critic, color="#2ecc71", linewidth=1.5, label="Critic Loss")
    axes[1, 0].set_title("Joint Critic Loss", fontsize=12, fontweight="bold")
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
    gnn_model = SchedulerGNN(job_in_dim=12, amr_in_dim=8, hidden_dim=128, gin_layers=3).to(device)
    optimizer_actor = optim.Adam(_actor_params(gnn_model), lr=args.lr_actor)
    load_training_checkpoint(
        gnn_model,
        {"optimizer_actor": optimizer_actor},
        args.init_checkpoint,
        device,
    )

    print(
        "Starting GNN Precise REINFORCE training "
        f"with dispatch baseline '{args.baseline_rule}' ({args.baseline_mode})."
    )

    best_makespan = float("inf")
    losses_job = []
    losses_machine = []
    sampled_makespans = []
    baseline_makespans = []
    improvements = []
    win_rates = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(script_dir, "gnn_precise_training_metrics.png")

    for epoch in range(1, args.epochs + 1):
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

        for batch_idx in range(args.batch_size):
            jobs = _select_jobs(dispatch_events)
            individual, _, _ = solve_with_gnn(jobs, gnn_model, deterministic=False)
            job_action_seq, machine_action_seq = action_sequences_from_individual(individual, jobs)
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
                    job_action_seq,
                    machine_action_seq,
                    comparison.step_advantages,
                    load_balance_step_advantages(machine_action_seq),
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

        optimizer_actor.zero_grad()
        epoch_job_loss = 0.0
        epoch_machine_loss = 0.0
        epoch_entropy = 0.0

        for (jobs, job_action_seq, machine_action_seq, _, load_advantages), advantages in zip(
            trajectories, normalized_advantages
        ):
            job_lp, machine_lp, job_entropy, machine_entropy, _ = evaluate_action_steps_multi(
                jobs, gnn_model, job_action_seq, machine_action_seq
            )
            advantage_tensor = torch.tensor(advantages, dtype=torch.float32, device=job_lp.device)
            load_advantage_tensor = torch.tensor(load_advantages, dtype=torch.float32, device=machine_lp.device)
            machine_advantage_tensor = advantage_tensor + args.load_balance_coef * load_advantage_tensor
            job_loss = -(job_lp * advantage_tensor).sum()
            machine_loss = -(machine_lp * machine_advantage_tensor).sum()
            entropy_bonus = job_entropy.sum() + machine_entropy.sum()
            total_loss = (job_loss + machine_loss - args.entropy_coef * entropy_bonus) / args.batch_size
            if not torch.isfinite(total_loss).item():
                raise RuntimeError("Non-finite GNN precise REINFORCE loss before backward.")
            total_loss.backward()

            epoch_job_loss += job_loss.item()
            epoch_machine_loss += machine_loss.item()
            epoch_entropy += (job_entropy.mean() + machine_entropy.mean()).item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(_actor_params(gnn_model), args.grad_clip)
        optimizer_actor.step()

        avg_job_loss = epoch_job_loss / args.batch_size
        avg_machine_loss = epoch_machine_loss / args.batch_size
        avg_sampled = sum(batch_sampled) / args.batch_size
        avg_baseline = sum(batch_baseline) / args.batch_size
        avg_improvement = sum(batch_improvement) / args.batch_size
        win_rate = sum(batch_wins) / args.batch_size
        avg_entropy = epoch_entropy / args.batch_size
        avg_sampled_invalid = sum(batch_sampled_invalid) / args.batch_size
        avg_baseline_invalid = sum(batch_baseline_invalid) / args.batch_size
        avg_max_load = sum(batch_max_load) / args.batch_size
        avg_load_gap = sum(batch_load_gap) / args.batch_size

        losses_job.append(avg_job_loss)
        losses_machine.append(avg_machine_loss)
        sampled_makespans.append(avg_sampled)
        baseline_makespans.append(avg_baseline)
        improvements.append(avg_improvement)
        win_rates.append(win_rate)

        print(
            f"Epoch [{epoch}/{args.epochs}] | Sampled: {avg_sampled:.2f} "
            f"| Baseline: {avg_baseline:.2f} | Improvement: {avg_improvement:.2f} "
            f"| Win Rate: {win_rate:.2%} | Job Loss: {avg_job_loss:.4f} "
            f"| AMR Loss: {avg_machine_loss:.4f} | Entropy: {avg_entropy:.4f} "
            f"| Invalid S/B: {avg_sampled_invalid:.2f}/{avg_baseline_invalid:.2f} "
            f"| Max Load: {avg_max_load:.1f} | Load Gap: {avg_load_gap:.1f}"
        )

        if _should_validate(epoch, args, validation_events):
            validation = evaluate_validation_events(
                validation_events,
                gnn_model,
                solve_with_gnn,
                evaluate_makespan,
            )
            print(
                f"   -> Validation | Samples: {validation['samples']} "
                f"| Makespan: {validation['makespan']:.2f} "
                f"| Invalid Jobs: {validation['invalid_jobs']:.2f}"
            )
            best_makespan = maybe_save_best_model(
                model=gnn_model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=validation["makespan"],
                best_metric=best_makespan,
                metric_label="Val Makespan",
            )
        elif not validation_events:
            best_makespan = maybe_save_best_model(
                model=gnn_model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=avg_sampled,
                best_metric=best_makespan,
                metric_label="Makespan",
            )

        save_training_checkpoint(
            args.latest_checkpoint_path,
            gnn_model,
            {"optimizer_actor": optimizer_actor},
            epoch,
            best_makespan,
            args,
        )

        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            _save_reinforce_chart(
                chart_path,
                "GNN Precise",
                losses_job,
                losses_machine,
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
    gnn_model = SchedulerGNN(job_in_dim=12, amr_in_dim=8, hidden_dim=128, gin_layers=3).to(device)
    optimizer_actor = optim.Adam(_actor_params(gnn_model), lr=args.lr_actor)
    optimizer_critic = optim.Adam(gnn_model.critic.parameters(), lr=args.lr_critic)
    load_training_checkpoint(
        gnn_model,
        {"optimizer_actor": optimizer_actor, "optimizer_critic": optimizer_critic},
        args.init_checkpoint,
        device,
    )

    print("Starting GNN Precise Multi-PPO training with critic-based advantage.")

    best_makespan = float("inf")
    losses_job = []
    losses_machine = []
    losses_critic = []
    makespans = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chart_path = os.path.join(script_dir, "gnn_precise_training_metrics.png")

    for epoch in range(1, args.epochs + 1):
        batch_makespans = []
        trajectories = []

        for _ in range(args.batch_size):
            jobs = _select_jobs(dispatch_events)
            individual, (old_job_lp, old_machine_lp), _ = solve_with_gnn(
                jobs, gnn_model, deterministic=False
            )
            stochastic_makespan, _ = evaluate_makespan(individual, jobs)
            batch_makespans.append(stochastic_makespan)
            job_action_seq, machine_action_seq = action_sequences_from_individual(individual, jobs)
            trajectories.append(
                (
                    jobs,
                    job_action_seq,
                    machine_action_seq,
                    old_job_lp.detach(),
                    old_machine_lp.detach(),
                    -stochastic_makespan,
                )
            )

        epoch_job_loss = 0.0
        epoch_machine_loss = 0.0
        epoch_critic_loss = 0.0

        for _ in range(args.ppo_epochs):
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            batch_job_loss = 0.0
            batch_machine_loss = 0.0
            batch_critic_loss = 0.0

            for (
                jobs,
                job_action_seq,
                machine_action_seq,
                old_job_lp,
                old_machine_lp,
                value_target,
            ) in trajectories:
                new_job_lp, new_machine_lp, values = evaluate_actions_multi_ppo(
                    jobs, gnn_model, job_action_seq, machine_action_seq
                )
                value_target_tensor = torch.tensor(value_target, dtype=torch.float32, device=values.device)
                value_targets = value_target_tensor.expand_as(values)
                advantage = value_target_tensor - values.detach().mean()

                job_ratio = torch.exp(new_job_lp - old_job_lp)
                job_surr1 = job_ratio * advantage
                job_surr2 = torch.clamp(job_ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantage
                job_loss = -torch.min(job_surr1, job_surr2)

                machine_ratio = torch.exp(new_machine_lp - old_machine_lp)
                machine_surr1 = machine_ratio * advantage
                machine_surr2 = torch.clamp(machine_ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantage
                machine_loss = -torch.min(machine_surr1, machine_surr2)

                critic_loss = F.mse_loss(values, value_targets)
                total_loss = (job_loss + machine_loss + args.value_loss_coef * critic_loss) / args.batch_size
                total_loss.backward()

                batch_job_loss += job_loss.item()
                batch_machine_loss += machine_loss.item()
                batch_critic_loss += critic_loss.item()

            optimizer_actor.step()
            optimizer_critic.step()

            epoch_job_loss += batch_job_loss / args.batch_size
            epoch_machine_loss += batch_machine_loss / args.batch_size
            epoch_critic_loss += batch_critic_loss / args.batch_size

        epoch_job_loss /= args.ppo_epochs
        epoch_machine_loss /= args.ppo_epochs
        epoch_critic_loss /= args.ppo_epochs
        avg_batch_makespan = sum(batch_makespans) / args.batch_size

        losses_job.append(epoch_job_loss)
        losses_machine.append(epoch_machine_loss)
        losses_critic.append(epoch_critic_loss)
        makespans.append(avg_batch_makespan)

        print(
            f"Epoch [{epoch}/{args.epochs}] | Avg Makespan: {avg_batch_makespan:.2f} "
            f"| Job Loss: {epoch_job_loss:.4f} | Machine Loss: {epoch_machine_loss:.4f} "
            f"| Critic Loss: {epoch_critic_loss:.4f}"
        )

        if _should_validate(epoch, args, validation_events):
            validation = evaluate_validation_events(
                validation_events,
                gnn_model,
                solve_with_gnn,
                evaluate_makespan,
            )
            print(
                f"   -> Validation | Samples: {validation['samples']} "
                f"| Makespan: {validation['makespan']:.2f} "
                f"| Invalid Jobs: {validation['invalid_jobs']:.2f}"
            )
            best_makespan = maybe_save_best_model(
                model=gnn_model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=validation["makespan"],
                best_metric=best_makespan,
                metric_label="Val Makespan",
            )
        elif not validation_events:
            best_makespan = maybe_save_best_model(
                model=gnn_model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=avg_batch_makespan,
                best_metric=best_makespan,
                metric_label="Makespan",
            )

        save_training_checkpoint(
            args.latest_checkpoint_path,
            gnn_model,
            {"optimizer_actor": optimizer_actor, "optimizer_critic": optimizer_critic},
            epoch,
            best_makespan,
            args,
        )

        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            _save_ppo_chart(chart_path, losses_job, losses_machine, losses_critic, makespans)
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
    parser.add_argument("--lr_actor", type=float, default=1e-3, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="Critic learning rate for PPO")
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
        help="Machine-actor penalty for assigning jobs to already overloaded AMRs during REINFORCE replay",
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
