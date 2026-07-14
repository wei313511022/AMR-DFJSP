from __future__ import annotations

import argparse
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
from operation_policy import (  # noqa: E402
    action_mask,
    apply_fast_action as apply_operation_action,
    decode_action_id,
    initial_operation_state,
)
from reinforce_baseline import (  # noqa: E402
    DEFAULT_BASELINE_RULE,
    compute_dispatch_baseline_comparison,
    evaluate_makespan,
    load_training_events,
    normalize_advantage_batches,
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
    step_log_probs, _, values = evaluate_action_steps_extend(
        jobs,
        model,
        action_seq,
        init_state=init_state,
        include_values=True,
    )
    return step_log_probs.sum(), values


def _load_validation_events_extend(args):
    return _load_validation_events(args)


def train_reinforce(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

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

    print(
        "Starting hybrid dock-aware extend_GNN REINFORCE training "
        f"with dispatch baseline '{args.baseline_rule}' ({args.baseline_mode})."
    )

    best_makespan = float("inf")
    actor_losses = []
    sampled_makespans = []
    baseline_makespans = []
    improvements = []
    win_rates = []
    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extend_gnn_training_metrics.png")

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
            individual, _, _ = solve_with_extend_gnn(jobs, model, deterministic=False)
            action_seq = action_sequences_from_individual(individual, jobs)
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

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(_actor_params(model), args.grad_clip)
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
                model,
                solve_with_extend_gnn,
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
                model=model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=validation_score,
                best_metric=best_makespan,
                metric_label="Val Score",
            )
        elif not validation_events:
            best_makespan = maybe_save_best_model(
                model=model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=avg_sampled,
                best_metric=best_makespan,
                metric_label="Makespan",
            )

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
            print(f"   -> Saved updated training chart to {chart_path}")


def train_ppo(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

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

    print("Starting hybrid dock-aware extend_GNN PPO training with critic-based advantage.")

    best_makespan = float("inf")
    losses_actor = []
    losses_critic = []
    makespans = []
    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extend_gnn_training_metrics.png")

    for epoch in range(1, args.epochs + 1):
        batch_makespans = []
        trajectories = []

        for _ in range(args.batch_size):
            jobs = _select_jobs(dispatch_events)
            individual, old_log_prob, solve_dur = solve_with_extend_gnn(jobs, model, deterministic=False)
            action_seq = action_sequences_from_individual(individual, jobs)

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

        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0

        for _ in range(args.ppo_epochs):
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            batch_actor_loss = 0.0
            batch_critic_loss = 0.0

            for jobs, action_seq, old_log_prob, value_target in trajectories:
                new_log_prob, values = evaluate_actions_extend_ppo(jobs, model, action_seq)
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
                batch_actor_loss += actor_loss.item()
                batch_critic_loss += critic_loss.item()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(_actor_params(model), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), args.grad_clip)
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

        print(
            f"Epoch [{epoch}/{args.epochs}] | Avg Makespan: {avg_batch_makespan:.2f} "
            f"| Actor Loss: {epoch_actor_loss:.4f} | Critic Loss: {epoch_critic_loss:.4f}"
        )

        if _should_validate(epoch, args, validation_events):
            validation = evaluate_validation_events(
                validation_events,
                model,
                solve_with_extend_gnn,
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
                model=model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=validation_score,
                best_metric=best_makespan,
                metric_label="Val Score",
            )
        elif not validation_events:
            best_makespan = maybe_save_best_model(
                model=model,
                best_model_path=args.best_model_path,
                fallback_model_path=LEGACY_BEST_MODEL_PATH,
                current_metric=avg_batch_makespan,
                best_metric=best_makespan,
                metric_label="Makespan",
            )

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
            print(f"   -> Saved updated training chart to {chart_path}")


def train(args):
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
    parser.add_argument("--inboxes", type=str, default="", help="Comma-separated dispatch JSONL files used as one combined training pool")
    parser.add_argument("--validation_inbox", type=str, default="", help="Optional fixed validation dispatch JSONL file")
    parser.add_argument("--validation_inboxes", type=str, default="", help="Comma-separated fixed validation dispatch JSONL files")
    parser.add_argument("--validation_interval", type=int, default=50, help="Epoch interval for fixed validation scoring")
    parser.add_argument("--validation_invalid_penalty", type=float, default=1000.0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_checkpoint", type=str, default="")
    parser.add_argument("--latest_checkpoint_path", type=str, default="extend_gnn_training_checkpoint.pth")
    parser.add_argument("--best_model_path", type=str, default="extend_gnn_scheduler_best.pth")
    parser.add_argument("--rl_method", choices=["reinforce", "ppo"], default="reinforce")
    parser.add_argument("--baseline_rule", type=str, default=DEFAULT_BASELINE_RULE)
    parser.add_argument("--baseline_mode", choices=["stepwise", "episode"], default="stepwise")
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--load_balance_coef", type=float, default=0.1)
    parser.add_argument("--normalize_advantage", dest="normalize_advantage", action="store_true", default=True)
    parser.add_argument("--no_normalize_advantage", dest="normalize_advantage", action="store_false")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
