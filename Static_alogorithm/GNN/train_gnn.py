import torch
import torch.optim as optim
import time
import numpy as np
import random
import matplotlib.pyplot as plt

# Import from the GNN script
from GNN import (
    SchedulerGNN, solve_with_gnn, extract_state_gnn,
    AMR_KEYS, STATIONS, SUPPLY_LOCATIONS, TYPE_DURATION, heuristic, AMR_STARTS
)
from GA.GA import make_jobs, describe_solution, local_improve, routing_iters, collision_routing_iters
import torch.nn.functional as F


def evaluate_actions_multi_ppo(jobs, model, job_action_seq, machine_action_seq, init_state=None):
    """
    Re-evaluates a trajectory under the current model parameters.
    Returns:
      - job_log_probs_sum: total log prob of job selections
      - machine_log_probs_sum: total log prob of machine selections
      - values: list of critic state-value estimates at each step
    """
    if init_state:
        amr_positions = {amr: init_state["positions"].get(amr, AMR_STARTS[amr]) for amr in AMR_KEYS}
        amr_availabilities = {amr: float(init_state["availability"].get(amr, 0.0)) for amr in AMR_KEYS}
        station_availabilities = {s: float(init_state["time"]) for s in STATIONS.keys()}
        amr_inventory = {amr: init_state["inventory"].get(amr, {mat: 0 for mat in TYPE_DURATION.keys()}).copy() for amr in AMR_KEYS}
    else:
        amr_positions = {amr: AMR_STARTS[amr] for amr in AMR_KEYS}
        amr_availabilities = {amr: 0.0 for amr in AMR_KEYS}
        station_availabilities = {s: 0.0 for s in STATIONS.keys()}
        amr_inventory = {amr: {mat: 0 for mat in TYPE_DURATION.keys()} for amr in AMR_KEYS}
        amr_inventory["AMR1"]["A"] = 3
        amr_inventory["AMR2"]["B"] = 3
        amr_inventory["AMR3"]["C"] = 3

    assigned_jobs_set = set()
    amr_assignment_map = {}
    order_seq = []

    job_log_probs = []
    machine_log_probs = []
    values = []

    model.train()
    device = next(model.parameters()).device

    for step in range(len(jobs)):
        amr_feat, job_feat, job_mask, adj = extract_state_gnn(
            jobs, assigned_jobs_set, amr_positions, amr_availabilities,
            amr_inventory, amr_assignment_map, station_availabilities, order_seq
        )
        amr_feat = amr_feat.to(device)
        job_feat = job_feat.to(device)
        job_mask = job_mask.to(device)
        adj = adj.to(device)

        # Encode
        job_embeddings = model.encode_jobs(job_feat, adj)
        amr_embeddings = model.encode_amrs(amr_feat)

        # Critic value
        value = model.forward_critic(job_embeddings, amr_embeddings, job_mask)
        values.append(value.squeeze())

        # Job Actor
        job_logits = model.forward_job_actor(job_embeddings, job_mask)
        job_logits_flat = job_logits.view(-1)
        step_job_log_probs = F.log_softmax(job_logits_flat, dim=0)

        chosen_job_list_idx = job_action_seq[step]
        job_log_probs.append(step_job_log_probs[chosen_job_list_idx])

        chosen_job = jobs[chosen_job_list_idx]
        selected_job_emb = job_embeddings[:, chosen_job_list_idx, :]

        # Machine Actor
        machine_logits = model.forward_machine_actor(selected_job_emb, amr_embeddings)
        machine_logits_flat = machine_logits.view(-1)
        step_machine_log_probs = F.log_softmax(machine_logits_flat, dim=0)

        chosen_amr_idx = machine_action_seq[step]
        machine_log_probs.append(step_machine_log_probs[chosen_amr_idx])

        chosen_amr = AMR_KEYS[chosen_amr_idx]

        # Record
        order_seq.append(chosen_job.idx)
        amr_assignment_map[chosen_job.idx] = chosen_amr
        assigned_jobs_set.add(chosen_job.idx)

        # Fast heuristic state update
        material = chosen_job.type_
        curr_pos = amr_positions[chosen_amr]
        avail = amr_availabilities[chosen_amr]

        if amr_inventory[chosen_amr][material] == 0:
            supply_location = SUPPLY_LOCATIONS[material]
            avail += heuristic(curr_pos, supply_location)
            curr_pos = supply_location
            amr_inventory[chosen_amr][material] = 3

        target_station = STATIONS[chosen_job.station]
        avail += heuristic(curr_pos, target_station)

        process_start = max(avail, station_availabilities[chosen_job.station])
        process_end = process_start + chosen_job.duration
        amr_inventory[chosen_amr][material] -= 1
        station_availabilities[chosen_job.station] = process_end

        next_dest = AMR_STARTS[chosen_amr]
        return_end = process_end + heuristic(target_station, next_dest)
        amr_availabilities[chosen_amr] = return_end
        amr_positions[chosen_amr] = next_dest

    return (
        torch.stack(job_log_probs).sum(),
        torch.stack(machine_log_probs).sum(),
        torch.stack(values),
    )


def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Hyperparameters
    num_epochs = args.epochs
    batch_size = 16
    lr_actor = 1e-3
    lr_critic = 1e-3

    # PPO Hyperparameters
    ppo_epochs = 4
    clip_eps = 0.2
    value_loss_coef = 0.5

    # Load test case if specified
    from GA.GA import load_dispatch_events
    dispatch_events = []
    if args.inbox:
        from pathlib import Path
        inbox_path = Path(args.inbox)
        if inbox_path.exists():
            dispatch_events = load_dispatch_events(inbox_path)
            print(f"Loaded {len(dispatch_events)} dispatch events from {args.inbox}")

    # Initialize Model and Optimizers
    gnn_model = SchedulerGNN(job_in_dim=12, amr_in_dim=8, hidden_dim=128, gin_layers=3).to(device)

    # Separate parameter groups for independent actor updates + shared critic
    actor_params = (
        list(gnn_model.job_emb.parameters()) +
        list(gnn_model.gin_layers.parameters()) +
        list(gnn_model.amr_emb.parameters()) +
        list(gnn_model.job_actor.parameters()) +
        list(gnn_model.machine_actor.parameters())
    )
    critic_params = list(gnn_model.critic.parameters())

    optimizer_actor = optim.Adam(actor_params, lr=lr_actor)
    optimizer_critic = optim.Adam(critic_params, lr=lr_critic)

    print("Starting Multi-PPO training with Critic-Based Advantage...")

    best_makespan = float('inf')
    losses_job = []
    losses_machine = []
    losses_critic = []
    makespans = []

    for epoch in range(1, num_epochs + 1):
        batch_makespans = []
        trajectories = []

        # ====== 1. Rollout Phase ======
        for b in range(batch_size):
            if dispatch_events:
                event = random.choice(dispatch_events)
                jobs = event["jobs"]
            else:
                jobs = make_jobs()

            # Stochastic Rollout
            ind, (old_job_lp, old_machine_lp), solve_dur = solve_with_gnn(
                jobs, gnn_model, deterministic=False
            )

            # Reconstruct model-sampled action indices before local improvement mutates the schedule.
            job_id_to_list_idx = {job.idx: idx for idx, job in enumerate(jobs)}
            job_action_seq = []
            machine_action_seq = []
            for job_id in ind.order:
                job_action_seq.append(job_id_to_list_idx[job_id])
                amr = ind.amr_assignment[job_id]
                machine_action_seq.append(AMR_KEYS.index(amr))

            # Apply Local Improve
            improve_start = time.perf_counter()
            ind = local_improve(ind, jobs, max_iters=routing_iters)
            if collision_routing_iters > 0:
                ind = local_improve(ind, jobs, max_iters=collision_routing_iters, check_collision=True)
            solve_dur += (time.perf_counter() - improve_start)

            stochastic_makespan, _ = describe_solution(ind, jobs, solve_time=solve_dur, show_gantt=False)
            batch_makespans.append(stochastic_makespan)

            # Value target: negative makespan (lower makespan = higher value)
            value_target = -stochastic_makespan

            trajectories.append((
                jobs, job_action_seq, machine_action_seq,
                old_job_lp.detach(), old_machine_lp.detach(),
                value_target
            ))

        # ====== 2. Multi-PPO Optimization Phase ======
        epoch_job_loss = 0.0
        epoch_machine_loss = 0.0
        epoch_critic_loss = 0.0

        for ppo_epoch in range(ppo_epochs):
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()

            batch_job_loss = 0.0
            batch_machine_loss = 0.0
            batch_critic_loss = 0.0

            for (jobs, job_action_seq, machine_action_seq,
                 old_job_lp, old_machine_lp,
                 value_target) in trajectories:

                new_job_lp, new_machine_lp, values = evaluate_actions_multi_ppo(
                    jobs, gnn_model, job_action_seq, machine_action_seq
                )
                value_target_tensor = torch.tensor(value_target, dtype=torch.float32, device=values.device)
                value_targets = value_target_tensor.expand_as(values)
                advantage = value_target_tensor - values.detach().mean()

                # --- Job Actor Loss (L_CLIP) ---
                job_ratio = torch.exp(new_job_lp - old_job_lp)
                job_surr1 = job_ratio * advantage
                job_surr2 = torch.clamp(job_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
                job_loss = -torch.min(job_surr1, job_surr2)

                # --- Machine Actor Loss (L_CLIP) ---
                machine_ratio = torch.exp(new_machine_lp - old_machine_lp)
                machine_surr1 = machine_ratio * advantage
                machine_surr2 = torch.clamp(machine_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
                machine_loss = -torch.min(machine_surr1, machine_surr2)

                critic_loss = F.mse_loss(values, value_targets)

                # Combine and backprop
                total_loss = (job_loss + machine_loss + value_loss_coef * critic_loss) / batch_size
                total_loss.backward()

                batch_job_loss += job_loss.item()
                batch_machine_loss += machine_loss.item()
                batch_critic_loss += critic_loss.item()

            optimizer_actor.step()
            optimizer_critic.step()

            epoch_job_loss += batch_job_loss / batch_size
            epoch_machine_loss += batch_machine_loss / batch_size
            epoch_critic_loss += batch_critic_loss / batch_size

        epoch_job_loss /= ppo_epochs
        epoch_machine_loss /= ppo_epochs
        epoch_critic_loss /= ppo_epochs

        avg_batch_makespan = sum(batch_makespans) / batch_size
        losses_job.append(epoch_job_loss)
        losses_machine.append(epoch_machine_loss)
        losses_critic.append(epoch_critic_loss)
        makespans.append(avg_batch_makespan)

        # Logging
        print(f"Epoch [{epoch}/{num_epochs}] | Avg Makespan: {avg_batch_makespan:.2f} "
              f"| Job Loss: {epoch_job_loss:.4f} | Machine Loss: {epoch_machine_loss:.4f} "
              f"| Critic Loss: {epoch_critic_loss:.4f}")

        # Save best model
        if avg_batch_makespan < best_makespan:
            best_makespan = avg_batch_makespan
            torch.save(gnn_model.state_dict(), "gnn_mpn_scheduler_best.pth")
            print(f"   -> Saved new best model (Makespan: {best_makespan:.2f})")

        # Save training metrics periodically
        if epoch == 1 or epoch % 100 == 0 or epoch == num_epochs:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            axes[0, 0].plot(losses_job, color='#e74c3c', linewidth=1.5, label='Job Actor Loss')
            axes[0, 0].set_title('Job Actor Loss', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, linestyle='--', alpha=0.5)
            axes[0, 0].legend()

            axes[0, 1].plot(losses_machine, color='#3498db', linewidth=1.5, label='Machine Actor Loss')
            axes[0, 1].set_title('Machine Actor Loss', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, linestyle='--', alpha=0.5)
            axes[0, 1].legend()

            axes[1, 0].plot(losses_critic, color='#2ecc71', linewidth=1.5, label='Critic Loss')
            axes[1, 0].set_title('Joint Critic Loss', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, linestyle='--', alpha=0.5)
            axes[1, 0].legend()

            axes[1, 1].plot(makespans, color='#ff7f0e', linewidth=1.5, label='Avg Makespan (s)')
            axes[1, 1].set_title('Average Batch Makespan Trend', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Makespan (s)')
            axes[1, 1].grid(True, linestyle='--', alpha=0.5)
            axes[1, 1].legend()

            plt.tight_layout()

            import os
            SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
            chart_path = os.path.join(SCRIPT_DIR, "gnn_training_metrics.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   -> Saved updated training chart to {chart_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    args = parser.parse_args()

    train(args)
