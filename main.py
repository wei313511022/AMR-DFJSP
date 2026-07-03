import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #solve OpenmMP library repeat problem

import torch
import torch.nn as nn
import torch.optim as optim

from core.env import TaskSchedulingEnv
from core.model import QNetwork
from inference.checkpoint_io import export_contract_checkpoint
from training.evaluator import print_batch_results, run_test_and_plot
from training.trainer import prepare_scenarios, train_ddqn


def main(): 
    # ---- FJSSP training data (scripts/Generate_training_data.py, unmodified) ----
    # Each JSONL line is one instance: {"machines": 6, "jobs": [{"operations":
    # [[{"machine", "processing"}, ...], ...], "material": "A|B|C"}, ...]}.
    # One instance = one training scenario (multi-operation jobs, machine choice).


    use_fjssp_dataset = True #True : call the Generate_training_data.py gen data
    fjssp_dataset_file = os.path.join("data", "fjssp_training_dataset.jsonl")
    fjssp_num_instances = 100
    fjssp_regenerate = False  # True: regenerate the dataset before training

    # ---- Legacy dispatch-batch data source (used when use_fjssp_dataset=False) ----**
    task_file = os.path.join("data", "dispatch_batches.jsonl")
    train_data_dir = os.path.join(os.getcwd(), "data", "train_data")
    auto_generate_data = True
    gen_batches = 15
    gen_size = None
    gen_min_size = 1
    gen_max_size = 25
    gen_arrival_mean = 100
    gen_seed = None
    multi_streams = True
    num_streams = 200
    base_seed = 100
    stream_file_template = "dispatch_batches_{i}.jsonl"

    # Scenario sampling
    sampling_mode = "full"  # full | window | subset
    window_size = 10
    subset_size = 10

    # Train / test switches
    do_train = True
    do_test = True

    # Model checkpoint
    save_model_path = os.path.join("checkpoints", "ddqn_policy.pt") #the model checkpoint
    # Contract-format checkpoint (Phase III section 6): the deliverable that
    # plugs into the target architecture via inference.load_model().(future job to connect the main)
    export_contract_path = os.path.join("checkpoints", "my_scheduler_v1.pth")
    load_model_path = None

    # Optimization and DDQN
    action_dim = 4  # (travel, station_wait, proc, dock_wait)
    lr = 1e-3
    adam_eps = 1.5e-4
    num_episodes = 30
    batch_size = 128
    gamma = 0.99
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    # Batch pickup / proactive replenishment: dock operations may take extra
    # units of the job's material (up to capacity 3/type) so later same-material
    # jobs are served from onboard stock. Score = Q + cover/load/wait bonuses.
    allow_proactive_replenish = True
    proactive_replenish_bias_weight = 2.5
    proactive_full_load_bias_weight = 1.8
    proactive_waiting_replenish_bias_weight = 1.5
    enable_collision_avoidance = True
    # rainbow DDQN settinng
    use_rainbow = True
    rainbow_num_atoms = 51
    rainbow_v_min = -10000.0
    rainbow_v_max = 0.0
    rainbow_noisy_std = 0.5
    rainbow_replay_capacity = 50000
    min_replay_size = 2048
    train_every_steps = 4
    target_sync_steps = 2000
    sample_with_replacement = True
    rainbow_n_step = 3
    per_alpha = 0.5
    per_beta_start = 0.4
    per_beta_end = 1.0
    priority_eps = 1e-5
    grad_clip_norm = 10.0
    use_noisy_exploration = True

    # Training visualization and profiling
    show_train_schedule = True
    train_schedule_every_episodes = 1
    train_schedule_every_steps = 10
    train_schedule_window = 120.0
    train_schedule_window_all_axes = False
    train_schedule_pause = 0.01
    train_schedule_show_labels = False
    train_schedule_figsize = (14, 8)
    show_train_route_map = True
    train_route_map_every_episodes = 1
    train_route_map_every_steps = 1
    train_route_map_pause = 0.01
    train_route_map_figsize = (9, 8)
    train_route_map_animate = True
    train_route_map_time_step = 0.5
    train_route_map_max_frames_per_update = 120
    train_route_map_delay_seconds = 20.0
    enable_profile = True
    profile_cuda_sync = True

    # Test and plotting
    test_scenario_file = os.path.join("data", "test_scenario_one_time.jsonl")
    show_live = False
    show_live_stream = False
    show_interactive = False
    show_route_map = True
    show_plotly = True
    show_test_plots = True
    live_pause = 0.05
    live_job_file = os.path.join("data", "live_jobs.jsonl")
    live_start_at_end = True
    live_poll_interval = 0.5
    live_idle_sleep = 0.1
    live_max_steps = 100
    live_max_sim_time = None
    live_init_scenario = []
    live_record_dir = None
    live_record_every = 5
    live_record_dpi = 140
    live_make_gif = False
    live_gif_path = "live_schedule.gif"
    route_play_step = 0.5
    route_play_interval_ms = 120

    if use_fjssp_dataset:
        # Integrates scripts/Generate_training_data.py as-is: import and call
        # its generator, then load each instance as an independent scenario.
        from core.data_io import load_records
        from scripts.Generate_training_data import create_jsonl_dataset

        if fjssp_regenerate or not os.path.exists(fjssp_dataset_file):
            os.makedirs(os.path.dirname(fjssp_dataset_file) or ".", exist_ok=True)
            create_jsonl_dataset(fjssp_dataset_file, fjssp_num_instances)
        scenario_list = load_records(fjssp_dataset_file)
        if not scenario_list:
            raise RuntimeError(f"No FJSSP instances found in {fjssp_dataset_file}")
        print(f"Loaded {len(scenario_list)} FJSSP instances from {fjssp_dataset_file}")
    else:
        scenario_list = prepare_scenarios(
            task_file=task_file,
            train_data_dir=train_data_dir,
            auto_generate_data=auto_generate_data,
            gen_batches=gen_batches,
            gen_size=gen_size,
            gen_min_size=gen_min_size,
            gen_max_size=gen_max_size,
            gen_arrival_mean=gen_arrival_mean,
            gen_seed=gen_seed,
            multi_streams=multi_streams,
            num_streams=num_streams,
            base_seed=base_seed,
            stream_file_template=stream_file_template,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    env = TaskSchedulingEnv()
    env.allow_proactive_replenish = allow_proactive_replenish
    env.proactive_replenish_bias_weight = proactive_replenish_bias_weight
    env.proactive_full_load_bias_weight = proactive_full_load_bias_weight
    env.proactive_waiting_replenish_bias_weight = proactive_waiting_replenish_bias_weight
    env.enable_collision_avoidance = enable_collision_avoidance

    # State layout is defined by the env spec (5 AMRs / 5 stations / 5 docks -> 47).
    state_dim = len(env.reset([]))
    input_dim = state_dim + action_dim
    print(f"state_dim = {state_dim}, action_dim = {action_dim}")

    policy_net = QNetwork(
        input_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        use_rainbow=use_rainbow,
        num_atoms=rainbow_num_atoms,
        v_min=rainbow_v_min,
        v_max=rainbow_v_max,
        noisy_std=rainbow_noisy_std,
    ).to(device)
    target_net = QNetwork(
        input_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        use_rainbow=use_rainbow,
        num_atoms=rainbow_num_atoms,
        v_min=rainbow_v_min,
        v_max=rainbow_v_max,
        noisy_std=rainbow_noisy_std,
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer_state = None
    if load_model_path and os.path.exists(load_model_path):
        ckpt = torch.load(load_model_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            policy_net.load_state_dict(ckpt["model_state"])
            if "target_state" in ckpt:
                target_net.load_state_dict(ckpt["target_state"])
            else:
                target_net.load_state_dict(policy_net.state_dict())
            if "optimizer_state" in ckpt:
                optimizer_state = ckpt["optimizer_state"]
        else:
            policy_net.load_state_dict(ckpt)
            target_net.load_state_dict(policy_net.state_dict())
        print(f"Loaded model from {load_model_path}")

    optimizer = optim.Adam(policy_net.parameters(), lr=lr, eps=adam_eps)
    criterion = nn.MSELoss()
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    if do_train:
        train_ddqn(
            env=env,
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            criterion=criterion,
            scenario_list=scenario_list,
            device=device,
            num_episodes=num_episodes,
            batch_size=batch_size,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            sampling_mode=sampling_mode,
            window_size=window_size,
            subset_size=subset_size,
            show_train_schedule=show_train_schedule,
            train_schedule_every_episodes=train_schedule_every_episodes,
            train_schedule_every_steps=train_schedule_every_steps,
            train_schedule_window=train_schedule_window,
            train_schedule_window_all_axes=train_schedule_window_all_axes,
            train_schedule_pause=train_schedule_pause,
            train_schedule_show_labels=train_schedule_show_labels,
            train_schedule_figsize=train_schedule_figsize,
            show_train_route_map=show_train_route_map,
            train_route_map_every_episodes=train_route_map_every_episodes,
            train_route_map_every_steps=train_route_map_every_steps,
            train_route_map_pause=train_route_map_pause,
            train_route_map_figsize=train_route_map_figsize,
            train_route_map_animate=train_route_map_animate,
            train_route_map_time_step=train_route_map_time_step,
            train_route_map_max_frames_per_update=train_route_map_max_frames_per_update,
            train_route_map_delay_seconds=train_route_map_delay_seconds,
            enable_profile=enable_profile,
            profile_cuda_sync=profile_cuda_sync,
            replay_capacity=rainbow_replay_capacity,
            rainbow_n_step=rainbow_n_step,
            per_alpha=per_alpha,
            per_beta_start=per_beta_start,
            per_beta_end=per_beta_end,
            priority_eps=priority_eps,
            grad_clip_norm=grad_clip_norm,
            use_noisy_exploration=use_noisy_exploration,
            min_replay_size=min_replay_size,
            train_every_steps=train_every_steps,
            target_sync_steps=target_sync_steps,
            sample_with_replacement=sample_with_replacement,
        )

        if save_model_path:
            os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
            ckpt = {
                "model_state": policy_net.state_dict(),
                "target_state": target_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "input_dim": input_dim,
                "model_kind": policy_net.model_kind,
                "rainbow_config": policy_net.rainbow_config(),
            }
            torch.save(ckpt, save_model_path)
            print(f"Saved model to {save_model_path}")

        if export_contract_path:
            os.makedirs(os.path.dirname(export_contract_path) or ".", exist_ok=True)
            export_contract_checkpoint(
                policy_net,
                export_contract_path,
                env_spec=env.env_spec,
                selection_bias={
                    "cover": proactive_replenish_bias_weight,
                    "load": proactive_full_load_bias_weight,
                    "wait": proactive_waiting_replenish_bias_weight,
                },
            )
            print(f"Exported Phase III contract checkpoint to {export_contract_path}")

    if do_test:
        run_test_and_plot(
            env=env,
            policy_net=policy_net,
            device=device,
            scenario_list=scenario_list,
            test_scenario_file=test_scenario_file,
            show_live=show_live,
            show_live_stream=show_live_stream,
            show_interactive=show_interactive,
            show_route_map=show_route_map,
            show_plotly=show_plotly,
            show_test_plots=show_test_plots,
            live_pause=live_pause,
            live_job_file=live_job_file,
            live_start_at_end=live_start_at_end,
            live_poll_interval=live_poll_interval,
            live_idle_sleep=live_idle_sleep,
            live_max_steps=live_max_steps,
            live_max_sim_time=live_max_sim_time,
            live_init_scenario=live_init_scenario,
            live_record_dir=live_record_dir,
            live_record_every=live_record_every,
            live_record_dpi=live_record_dpi,
            live_make_gif=live_make_gif,
            live_gif_path=live_gif_path,
            route_play_step=route_play_step,
            route_play_interval_ms=route_play_interval_ms,
        )
        print_batch_results(env=env, policy_net=policy_net, device=device, scenario_list=scenario_list)


if __name__ == "__main__":
    main()
