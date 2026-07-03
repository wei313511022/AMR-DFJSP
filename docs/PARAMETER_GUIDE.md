# Project Parameter Guide

This file summarizes the control parameters used across the DDQN project.

Primary entry points:
- `ddqn_model.ipynb`: main training and evaluation control block via `CFG`
- `main.py`: script version of the same pipeline, with a different default set in some fields
- `ddqn_test.ipynb`: offline test and batch-review control block via `TEST_CFG`
- `env.py`: environment and simulation constants
- `trainer.py`: trainer-side fixed constants
- `random_job_gen.py`: synthetic dispatch data generation parameters
- `live_job_feeder.py`: live stream job injection parameters

## 1. `ddqn_model.ipynb` / `main.py` main controls

`ddqn_model.ipynb` uses a `CFG` dictionary as the main manual parameter block.
`main.py` exposes nearly the same controls as local variables.

### Data source and training-scenario generation

| Parameter | Meaning |
| --- | --- |
| `task_file` | Base JSONL filename used when training data is read from a single file instead of many streams. |
| `train_data_dir` | Folder that stores generated or discovered training scenario files. |
| `auto_generate_data` | If `True`, call `random_job_gen.generate_data()` before training/testing. If `False`, reuse existing JSONL files. |
| `gen_batches` | Number of dispatch batches generated for each training stream/file. |
| `gen_size` | Fixed number of jobs per dispatch batch. If `None`, the generator uses `gen_min_size` and `gen_max_size`. |
| `gen_min_size` | Minimum jobs per dispatch when `gen_size` is not fixed. |
| `gen_max_size` | Maximum jobs per dispatch when `gen_size` is not fixed. |
| `gen_arrival_mean` | Mean inter-arrival time between dispatch batches. The generator samples exponentially around this mean. |
| `gen_seed` | Random seed for single-stream generation mode. |
| `multi_streams` | If `True`, training data is treated as multiple JSONL streams. If `False`, only one file is used. |
| `num_streams` | Number of stream files to generate or probe in multi-stream mode. |
| `base_seed` | Base seed for multi-stream generation. Stream `i` uses `base_seed + i`. |
| `stream_file_template` | Naming pattern for multi-stream files, for example `dispatch_batches_{i}.jsonl`. |

Important behavior:
- `gen_size` overrides `gen_min_size` and `gen_max_size`.
- When `multi_streams=True` and `auto_generate_data=True`, old matching JSONL stream files are deleted first, then regenerated.
- Scenario discovery prefers globbing existing files under `train_data_dir`, then falls back to explicit `0..num_streams-1`.

### Scenario sampling

| Parameter | Meaning |
| --- | --- |
| `sampling_mode` | Training-scenario sampling mode. `full` = use the whole stream, `window` = random contiguous window, `subset` = random subset sorted by dispatch time. |
| `window_size` | Number of records kept when `sampling_mode="window"`. |
| `subset_size` | Number of records sampled when `sampling_mode="subset"`. |

### Train/test switches

| Parameter | Meaning |
| --- | --- |
| `do_train` | If `True`, run the DDQN training loop. |
| `do_test` | If `True`, run greedy evaluation and plotting after loading/training. |

### Checkpoint and model-shape controls

| Parameter | Meaning |
| --- | --- |
| `save_model_path` | Output checkpoint path after training. Saved checkpoint includes policy, target, optimizer, and input dimensions. |
| `load_model_path` | Optional checkpoint path to load before training/testing. |
| `state_dim` | Expected environment state dimension. In the current environment this is `28`. Used to compute `input_dim`. |
| `action_dim` | Action-feature dimension. In the current project this is `4`: `travel`, `wait`, `proc`, `replenish`. |
| `input_dim` | Network input size = `state_dim + action_dim`. In `main.py` this is set explicitly; in the notebook it is recomputed before model creation. |

Important behavior:
- Checkpoint compatibility depends on `input_dim` matching the current environment and action feature definition.
- The current environment state layout is fixed in `env.py`; if that layout changes, `state_dim` must change too.

### DDQN optimization parameters

| Parameter | Meaning |
| --- | --- |
| `lr` | Adam learning rate. |
| `adam_eps` | Adam epsilon (`optim.Adam(..., eps=adam_eps)`). Rainbow paper uses `1.5e-4`. |
| `num_episodes` | Number of training episodes. Each episode samples one scenario/stream and runs until done. |
| `batch_size` | Replay batch size used once replay memory has at least this many transitions. |
| `gamma` | Discount factor in the DDQN target `r + gamma * Q_target(...)`. |
| `epsilon` | Initial epsilon-greedy exploration rate. |
| `epsilon_end` | Minimum exploration rate after decay. |
| `epsilon_decay` | Per-episode multiplicative decay applied to `epsilon`. |
| `allow_proactive_replenish` | Controls action-space design. If `True`, an AMR may replenish even when it still has inventory. If `False`, replenishment is only allowed when inventory is empty. |
| `proactive_replenish_bias_weight` | Action-selection bias weight used as `score = Q + weight * coverage_gain`. Larger values more strongly prefer replenishing now to cover same-type future jobs and reduce source round-trips. Set `0` to disable bias. |
| `proactive_full_load_bias_weight` | Extra action-selection bias to prefer larger replenish amounts (closer to full load). Helps enforce the "take more in one trip" policy. |
| `proactive_waiting_replenish_bias_weight` | Extra bias when station wait time exists; encourages using waiting windows to proactively replenish. |
| `enable_collision_avoidance` | Toggles time-aware AMR path planning. If `False`, robots use shortest paths and route overlap is allowed. If `True`, reservation-based collision avoidance is used. |
| `use_rainbow` | If `True`, use the Rainbow-capable network/trainer path instead of the classic scalar DDQN MLP. |
| `rainbow_num_atoms` | Number of C51 atoms used by the distributional head. |
| `rainbow_v_min` | Lower bound of the distributional value support. |
| `rainbow_v_max` | Upper bound of the distributional value support. |
| `rainbow_noisy_std` | Initial standard deviation used by NoisyLinear layers. |
| `rainbow_replay_capacity` | Replay buffer capacity used by the prioritized replay buffer. |
| `min_replay_size` | Minimum replay size before optimization starts (warmup phase). |
| `train_every_steps` | Perform one optimizer update every N environment steps. |
| `target_sync_steps` | If `>0`, sync target network every N environment steps. If `<=0`, fallback to legacy every-5-episodes sync. |
| `sample_with_replacement` | Whether PER sampling is done with replacement (Rainbow style). |
| `rainbow_n_step` | N-step return length used by Rainbow training. |
| `per_alpha` | Priority exponent for prioritized replay. `0` reduces to uniform replay. |
| `per_beta_start` | Initial importance-sampling correction exponent for prioritized replay. |
| `per_beta_end` | Final importance-sampling correction exponent, annealed across episodes. |
| `priority_eps` | Small positive floor added to replay priorities to avoid zero-probability samples. |
| `grad_clip_norm` | Gradient clipping threshold used during optimization. |
| `use_noisy_exploration` | If `True`, use NoisyNet exploration instead of epsilon-greedy action sampling. |

Rainbow-specific behavior:
- Distributional RL uses C51 style categorical support between `rainbow_v_min` and `rainbow_v_max`.
- Prioritized replay is enabled when `use_rainbow=True`.
- N-step returns are enabled when `use_rainbow=True`.
- NoisyNet exploration is available through `use_noisy_exploration`.
- With `sample_with_replacement=True`, replay sampling follows the standard proportional PER setting.
- `min_replay_size`, `train_every_steps`, and `target_sync_steps` allow Rainbow-style warmup / update cadence.

Trainer-side fixed behavior not exposed in `CFG`:
- If `target_sync_steps <= 0`, the target network is synced every `5` episodes (legacy fallback).
- Intermediate reward is always `0`; terminal reward is always `-makespan`.

### Training visualization and profiling

| Parameter | Meaning |
| --- | --- |
| `show_train_schedule` | If `True`, draw the dispatch queue, AMR schedule, and input queue during training. |
| `train_schedule_every_episodes` | Draw schedule every N episodes. `1` means every episode. |
| `train_schedule_every_steps` | Draw schedule every N dispatch decisions within the selected episode. |
| `train_schedule_window` | X-axis time window centered around current time. If `0` or falsy, no local zoom is applied. |
| `train_schedule_window_all_axes` | If `True`, apply the same time window to all three training schedule plots; otherwise only the AMR schedule plot is zoomed. |
| `train_schedule_pause` | `matplotlib` pause after each schedule refresh. Controls visualization speed. |
| `train_schedule_show_labels` | Whether plot labels are rendered on training schedule charts. |
| `train_schedule_figsize` | Figure size for the 3-row training schedule plot. |
| `show_train_route_map` | If `True`, render the route-map replay during training. |
| `train_route_map_every_episodes` | Draw route map every N episodes. |
| `train_route_map_every_steps` | Draw route map every N decisions inside a selected episode. |
| `train_route_map_pause` | Pause between route-map frames. |
| `train_route_map_figsize` | Figure size for route-map display. |
| `train_route_map_animate` | If `True`, animate route-map frames between two time stamps; if `False`, draw only one frame per refresh. |
| `train_route_map_time_step` | Time resolution used when animating the route map. Smaller values create more frames. |
| `train_route_map_max_frames_per_update` | Upper bound on frames rendered in one route-map refresh. Prevents runaway rendering cost. |
| `train_route_map_delay_seconds` | Route-map replay lags behind scheduler time by this many seconds. Useful to avoid replaying a path segment that has not "completed" yet. |
| `enable_profile` | If `True`, print aggregated timing breakdown for action-feature building, Q selection, environment stepping, and DDQN update. |
| `profile_cuda_sync` | If `True` and CUDA is used, synchronize before timing GPU work to get more accurate profiling. |

### Testing and plotting

| Parameter | Meaning |
| --- | --- |
| `test_scenario_file` | Optional JSON/JSONL test scenario used by `run_test_and_plot()`. If omitted, the first training scenario is reused. |
| `show_live` | Use the live animated greedy rollout on a fixed scenario. |
| `show_live_stream` | Use live stream mode: poll `live_job_file`, enqueue new jobs, and keep dispatching greedily. This branch takes priority over `show_live`. |
| `show_interactive` | Show the matplotlib interactive schedule viewer after testing. |
| `show_route_map` | Show route-map replay after testing. |
| `show_plotly` | Export/show the Plotly interactive schedule HTML. |
| `show_test_plots` | Save the standard PNG summary plots (`dispatch_queue.png`, `amr_schedule.png`, `input_queue.png`). |
| `live_pause` | Pause per visual refresh in live rollout modes. |
| `live_job_file` | JSONL file polled by live-stream mode for appended batches. |
| `live_start_at_end` | If `True`, the live-stream reader starts from the end of the file and only consumes newly appended jobs. |
| `live_poll_interval` | Wall-clock polling interval for new live batches. |
| `live_idle_sleep` | Sleep time when no dispatch can be made in live-stream mode. |
| `live_max_steps` | Optional cap on dispatch decisions in live-stream mode. |
| `live_max_sim_time` | Optional cap on simulation time in live-stream mode. |
| `live_init_scenario` | Initial jobs already present before live-stream polling starts. |
| `live_record_dir` | If set, save live rollout frames into this folder. |
| `live_record_every` | Save one frame every N live steps. |
| `live_record_dpi` | DPI for saved live frames. |
| `live_make_gif` | In fixed-scenario live mode, optionally combine saved frames into a GIF. |
| `live_gif_path` | Output GIF path when `live_make_gif=True`. |
| `route_play_step` | Route-map replay time step during test replay. |
| `route_play_interval_ms` | UI refresh interval in milliseconds for route-map replay. |

Important behavior:
- `show_live_stream=True` ignores the regular test scenario and instead uses `live_init_scenario` plus batches appended to `live_job_file`.
- `show_route_map` and `show_plotly` are post-run visualization controls; they do not change policy behavior.

### Notebook defaults vs `main.py` defaults

The current `ddqn_model.ipynb` and `main.py` do not use exactly the same defaults.
The most important differences are:

| Parameter | Notebook default | `main.py` default |
| --- | --- | --- |
| `auto_generate_data` | `False` | `True` |
| `gen_batches` | `1` | `15` |
| `num_streams` | `2000` | `200` |
| `save_model_path` | `ddqn_policy_static_sch_no_avoidance.pt` | `ddqn_policy.pt` |
| `num_episodes` | `1000` | `30` |
| `enable_collision_avoidance` | `False` | `True` |
| `show_train_schedule` | `True` | `True` |
| `show_train_route_map` | `False` | `True` |
| `train_route_map_animate` | `False` | `True` |
| `train_route_map_max_frames_per_update` | `1000` | `120` |
| `train_route_map_delay_seconds` | `0` | `20.0` |

If you mainly work in the notebook, treat `CFG` as the current primary control block.

## 2. `ddqn_test.ipynb` test controls (`TEST_CFG`)

`ddqn_test.ipynb` is focused on offline case review, per-file testing, route export, and statistics export.

| Parameter | Meaning |
| --- | --- |
| `target_path` | Path to one `.jsonl` file or to a folder containing many `.jsonl` files. |
| `model_path` | Checkpoint path to load for evaluation. |
| `case_mode` | `full_stream` treats an entire JSONL file as one multi-dispatch scenario. `each_line` treats each line as an independent single-dispatch case. |
| `max_files` | When `target_path` is a folder, only evaluate the first N JSONL files. |
| `max_cases_per_file` | When `case_mode="each_line"`, only evaluate the first N lines/cases in each file. |
| `plot` | Show matplotlib figures interactively while testing. |
| `save_plots` | Save PNG summary plots for each case. |
| `plot_dir` | Output directory for saved PNG plots. |
| `show_route_map` | Replay route map after each test case. |
| `show_route_map_use_saved_jsonl` | If route JSONL export exists, replay from the saved route file instead of using in-memory trace. Present in function signatures even if not shown in the sample `TEST_CFG`. |
| `show_plotly` | Generate Plotly HTML schedules. |
| `plotly_html_dir` | Output directory for Plotly HTML files. |
| `plotly_window` | Time window width for Plotly schedule view. |
| `plotly_step` | Sampling/step size used by the Plotly schedule helper. |
| `save_route_jsonl` | Export per-time-step AMR route traces into JSONL files. |
| `route_jsonl_dir` | Output directory for saved route JSONL logs. |
| `route_time_step` | Time step used when exporting route JSONL. Smaller values produce denser route logs. |
| `save_stats_txt` | Export case-level and summary statistics to a text file. |
| `stats_txt_path` | Output path for the stats text file. |
| `stats_txt_append` | If `True`, append to existing stats file instead of overwriting it. |
| `allow_proactive_replenish` | Same semantics as training: controls whether proactive refill actions are available during evaluation. |
| `proactive_replenish_bias_weight` | Same semantics as training. Controls how much evaluation policy prefers proactive refill when Q-values are close. |
| `proactive_full_load_bias_weight` | Same semantics as training. Controls preference for larger refill amounts during evaluation. |
| `proactive_waiting_replenish_bias_weight` | Same semantics as training. Controls extra refill preference when selected task has station wait. |
| `enable_collision_avoidance` | Same semantics as training: toggles time-aware collision avoidance. |
| `print_predict_time` | Measure and print inference latency statistics for each case. |

Important behavior:
- Folder mode reuses one loaded model/runtime across all files.
- `case_mode="each_line"` is the right mode when a file contains many independent test cases.
- `route_time_step` only affects exported route logs, not the actual scheduling simulation.

## 3. Environment constants in `env.py`

These are not exposed through `CFG`, but they directly determine the scheduling problem.

| Constant | Current value | Meaning |
| --- | --- | --- |
| `W`, `H` | `10`, `10` | Grid size is 10 columns by 10 rows. |
| `obstacles` | wall at `x=6`, blocked except gaps at `y=3` and `y=7` | Defines the map topology and corridor bottlenecks. |
| `source_locs` | `A:(0,7)`, `B:(0,4)`, `C:(0,1)` | Pickup points for each material type. |
| `station_locs` | `S1:(9,8)`, `S2:(9,6)`, `S3:(9,4)`, `S4:(9,2)`, `S5:(9,0)` | Processing station coordinates. |
| `num_robots` | `3` | Number of AMRs. |
| `material_types` | `["A","B","C"]` | Supported material classes. |
| `capacity_per_type` | `3` | Max per-material inventory carried by each AMR. |
| `initial_robot_positions` | `(2,1)`, `(2,4)`, `(2,7)` | Starting AMR positions. |

Behavioral rules tied to these constants:
- Earliest dispatch time is normalized to `t=0` in every scenario.
- State vector length is fixed at `28`:
  - current robot one-hot: `3`
  - available task count: `1`
  - current time: `1`
  - each robot `[free_time, x, y, invA, invB, invC]`: `18`
  - station busy-until values: `5`
- Action feature length is fixed at `4`: `[travel, wait, proc, replenish_total]`.
- A grid move or wait step in the time-aware planner costs `1.0` simulated second.
- If collision avoidance is on, route planning uses reservation tables for occupied cells, occupied intervals, and edge conflicts.
- If collision avoidance is off, route planning uses shortest paths only and allows overlap.
- Processing at a station is mutually exclusive because `station_busy_until[station]` blocks overlapping process segments.
- After each dispatch, the chosen AMR consumes exactly one unit of the task material.

Action-space rule:
- One action is `(task_idx, replenish_plan)` where `replenish_plan` is a dict over `A/B/C`.
- Task material follows hard feasibility:
  - if task-type inventory is zero, task-type replenish must be at least `1`.
  - if task-type inventory is positive and `allow_proactive_replenish=False`, task-type replenish is forced to `0`.
- Cross-type proactive refill:
  - if `allow_proactive_replenish=True`, other types can also be refilled in the same action.
  - current implementation uses compact options for other types: `0` or full top-up to `capacity_per_type`.

Reward rule:
- Step reward: `-(makespan_t - makespan_{t-1})`
- Episode return equals `-final_makespan`
- Therefore objective is still minimizing final makespan, but with denser learning signal.

## 4. Trainer and model fixed parameters

These values are hard-coded in code, not exposed in the notebook config block.

### `trainer.py`

| Fixed parameter | Current value | Meaning |
| --- | --- | --- |
| Target-network sync period | every `5` episodes | Frequency of copying policy weights into target network. |
| Makespan moving average | `50` episodes | Used for displayed moving-average plots. |
| Loss moving average | `100` updates | Used for displayed loss smoothing. |

### `model.py`

`QNetwork` is now a wrapper with two modes:

- Classic mode: scalar Q-value MLP for old DDQN checkpoints
- Rainbow mode: state encoder + action encoder + NoisyLinear dueling distributional head

Classic mode default hidden sizes:

`input_dim -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 1`

Rainbow mode current fixed structure:

- State encoder: `state_dim -> 256 -> 128`
- Action encoder: `action_dim -> 64 -> 64`
- Value stream: `128 -> 128 -> num_atoms`
- Advantage stream: `(128 + 64) -> 128 -> num_atoms`

This means:
- `input_dim`, `state_dim`, `action_dim`, and Rainbow distributional settings are externally controllable.
- Classic hidden sizes are still hard-coded unless the wrapper reconstructs them from an older checkpoint.
- Old classic checkpoints and new Rainbow checkpoints are both supported by the current wrapper.

## 5. Synthetic data generation parameters (`random_job_gen.py`)

These are used by `prepare_scenarios()` when auto-generation is enabled, and also available from the command line.

| Parameter | Meaning |
| --- | --- |
| `num_batches` / `--batches` | Number of dispatch events to generate. |
| `fixed_size` / `--size` | Fixed jobs per dispatch batch. |
| `min_size` / `--min-size` | Minimum jobs per dispatch if size is random. |
| `max_size` / `--max-size` | Maximum jobs per dispatch if size is random. |
| `arrival_mean` / `--arrival-mean` | Mean exponential inter-arrival time between dispatches. |
| `output_file` / `--out` | Output JSONL file path. |
| `station_count` / `--stations` | Random station index range `1..station_count`. |
| `seed` / `--seed` | Random seed for reproducibility. |

Generator-side fixed constants:

| Constant | Current value | Meaning |
| --- | --- | --- |
| `DEFAULT_STATION_COUNT` | `5` | Default number of stations used by the generator CLI. |
| `JOB_TYPES["A"]["time"]` | `10` | Processing time for type A jobs. |
| `JOB_TYPES["B"]["time"]` | `15` | Processing time for type B jobs. |
| `JOB_TYPES["C"]["time"]` | `20` | Processing time for type C jobs. |

## 6. Live stream job feeder parameters (`live_job_feeder.py`)

These constants drive the script that appends new batches into `live_jobs.jsonl`.

| Parameter | Meaning |
| --- | --- |
| `OUTPUT_FILE` | JSONL file receiving appended live batches. |
| `INTERVAL_SEC` | Wall-clock interval between appended batches. |
| `JOBS_PER_BATCH` | Number of jobs appended each time. |
| `TYPES` | Job types sampled by the feeder. |
| `STATIONS` | Candidate stations sampled by the feeder. |
| `PROC_TIME_BY_TYPE` | Processing time assigned to each job type. |
| `USE_ELAPSED_TIME` | If `True`, `dispatch_time` equals elapsed wall-clock time since feeder start. If `False`, all appended jobs use `dispatch_time=0.0` and the environment clamps them to current time. |

## 7. Practical reading order

If you want to tune the project without changing code structure, the usual order is:

1. Edit `ddqn_model.ipynb` `CFG` or `main.py`.
2. If the scheduling problem itself should change, edit `env.py`.
3. If generated data should change, edit `random_job_gen.py` inputs.
4. If offline evaluation output should change, edit `ddqn_test.ipynb` `TEST_CFG`.

## 8. Highest-impact parameters in practice

If the goal is to influence learning behavior or scheduling behavior quickly, the most important controls are:

- `enable_collision_avoidance`
- `allow_proactive_replenish`
- `num_episodes`
- `batch_size`
- `gamma`
- `epsilon`, `epsilon_end`, `epsilon_decay`
- `sampling_mode`, `window_size`, `subset_size`
- `gen_batches`, `gen_size`, `gen_min_size`, `gen_max_size`, `gen_arrival_mean`
- Environment constants in `env.py`: `num_robots`, `capacity_per_type`, map layout, source/station positions

## 9. Suggested next cleanup

The project currently keeps parameters in three places:
- notebook `CFG`
- script variables in `main.py`
- test notebook `TEST_CFG`

If you want, the next practical step is to centralize them into one config file or one dataclass so notebook and script defaults cannot drift apart.
