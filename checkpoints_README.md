# Checkpoint map

Model weights (`*.pth`) are **not tracked in git**. They were committed until
`a007185`, which pushed `.git` to 741 MB; the history still carries those blobs but
nothing new is added. The directories below live on disk only — back them up
separately, they are not restored by a fresh clone.

## Live — do not rename

The paths below are cited by the pre-registered evaluation protocol (`e6022ef`).
Renaming them breaks the paper's audit trail.

| Directory | Contents | Backs |
|---|---|---|
| `checkpoints_v8/` | `ppo_gc1_s{42,43,44}`, `ppo_gc1.5_s{42,43,44}` (best + latest) | **Headline result.** Gradient-clip arms. Consumed by `experiments/policy_vs_rules/run_test.sh`, `run_ablations.sh`, `run_congestion_blind.sh`, `bench_compute.py`, and `Static_alogorithm/eval_extend_gnn.py` |
| `checkpoints_v7/` | `ppo_s42`, `clip50_s42`, `episode_s42`, `stepwise_s42`, `stepwise_clip50_s42`, `extend_gnn_s42` (best + latest) | PPO-vs-REINFORCE credit-assignment arms, seed 42 only. Consumed by `run_select.sh`, `run_test.sh` |
| `checkpoints_v6/` | `attention_s{42,43,44}`, `gnn_s{42,43,44}`, `extend_gnn_s{42,43,44}` (best + latest) | Architecture comparison — all three trained under one recipe, which is why it is the matched comparison. Consumed by `run_architectures.sh`, `eval_extend_gnn.py` |
| `checkpoints_v3/` | `attention_s42`, `gnn_s42`, `extend_gnn_s42` (best + latest) | The v3 training run described in `TRAINING_v3.md`; default output dir of `train_all_models_parallel.py` |

## Legacy 5-AMR weights (repo root)

The 11 `*.pth` at the repo root are the pre-v3, 5-AMR-era weights. They stay at the
root **on purpose**: several call sites load them by bare CWD-relative path, e.g.

- `Static_alogorithm/Attention/Attention.py:484` — `Path("attention_scheduler_best.pth")`
- `Static_alogorithm/GNN/GNN.py:494` — `Path("gnn_mpn_scheduler_best.pth")`
- `Static_alogorithm/GNN/GNN_precise.py:514`, `Attention_precise.py:518`

so those scripts only resolve their weights when run from the repo root. Moving the
files requires editing every literal plus the `_find_checkpoint` helper at
`Static_alogorithm/calibrate_fast_model.py:94` (which searches only `ROOT_DIR` and
`STATIC_DIR`). Not worth the regression risk for 28 MB — left as a deliberate
follow-up.

Note `Static_alogorithm/Attention/attention_scheduler_best.pth` is a **different file**
from the root copy of the same name (differing checksums); the
`Random_Job_Arrivals/models/Attention_DDQN_*.py` demos resolve that one via
`../../Static_alogorithm/Attention/`.

## Archived

Moved out of the repo to `~/Desktop/AMR-DFJSP-archive/`, all with zero code
references at the time of archiving:

| Archived | Size | Was |
|---|---|---|
| `checkpoints/checkpoints_v5/` | 80 MB | superseded 3-seed run |
| `checkpoints/checkpoints_v4/` | 31 MB | superseded, seed 43 only |
| `checkpoints/checkpoints_5amr_backup/` | 29 MB | partial stale copy of the root weights — 6 of 11 byte-identical, the rest diverged |
| `checkpoints/old_checkpoints_precalibration/` | 25 MB | pre-calibration weights |
| `training_logs/curriculum_20260615_175146_checkpoints/` | 137 MB | phase_01–04 curriculum weights |
| `root_stale/` | small | `Route_Map.png`, `results_fleet_sweep.jsonl` |

All four checkpoint directories remain recoverable from git history as well, since
they were tracked before `a007185`.
