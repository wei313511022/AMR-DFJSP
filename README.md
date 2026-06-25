Current cross-docking default: 5 AMRs, 5 inbound docks, and 5 outbound/processing stations.

Legacy FJSP generator default: 6 stations

10, 15, 20, 25, 30 jobs per instance

4 ~ 8 operations per job

1 ~ 6 feasible machines per operation

10 ~ 99 processing time


ref: https://github.com/SchedulingLab/fjsp-instances/tree/main

## Environment setup

On Windows PowerShell, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

Then run demos with:

```powershell
.\.venv\Scripts\python.exe run_all_demos.py
```

Run training with:

```powershell
.\.venv\Scripts\python.exe train_all_models_parallel.py
```

## Neural local-improvement study

Preview the default Attention/GNN experiment without loading the models:

```powershell
.\.venv\Scripts\python.exe Static_alogorithm\benchmark_neural_local_improvement.py --dry_run
```

Run the tuning and held-out validation phases:

```powershell
.\.venv\Scripts\python.exe Static_alogorithm\benchmark_neural_local_improvement.py --phase all
```

The study writes resumable run details, aggregated metrics, model-specific
recommendations, and Pareto plots under
`Static_alogorithm/benchmark_results/local_improvement/`. Add `--resume` to
continue a run with the same grid, samples, models, and seed.
