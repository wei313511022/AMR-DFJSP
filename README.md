6 stations

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
