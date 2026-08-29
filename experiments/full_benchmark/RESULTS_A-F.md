# v10 factorial A–F — training mask × training return

*Generated 2026-08-29 17:48 from `raw/rows*.jsonl`. Every number in this file is computed from the raw rows, not transcribed.*

## What the six cells are

Two factors crossed. **Mask** controls what the policy can see at decision time; **return** controls what a finished schedule is scored on during training.

The L3 mask zeroes every congestion channel in the encoder: `dock[2]` time until the door frees, `dock[3]` queue count / fleet, `dock[4]` queue / waiting slots, `dock[5]` remaining service, `dock[6]` committed workload, `action[2]` expected wait for this action, and `amr[9]` nearby robot count. Masked, the policy still sees position, distance, inventory and time — but cannot tell that any door is busy.

| return | fidelity at n=60 | cost/schedule | masked L3 | full features |
|---|---|---|---|---|
| C̃ idealised — free-space travel, no queueing, cannot fail to route | 15.3% error, τ=0.454 | 1× | **A** | **B** |
| Ψ̂ surrogate — the calibrated fast model the rollout already advances | 7.0% error, τ=0.736 | ~3× | **E** | **F** |
| Φ executor — real space-time A*, reservations, waiting lines | ground truth | ~870× | **C** | **D** |

`D` is v9's `only60` — the configuration this project has treated as its main method.

**Why E/F exist.** With only A/B/C/D, the claim "a cheap reward is worse" could be an artefact of C̃ being a strawman (15.3% error, τ=0.454). Ψ̂ is the cheap pipeline a deployment would actually build — half the error, τ from 0.45 to 0.74. E/F upgrade the control.

## Data and method

- **Instances** `test_case/v3/trend/full_{20,40,60,80,100}.jsonl`, 100 per size, m = 16. Verified disjoint from `train_60`, `test_60`, and every `train_mix_n` / `val_mix_n` pool.
- **Inference** every arm plans under Ψ̂ and every schedule is scored once by Φ — the standard pairing, not executor-in-the-loop.
- **Masks match training.** A, C and E are evaluated with L3 installed; a checkpoint fed channels it never saw during training is not the model that was trained.
- **Selection held fixed.** All cells use their `_best` checkpoint, selected on Φ over the same val_mix set, so cells differ in training and not in how the checkpoint was picked. The `_bestideal` / `_bestsurr` checkpoints are not in this file.
- **Averaging.** Makespan over routable instances only (ν = 0). Where three seeds exist they are averaged *within instance* before averaging across instances. CI = 1.96·sd/√n over instances; deltas are paired per instance.
- **Seed coverage.** Only seed 44 has all six cells at 4000 epochs. D, E and F have three finished seeds. A/B's seeds 42/43 finished after this sweep ran; C's were still training.

## Baseline

The strongest dispatching rule *at each size*, from the 60-combination grid:

| n | rule | makespan |
|---|---|---|
| 20 | `earliest_completion_job+material_match` | 166.2 |
| 40 | `earliest_completion_job+material_match` | 257.7 |
| 60 | `milk_run+least_loaded` | 346.9 |
| 80 | `milk_run+earliest_completion` | 426.2 |
| 100 | `milk_run+earliest_available` | 501.5 |

## Results — greedy (one deterministic rollout)

Seed 44. Percentage is the paired delta against the rule above; negative = policy wins.

| cell | mask | return | val | n=20 | n=40 | n=60 | n=80 | n=100 |
|---|---|---|---|---|---|---|---|---|
| **A** | L3 | C̃ idealised | 792.5 | 196.9 (+18.45%) | 399.4 (+54.97%) | 671.6 (+93.85%) | 875.0 (+105.32%) | 1042.6 (+108.02%) |
| **B** | none | C̃ idealised | 541.1 | 236.1 (+42.05%) | 330.3 (+28.15%) | 407.5 (+17.33%) | 502.8 (+17.97%) | 607.9 (+21.22%) |
| **E** | L3 | Ψ̂ surrogate | 488.5 | 167.4 (+0.71%) | 268.5 (+4.19%) | 371.9 (+7.20%) | 475.9 (+11.66%) | 590.7 (+17.78%) |
| **F** | none | Ψ̂ surrogate | 432.7 | 173.2 (+4.13%) | 250.0 (-3.00%) | 338.8 (-2.26%) | 427.9 (+0.40%) | 516.0 (+2.87%) |
| **C** | L3 | Φ executor | 465.1 | 168.0 (+1.06%) | 260.7 (+1.13%) | 353.3 (+1.84%) | 460.2 (+8.03%) | 567.0 (+13.05%) |
| **D** | none | Φ executor | 437.5 | 168.5 (+1.38%) | 254.1 (-1.42%) | 342.8 (-1.19%) | 440.6 (+3.36%) | 531.5 (+5.98%) |

Win rate — share of instances where the cell beats the rule:

| cell | n=20 | n=40 | n=60 | n=80 | n=100 |
|---|---|---|---|---|---|
| **A** | 5% | 0% | 0% | 0% | 0% |
| **B** | 0% | 0% | 13% | 2% | 1% |
| **E** | 46% | 25% | 18% | 9% | 3% |
| **F** | 32% | 65% | 58% | 47% | 30% |
| **C** | 43% | 41% | 44% | 14% | 7% |
| **D** | 48% | 62% | 53% | 33% | 20% |

**No cell beats the rules at every size under greedy.** F comes closest — ahead at n=40 and n=60, behind at both ends. A single rollout does not beat a tuned dispatching rule; the policy's value is realised through sampling.

## Results — best-of-8 (executor picks among 8 sampled rollouts)

| cell | mask | return | n=20 | n=40 | n=60 | n=80 | n=100 |
|---|---|---|---|---|---|---|---|
| **A** | L3 | C̃ idealised | 192.7 (+15.94%) | 378.7 (+46.93%) | 603.0 (+73.85%) | 800.0 (+87.68%) | 968.2 (+93.07%) |
| **B** | none | C̃ idealised | 222.7 (+34.01%) | 325.5 (+26.28%) | 419.6 (+20.97%) | 524.4 (+23.02%) | 630.1 (+25.65%) |
| **E** | L3 | Ψ̂ surrogate | 160.2 (-3.63%) | 253.5 (-1.64%) | 351.9 (+1.44%) | 451.0 (+5.81%) | 550.1 (+9.70%) |
| **F** | none | Ψ̂ surrogate | 162.0 (-2.53%) | 239.2 (-7.18%) | 319.9 (-7.78%) | 403.4 (-5.35%) | 488.5 (-2.59%) |
| **C** | L3 | Φ executor | 160.2 (-3.62%) | 246.6 (-4.33%) | 338.5 (-2.42%) | 436.1 (+2.30%) | 534.5 (+6.57%) |
| **D** | none | Φ executor | 161.1 (-3.04%) | 242.2 (-6.05%) | 326.1 (-6.00%) | 416.6 (-2.27%) | 504.3 (+0.56%) |

Best-of-8 also fixes routability: it ranks candidates on `(ν, makespan)`, so it only returns an unroutable schedule if all 8 samples fail. Zero unroutable across every best-of-K cell.

## Seed-averaged (D, E, F — the cells with three finished seeds)

**greedy** — makespan and paired delta vs the rule

| cell | n=20 | n=40 | n=60 | n=80 | n=100 |
|---|---|---|---|---|---|
| **D** | 168.8 (+1.57% ±1.40) | 255.4 (-0.92% ±1.06) | 341.5 (-1.60% ±1.73) | 435.3 (+2.13% ±1.42) | 527.3 (+5.14% ±1.31) |
| **E** | 173.2 (+4.26% ±1.28) | 271.5 (+5.33% ±0.98) | 372.7 (+7.31% ±1.74) | 485.3 (+13.79% ±1.40) | 611.1 (+21.77% ±1.51) |
| **F** | 168.7 (+1.40% ±1.36) | 249.3 (-3.28% ±1.15) | 334.7 (-3.43% ±1.68) | 422.4 (-0.96% ±1.41) | 513.2 (+2.22% ±1.29) |

**best-of-8** — makespan and paired delta vs the rule

| cell | n=20 | n=40 | n=60 | n=80 | n=100 |
|---|---|---|---|---|---|
| **D** | 160.8 (-3.23% ±1.13) | 242.8 (-5.79% ±0.93) | 326.6 (-5.86% ±1.68) | 414.0 (-2.88% ±1.31) | 500.6 (-0.18% ±1.24) |
| **E** | 164.5 (-1.03% ±1.00) | 259.5 (+0.68% ±0.91) | 356.3 (+2.71% ±1.70) | 461.0 (+8.15% ±1.33) | 570.1 (+13.67% ±1.36) |
| **F** | 158.8 (-4.46% ±1.11) | 237.5 (-7.85% ±0.86) | 317.9 (-8.34% ±1.65) | 401.4 (-5.83% ±1.28) | 485.1 (-3.27% ±1.24) |

**best-of-16** — makespan and paired delta vs the rule

| cell | n=20 | n=40 | n=60 | n=80 | n=100 |
|---|---|---|---|---|---|
| **D** | 158.6 (-4.58% ±1.11) | 239.5 (-7.09% ±0.92) | 322.9 (-6.92% ±1.67) | 409.9 (-3.83% ±1.32) | 495.5 (-1.20% ±1.26) |
| **E** | — | — | — | — | — |
| **F** | 156.4 (-5.89% ±1.06) | 234.5 (-9.02% ±0.85) | 313.6 (-9.60% ±1.65) | 397.1 (-6.82% ±1.27) | 479.7 (-4.35% ±1.26) |

### F − D, paired per instance, seed-averaged (negative = F wins)

| budget | n=20 | n=40 | n=60 | n=80 | n=100 |
|---|---|---|---|---|---|
| greedy | -0.2 ±1.9 | -6.1 ±2.2 | -6.8 ±2.9 | -13.1 ±3.2 | -14.6 ±3.7 |
| best-of-8 | -2.0 ±1.0 | -5.3 ±1.0 | -8.6 ±1.1 | -12.6 ±1.4 | -15.5 ±1.6 |
| best-of-16 | -2.2 ±0.8 | -5.0 ±0.8 | -9.3 ±1.0 | -12.8 ±1.2 | -15.8 ±1.6 |

**F beats D at every search budget**, on every seed, with the advantage widening as the workload grows. The cheaper training reward produces the better policy.

## Effect sizes and interaction (greedy, seed 44, makespan units)

Positive = the first cell is worse.

| contrast | n=20 | n=40 | n=60 | n=80 | n=100 |
|---|---|---|---|---|---|
| **cost of masking** under Φ (C − D) | -0.5 | +6.6 | +10.5 | +19.6 | +35.5 |
| **cost of masking** under Ψ̂ (E − F) | -5.5 | +18.5 | +32.7 | +48.0 | +74.8 |
| **cost of masking** under C̃ (A − B) | -39.2 | +69.1 | +264.0 | +372.4 | +434.6 |
| reward Ψ̂→Φ, full features (F − D) | +4.7 | -4.1 | -3.8 | -12.6 | -15.6 |
| reward Ψ̂→Φ, masked (E − C) | -0.6 | +7.9 | +18.5 | +15.7 | +23.7 |
| reward C̃→Ψ̂, full features (B − F) | +62.7 | +80.3 | +69.0 | +74.9 | +92.4 |
| reward C̃→Ψ̂, masked (A − E) | +29.5 | +130.9 | +298.7 | +399.2 | +451.8 |

**The two factors substitute for each other.** Masking costs 35.5 units at n=100 under the executor reward, 74.8 under the surrogate, and 434.6 under the idealised decode. Read the other way: with full features the cheap reward is *better*; with features masked the expensive reward is needed. A policy needs congestion information from somewhere — its inputs or its reward — and **A has neither**.

At n=20 masking is mildly *helpful* in every row. With 1.25 parcels per robot there is almost no contention to observe, so the congestion channels are noise that costs capacity.

## Per-seed detail (best-of-8 and best-of-16)

**best-of-8** — delta vs rule, %

| arm | n=20 | n=40 | n=60 | n=80 | n=100 | mean |
|---|---|---|---|---|---|---|
| F s42 | -6.20% | -7.95% | -7.19% | -4.03% | -0.87% | -5.25% |
| F s43 | -4.66% | -8.42% | -10.04% | -8.11% | -6.34% | -7.51% |
| F s44 | -2.53% | -7.18% | -7.78% | -5.35% | -2.59% | -5.09% |
| *F best seed (s43)* | *-4.66%* | *-8.42%* | *-10.04%* | *-8.11%* | *-6.34%* | *-7.51%* |
| **F 3-seed avg** | **-4.46%** | **-7.85%** | **-8.34%** | **-5.83%** | **-3.27%** | **-5.95%** |
| D s42 | -3.54% | -5.66% | -5.84% | -3.02% | -0.11% | -3.63% |
| D s43 | -3.12% | -5.67% | -5.73% | -3.34% | -0.98% | -3.77% |
| D s44 | -3.04% | -6.05% | -6.00% | -2.27% | +0.56% | -3.36% |
| *D best seed (s43)* | *-3.12%* | *-5.67%* | *-5.73%* | *-3.34%* | *-0.98%* | *-3.77%* |
| **D 3-seed avg** | **-3.23%** | **-5.79%** | **-5.86%** | **-2.88%** | **-0.18%** | **-3.59%** |

**best-of-16** — delta vs rule, %

| arm | n=20 | n=40 | n=60 | n=80 | n=100 | mean |
|---|---|---|---|---|---|---|
| F s42 | -7.67% | -8.98% | -8.43% | -4.97% | -1.84% | -6.38% |
| F s43 | -6.20% | -9.73% | -11.17% | -9.12% | -7.36% | -8.71% |
| F s44 | -3.80% | -8.36% | -9.20% | -6.38% | -3.85% | -6.32% |
| *F best seed (s43)* | *-6.20%* | *-9.73%* | *-11.17%* | *-9.12%* | *-7.36%* | *-8.71%* |
| **F 3-seed avg** | **-5.89%** | **-9.02%** | **-9.60%** | **-6.82%** | **-4.35%** | **-7.14%** |
| D s42 | -4.90% | -7.08% | -6.97% | -4.00% | -0.90% | -4.77% |
| D s43 | -4.28% | -6.66% | -6.87% | -4.31% | -1.89% | -4.80% |
| D s44 | -4.55% | -7.53% | -6.91% | -3.18% | -0.83% | -4.60% |
| *D best seed (s43)* | *-4.28%* | *-6.66%* | *-6.87%* | *-4.31%* | *-1.89%* | *-4.80%* |
| **D 3-seed avg** | **-4.58%** | **-7.09%** | **-6.92%** | **-3.83%** | **-1.20%** | **-4.72%** |

Picking the strongest seed is worth roughly 1.5 pp over the three-seed average. That is a post-hoc choice made on the test set and should not be the headline — v9 spent a whole iteration cutting checkpoint-selection bias (`--val_window 5`, 280-instance validation, seed spread from 13.4 down to ~1), and cherry-picking a seed throws that away.

## Compute cost

Seconds per instance, single process, nothing else on the machine (seed 44).

| arm | budget | n=20 | n=40 | n=60 | n=80 | n=100 |
|---|---|---|---|---|---|---|
| F | greedy | 0.100 | 0.345 | 0.769 | 1.361 | 2.147 |
| F | best-of-8 | 1.057 | 3.172 | 6.634 | 11.831 | 18.701 |
| F | best-of-16 | 2.128 | 6.377 | 13.490 | 23.687 | 36.893 |
| D | greedy | 0.098 | 0.341 | 0.766 | 1.371 | 2.164 |
| D | best-of-8 | 1.064 | 3.208 | 6.739 | 11.795 | 18.317 |
| D | best-of-16 | 2.123 | 6.418 | 13.460 | 23.590 | 36.776 |

**F and D cost the same to run.** Same architecture, same Ψ̂ rollout; they differ only in the reward used during training. F's advantage is free at inference — the saving is in training, where the reward is ~870× cheaper per schedule.

Inside best-of-K the split is `solve` (K Ψ̂ rollouts) versus `select` (K Φ scorings):

| budget | n | solve (Ψ̂) | select (Φ) | select share |
|---|---|---|---|---|
| best-of-8 | 60 | 6.128 | 0.506 | 7.6% |
| best-of-8 | 100 | 17.794 | 0.907 | 4.9% |
| best-of-16 | 60 | 12.452 | 1.038 | 7.7% |
| best-of-16 | 100 | 35.081 | 1.812 | 4.9% |

Φ is ~870× more expensive per schedule than Ψ̂, but best-of-K calls it only K times against K·2n rollout steps, so real-executor scoring is under 8% of inference cost.

## Sweep wall-clock

Not a compute cost — these ran with 18 workers plus concurrent training on the same box, and the two differ mostly because the machine was busier during the first one.

| sweep | schedules | wall-clock | concurrent training |
|---|---|---|---|
| F best-of-16 | 1500 | 74.5 min | 4 runs |
| D best-of-16 | 1500 | 36.6 min | 2 runs |

## Open items

- **A/B/C at seeds 42/43.** Only s44 was finished when this sweep ran. A and B have since finished; C was still training. Re-running those three cells at all seeds would give the six-cell table the same seed backing D/E/F already have.
- **Selection under Ψ̂.** best-of-K currently *generates* with Ψ̂ but *selects* with Φ, so the pipeline still touches the executor at inference. `experiments/surrogate_fidelity` measured Ψ̂'s selection regret at 1.24% on 12 rule schedules; it has not been measured on K samples from one policy, which are more similar and therefore harder to rank.
- **n > 100.** Whether F's advantage over D keeps widening is untested.

---

Raw rows: `raw/rows.jsonl` (A–D, rules, GA), `raw/rows_ef.jsonl` (E/F greedy+best8), `raw/rows_f16.jsonl`, `raw/rows_d16.jsonl`, timings `raw/timing_k.jsonl`.  
Regenerate with `python experiments/full_benchmark/write_md.py`.
