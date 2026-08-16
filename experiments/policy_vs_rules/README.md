# Experiment 2 — learned policy (extend_GNN + PPO) vs dispatching rules

**Question.** Six PPO checkpoints finished training on 2026-08-16 and none had ever been
evaluated outside its own training loop. Does the learned policy beat the best dispatching
rule on unseen instances, and does it hold up away from the conditions it was trained under?

---

## 1. What is being evaluated

| | |
|---|---|
| model | `ExtendSchedulerGNN(hidden_dim=128, gin_layers=3)`, `Static_alogorithm/extend_GNN/extend_GNN.py` |
| v8 checkpoints | `ppo_gc1_s{42,43,44}`, `ppo_gc1.5_s{42,43,44}` — PPO, 4000 epochs, grad_clip 1.0 vs 1.5 |
| v7 checkpoints | `ppo_s42` (PPO, 2000 ep), `episode_s42`, `clip50_s42` (REINFORCE, 2000 ep), `stepwise_s42`, `stepwise_clip50_s42` (REINFORCE, **killed at epoch 528/536 of 2000**) |
| trained on | `test_case/v3/train_60.jsonl` (5000 instances), m=16, n=60 |
| selected on | `test_case/v3/val_60.jsonl` (50 instances) |
| reported on | `test_case/v3/test_60.jsonl` (100 instances), plus `test_120`/`test_240` for zero-shot |

All eleven runs post-date commit `d2f014f` (2026-08-08), which introduced the current
geometry (inward waiting lines, aisle depot, θ=1 door clearance). `eval_extend_gnn.py`
asserts this mechanically from each checkpoint's sibling `_latest.pth` and `runs/*/args.json`
rather than taking it on trust.

## 2. Harness

`Static_alogorithm/eval_extend_gnn.py`. Rows are the unmodified `ideal_evaluator.evaluate`
dict plus join keys matching the dispatching-rule sweeps (`amrs`, `instance`,
`job_rule="policy"`, `rule=<run_key>`), so policy and rule rows pair on `(amrs, instance)`
and both pass through `ie.aggregate` unchanged.

Four guard rails, each because the corresponding failure is silent:

- **`load_state_dict(strict=False)`** in `operation_policy.py:563` drops shape-mismatched
  tensors and leaves them randomly initialised — the eval would still print a plausible
  number. The harness parses the "loaded X/Y tensors" status and refuses unless X == Y
  (currently 66/66 for all 17 checkpoints).
- **`scenario_v3.apply_layout`** patches `operation_policy` inside a bare
  `except Exception: pass`. A swallowed failure leaves the dock-queue features in the
  previous fleet's units. `DOCK_QUEUE_SCALE` is asserted after every layout call.
- **`solve_with_extend_gnn(deterministic=False)`** calls `model.train()` and never restores
  eval mode, so one sampled row would change the regime of every greedy row after it in the
  same process. Eval mode is restored explicitly after each sampled rollout.
- **`instances_60.jsonl` is entirely inside `train_60.jsonl`.** Passing it is refused outright.

**Harness self-test (blocking).** Recomputed val scores must equal the values logged during
training. All four spot-checked checkpoints reproduce exactly:
`ppo_gc1_s42_best` 344.38, `ppo_gc1_s44_best` 335.38, `ppo_gc1.5_s42_best` 340.50,
`ppo_gc1.5_s44_best` 340.90. Re-run this whenever torch, the driver, or the GPU changes —
greedy decoding is deterministic only up to float ties in `torch.argmax`.

## 3. PRE-REGISTRATION

> Committed before any `test_60` policy row existed. The commit containing this section is
> quoted in `RESULTS.md`. Everything below was fixed on `val_60` evidence alone.

`_best.pth` is `argmin` over 201 validation evaluations on the same 50 instances, so a val
score is an optimistic order statistic, **not** a generalisation estimate. `val_60` decides
two things and reports none.

**3.1 Arm selection.** `A(g) = mean over seeds of val_60 executed makespan`, tie band
**2.0 makespan units**, tie-break **gc1** (the library default). Measured:

```
A(gc1)   = 342.27   seeds [335.38, 344.38, 347.04]
A(gc1.5) = 341.11   seeds [340.50, 340.90, 341.92]
gap      = 1.16  ->  TIE
```

The tie branch fired, as anticipated. **Headline arm = gc1**; the gc1.5 arm is reported on
test alongside it and labelled exploratory. Between-seed sd within gc1 is **6.11** against an
arm gap of **1.16**, so grad_clip is **not resolvable at 3 seeds** and no claim will be made
about it. (Worth noting separately: gc1.5's seed range is 1.42 versus gc1's 11.66 — the arms
tie on mean but not on stability.)

**3.2 Comparison rule.** `argmin` over the 12 (job rule × AMR rule) combinations of mean
executed makespan on `val_60`:

```
milk_run+earliest_completion  356.78   <- selected
milk_run+nearest_amr          356.78   (byte-identical per instance to the above)
milk_run+earliest_available   358.08
milk_run+least_loaded         358.08
```

The selected rule is `milk_run+earliest_completion`. This is also the rule the policy was
trained against, which is a weaker bar than the strongest rule on test
(`milk_run+earliest_available`, 347.2). **Both are reported**, with the training-baseline
comparison marked as such, so the gap between "beats its own baseline" and "beats the best
rule" stays visible. Note that `earliest_completion` and `nearest_amr` produce identical
schedules on every instance, so the val tie is a duplicate rather than an arbitrary choice.

**3.3 Primary endpoint.** Seed-averaged paired difference in **executed makespan** between
the gc1 arm (3 seeds, `_best.pth`, greedy, no local improvement, m=16) and the val-selected
rule, over the 100 `test_60` instances.

**3.4 Go/no-go.** If the seed-averaged paired delta is not significantly negative, the
headline becomes *"parity with the best dispatching rule at comparable compute"*. It does
**not** become a search for a configuration that wins.

**3.5 Post-hoc labelling.** Any configuration examined after test rows exist — the other arm,
a different rule, `_latest.pth`, best-of-K, local improvement — is reported under a heading
marked post-hoc, with the number of comparisons stated.

## 4. Primary configuration vs ablations

| Confound | Primary | Ablation | Why |
|---|---|---|---|
| local improvement | **OFF** | ON at (1000, 100), seeds 42/42 | Validation never used it, so ON breaks comparability with every logged val score; probes show it can make executed makespan *worse* (340→357) because its simplified stage optimises the collision-free decode |
| decoding | **deterministic greedy, 1 rollout** | best-of-8, beside a matched-budget oracle-rule column | K rollouts is a search budget the single-shot rules do not have |
| checkpoint | **`_best.pth`** | `_latest.pth` (epoch 4000) | `_latest` is val-independent, so `test(_best) − test(_latest)` measures what the selection actually bought |
| baseline rule | **val-selected** (`milk_run+earliest_completion`) | strongest test rule (`milk_run+earliest_available`) shown alongside | Beating only its own training baseline is the least informative result |

## 5. Statistics

Unit of analysis is the **instance**; 3 seeds × 100 instances are never pooled into 300 rows.
Seeds are averaged *within* instance first, then across instances.

Pairing is on `(amrs, instance)` with set equality asserted before anything is computed.
Only instances where **both** sides routed cleanly (`nu == 0`) enter a timing mean; dropped
ids are listed. Per-instance deltas get a normal-approximation CI (`1.96·sd/√n`, matching
`experiments/congestion_penalty/analyze.py`), a percentile bootstrap CI (10 000 resamples),
win/tie/loss counts, and a paired permutation test (10 000 sign flips). Stdlib only.

**Λ is a diagnostic, not the endpoint.** The policy changes the assignment and therefore
changes `ideal` as well as `executed`, so it can lower Λ by inflating the ideal. The
objective is `executed`; Λ, Ω_q, Ω_r describe where the time goes.

## 6. An expectation that Run 0 corrected

Before the val rule baseline existed, the policy's val scores (335–347) were being compared
against the *test* rule field (347.2–350.6), which suggested parity. That was an
apples-to-oranges comparison across two different instance sets. `val_60` turns out to be
**harder** than `test_60` — the same rule scores 356.8 on val versus 350.6 on test, a 1.8%
gap. Against the val rule baseline the gc1 arm leads by 14.5 units (4.1%) on val. The test
result is the one that counts, but "parity" was the wrong prior.

## 7. Files

| file | contents |
|---|---|
| `run_select.sh` | Runs 0–1, val only, safe to re-run |
| `run_test.sh` | Run 2 + zero-shot + v7; refuses to run if this file is dirty in git |
| `run_ablations.sh` | fleet transfer, best-of-8, local improvement |
| `analyze_policy.py` | selection and headline stages |
| `raw/*.jsonl` | one file per run, plus `manifest_*.json` (sha256, geometry, git HEAD) |
| `selection.json` | the frozen output of §3 |
| `RESULTS.md` | findings and limitations |
