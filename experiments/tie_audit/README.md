# Tie audit — how much of the result is decided by an arbitrary tie-break

Two `argmin`s in this codebase are not single-valued, and in both places something other
than the model settles the outcome.

| where | the tie | who actually decides | script |
|---|---|---|---|
| evaluation | several candidate schedules share the same integer C̃ | the order the candidate loop emitted them in | `../surrogate_fidelity/analyze_fidelity.py --ties` |
| rollout | several legal actions share the same substantive rule score | `job.idx`, then the alphabetical AMR name | `rule_tie_depth.py` |

## 1. Evaluation layer — selection regret

`min(g, key=lambda r: r["c_tilde"])` returns the first minimum in list order, so on a tied
instance the reported regret is a property of the enumeration order. Bracketing it over
the minimiser set M_i:

| n | C̃ tie rate | consequential | best | as-coded | mean | worst |
|---|---|---|---|---|---|---|
| 20 | 56.7% | 13/30 | 7.63% | 9.40% | 9.35% | 11.35% |
| 60 | 16.7% | 4/30 | 3.73% | 4.96% | 4.29% | 4.98% |
| 100 | 0.0% | 0/30 | 2.06% | 2.06% | 2.06% | 2.06% |

Ties are split by consequence: *benign* ones (every member of M_i executes to the same
makespan — usually two rule combos emitting one schedule) cannot move the number and are
excluded from *consequential*. Only 9.7 of the 12 candidates are distinct at n=20.

The summary's "tau" is Goodman–Kruskal γ, not Kendall τ — it drops tied pairs instead of
counting them, and it drops more of them for C̃ (54.9/66 comparable) than for Ψ̂ (57.5/66),
so the two columns are not computed on the same base. τ_b is reported alongside.

Full output: `../surrogate_fidelity/tie_sensitivity.{txt,csv}`.

## 2. Rollout layer — dispatching rules

Every branch of `_job_rule_score` ends with `job.idx` and every branch of
`_amr_rule_score` ends with `action.amr`. These make the order total, so rollouts are
deterministic and reproducible — but they are identifiers, not scheduling arguments.

**How often they decide** (share of decisions where the substantive keys tie):

| AMR rule | n=20 | n=60 | n=100 | AMRs tied |
|---|---|---|---|---|
| `earliest_available` | 28.0% | 9.8% | 6.0% | 5.38 |
| `least_loaded` | 28.0% | 9.8% | 6.0% | 5.38 |
| `nearest_amr` | 8.1% | 9.3% | 9.3% | 2.40 |
| `earliest_completion` | 7.6% | 5.8% | 5.5% | 2.10 |

| job rule | n=20 | n=60 | n=100 | jobs tied |
|---|---|---|---|---|
| `fifo` | 95.0% | 98.3% | 99.0% | 31.00 |
| `spt` / `lpt` | 85.0% | 95.0% | 97.0% | 11.43 |
| `earliest_completion_job` | 15.8% | 27.4% | 32.1% | 2.66 |
| `milk_run` | 13.5% | 26.5% | 34.6% | 2.56 |
| `material_match` | 11.4% | 21.3% | 26.8% | 2.66 |

For `fifo` the index *is* the rule (arrival order), so its 99% is intent. For `spt`/`lpt`
it is a tie-break between equal-duration jobs, and durations are coarse enough that ~11
jobs tie at once. No AMR rule intends `action.amr` as anything but a tie-break.

**Whether it matters** — rerun with the identifiers reversed, rule intent untouched:

| perturbation | n | instances whose makespan moves | mean abs Δ | mean signed Δ | worst |
|---|---|---|---|---|---|
| AMR name reversed | 20 | 65.8% | 1.68% | +0.07% | 11.69% |
| | 60 | 89.6% | 2.13% | −0.43% | 17.28% |
| | 100 | 96.7% | 2.14% | +0.04% | 18.80% |
| both reversed | 20 | 85.0% | 4.78% | +0.24% | 32.63% |
| | 60 | 97.9% | 4.85% | +0.44% | 54.07% |
| | 100 | 99.2% | 4.72% | +1.19% | 39.78% |

Worst hit at n=100: `milk_run+earliest_completion` and `milk_run+nearest_amr`, mean abs
Δ = 5.12%.

**Reading.** The signed mean is near zero, so the tie-break is noise rather than bias and
survives averaging over instances; a paired comparison under one fixed convention stays
internally valid. What it does mean is that a single instance's makespan carries a ±2%
(AMR only) to ±5% (both) arbitrary component, so a rule-vs-rule gap below roughly 2% is a
property of *this implementation* of the rules, not of the rules. The policy-vs-rule
margins (23–34%) are an order of magnitude clear of it; the selection-regret gaps in
`congestion_penalty` (0.03–1.37%) are not.

The instrumented chooser is verified to reproduce the stock rollout exactly when the
perturbations are off.

## Reproduce

```
cd ../surrogate_fidelity && python analyze_fidelity.py --ties
python rule_tie_depth.py --sizes 20,60,100 --instances 10
```
