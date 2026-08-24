# Surrogate fidelity — what Ψ̂ buys over the idealised transport model

**Question.** Section IV conditions the policy on a calibrated surrogate Ψ̂ rather than on
the idealised decode C̃ of eq. (5), and asserts that running the executor Φ per epoch is
not affordable. Neither claim was quantified for Ψ̂ itself. This measures all three
evaluators on identical schedules.

| evaluator | what it models |
|---|---|
| C̃ `per_robot_ideal` | free-space travel, service on arrival, no interference |
| Ψ̂ `apply_fast_action` | calibrated travel (`0.892·d + 4.49`) + fitted queue-depth penalty `[0.32, 0.32, 2.24, 6.41]`, no routing |
| Φ `decode_schedule` | space–time A*, exclusivity, waiting lines, collisions |

## Design

30 instances at each of n = 20/60/100, m=16, from `test_case/v3/trend/full_*.jsonl`.
Per instance, **12 candidate schedules** from 6 job rules × 2 AMR rules spanning the rule
families. Every schedule is scored by all three evaluators, and the time to score is
recorded. 1080 schedules, all routable.

**Decision error is computed within an instance.** Ranking candidates pooled across
instances would largely recover "more parcels take longer" and inflate τ for every
evaluator, including the idealised one — the comparison has to hold the instance fixed.

## Reproduce

```
python run_fidelity.py --sizes 20,60,100 --instances 30
python analyze_fidelity.py
```

Outputs `fidelity.jsonl` (per-schedule rows), `fidelity.csv`, `fidelity_summary.txt`.

## Known limitation

The 12 candidates are produced by dispatching rules that themselves plan with Ψ̂, so Ψ̂ is
being asked to rank schedules it helped construct. The rules differ enough to give a wide
spread, but a robustness check with schedules from another source — sampled policy
rollouts, or the GA — would remove the concern. Not yet run.
