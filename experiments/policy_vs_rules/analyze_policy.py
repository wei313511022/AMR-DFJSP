"""Policy-vs-rules analysis: paired per instance, seed-aware, executed-makespan first.

Mirrors experiments/congestion_penalty/analyze.py in conventions (ci95 = 1.96*sd/sqrt(n),
never average over nu > 0, pair within instance) and adds what a learned policy needs:

  * Pairing is on (amrs, instance) between policy rows and rule rows from the congestion
    sweep. Set equality is asserted before anything is computed -- a --start offset on one
    side and not the other would otherwise produce a silently misaligned mean and CI.
  * Three seeds are averaged WITHIN instance first, then over instances. Pooling
    3 seeds x 100 instances into 300 "independent" rows would understate the CI by ~sqrt(3).
  * Skewed makespan differences get a bootstrap CI and a permutation p-value alongside the
    normal-approximation CI. Stdlib only; no scipy dependency is added.
  * Lambda is reported but is NOT the endpoint. A policy changes the assignment and hence
    changes `ideal` as well as `executed`, so it can lower Lambda by inflating the ideal.
    The objective is executed makespan; Lambda / Omega are diagnostics of where time goes.

    python analyze_policy.py --stage select     # val_60: arm + baseline-rule choice
    python analyze_policy.py --stage headline   # test_60 policy vs the selected rule
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
CONGESTION = HERE.parent / "congestion_penalty" / "raw"

BOOTSTRAP_N = 10000
PERMUTATION_N = 10000
SEEDS = (42, 43, 44)


def load(path: Path) -> list:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def ci95(xs: list) -> float:
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def bootstrap_ci(xs: list, n: int = BOOTSTRAP_N, seed: int = 7) -> tuple:
    """Percentile bootstrap of the mean. Makespan deltas are skewed, so the normal
    approximation alone can misstate the interval."""
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(xs)
    means = []
    for _ in range(n):
        means.append(sum(xs[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return (means[int(0.025 * n)], means[int(0.975 * n)])


def permutation_p(xs: list, n: int = PERMUTATION_N, seed: int = 11) -> float:
    """Two-sided paired permutation test: random sign flips of the per-instance deltas."""
    if len(xs) < 2:
        return float("nan")
    rng = random.Random(seed)
    obs = abs(statistics.mean(xs))
    hits = 0
    for _ in range(n):
        flipped = sum(x if rng.random() < 0.5 else -x for x in xs)
        if abs(flipped / len(xs)) >= obs:
            hits += 1
    return (hits + 1) / (n + 1)


# --------------------------------------------------------------------------
# rule side
# --------------------------------------------------------------------------

def rule_table(rows: list, amrs: int) -> dict:
    """{(job_rule, amr_rule): {instance: row}} for cleanly-routed rows at one fleet size."""
    out = defaultdict(dict)
    for r in rows:
        if r["amrs"] == amrs and r.get("family") != "policy":
            out[(r["job_rule"], r["rule"])][r["instance"]] = r
    return out


def rank_rules(table: dict) -> list:
    """Rule combinations by mean executed makespan over cleanly-routed instances."""
    ranked = []
    for combo, by_inst in table.items():
        clean = [r for r in by_inst.values() if r["nu"] == 0]
        if not clean:
            continue
        ranked.append({
            "job_rule": combo[0], "amr_rule": combo[1],
            "combo": "+".join(combo),
            "executed": round(statistics.mean([r["executed"] for r in clean]), 2),
            "ideal": round(statistics.mean([r["ideal"] for r in clean]), 2),
            "penalty_pct": round(100 * statistics.mean([r["penalty"] for r in clean]), 2),
            "clean": len(clean), "instances": len(by_inst),
        })
    return sorted(ranked, key=lambda r: r["executed"])


def policy_table(rows: list, amrs: int) -> dict:
    """{run_key: {instance: row}}."""
    out = defaultdict(dict)
    for r in rows:
        if r["amrs"] == amrs:
            out[r["run_key"]][r["instance"]] = r
    return out


# --------------------------------------------------------------------------
# pairing
# --------------------------------------------------------------------------

def pair(policy: dict, rule: dict, label: str) -> tuple:
    """Instances where BOTH sides routed cleanly. Asserts the instance sets match."""
    if set(policy) != set(rule):
        only_p = sorted(set(policy) - set(rule))[:5]
        only_r = sorted(set(rule) - set(policy))[:5]
        raise SystemExit(
            f"{label}: instance sets differ -- policy-only {only_p}, rule-only {only_r}. "
            f"Refusing to pair misaligned rows."
        )
    common = sorted(policy)
    dropped = [i for i in common if policy[i]["nu"] != 0 or rule[i]["nu"] != 0]
    kept = [i for i in common if i not in set(dropped)]
    return kept, dropped


def paired_stats(policy: dict, rule: dict, kept: list, field: str = "executed") -> dict:
    d = [policy[i][field] - rule[i][field] for i in kept]
    rule_mean = statistics.mean([rule[i][field] for i in kept])
    lo, hi = bootstrap_ci(d)
    wins = sum(1 for x in d if x < 0)
    ties = sum(1 for x in d if x == 0)
    return {
        "n_pairs": len(kept),
        "policy_mean": round(statistics.mean([policy[i][field] for i in kept]), 2),
        "rule_mean": round(rule_mean, 2),
        "delta": round(statistics.mean(d), 2),
        "delta_ci95": round(ci95(d), 2),
        "delta_pct": round(100 * statistics.mean(d) / rule_mean, 2),
        "boot_lo": round(lo, 2), "boot_hi": round(hi, 2),
        "wins": wins, "ties": ties, "losses": len(d) - wins - ties,
        "win_rate_pct": round(100 * wins / len(d), 1) if d else float("nan"),
        "perm_p": round(permutation_p(d), 4),
        "better": "policy" if statistics.mean(d) < 0 else "rule",
        "significant": "yes" if abs(statistics.mean(d)) > ci95(d) else "no",
    }


def seed_averaged(policy_by_seed: dict, rule: dict, field: str = "executed") -> dict:
    """Average over seeds WITHIN instance, then across instances.

    The correct paired estimate for "the policy family this recipe produces". Pooling
    seeds x instances would treat three correlated measurements of the same instance as
    independent samples.
    """
    inst = set.intersection(*[set(p) for p in policy_by_seed.values()]) & set(rule)
    kept = [i for i in sorted(inst)
            if rule[i]["nu"] == 0 and all(p[i]["nu"] == 0 for p in policy_by_seed.values())]
    if not kept:
        return {}
    d = [statistics.mean([p[i][field] for p in policy_by_seed.values()]) - rule[i][field]
         for i in kept]
    pol = [statistics.mean([p[i][field] for p in policy_by_seed.values()]) for i in kept]
    rule_mean = statistics.mean([rule[i][field] for i in kept])
    lo, hi = bootstrap_ci(d)
    wins = sum(1 for x in d if x < 0)
    return {
        "n_pairs": len(kept), "n_seeds": len(policy_by_seed),
        "policy_mean": round(statistics.mean(pol), 2), "rule_mean": round(rule_mean, 2),
        "delta": round(statistics.mean(d), 2), "delta_ci95": round(ci95(d), 2),
        "delta_pct": round(100 * statistics.mean(d) / rule_mean, 2),
        "boot_lo": round(lo, 2), "boot_hi": round(hi, 2),
        "wins": wins, "losses": len(d) - wins,
        "win_rate_pct": round(100 * wins / len(d), 1),
        "perm_p": round(permutation_p(d), 4),
        "significant": "yes" if abs(statistics.mean(d)) > ci95(d) else "no",
    }


def variance_decomposition(policy_by_seed: dict) -> dict:
    """Between-seed spread of mean makespan vs within-seed spread across instances."""
    per_seed = {}
    for key, by_inst in policy_by_seed.items():
        clean = [r["executed"] for r in by_inst.values() if r["nu"] == 0]
        per_seed[key] = statistics.mean(clean)
    means = list(per_seed.values())
    any_seed = next(iter(policy_by_seed.values()))
    within = statistics.stdev([r["executed"] for r in any_seed.values() if r["nu"] == 0])
    return {
        "seed_means": {k: round(v, 2) for k, v in per_seed.items()},
        "between_seed_sd": round(statistics.stdev(means), 2) if len(means) > 1 else 0.0,
        "between_seed_range": round(max(means) - min(means), 2),
        "within_seed_instance_sd": round(within, 2),
    }


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.name}")


def fmt_rules(ranked: list, title: str) -> str:
    lines = [title, "-" * len(title),
             f"  {'combo':>34} {'executed':>9} {'ideal':>8} {'Lambda':>8} {'clean':>9}"]
    for r in ranked:
        lines.append(f"  {r['combo']:>34} {r['executed']:>9.1f} {r['ideal']:>8.1f} "
                     f"{r['penalty_pct']:>7.2f}% "
                     f"{str(r['clean']) + '/' + str(r['instances']):>9}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_select() -> None:
    """val_60 only. Chooses the grad_clip arm and the comparison rule; reports nothing."""
    rules = load(RAW / "rules_val60_m16.jsonl")
    policy = load(RAW / "policy_val60_m16.jsonl")

    ranked = rank_rules(rule_table(rules, 16))
    out = [fmt_rules(ranked, "\nRun 0 -- dispatching rules on val_60 (50 instances, m=16)")]

    ptab = policy_table(policy, 16)
    rows = []
    for key in sorted(ptab):
        clean = [r for r in ptab[key].values() if r["nu"] == 0]
        rows.append({
            "run_key": key,
            "family": "v7" if key.startswith("v7_") else "v8",
            "arm": next(iter(ptab[key].values())).get("arm"),
            "seed": next(iter(ptab[key].values())).get("train_seed"),
            "ckpt_kind": next(iter(ptab[key].values())).get("ckpt_kind"),
            "executed": round(statistics.mean([r["executed"] for r in clean]), 2),
            "ideal": round(statistics.mean([r["ideal"] for r in clean]), 2),
            "penalty_pct": round(100 * statistics.mean([r["penalty"] for r in clean]), 2),
            "clean": len(clean), "instances": len(ptab[key]),
            "solve_s": round(statistics.mean([r["solve_s"] for r in ptab[key].values()]), 3),
        })
    rows.sort(key=lambda r: r["executed"])

    out.append("\n\nRun 1 -- every candidate checkpoint on val_60 (SELECTION SURFACE, not a result)")
    out.append("-" * 78)
    out.append(f"  {'run_key':>26} {'fam':>4} {'arm':>6} {'seed':>5} {'kind':>7} "
               f"{'executed':>9} {'ideal':>8} {'clean':>8} {'s/inst':>7}")
    for r in rows:
        out.append(f"  {r['run_key']:>26} {r['family']:>4} {str(r['arm']):>6} "
                   f"{str(r['seed']):>5} {r['ckpt_kind']:>7} {r['executed']:>9.2f} "
                   f"{r['ideal']:>8.1f} "
                   f"{str(r['clean']) + '/' + str(r['instances']):>8} {r['solve_s']:>7.2f}")

    # --- harness self-test: must reproduce the training logs exactly ---
    expected = {"ppo_gc1_s42_best": 344.38, "ppo_gc1.5_s44_best": 340.90,
                "ppo_gc1_s44_best": 335.38, "ppo_gc1.5_s42_best": 340.50}
    out.append("\n\nHARNESS SELF-TEST -- recomputed val vs the value logged during training")
    out.append("-" * 78)
    got = {r["run_key"]: r["executed"] for r in rows}
    failures = []
    for key, want in sorted(expected.items()):
        have = got.get(key)
        ok = have is not None and abs(have - want) < 0.005
        if not ok:
            failures.append(f"{key}: recomputed {have}, training logged {want}")
        out.append(f"  {key:>26}  logged {want:8.2f}  recomputed "
                   f"{have if have is not None else float('nan'):8.2f}  "
                   f"{'PASS' if ok else 'FAIL'}")

    # --- arm selection ---
    TIE_BAND = 2.0
    arms = defaultdict(list)
    for r in rows:
        if r["family"] == "v8" and r["ckpt_kind"] == "best" and r["arm"]:
            arms[r["arm"]].append(r["executed"])
    arm_scores = {a: statistics.mean(v) for a, v in arms.items()}
    best_arm = min(arm_scores, key=arm_scores.get)
    gap = abs(arm_scores["gc1"] - arm_scores["gc1.5"]) if len(arm_scores) == 2 else float("inf")
    tie = gap < TIE_BAND
    headline_arm = "gc1" if tie else best_arm

    out.append("\n\nARM SELECTION (pre-registered: argmin of mean val executed, "
               f"tie band {TIE_BAND} makespan units, tie-break = gc1)")
    out.append("-" * 78)
    for a in sorted(arm_scores):
        out.append(f"  A({a:>6}) = {arm_scores[a]:7.2f}   seeds {sorted(arms[a])}")
    out.append(f"  gap = {gap:.2f}  ->  {'TIE' if tie else 'DECISIVE'}")
    out.append(f"  headline arm = {headline_arm}"
               + ("  (tie-break; report both arms on test, label the other exploratory)" if tie else ""))
    between = statistics.stdev(arms[headline_arm]) if len(arms[headline_arm]) > 1 else 0.0
    out.append(f"  between-seed sd within {headline_arm} = {between:.2f} vs arm gap {gap:.2f} "
               f"-> grad_clip is {'NOT resolvable' if between > gap else 'resolvable'} at 3 seeds")

    baseline = ranked[0]["combo"]
    out.append(f"\n  comparison rule (val-selected) = {baseline}  "
               f"[val executed {ranked[0]['executed']:.1f}]")

    write_csv(HERE / "val_rules.csv", ranked)
    write_csv(HERE / "val_candidates.csv", rows)
    (HERE / "selection.json").write_text(json.dumps({
        "headline_arm": headline_arm, "arm_scores": arm_scores, "arm_gap": gap,
        "tie": tie, "tie_band": TIE_BAND, "between_seed_sd": between,
        "baseline_rule": baseline, "baseline_val_executed": ranked[0]["executed"],
        "self_test_failures": failures,
    }, indent=2) + "\n")

    text = "\n".join(out)
    (HERE / "selection_summary.txt").write_text(text + "\n")
    print(text)
    if failures:
        raise SystemExit("\nSELF-TEST FAILED -- the environment does not match training:\n  "
                         + "\n  ".join(failures))
    print("\nself-test passed; selection written to selection.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("select", "headline"), required=True)
    args = ap.parse_args()
    if args.stage == "select":
        stage_select()


if __name__ == "__main__":
    main()
