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
    """Union of keys across rows, in first-seen order.

    Per-seed and seed-averaged rows carry different fields (the latter has n_seeds and no
    tie/better columns), so taking fieldnames from rows[0] would raise on the second kind.
    """
    if not rows:
        return
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
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


def _load_selection() -> dict:
    path = HERE / "selection.json"
    if not path.exists():
        raise SystemExit("run --stage select first; selection.json is missing")
    return json.loads(path.read_text())


def stage_headline() -> None:
    """test_60. Everything here is a reported number."""
    sel = _load_selection()
    arm, baseline = sel["headline_arm"], sel["baseline_rule"]
    rules16 = rule_table(load(CONGESTION / "final_fleet_test60.jsonl"), 16)
    ranked = rank_rules(rules16)
    strongest = ranked[0]["combo"]

    pol = policy_table(load(RAW / "policy_test60_m16.jsonl"), 16)
    out = [f"\nHEADLINE -- test_60, 100 instances, m=16, greedy, no local improvement",
           f"pre-registered arm = {arm} | val-selected rule = {baseline}",
           f"strongest rule on test = {strongest} ({ranked[0]['executed']:.1f})", "=" * 96]

    tables = {}
    for rule_combo, tag in ((baseline, "val-selected (PRIMARY)"), (strongest, "strongest on test")):
        rule = rules16[tuple(rule_combo.split("+", 1))]
        out.append(f"\nvs {rule_combo}  [{tag}]")
        out.append(f"  {'run_key':>24} {'policy':>8} {'rule':>8} {'delta':>17} "
                   f"{'bootstrap 95%':>17} {'win%':>6} {'perm p':>8} {'n':>5} {'nu>0':>6}")
        rows = []
        for key in sorted(pol):
            if f"_{arm}_" not in key or not key.endswith("_best"):
                continue
            kept, dropped = pair(pol[key], rule, key)
            st = paired_stats(pol[key], rule, kept)
            p_nu = sum(1 for r in pol[key].values() if r["nu"] > 0)
            r_nu = sum(1 for r in rule.values() if r["nu"] > 0)
            rows.append({"run_key": key, "rule": rule_combo, "tag": tag,
                         "dropped": len(dropped), "policy_unroutable": p_nu,
                         "rule_unroutable": r_nu, **st})
            out.append(f"  {key:>24} {st['policy_mean']:>8.1f} {st['rule_mean']:>8.1f} "
                       f"{st['delta']:>+8.2f} +/-{st['delta_ci95']:<5.2f} "
                       f"[{st['boot_lo']:>+7.2f},{st['boot_hi']:>+7.2f}] "
                       f"{st['win_rate_pct']:>5.1f}% {st['perm_p']:>8.4f} {st['n_pairs']:>5} "
                       f"{str(p_nu) + '/' + str(r_nu):>6}")

        by_seed = {k: pol[k] for k in pol if f"_{arm}_" in k and k.endswith("_best")}
        agg = seed_averaged(by_seed, rule)
        if agg:
            rows.append({"run_key": f"SEED-AVERAGED ({agg['n_seeds']} seeds)", "rule": rule_combo,
                         "tag": tag, "dropped": 0, **agg})
            out.append(f"  {'SEED-AVERAGED':>24} {agg['policy_mean']:>8.1f} {agg['rule_mean']:>8.1f} "
                       f"{agg['delta']:>+8.2f} +/-{agg['delta_ci95']:<5.2f} "
                       f"[{agg['boot_lo']:>+7.2f},{agg['boot_hi']:>+7.2f}] "
                       f"{agg['win_rate_pct']:>5.1f}% {agg['perm_p']:>8.4f} {agg['n_pairs']:>5}")
            out.append(f"  -> {agg['delta_pct']:+.2f}% of rule makespan, "
                       f"{'SIGNIFICANT' if agg['significant'] == 'yes' else 'not significant'}")
        tables[tag] = rows

    # seed variance vs treatment effect
    by_seed = {k: pol[k] for k in pol if f"_{arm}_" in k and k.endswith("_best")}
    var = variance_decomposition(by_seed)
    out.append("  nu>0 column is policy/rule: instances the executor could not route. "
               "A policy explores a wider")
    out.append("  schedule distribution than a dispatching rule, so this is not "
               "expected to be symmetric.")

    out.append(f"\nSEED VARIANCE ({arm} arm)")
    out.append("-" * 96)
    for k, v in sorted(var["seed_means"].items()):
        out.append(f"  {k:>24} {v:>8.2f}")
    out.append(f"  between-seed sd {var['between_seed_sd']:.2f} | range {var['between_seed_range']:.2f} "
               f"| within-seed instance sd {var['within_seed_instance_sd']:.2f}")

    # post-hoc: other arm, and _best vs _latest
    out.append("\nPOST-HOC (examined after test rows existed; 2 extra comparisons)")
    out.append("-" * 96)
    rule = rules16[tuple(baseline.split("+", 1))]
    posthoc = []
    for label, pred in (("other arm gc1.5 _best", lambda k: "_gc1.5_" in k and k.endswith("_best")),
                        (f"{arm} _latest", lambda k: f"_{arm}_" in k and k.endswith("_latest"))):
        sub = {k: pol[k] for k in pol if pred(k)}
        if not sub:
            continue
        agg = seed_averaged(sub, rule)
        if agg:
            posthoc.append({"label": label, "rule": baseline, **agg})
            out.append(f"  {label:>24} policy {agg['policy_mean']:>7.1f} "
                       f"delta {agg['delta']:>+7.2f} +/-{agg['delta_ci95']:<5.2f} "
                       f"win {agg['win_rate_pct']:>5.1f}%  ({agg['n_seeds']} seeds)")

    write_csv(HERE / "headline.csv", tables["val-selected (PRIMARY)"] + tables["strongest on test"])
    write_csv(HERE / "posthoc.csv", posthoc)
    write_csv(HERE / "by_seed.csv", [{"arm": arm, "run_key": k, "mean_executed": v,
                                      **{kk: vv for kk, vv in var.items() if kk != "seed_means"}}
                                     for k, v in sorted(var["seed_means"].items())])
    text = "\n".join(out)
    (HERE / "headline_summary.txt").write_text(text + "\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",
                    choices=("select", "headline", "robustness", "scaling", "v7"),
                    required=True)
    args = ap.parse_args()
    if args.stage == "select":
        stage_select()
    elif args.stage == "headline":
        stage_headline()
    elif args.stage == "robustness":
        stage_robustness()
    elif args.stage == "scaling":
        stage_scaling()
    elif args.stage == "v7":
        stage_v7()




def stage_robustness() -> None:
    """Leave-one-seed-out and the shape of the per-instance delta distribution.

    Two things the headline mean hides. First, whether the result depends on a single seed:
    with 3 seeds and a between-seed sd comparable to the effect, one lucky run can carry the
    average. Second, whether the policy is uniformly better or merely higher-variance -- a
    mean improvement built from rare large wins is a different claim from one built from
    consistent small ones, and only the second would survive a risk-averse deployment.
    """
    sel = _load_selection()
    arm, baseline = sel["headline_arm"], sel["baseline_rule"]
    rules = rule_table(load(CONGESTION / "final_fleet_test60.jsonl"), 16)
    strongest = rank_rules(rules)[0]["combo"]
    pol = policy_table(load(RAW / "policy_test60_m16.jsonl"), 16)
    keys = [k for k in sorted(pol) if f"_{arm}_" in k and k.endswith("_best")]

    out, rows = ["\nROBUSTNESS -- test_60, m=16"], []
    for combo, tag in ((baseline, "val-selected (PRIMARY)"), (strongest, "strongest on test")):
        rule = rules[tuple(combo.split("+", 1))]
        out.append(f"\nvs {combo}  [{tag}]")
        out.append("  leave-one-seed-out:")
        for drop in keys:
            sub = [k for k in keys if k != drop]
            inst = [i for i in sorted(rule)
                    if rule[i]["nu"] == 0 and all(pol[k][i]["nu"] == 0 for k in sub)]
            d = [statistics.mean([pol[k][i]["executed"] for k in sub]) - rule[i]["executed"]
                 for i in inst]
            m, h, p = statistics.mean(d), ci95(d), permutation_p(d)
            sig = abs(m) > h
            rows.append({"rule": combo, "tag": tag, "dropped_seed": drop, "n": len(d),
                         "delta": round(m, 2), "ci95": round(h, 2), "perm_p": round(p, 4),
                         "significant": "yes" if sig else "no"})
            out.append(f"    without {drop:>24}: {m:>+7.2f} +/-{h:<5.2f} "
                       f"p={p:.4f}  {'SIG' if sig else 'ns'}")

        inst = [i for i in sorted(rule)
                if rule[i]["nu"] == 0 and all(pol[k][i]["nu"] == 0 for k in keys)]
        d = sorted(statistics.mean([pol[k][i]["executed"] for k in keys]) - rule[i]["executed"]
                   for i in inst)
        n = len(d)
        wins = [x for x in d if x < 0]
        losses = [x for x in d if x > 0]
        out.append(f"  delta distribution (n={n}): min {d[0]:+.1f} | p10 {d[n // 10]:+.1f} "
                   f"| median {statistics.median(d):+.1f} | p90 {d[9 * n // 10]:+.1f} "
                   f"| max {d[-1]:+.1f}")
        out.append(f"  wins n={len(wins)} mean {statistics.mean(wins):+.1f} | "
                   f"losses n={len(losses)} mean {statistics.mean(losses):+.1f}")
        rows.append({"rule": combo, "tag": tag, "dropped_seed": "(none) distribution",
                     "n": n, "delta": round(statistics.mean(d), 2),
                     "median": round(statistics.median(d), 2),
                     "p10": round(d[n // 10], 2), "p90": round(d[9 * n // 10], 2),
                     "win_mean": round(statistics.mean(wins), 2), "n_wins": len(wins),
                     "loss_mean": round(statistics.mean(losses), 2), "n_losses": len(losses)})

    write_csv(HERE / "robustness.csv", rows)
    text = "\n".join(out)
    (HERE / "robustness_summary.txt").write_text(text + "\n")
    print(text)


def stage_scaling() -> None:
    """Zero-shot parcel count: does a policy trained only at n=60 hold at 2x and 4x?"""
    sel = _load_selection()
    arm, baseline = sel["headline_arm"], sel["baseline_rule"]
    combo = tuple(baseline.split("+", 1))
    sources = {
        60: (RAW / "policy_test60_m16.jsonl", CONGESTION / "final_fleet_test60.jsonl"),
        120: (RAW / "policy_test120_m16.jsonl", CONGESTION / "parcels_120_m16.jsonl"),
        240: (RAW / "policy_test240_m16.jsonl", CONGESTION / "parcels_240_m16.jsonl"),
    }
    out = ["\nZERO-SHOT PARCEL COUNT -- m=16, policy trained only at n=60",
           f"baseline rule = {baseline}", "-" * 92,
           f"  {'n':>5} {'n/robot':>8} {'policy':>9} {'rule':>9} {'delta':>17} "
           f"{'delta%':>8} {'win%':>6} {'perm p':>8} {'nu>0':>6}"]
    rows = []
    for n, (ppath, rpath) in sources.items():
        if not ppath.exists():
            out.append(f"  {n:>5}  (missing {ppath.name})")
            continue
        rule = rule_table(load(rpath), 16)[combo]
        pol = policy_table(load(ppath), 16)
        by_seed = {k: pol[k] for k in pol if f"_{arm}_" in k and k.endswith("_best")}
        if not by_seed:
            continue
        agg = seed_averaged(by_seed, rule)
        p_nu = sum(1 for p in by_seed.values() for r in p.values() if r["nu"] > 0)
        rows.append({"n_jobs": n, "parcels_per_robot": round(n / 16, 2),
                     "policy_unroutable": p_nu, **agg})
        out.append(f"  {n:>5} {n / 16:>8.2f} {agg['policy_mean']:>9.1f} {agg['rule_mean']:>9.1f} "
                   f"{agg['delta']:>+8.2f} +/-{agg['delta_ci95']:<5.2f} "
                   f"{agg['delta_pct']:>+7.2f}% {agg['win_rate_pct']:>5.1f}% "
                   f"{agg['perm_p']:>8.4f} {p_nu:>6}")
    write_csv(HERE / "parcel_scaling.csv", rows)
    text = "\n".join(out)
    (HERE / "scaling_summary.txt").write_text(text + "\n")
    print(text)


def stage_v7() -> None:
    """PPO vs REINFORCE at a matched 2000-epoch budget, plus 2000 vs 4000 epochs of PPO.

    stepwise_s42 and stepwise_clip50_s42 are deliberately absent: those runs were killed at
    epoch 528 and 536 of 2000, so including them would understate REINFORCE and inflate the
    apparent advantage of PPO.
    """
    sel = _load_selection()
    arm, baseline = sel["headline_arm"], sel["baseline_rule"]
    rule = rule_table(load(CONGESTION / "final_fleet_test60.jsonl"), 16)[tuple(baseline.split("+", 1))]

    v7 = policy_table(load(RAW / "policy_test60_v7.jsonl"), 16)
    v8 = policy_table(load(RAW / "policy_test60_m16.jsonl"), 16)
    cand = {k: v for k, v in v7.items()}
    cand[f"v8_ppo_{arm}_s42_best (4000ep)"] = v8[f"ppo_{arm}_s42_best"]

    out = ["\nPPO vs REINFORCE -- test_60, m=16, seed 42, greedy, no local improvement",
           f"baseline rule = {baseline}", "-" * 96,
           f"  {'checkpoint':>30} {'method':>10} {'epochs':>7} {'policy':>8} "
           f"{'delta':>17} {'win%':>6} {'perm p':>8} {'nu>0':>5}"]
    meta = {
        "v7_ppo_s42_best": ("ppo", 2000),
        "v7_episode_s42_best": ("reinforce/episode", 2000),
        "v7_clip50_s42_best": ("reinforce/multisample", 2000),
        f"v8_ppo_{arm}_s42_best (4000ep)": ("ppo", 4000),
    }
    rows = []
    for key in sorted(cand, key=lambda k: statistics.mean(
            [r["executed"] for r in cand[k].values() if r["nu"] == 0])):
        kept, _ = pair(cand[key], rule, key)
        st = paired_stats(cand[key], rule, kept)
        method, epochs = meta.get(key, ("?", 0))
        nu = sum(1 for r in cand[key].values() if r["nu"] > 0)
        rows.append({"checkpoint": key, "method": method, "epochs": epochs,
                     "unroutable": nu, **st})
        out.append(f"  {key:>30} {method:>10} {epochs:>7} {st['policy_mean']:>8.1f} "
                   f"{st['delta']:>+8.2f} +/-{st['delta_ci95']:<5.2f} "
                   f"{st['win_rate_pct']:>5.1f}% {st['perm_p']:>8.4f} {nu:>5}")
    out.append("  stepwise_s42 / stepwise_clip50_s42 excluded: killed at epoch 528/536 of 2000.")
    write_csv(HERE / "ppo_vs_reinforce.csv", rows)
    text = "\n".join(out)
    (HERE / "v7_summary.txt").write_text(text + "\n")
    print(text)



if __name__ == "__main__":
    main()
