"""REPORTING STAGE -- the frozen val choice, scored on trend/full_{n}.

Reads `selection_val.json` (written by select_on_val.py, val_mix only) and the existing test
rows in `raw/rows.jsonl`. Nothing is chosen here; every argmin ran on the other split.

Three numbers per cell, kept apart because they answer different questions:

  PRIMARY     seed-averaged recipe vs val-selected rule. "How good is this training recipe,
              against the rule a practitioner would have picked in advance?" Seeds are
              averaged *within* instance before differencing -- 3 seeds x N instances are
              three correlated measurements of each instance, not 3N samples.
  DEPLOYED    val-selected seed vs val-selected rule. "What would I actually have shipped?"
              Legitimate, because the seed was chosen on val, but it is one draw and its
              gap to PRIMARY is the selection luck.
  ORACLE      best seed and best rule on test, i.e. what summarise.py's headline reports.
              Shown only as the bias it is: ORACLE - PRIMARY is what picking on the
              reporting set buys each side.

Statistics follow experiments/policy_vs_rules/analyze_policy.py: instance is the unit,
pairing is asserted on the instance id, only instances both sides routed cleanly (nu == 0)
enter a mean, and each delta gets a normal CI, a percentile bootstrap CI, win/loss counts
and a paired permutation test. Stdlib only.
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
SIZES = (20, 40, 60, 80, 100)
BOOTSTRAP_N = 10_000
PERMUTATION_N = 10_000

RECIPE_OF = {}
for _rec, _members in {
    "v9_only60": ["v9_only60_s42", "v9_only60_s43", "v9_only60_s44"],
    "v9_mix": ["v9_mix_s42", "v9_mix_s43", "v9_mix_s44"],
}.items():
    for _m in _members:
        RECIPE_OF[_m] = _rec


def load(path: Path) -> list:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def ci95(xs: list) -> float:
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def bootstrap_ci(xs: list, n: int = BOOTSTRAP_N, seed: int = 7) -> tuple:
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n)])


def permutation_p(xs: list, n: int = PERMUTATION_N, seed: int = 11) -> float:
    if len(xs) < 2:
        return float("nan")
    rng = random.Random(seed)
    obs = abs(statistics.mean(xs))
    hits = sum(1 for _ in range(n)
               if abs(sum(x if rng.random() < 0.5 else -x for x in xs) / len(xs)) >= obs)
    return (hits + 1) / (n + 1)


def by_instance(rows: list, **match) -> dict:
    out = {}
    for r in rows:
        if all(r.get(k) == v for k, v in match.items()):
            if r["instance"] in out:
                raise SystemExit(f"duplicate instance {r['instance']} for {match}")
            out[r["instance"]] = r
    return out


def paired(policy_sides: list, rule: dict, label: str) -> dict:
    """`policy_sides` is a list of per-instance dicts; they are averaged within instance
    first. One element = a single checkpoint, three = a seed-averaged recipe."""
    sets = [set(p) for p in policy_sides] + [set(rule)]
    common = set.intersection(*sets)
    union = set.union(*sets)
    if common != union:
        raise SystemExit(f"{label}: instance sets differ by {sorted(union - common)[:8]}")
    kept = [i for i in sorted(common)
            if rule[i]["nu"] == 0 and all(p[i]["nu"] == 0 for p in policy_sides)]
    if len(kept) < 2:
        return {}
    pol = [statistics.mean(p[i]["executed"] for p in policy_sides) for i in kept]
    ref = [rule[i]["executed"] for i in kept]
    d = [a - b for a, b in zip(pol, ref)]
    lo, hi = bootstrap_ci(d)
    wins = sum(1 for x in d if x < 0)
    ties = sum(1 for x in d if x == 0)
    mean_d = statistics.mean(d)
    return {
        "n_pairs": len(kept), "dropped": len(common) - len(kept), "n_seeds": len(policy_sides),
        "policy_mean": round(statistics.mean(pol), 2), "rule_mean": round(statistics.mean(ref), 2),
        "delta": round(mean_d, 2), "delta_ci95": round(ci95(d), 2),
        "delta_pct": round(100 * mean_d / statistics.mean(ref), 2),
        "boot_lo": round(lo, 2), "boot_hi": round(hi, 2),
        "wins": wins, "ties": ties, "losses": len(d) - wins - ties,
        "win_rate_pct": round(100 * wins / len(d), 1),
        "perm_p": round(permutation_p(d), 4),
        "significant": "yes" if abs(mean_d) > ci95(d) else "no",
    }


def oracle_rule(rows: list, n: int) -> tuple:
    """argmin over rule combos ON TEST -- the diagnostic, never the baseline."""
    by = defaultdict(list)
    for r in rows:
        if r["family"] == "rule" and r["n_jobs"] == n:
            by[r["method"]].append(r)
    scored = []
    for combo, rs in by.items():
        ok = [x["executed"] for x in rs if x["nu"] == 0]
        if ok:
            scored.append((statistics.mean(ok), combo))
    scored.sort()
    return scored[0][1], round(scored[0][0], 2)


def main() -> None:
    sel = json.loads((HERE / "selection_val.json").read_text())
    rows = [r for r in load(RAW / "rows.jsonl") if r.get("split", "test") == "test"]

    out_rows, lines = [], []
    P = lines.append
    P("=" * 118)
    P("VAL-SELECTED BASELINE vs POLICY -- reported on test_case/v3/trend/full_{n}.jsonl, "
      "100 instances/size, m=16")
    P("Rule and seed both chosen on val_mix (40 instances/size, disjoint). "
      "selection_val.json is the frozen choice.")
    P("=" * 118)

    for mode in ("greedy", "best8"):
        P("")
        P("")
        P(f"### policy mode = {mode}")
        for n in SIZES:
            e = sel["by_size"][str(n)]
            pol_sel = e["policy"].get(mode)
            if not pol_sel:
                continue
            rule_combo = e["rule"]["combo"]
            rule = by_instance(rows, family="rule", method=rule_combo, n_jobs=n)
            if not rule:
                raise SystemExit(f"val-selected rule {rule_combo} has no test rows at n={n}")

            picked = pol_sel["best_model"]
            recipe = RECIPE_OF.get(picked)
            members = ([m for m, r in RECIPE_OF.items() if r == recipe] if recipe
                       else [picked])
            sides = [by_instance(rows, family="policy", method=m, mode=mode, n_jobs=n)
                     for m in sorted(members)]

            primary = paired(sides, rule, f"n={n} {mode} PRIMARY")
            deployed = paired([by_instance(rows, family="policy", method=picked,
                                           mode=mode, n_jobs=n)], rule,
                              f"n={n} {mode} DEPLOYED")

            # --- oracle diagnostics: what picking on test would have bought each side
            o_combo, o_val = oracle_rule(rows, n)
            o_rule = by_instance(rows, family="rule", method=o_combo, n_jobs=n)
            seed_means = {}
            for m in sorted(members):
                d = by_instance(rows, family="policy", method=m, mode=mode, n_jobs=n)
                ok = [r["executed"] for r in d.values() if r["nu"] == 0]
                seed_means[m] = round(statistics.mean(ok), 2)
            best_seed_on_test = min(seed_means, key=seed_means.get)
            oracle_pair = paired([by_instance(rows, family="policy", method=best_seed_on_test,
                                              mode=mode, n_jobs=n)], o_rule,
                                 f"n={n} {mode} ORACLE")

            P("")
            P(f"n={n:<4} val-selected rule = {rule_combo}   "
              f"(val {e['rule']['val_executed']:.1f}, chosen from {e['rule']['n_candidates']})")
            P(f"      val-selected model = {picked}"
              + (f"   recipe = {recipe} ({len(members)} seeds)" if recipe else "   (single seed)"))
            P("-" * 118)
            P(f"      {'row':<12}{'policy':>9}{'rule':>9}{'delta':>19}"
              f"{'bootstrap 95%':>19}{'win%':>7}{'perm p':>9}{'n':>5}{'drop':>6}")
            for tag, st in (("PRIMARY", primary), ("DEPLOYED", deployed)):
                if not st:
                    continue
                P(f"      {tag:<12}{st['policy_mean']:>9.1f}{st['rule_mean']:>9.1f}"
                  f"{st['delta']:>11.2f} +/-{st['delta_ci95']:<5.2f}"
                  f"[{st['boot_lo']:>8.2f},{st['boot_hi']:>8.2f}]"
                  f"{st['win_rate_pct']:>7.1f}{st['perm_p']:>9.4f}"
                  f"{st['n_pairs']:>5}{st['dropped']:>6}")
                out_rows.append({"n_jobs": n, "mode": mode, "row": tag,
                                 "rule": rule_combo, "model": picked,
                                 "recipe": recipe or picked, **st})
            if primary:
                P(f"      -> {primary['delta_pct']:+.2f}% of rule makespan, "
                  f"{'SIGNIFICANT' if primary['significant'] == 'yes' else 'not significant'}"
                  f"  ({primary['wins']}W/{primary['ties']}T/{primary['losses']}L)")
            P(f"      seeds on test: "
              + "  ".join(f"{k.split('_')[-1]}={v:.1f}" for k, v in seed_means.items()))
            if oracle_pair:
                # Both quantities are "makespan units that picking on the reporting set
                # buys that side", so they are directly comparable to each other and to
                # the PRIMARY delta they would have inflated.
                rule_regret = round(primary["rule_mean"] - o_val, 2) if primary else None
                seed_gain = (round(primary["policy_mean"] - seed_means[best_seed_on_test], 2)
                             if primary else None)
                P(f"      ORACLE (summarise.py headline): best-seed {best_seed_on_test} "
                  f"{seed_means[best_seed_on_test]:.1f} vs best-on-test rule {o_combo} {o_val:.1f}"
                  f"  -> delta {oracle_pair['delta']:+.2f}")
                if primary:
                    P(f"      picking on test would buy: rule {rule_regret:.2f} units "
                      f"(val-selection regret), policy {seed_gain:.2f} units (seed cherry-pick)"
                      f"  | PRIMARY delta is {abs(primary['delta']):.2f}")
                out_rows.append({"n_jobs": n, "mode": mode, "row": "ORACLE",
                                 "rule": o_combo, "model": best_seed_on_test,
                                 "recipe": recipe or picked,
                                 "rule_selection_regret": rule_regret,
                                 "seed_cherry_pick_gain": seed_gain, **oracle_pair})

    text = "\n".join(lines)
    (HERE / "val_selected_summary.txt").write_text(text + "\n")
    cols = ["n_jobs", "mode", "row", "rule", "model", "recipe", "n_seeds", "policy_mean",
            "rule_mean", "delta", "delta_ci95", "delta_pct", "boot_lo", "boot_hi",
            "wins", "ties", "losses", "win_rate_pct", "perm_p", "significant",
            "n_pairs", "dropped", "rule_selection_regret", "seed_cherry_pick_gain"]
    with (HERE / "val_selected.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(text)
    print(f"\n-> {HERE / 'val_selected.csv'}")
    print(f"-> {HERE / 'val_selected_summary.txt'}")


if __name__ == "__main__":
    main()
