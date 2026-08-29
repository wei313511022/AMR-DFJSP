"""Matched-seed variant: every recipe evaluated at ONE seed, against the val-selected rule.

Motivation is comparability across recipes, not statistics. The v10 cells were only ever
trained at s44, so putting a 3-seed v9 average next to a 1-seed v10 number compares a recipe
mean against a single draw. Holding the seed fixed at s44 makes every row the same kind of
object; the price is that no row carries any between-seed information at all.

This does NOT replace compare_selected.py. Fixing the seed a priori is defensible only
because s44 is the seed v10 happens to have -- it is a matching constraint, not a
performance claim. Whether it flatters the policy is checked, not assumed: the script prints
where the fixed seed ranked among its recipe's seeds ON VAL, and refuses to stay quiet if it
was the val-best everywhere (which would make the "matched, not chosen" story false).

The dispatching-rule baseline is unchanged -- still the combination frozen in
selection_val.json by select_on_val.py, chosen on val_mix before any test row was read.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from compare_selected import RAW, SIZES, by_instance, load, oracle_rule, paired

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="s44", help="seed tag every recipe is held at")
    args = ap.parse_args()
    tag = f"_{args.seed}"

    sel = json.loads((HERE / "selection_val.json").read_text())
    rows = [r for r in load(RAW / "rows.jsonl") if r.get("split", "test") == "test"]

    out_rows, lines = [], []
    P = lines.append
    P("=" * 118)
    P(f"MATCHED-SEED COMPARISON -- every recipe at {args.seed}, vs the val-selected rule")
    P("Reported on test_case/v3/trend/full_{n}.jsonl, 100 instances/size, m=16.")
    P("Rule frozen in selection_val.json (chosen on val_mix). One seed per row: no seed")
    P("averaging is possible here, so these rows carry no between-seed information.")
    P("=" * 118)

    # --- where did the fixed seed rank on val? the honesty check -----------------
    P("")
    P(f"Rank of {args.seed} among its recipe's seeds ON VAL (1 = best; 3 = worst)")
    P("-" * 118)
    P(f"{'mode':>8}{'n=20':>10}{'n=40':>10}{'n=60':>10}{'n=80':>10}{'n=100':>10}")
    was_always_best = True
    for mode in ("greedy", "best8"):
        cells = []
        for n in SIZES:
            rec = sel["by_size"][str(n)]["policy"][mode]["by_recipe"].get("v9_only60", {})
            order = list(rec.get("seeds", {}))
            if not order:
                cells.append("-")
                continue
            rank = order.index(f"v9_only60{tag}") + 1
            cells.append(f"{rank}/{len(order)}")
            if rank != 1:
                was_always_best = False
        P(f"{mode:>8}" + "".join(f"{c:>10}" for c in cells))
    P("")
    if was_always_best:
        P(f"!! {args.seed} was the val-best seed at every size. The 'matched, not chosen'")
        P("!! justification does not hold here -- report this as a seed selection instead.")
    else:
        P(f"-> {args.seed} is not uniformly the val-best seed, so holding it fixed is a")
        P("   matching constraint rather than a pick in the policy's favour.")

    for mode in ("greedy", "best8"):
        P("")
        P("")
        P(f"### policy mode = {mode}")
        for n in SIZES:
            e = sel["by_size"][str(n)]
            pol = e["policy"].get(mode)
            if not pol:
                continue
            rule_combo = e["rule"]["combo"]
            rule = by_instance(rows, family="rule", method=rule_combo, n_jobs=n)

            pool = [d for d in pol["ranking"] if tag in d["model"]]
            if not pool:
                raise SystemExit(f"no {args.seed} models at n={n} {mode}")
            val_pick = pool[0]["model"]

            rule_test = statistics.mean(
                [r["executed"] for r in rule.values() if r["nu"] == 0])
            P("")
            P(f"n={n:<4} val-selected rule = {rule_combo}  (test {rule_test:.1f})"
              f"   |  best {args.seed} model on val = {val_pick}")
            P("-" * 118)
            P(f"      {'model':<24}{'policy':>9}{'rule':>9}{'delta':>19}"
              f"{'bootstrap 95%':>19}{'win%':>7}{'perm p':>9}{'n':>5}")
            for cand in pool:
                name = cand["model"]
                side = by_instance(rows, family="policy", method=name, mode=mode, n_jobs=n)
                st = paired([side], rule, f"n={n} {mode} {name}")
                if not st:
                    continue
                mark = " <- val pick" if name == val_pick else ""
                P(f"      {name:<24}{st['policy_mean']:>9.1f}{st['rule_mean']:>9.1f}"
                  f"{st['delta']:>11.2f} +/-{st['delta_ci95']:<5.2f}"
                  f"[{st['boot_lo']:>8.2f},{st['boot_hi']:>8.2f}]"
                  f"{st['win_rate_pct']:>7.1f}{st['perm_p']:>9.4f}{st['n_pairs']:>5}{mark}")
                out_rows.append({"n_jobs": n, "mode": mode, "seed": args.seed,
                                 "model": name, "rule": rule_combo,
                                 "is_val_pick": "yes" if name == val_pick else "no", **st})

            # what the fixed seed costs against its own recipe average, where one exists
            members = [d["model"] for d in pol["ranking"]
                       if d["model"].startswith("v9_only60_")]
            if len(members) > 1:
                sides = [by_instance(rows, family="policy", method=m, mode=mode, n_jobs=n)
                         for m in sorted(members)]
                avg = paired(sides, rule, f"n={n} {mode} recipe-avg")
                one = paired([by_instance(rows, family="policy", method=f"v9_only60{tag}",
                                          mode=mode, n_jobs=n)], rule, f"n={n} {mode} one")
                if avg and one:
                    P(f"      v9_only60: {args.seed} alone {one['delta']:+.2f} vs "
                      f"3-seed average {avg['delta']:+.2f}"
                      f"  -> fixing the seed shifts the delta by {one['delta'] - avg['delta']:+.2f}")

    text = "\n".join(lines)
    (HERE / f"fixed_{args.seed}_summary.txt").write_text(text + "\n")
    cols = ["n_jobs", "mode", "seed", "model", "rule", "is_val_pick", "policy_mean",
            "rule_mean", "delta", "delta_ci95", "delta_pct", "boot_lo", "boot_hi",
            "wins", "ties", "losses", "win_rate_pct", "perm_p", "significant", "n_pairs"]
    with (HERE / f"fixed_{args.seed}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(text)
    print(f"\n-> {HERE / f'fixed_{args.seed}.csv'}")
    print(f"-> {HERE / f'fixed_{args.seed}_summary.txt'}")


if __name__ == "__main__":
    main()
