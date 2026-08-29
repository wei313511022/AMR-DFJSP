"""SELECTION STAGE -- freezes every choice on val_mix. Nothing here may be reported.

The full_benchmark headline (summarise.py:57) takes `min(..., key=makespan_mean)` over the
rule field AND over the policy seeds, both on the reporting set. Two problems:

  * a minimum over 60 rule combinations on test is an optimistic order statistic, not the
    score of a rule anyone could have chosen in advance;
  * a minimum over 3 seeds on test is the same move applied to the policy, and at n=60 in
    experiments/policy_vs_rules it is worth 8.2 makespan units -- larger than the entire
    effect being claimed there (-6.94 vs the strongest test rule).

This script makes both choices on `test_case/v3/mix/val_mix_{n}.jsonl` instead: 40 instances
per size, disjoint from train_mix and from trend/full_{n} (asserted in run_val_selection.sh),
and already the validation set every v9/v10 checkpoint was selected against
(launch_v9.sh:23, launch_v10_2x3.sh:46). No dispatching rule has ever seen it.

One caveat this cannot remove, and which is why val scores are never reported: each
`_best.pth` is itself an argmin over many val_mix evaluations, so a policy's val score is an
optimistic order statistic on this set. That is fine for *ranking* candidates -- it is what a
validation set is for -- but it makes the val number useless as a generalisation estimate.

Output `selection_val.json` is the frozen input to compare_selected.py.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
SIZES = (20, 40, 60, 80, 100)

# Rules within this many makespan units of the leader are treated as indistinguishable on 40
# instances; the tie is broken alphabetically so the choice does not depend on row order.
# Reported alongside the pick so the reader can see how wide the band was.
RULE_TIE_BAND = 0.5

# A "recipe" is a training configuration; its seeds are interchangeable draws from it.
# Selecting among seeds and selecting among recipes are different claims, so they are
# recorded separately.
RECIPES = {
    "v9_only60": ["v9_only60_s42", "v9_only60_s43", "v9_only60_s44"],
    "v9_mix": ["v9_mix_s42", "v9_mix_s43", "v9_mix_s44"],
    "v10_A_s44": ["v10_A_s44"],
    "v10_B_s44": ["v10_B_s44"],
    "v10_C_s44": ["v10_C_s44"],
    "v10_A_s44_bestideal": ["v10_A_s44_bestideal"],
    "v10_B_s44_bestideal": ["v10_B_s44_bestideal"],
}


def load(path: Path) -> list:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def mean_executed(rows: list) -> tuple:
    """Mean over instances the executor routed cleanly, plus the unroutable count.

    Averaging executed makespan over an unroutable instance would mix a failure into a
    timing mean; analyze.py and analyze_policy.py both drop them, so this does too.
    """
    ok = [r["executed"] for r in rows if r["nu"] == 0]
    if not ok:
        return (float("inf"), len(rows), 0)
    return (statistics.mean(ok), len(rows) - len(ok), len(ok))


def rule_ranking(rows: list, n: int) -> list:
    by = defaultdict(list)
    for r in rows:
        if r["family"] == "rule" and r["n_jobs"] == n:
            by[r["method"]].append(r)
    out = []
    for combo, rs in by.items():
        m, nu, k = mean_executed(rs)
        out.append({"combo": combo, "val_executed": round(m, 2), "unroutable": nu,
                    "instances": len(rs), "clean": k})
    out.sort(key=lambda d: (d["val_executed"], d["combo"]))
    return out


def policy_ranking(rows: list, n: int, mode: str) -> list:
    by = defaultdict(list)
    for r in rows:
        if r["family"] == "policy" and r["n_jobs"] == n and r["mode"] == mode:
            by[r["method"]].append(r)
    out = []
    for name, rs in by.items():
        m, nu, k = mean_executed(rs)
        out.append({"model": name, "val_executed": round(m, 2), "unroutable": nu,
                    "instances": len(rs), "clean": k})
    out.sort(key=lambda d: (d["val_executed"], d["model"]))
    return out


def pick_rule(ranked: list) -> dict:
    best = ranked[0]["val_executed"]
    band = [d for d in ranked if d["val_executed"] <= best + RULE_TIE_BAND]
    return {
        "combo": sorted(d["combo"] for d in band)[0],
        "val_executed": best,
        "tie_band": RULE_TIE_BAND,
        "tied_with": sorted(d["combo"] for d in band),
        "runner_up_gap": round(ranked[1]["val_executed"] - best, 2) if len(ranked) > 1 else None,
        "n_candidates": len(ranked),
    }


def pick_seed(ranked: list, recipe: str) -> dict:
    """argmin over the seeds of one recipe. Between-seed sd is reported next to the gap so
    a pick made inside the noise band is visible as such."""
    members = {d["model"]: d for d in ranked if d["model"] in RECIPES[recipe]}
    if not members:
        return {}
    order = sorted(members.values(), key=lambda d: (d["val_executed"], d["model"]))
    vals = [d["val_executed"] for d in order]
    return {
        "recipe": recipe,
        "seeds": {d["model"]: d["val_executed"] for d in order},
        "picked": order[0]["model"],
        "val_executed": order[0]["val_executed"],
        "between_seed_sd": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
        "gap_to_runner_up": round(vals[1] - vals[0], 2) if len(vals) > 1 else None,
    }


def main() -> None:
    rows = load(RAW / "rows_val.jsonl")
    bad = [r for r in rows if r.get("split") != "val"]
    if bad:
        raise SystemExit(f"rows_val.jsonl contains {len(bad)} non-val rows -- refusing")

    sel = {"split": "val", "dataset": "test_case/v3/mix/val_mix_{n}.jsonl",
           "rule_tie_band": RULE_TIE_BAND, "sizes": list(SIZES), "by_size": {}}

    for n in SIZES:
        rr = rule_ranking(rows, n)
        if not rr:
            raise SystemExit(f"no rule rows at n={n}")
        entry = {"rule": pick_rule(rr), "rule_ranking": rr, "policy": {}}
        for mode in ("greedy", "best8"):
            pr = policy_ranking(rows, n, mode)
            if not pr:
                continue
            entry["policy"][mode] = {
                "ranking": pr,
                "best_model": pr[0]["model"],
                "best_val_executed": pr[0]["val_executed"],
                "by_recipe": {rec: pick_seed(pr, rec) for rec in RECIPES
                              if any(d["model"] in RECIPES[rec] for d in pr)},
            }
        sel["by_size"][str(n)] = entry

    out = HERE / "selection_val.json"
    out.write_text(json.dumps(sel, indent=2) + "\n")

    # ---- human-readable echo -------------------------------------------------
    lines = []
    P = lines.append
    P("=" * 96)
    P("SELECTION ON val_mix -- 40 instances/size, m=16. NOT RESULTS.")
    P("=" * 96)
    P("")
    P(f"{'n':>4}  {'val-selected rule':<44}{'val':>9}{'gap to #2':>11}{'tied':>6}{'of':>5}")
    P("-" * 96)
    for n in SIZES:
        e = sel["by_size"][str(n)]["rule"]
        gap = e["runner_up_gap"]
        P(f"{n:>4}  {e['combo']:<44}{e['val_executed']:>9.1f}"
          f"{(gap if gap is not None else float('nan')):>11.2f}"
          f"{len(e['tied_with']):>6}{e['n_candidates']:>5}")
    P("")
    for mode in ("greedy", "best8"):
        P("")
        P(f"val-selected policy -- {mode}")
        P("-" * 96)
        P(f"{'n':>4}  {'best model on val':<28}{'val':>9}   "
          f"{'v9_only60 seed pick':<22}{'seeds on val':>34}{'sd':>7}")
        for n in SIZES:
            pol = sel["by_size"][str(n)]["policy"].get(mode)
            if not pol:
                continue
            rec = pol["by_recipe"].get("v9_only60", {})
            seeds = rec.get("seeds", {})
            seedtxt = " ".join(f"{k.split('_')[-1]}={v:.1f}" for k, v in seeds.items())
            P(f"{n:>4}  {pol['best_model']:<28}{pol['best_val_executed']:>9.1f}   "
              f"{rec.get('picked', '-'):<22}{seedtxt:>34}{rec.get('between_seed_sd', 0):>7.2f}")
    P("")
    P(f"frozen -> {out}")
    text = "\n".join(lines)
    (HERE / "selection_val_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
