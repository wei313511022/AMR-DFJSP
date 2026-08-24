"""Full run: makespan, runtime and deviation for every scheduler at n = 20..100.

Schedulers
----------
  6 dispatching-rule combinations, chosen on the 10-instance grid in `analyze_trend.py`
  by mean relative regret across all five sizes (see `selected_combos.json`). The set
  contains the per-size winner at every size, which is the property that matters when the
  comparison spans a range over which the rule ranking moves.

  extend_GNN + PPO, `ppo_gc1_s{42,43,44}_best`, three decode budgets:
    greedy       1 deterministic rollout, all 3 seeds (the pre-registered endpoint)
    best-of-8    8 sampled rollouts, seed 42
    best-of-16  16 sampled rollouts, seed 42
  Best-of-K is seed 42 only, so the decode ladder is compared at a fixed seed: the
  greedy s42 row is printed beside them for that purpose, while the seed-averaged greedy
  row is the one to quote against the rules.

Reported quantities
-------------------
  makespan   mean executed makespan over cleanly-routed instances
  deviation  BOTH senses, because they answer different questions:
               sd / CI over instances -- spread of the scheduler's outcomes
               Lambda = (executed - ideal) / ideal -- how far each schedule lands above
                        its own collision-free lower bound
  runtime    solve_s  scheduler time to produce a schedule
             select_s executor time best-of-K spends scoring its K candidates to pick one
                      (zero for greedy and for rules -- they have nothing to choose between)
             eval_s   one final evaluation, paid identically by every scheduler
             total_s  solve + select + eval

    python analyze_full.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
sys.path.insert(0, str(HERE.parent / "policy_vs_rules"))

from analyze_policy import bootstrap_ci, ci95, permutation_p, write_csv  # noqa: E402

SIZES = (20, 40, 60, 80, 100)
AMRS = 16
SEEDS = ("ppo_gc1_s42_best", "ppo_gc1_s43_best", "ppo_gc1_s44_best")
COMBOS = json.loads((HERE / "selected_combos.json").read_text())


EXPECTED_INSTANCES = 100


def rows(path: Path, expect: int | None = None) -> list:
    """Load rows, refusing a file that is short of `expect`.

    These files are appended to while a run is in progress, so "exists" does not mean
    "finished". A partially-written file loads cleanly and produces a perfectly plausible
    mean over whatever prefix happens to be on disk -- the failure is invisible in the
    output. Every caller therefore states how many rows it requires.
    """
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    out = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if expect is not None and len(out) != expect:
        raise SystemExit(
            f"{path.name}: {len(out)} rows, expected {expect}. The run that writes it is "
            f"probably still in progress; refusing to report a mean over a partial file."
        )
    return out


def stats(by_inst: dict, label: str, family: str, n: int, seed_avg: bool = False) -> dict:
    """Makespan / deviation / runtime for one scheduler at one size.

    `by_inst` maps instance -> row, or instance -> list of rows when several seeds are
    being collapsed. Means are taken over cleanly-routed instances only: on a failed leg
    the decoder adds a penalty constant rather than elapsed time, so a mean including
    those is not a duration.
    """
    clean, ex, lam, solve, select, ev = [], [], [], [], [], []
    for inst, r in sorted(by_inst.items()):
        rs = r if isinstance(r, list) else [r]
        if any(x["nu"] != 0 for x in rs):
            continue
        clean.append(inst)
        ex.append(statistics.mean([x["executed"] for x in rs]))
        lam.append(statistics.mean([x["penalty"] for x in rs]))
        solve.append(statistics.mean([x["solve_s"] for x in rs]))
        select.append(statistics.mean([x.get("select_s", 0.0) for x in rs]))
        ev.append(statistics.mean([x["eval_s"] for x in rs]))

    def sd(xs):
        return statistics.stdev(xs) if len(xs) > 1 else 0.0

    total = [a + b + c for a, b, c in zip(solve, select, ev)]
    return {
        "n_jobs": n, "scheduler": label, "family": family,
        "seed_averaged": "yes" if seed_avg else "no",
        "instances": len(by_inst), "clean": len(clean),
        "makespan_mean": round(statistics.mean(ex), 2),
        "makespan_sd": round(sd(ex), 2),
        "makespan_ci95": round(ci95(ex), 2),
        "makespan_min": round(min(ex), 1), "makespan_max": round(max(ex), 1),
        "lambda_pct_mean": round(100 * statistics.mean(lam), 2),
        "lambda_pct_sd": round(100 * sd(lam), 2),
        "solve_s_mean": round(statistics.mean(solve), 4),
        "solve_s_sd": round(sd(solve), 4),
        "select_s_mean": round(statistics.mean(select), 4),
        "eval_s_mean": round(statistics.mean(ev), 4),
        "total_s_mean": round(statistics.mean(total), 4),
        "total_s_sd": round(sd(total), 4),
    }


def paired(pol: dict, rule: dict, field: str = "executed") -> dict:
    """Paired delta on instances BOTH sides routed cleanly. Asserts the sets match."""
    if set(pol) != set(rule):
        raise SystemExit("instance sets differ; refusing to pair misaligned rows")
    kept = [i for i in sorted(pol)
            if all(x["nu"] == 0 for x in _aslist(pol[i]))
            and all(x["nu"] == 0 for x in _aslist(rule[i]))]
    d = [statistics.mean([x[field] for x in _aslist(pol[i])])
         - statistics.mean([x[field] for x in _aslist(rule[i])]) for i in kept]
    rm = statistics.mean([statistics.mean([x[field] for x in _aslist(rule[i])]) for i in kept])
    lo, hi = bootstrap_ci(d)
    wins = sum(1 for x in d if x < 0)
    return {
        "n_pairs": len(kept), "delta": round(statistics.mean(d), 2),
        "delta_ci95": round(ci95(d), 2),
        "delta_pct": round(100 * statistics.mean(d) / rm, 2),
        "boot_lo": round(lo, 2), "boot_hi": round(hi, 2),
        "wins": wins, "losses": len(d) - wins - sum(1 for x in d if x == 0),
        "win_rate_pct": round(100 * wins / len(d), 1) if d else float("nan"),
        "perm_p": round(permutation_p(d), 4),
        "significant": "yes" if abs(statistics.mean(d)) > ci95(d) else "no",
    }


def _aslist(v):
    return v if isinstance(v, list) else [v]


def collect(n: int) -> tuple:
    """(scheduler label -> {instance: row|[rows]}, ordered labels) at one size."""
    tables, order = {}, []

    for r in rows(RAW / f"full_rules_n{n}.jsonl", EXPECTED_INSTANCES * len(COMBOS)):
        combo = f"{r['job_rule']}+{r['rule']}"
        tables.setdefault(combo, {})[r["instance"]] = r
    order += [c for c in COMBOS if c in tables]

    greedy = defaultdict(dict)
    for r in rows(RAW / f"full_greedy_n{n}.jsonl", EXPECTED_INSTANCES * len(SEEDS)):
        greedy[r["run_key"]][r["instance"]] = r
    for k in SEEDS:
        tables[f"extend_GNN greedy {k[-9:-5]}"] = greedy[k]
    insts = sorted(set.intersection(*[set(greedy[k]) for k in SEEDS]))
    tables["extend_GNN greedy (3 seeds)"] = {i: [greedy[k][i] for k in SEEDS] for i in insts}
    order.append("extend_GNN greedy (3 seeds)")
    order += [f"extend_GNN greedy {k[-9:-5]}" for k in SEEDS]

    # Sampled decodes are optional so this script can be run against a partial sweep, but
    # "partial" must mean "this size is absent", never "this size is averaged over the
    # prefix that happened to be written when the analysis ran".
    # Two decode ladders. s42 is the seed `compute_time.csv` was measured on; s44 is the
    # seed `policy_vs_rules/policy_in_grid.csv` reports and is ~11-13 makespan units
    # stronger at n=60, so the two are not interchangeable and both are kept.
    for k, seed, path in ((8, "s42", f"full_k8_n{n}.jsonl"),
                          (16, "s42", f"full_k16_n{n}.jsonl"),
                          (8, "s44", f"full_k8_s44_n{n}.jsonl"),
                          (16, "s44", f"full_k16_s44_n{n}.jsonl")):
        p = RAW / path
        if not p.exists():
            continue
        n_rows = sum(1 for l in p.read_text().splitlines() if l.strip())
        if n_rows != EXPECTED_INSTANCES:
            print(f"  SKIP best-of-{k} {seed} n={n}: {n_rows}/{EXPECTED_INSTANCES} rows "
                  f"(in progress)")
            continue
        label = f"extend_GNN best-of-{k} {seed}"
        tables[label] = {r["instance"]: r for r in rows(p, EXPECTED_INSTANCES)}
        order.append(label)

    return tables, order


def main() -> None:
    main_rows, paired_rows = [], []

    for n in SIZES:
        tables, order = collect(n)
        fam = lambda lab: "policy" if lab.startswith("extend_GNN") else "rule"
        for lab in order:
            main_rows.append(stats(tables[lab], lab, fam(lab), n,
                                   seed_avg=lab.endswith("(3 seeds)")))

        # every policy variant against every one of the 6 rules
        best_rule = min(COMBOS, key=lambda c: statistics.mean(
            [r["executed"] for r in tables[c].values() if r["nu"] == 0]))
        for lab in order:
            if fam(lab) != "policy":
                continue
            for combo in COMBOS:
                paired_rows.append({
                    "n_jobs": n, "policy": lab, "rule": combo,
                    "is_best_rule_at_n": "yes" if combo == best_rule else "no",
                    **paired(tables[lab], tables[combo])})

    write_csv(HERE / "full_main.csv", main_rows)
    write_csv(HERE / "full_paired.csv", paired_rows)

    out = []
    out.append("\nFULL RUN -- 100 instances per size, m=16, serial (nothing else on the box)")
    out.append("makespan = executed, over cleanly-routed instances | Lambda = (executed-ideal)/ideal")
    out.append("=" * 118)
    for n in SIZES:
        sub = [r for r in main_rows if r["n_jobs"] == n]
        out.append(f"\nn = {n} parcels  ({n/AMRS:.2f} per robot)")
        out.append(f"  {'scheduler':<38}{'makespan':>10}{'sd':>8}{'+/-95%':>8}"
                   f"{'Lambda':>8}{'L sd':>7}{'solve_s':>9}{'sel_s':>8}{'total_s':>9}{'clean':>8}")
        for r in sub:
            out.append(f"  {r['scheduler']:<38}{r['makespan_mean']:>10.2f}"
                       f"{r['makespan_sd']:>8.2f}{r['makespan_ci95']:>8.2f}"
                       f"{r['lambda_pct_mean']:>7.2f}%{r['lambda_pct_sd']:>7.2f}"
                       f"{r['solve_s_mean']:>9.4f}{r['select_s_mean']:>8.4f}"
                       f"{r['total_s_mean']:>9.4f}"
                       f"{str(r['clean'])+'/'+str(r['instances']):>8}")

    out.append("\n" + "=" * 118)
    out.append("PAIRED vs the strongest of the 6 rules at each size (negative = policy faster)")
    out.append(f"  {'n':>5}  {'policy':<32}{'rule':<34}{'delta':>17}{'delta%':>9}"
               f"{'win%':>7}{'perm p':>9}")
    for r in paired_rows:
        if r["is_best_rule_at_n"] != "yes":
            continue
        out.append(f"  {r['n_jobs']:>5}  {r['policy']:<32}{r['rule']:<34}"
                   f"{r['delta']:>+8.2f} +/-{r['delta_ci95']:<5.2f}{r['delta_pct']:>+8.2f}%"
                   f"{r['win_rate_pct']:>6.1f}%{r['perm_p']:>9.4f}")

    # --- decode-budget ladder, one seed at a time -------------------------------------
    # Greedy / best-of-8 / best-of-16 must be read within a seed: the s42-s44 gap at n=60
    # is ~11 units, larger than the entire K=8 -> K=16 step, so a ladder that changes seed
    # between rungs measures the seed, not the budget.
    out.append("\n" + "=" * 118)
    out.append("DECODE-BUDGET LADDER (within seed; delta% is vs the strongest of the 6 rules)")
    for seed in ("s42", "s44"):
        out.append(f"\n  seed {seed}")
        out.append(f"  {'n':>5}{'K=1 greedy':>13}{'K=8':>10}{'K=16':>10}"
                   f"{'best rule':>11}{'K16 vs rule':>13}{'K1 total_s':>12}"
                   f"{'K8 total_s':>12}{'K16 total_s':>13}")
        for n in SIZES:
            sub = {r["scheduler"]: r for r in main_rows if r["n_jobs"] == n}
            g = sub.get(f"extend_GNN greedy _{seed}")
            k8 = sub.get(f"extend_GNN best-of-8 {seed}")
            k16 = sub.get(f"extend_GNN best-of-16 {seed}")
            rules = [r for r in main_rows if r["n_jobs"] == n and r["family"] == "rule"]
            br = min(rules, key=lambda r: r["makespan_mean"])
            f = lambda r, key="makespan_mean": f"{r[key]:.2f}" if r else "--"
            d = (f"{100*(k16['makespan_mean']-br['makespan_mean'])/br['makespan_mean']:+.2f}%"
                 if k16 else "--")
            out.append(f"  {n:>5}{f(g):>13}{f(k8):>10}{f(k16):>10}"
                       f"{br['makespan_mean']:>11.2f}{d:>13}"
                       f"{f(g,'total_s_mean'):>12}{f(k8,'total_s_mean'):>12}"
                       f"{f(k16,'total_s_mean'):>13}")

    text = "\n".join(out)
    (HERE / "full_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
