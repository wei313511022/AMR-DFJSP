"""The complete dispatching-rule grid on test_60, and where the policy sits in it.

Answers four things the milk_run/lpt subset could not:

  1. What is the strongest rule baseline, over all 60 combinations rather than the 12 that
     happened to be measured for the congestion sweep?
  2. Is the job rule or the AMR rule the bigger lever?
  3. Do the two interact, or is the grid roughly separable?
  4. Where does the learned policy land in the full ranking, at what compute?

Every mean is over cleanly-routed instances only (ie.aggregate enforces this); nu is
reported separately because on a failed leg the decoder charges MAX_DEPTH rather than
elapsed time, so a pooled mean would stop being a duration.

    python analyze_grid.py
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
GRID = RAW / "rule_grid_test60_m16.jsonl"
POLICY_FILES = {
    "greedy": RAW / "policy_test60_m16.jsonl",
    "best-of-8": RAW / "policy_test60_k8.jsonl",
    "best-of-16": RAW / "policy_test60_k16.jsonl",
}
BEST_MODEL = "ppo_gc1_s44_best"     # argmin over all 17 candidates on val_60, not on test


def load(path: Path) -> list:
    if not path.exists():
        raise SystemExit(f"missing: {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def summarise_combo(rows: list) -> dict:
    clean = [r for r in rows if r["nu"] == 0]
    if not clean:
        return {}
    return {
        "executed": round(statistics.mean([r["executed"] for r in clean]), 2),
        "executed_sd": round(statistics.stdev([r["executed"] for r in clean]), 2)
        if len(clean) > 1 else 0.0,
        "ideal": round(statistics.mean([r["ideal"] for r in clean]), 2),
        "penalty_pct": round(100 * statistics.mean([r["penalty"] for r in clean]), 2),
        "omega_q_pct": round(100 * statistics.mean([r["omega_q"] for r in clean]), 2),
        "omega_r_pct": round(100 * statistics.mean([r["omega_r"] for r in clean]), 2),
        "clean": len(clean), "instances": len(rows),
        "unroutable": len(rows) - len(clean),
        "solve_s": round(statistics.mean([r.get("solve_s", 0.0) for r in rows]), 4),
        "eval_s": round(statistics.mean([r.get("eval_s", 0.0) for r in rows]), 4),
    }


def main() -> None:
    rows = load(GRID)
    by_combo = defaultdict(list)
    for r in rows:
        by_combo[(r["job_rule"], r["rule"])].append(r)

    table = []
    for (jr, ar), rs in by_combo.items():
        s = summarise_combo(rs)
        if s:
            table.append({"job_rule": jr, "amr_rule": ar, "combo": f"{jr}+{ar}", **s})
    table.sort(key=lambda r: r["executed"])

    out = [f"\nCOMPLETE DISPATCHING-RULE GRID -- test_60, 100 instances, m=16 (eta=3.2)",
           f"{len(table)} combinations, ranked by executed makespan", "=" * 104,
           f"{'#':>3} {'combination':>42} {'executed':>9} {'sd':>7} {'ideal':>8} "
           f"{'Lambda':>8} {'Om_q':>7} {'Om_r':>7} {'nu':>5} {'solve_s':>8}"]
    for i, r in enumerate(table, 1):
        out.append(f"{i:>3} {r['combo']:>42} {r['executed']:>9.1f} {r['executed_sd']:>7.1f} "
                   f"{r['ideal']:>8.1f} {r['penalty_pct']:>7.2f}% {r['omega_q_pct']:>6.1f}% "
                   f"{r['omega_r_pct']:>6.1f}% {r['unroutable']:>5} {r['solve_s']:>8.3f}")

    # --- marginals: which axis is the bigger lever? ---
    by_job, by_amr = defaultdict(list), defaultdict(list)
    for r in table:
        by_job[r["job_rule"]].append(r["executed"])
        by_amr[r["amr_rule"]].append(r["executed"])

    out.append("\n\nMARGINALS -- mean executed over the other axis")
    out.append("-" * 104)
    out.append(f"  {'job rule':>26} {'mean':>8} {'best':>8} {'worst':>8} {'spread':>8}")
    for jr, v in sorted(by_job.items(), key=lambda kv: statistics.mean(kv[1])):
        out.append(f"  {jr:>26} {statistics.mean(v):>8.1f} {min(v):>8.1f} {max(v):>8.1f} "
                   f"{max(v) - min(v):>8.1f}")
    out.append(f"\n  {'AMR rule':>26} {'mean':>8} {'best':>8} {'worst':>8} {'spread':>8}")
    for ar, v in sorted(by_amr.items(), key=lambda kv: statistics.mean(kv[1])):
        out.append(f"  {ar:>26} {statistics.mean(v):>8.1f} {min(v):>8.1f} {max(v):>8.1f} "
                   f"{max(v) - min(v):>8.1f}")

    job_range = max(statistics.mean(v) for v in by_job.values()) - \
        min(statistics.mean(v) for v in by_job.values())
    amr_range = max(statistics.mean(v) for v in by_amr.values()) - \
        min(statistics.mean(v) for v in by_amr.values())
    out.append(f"\n  job-rule choice moves the mean by {job_range:.1f}; "
               f"AMR-rule choice by {amr_range:.1f} "
               f"-> the {'JOB' if job_range > amr_range else 'AMR'} rule is the bigger lever "
               f"({max(job_range, amr_range) / max(min(job_range, amr_range), 1e-9):.1f}x)")

    # Excluding `random` on both axes: it is a spread reference, not a candidate, and it
    # would otherwise dominate the marginal ranges and hide the real ordering.
    sane = [r for r in table if r["job_rule"] != "random" and r["amr_rule"] != "random"]
    out.append(f"\n  excluding `random` on both axes ({len(sane)} combinations): "
               f"best {sane[0]['combo']} {sane[0]['executed']:.1f}, "
               f"worst {sane[-1]['combo']} {sane[-1]['executed']:.1f} "
               f"({100 * (sane[-1]['executed'] - sane[0]['executed']) / sane[0]['executed']:.1f}% spread)")

    # --- policy placement ---
    out.append("\n\nWHERE THE LEARNED POLICY LANDS")
    out.append("-" * 104)
    out.append(f"  model {BEST_MODEL} (selected on val_60, not on test)")
    out.append(f"  {'configuration':>42} {'executed':>9} {'rank':>16} {'solve_s':>8} "
               f"{'vs best rule':>14}")
    best_rule = table[0]
    pol_rows = []
    for label, path in POLICY_FILES.items():
        if not path.exists():
            continue
        prs = [r for r in load(path) if r.get("run_key", "").startswith(BEST_MODEL)
               and r["amrs"] == 16]
        if not prs:
            continue
        s = summarise_combo(prs)
        k = prs[0].get("samples", 1)
        total = s["solve_s"] + k * s["eval_s"]
        rank = sum(1 for r in table if r["executed"] < s["executed"]) + 1
        pol_rows.append({"configuration": f"{BEST_MODEL} {label}", "k": k, **s,
                         "total_s": round(total, 3), "rank_in_grid": rank})
        out.append(f"  {BEST_MODEL + ' ' + label:>42} {s['executed']:>9.1f} "
                   f"{str(rank) + ' / ' + str(len(table) + 1):>16} {total:>8.3f} "
                   f"{s['executed'] - best_rule['executed']:>+13.1f}")
    out.append(f"  {'best rule: ' + best_rule['combo']:>42} {best_rule['executed']:>9.1f} "
               f"{'1 / ' + str(len(table) + 1):>16} "
               f"{best_rule['solve_s'] + best_rule['eval_s']:>8.3f} {0.0:>+13.1f}")

    with (HERE / "rule_grid.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    if pol_rows:
        keys = []
        for r in pol_rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with (HERE / "policy_in_grid.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, restval="")
            w.writeheader(); w.writerows(pol_rows)

    text = "\n".join(out)
    (HERE / "rule_grid_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
