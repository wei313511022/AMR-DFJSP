"""Build the deliverable CSV and the summary tables from the sweep + timing runs.

Conventions follow experiments/policy_vs_rules/analyze_policy.py:
  * never average executed/ideal/Lambda over rows the executor could not route (nu > 0);
    those instances are counted separately in `unroutable`
  * ci95 = 1.96 * sd / sqrt(n) over instances
  * Lambda is reported but is a diagnostic, not the endpoint -- a method that changes the
    assignment changes `ideal` too, so it can lower Lambda by inflating the denominator
"""
from __future__ import annotations
import csv, json, math, statistics, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def ci95(xs):
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")


def agg(rows, timing):
    """One aggregate per (family, method, mode, n_jobs)."""
    by = defaultdict(list)
    for r in rows:
        by[(r["family"], r["method"], r["mode"], r["n_jobs"])].append(r)
    tby = defaultdict(list)
    for r in timing:
        tby[(r["family"], r["method"], r["mode"], r["n_jobs"])].append(r)

    out = []
    for key in sorted(by, key=lambda k: (k[3], k[0], k[1], k[2])):
        fam, method, mode, n = key
        all_rows = by[key]
        ok = [r for r in all_rows if r["nu"] == 0]
        t = tby.get(key, [])
        rec = {
            "n_jobs": n, "family": fam, "method": method, "mode": mode,
            "instances": len(all_rows), "unroutable": len(all_rows) - len(ok),
        }
        if ok:
            ex = [r["executed"] for r in ok]
            rec.update({
                "makespan_mean": round(statistics.mean(ex), 2),
                "makespan_ci95": round(ci95(ex), 2),
                "makespan_median": round(statistics.median(ex), 2),
                "makespan_min": round(min(ex), 1), "makespan_max": round(max(ex), 1),
                "ideal_mean": round(statistics.mean(r["ideal"] for r in ok), 2),
                "gap_abs_mean": round(statistics.mean(r["executed"] - r["ideal"] for r in ok), 2),
                "gap_pct_mean": round(100 * statistics.mean(r["penalty"] for r in ok), 2),
                "omega_q": round(statistics.mean(r["omega_q"] for r in ok), 4),
                "omega_r": round(statistics.mean(r["omega_r"] for r in ok), 4),
            })
        if t:
            tok = t
            rec.update({
                "solve_s": round(statistics.mean(r["solve_s"] for r in tok), 4),
                "select_s": round(statistics.mean(r["select_s"] for r in tok), 4),
                "compute_s": round(statistics.mean(r["total_s"] for r in tok), 4),
                "eval_s": round(statistics.mean(r["eval_s"] for r in tok), 4),
                "timing_instances": len(tok),
            })
        out.append(rec)
    return out


COLS = ["n_jobs", "family", "method", "mode", "instances", "unroutable",
        "makespan_mean", "makespan_ci95", "makespan_median", "makespan_min", "makespan_max",
        "ideal_mean", "gap_abs_mean", "gap_pct_mean", "omega_q", "omega_r",
        "solve_s", "select_s", "compute_s", "eval_s", "timing_instances"]


def main():
    rows = load(RAW / "rows.jsonl")
    timing = load(RAW / "timing.jsonl") if (RAW / "timing.jsonl").exists() else []
    recs = agg(rows, timing)

    csv_path = HERE / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in recs:
            w.writerow({c: r.get(c, "") for c in COLS})
    print(f"wrote {csv_path}  ({len(recs)} aggregate rows from {len(rows)} raw rows)")

    per = HERE / "results_per_instance.csv"
    keys = ["n_jobs", "instance", "amrs", "family", "method", "mode", "dataset",
            "executed", "ideal", "penalty", "omega_q", "omega_r", "nu", "routable",
            "solve_s", "select_s", "eval_s", "total_s"]
    with per.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {per}  ({len(rows)} rows)")
    return recs


if __name__ == "__main__":
    main()
