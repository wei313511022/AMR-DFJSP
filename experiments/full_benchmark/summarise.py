"""Human-readable summary tables from results.csv."""
from __future__ import annotations
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIZES = (20, 40, 60, 80, 100)


def load():
    rows = []
    with (HERE / "results.csv").open() as fh:
        for r in csv.DictReader(fh):
            for k in ("makespan_mean","makespan_ci95","ideal_mean","gap_abs_mean","gap_pct_mean",
                      "omega_q","omega_r","solve_s","select_s","compute_s","eval_s"):
                r[k] = float(r[k]) if r[k] not in ("", None) else None
            for k in ("n_jobs","instances","unroutable"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def key(r):
    return f"{r['method']}" + ("" if r["mode"] in ("single",) else f" [{r['mode']}]")


def main():
    rows = load()
    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n_jobs"]].append(r)

    out = []
    P = out.append
    P("=" * 108)
    P("FULL METHOD BENCHMARK -- scenario v3, m=16, 100 instances per size, dataset test_case/v3/trend/full_{n}.jsonl")
    P("Planning under the calibrated surrogate; every schedule scored once by the collision-aware executor.")
    P("Makespan/gap averaged over routable instances only (nu=0). Compute time from the serial pass, no contention.")
    P("=" * 108)

    # ---- headline: best of each family at each size
    P("")
    P("HEADLINE -- best method in each family, per parcel count")
    P("-" * 108)
    P(f"{'n':>4} {'family':>8} {'method':<30}{'makespan':>10}{'ci95':>8}{'ideal':>9}"
      f"{'gap':>9}{'gap%':>8}{'compute':>10}{'nu>0':>6}")
    for n in SIZES:
        rs = by_n.get(n, [])
        if not rs: continue
        fams = [("rule", [r for r in rs if r["family"] == "rule"]),
                ("policy", [r for r in rs if r["family"] == "policy" and r["mode"] == "greedy"]),
                ("policy", [r for r in rs if r["family"] == "policy" and r["mode"] == "best8"]),
                ("ga", [r for r in rs if r["family"] == "ga"])]
        for fam, group in fams:
            g = [r for r in group if r["makespan_mean"] is not None]
            if not g: continue
            b = min(g, key=lambda r: r["makespan_mean"])
            P(f"{n:>4} {fam:>8} {key(b):<30}{b['makespan_mean']:>10.1f}{b['makespan_ci95']:>8.2f}"
              f"{b['ideal_mean']:>9.1f}{b['gap_abs_mean']:>9.1f}{b['gap_pct_mean']:>7.1f}%"
              f"{(b['compute_s'] if b['compute_s'] is not None else float('nan')):>10.3f}{b['unroutable']:>6}")
        P("")

    # ---- every model, every size
    P("")
    P("ALL LEARNED MODELS -- makespan (routable instances)")
    P("-" * 108)
    hdr = f"{'model':<24}{'mode':>7}" + "".join(f"{'n='+str(n):>13}" for n in SIZES)
    P(hdr); P("-" * 108)
    models = sorted({r["method"] for r in rows if r["family"] == "policy"})
    for mode in ("greedy", "best8"):
        for m in models:
            cells = []
            for n in SIZES:
                hit = [r for r in by_n.get(n, []) if r["method"] == m and r["mode"] == mode]
                cells.append(f"{hit[0]['makespan_mean']:>8.1f}+-{hit[0]['makespan_ci95']:<4.1f}"
                             if hit and hit[0]["makespan_mean"] is not None else f"{'--':>13}")
            P(f"{m:<24}{mode:>7}" + "".join(cells))
        P("")

    # ---- rule grid: top 8 at each size
    P("")
    P("DISPATCHING RULES -- top 8 combinations at each size (of 60)")
    P("-" * 108)
    for n in SIZES:
        rs = sorted([r for r in by_n.get(n, []) if r["family"] == "rule" and r["makespan_mean"] is not None],
                    key=lambda r: r["makespan_mean"])
        if not rs: continue
        P(f"  n={n}")
        P(f"    {'rank':>4}  {'combination':<44}{'makespan':>10}{'gap%':>8}{'compute':>10}")
        for i, r in enumerate(rs[:8], 1):
            c = r["compute_s"] if r["compute_s"] is not None else float("nan")
            P(f"    {i:>4}  {r['method']:<44}{r['makespan_mean']:>10.1f}{r['gap_pct_mean']:>7.1f}%{c:>10.3f}")
        P(f"    {'worst':>4}  {rs[-1]['method']:<44}{rs[-1]['makespan_mean']:>10.1f}{rs[-1]['gap_pct_mean']:>7.1f}%")
        P("")

    # ---- compute time table
    P("")
    P("COMPUTE TIME -- seconds per instance, single process, no contention")
    P("-" * 108)
    P(f"{'method':<34}" + "".join(f"{'n='+str(n):>13}" for n in SIZES))
    P("-" * 108)
    def timing_rows():
        seen = []
        for fam, mode in (("rule", "single"), ("policy", "greedy"), ("policy", "best8"), ("ga", "single")):
            names = sorted({r["method"] for r in rows if r["family"] == fam and r["mode"] == mode})
            for nm in names:
                seen.append((fam, nm, mode))
        return seen
    for fam, nm, mode in timing_rows():
        cells = []
        any_t = False
        for n in SIZES:
            hit = [r for r in by_n.get(n, []) if r["method"] == nm and r["mode"] == mode and r["family"] == fam]
            if hit and hit[0]["compute_s"] is not None:
                any_t = True
                cells.append(f"{hit[0]['compute_s']:>13.3f}")
            else:
                cells.append(f"{'--':>13}")
        if any_t:
            label = nm + ("" if mode == "single" else f" [{mode}]")
            P(f"{label:<34}" + "".join(cells))

    text = "\n".join(out)
    (HERE / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
