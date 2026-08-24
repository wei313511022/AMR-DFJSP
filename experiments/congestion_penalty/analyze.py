"""Analyse the congestion sweep: the curve Lambda(eta), paired across instances.

Reads a raw jsonl (one row per fleet x instance x job rule x AMR rule) and
writes curve.csv, by_rule.csv and summary.txt beside it, plus a figure.

    python analyze.py                                   # the original sweep
    python analyze.py --raw raw/final_fleet_test60.jsonl --prefix final_

`--prefix` keeps several analyses side by side without clobbering each other.

Two things this script is careful about, both of which have burned this project
before:

1. It never pools runs that had routing failures into a makespan mean. On
   failure the executor charges `availability[amr] += MAX_DEPTH`, a penalty
   constant rather than elapsed time, so a single failed run drags a mean
   towards a number that is not a makespan at all. Rows with nu > 0 are
   excluded from the timing columns and counted separately.

2. It pairs by instance. Every fleet size is scored on the SAME instances, so
   the fleet-to-fleet difference is a within-instance quantity and its
   confidence interval should be computed per instance, not from the pooled
   spread of absolute makespans (which is dominated by instance difficulty).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "sweep.jsonl"
PREFIX = ""

# The dominant AMR rule in the full rule grid, and the headline column here.
REFERENCE_AMR = "earliest_completion"
DOORS = 5


def load() -> list:
    return [json.loads(l) for l in RAW.read_text().splitlines() if l.strip()]


def ci95(xs: list) -> float:
    """Half-width of the 95% CI of the mean. 0 for n<2 rather than a crash."""
    n = len(xs)
    if n < 2:
        return 0.0
    return 1.96 * statistics.stdev(xs) / math.sqrt(n)


def curve(rows: list, job_rule: str, amr_rule: str) -> list:
    """One row per fleet size, paired over instances."""
    sel = [r for r in rows if r["job_rule"] == job_rule and r["rule"] == amr_rule]
    by_m = defaultdict(list)
    for r in sel:
        by_m[r["amrs"]].append(r)

    out = []
    for m in sorted(by_m):
        runs = by_m[m]
        clean = [r for r in runs if r["nu"] == 0]
        # Per-instance penalty, then average -- not (mean exec)/(mean ideal),
        # which would weight easy and hard instances differently.
        pen = [100 * r["penalty"] for r in clean]
        oq = [100 * r["omega_q"] for r in clean]
        orr = [100 * r["omega_r"] for r in clean]
        q, rr = statistics.mean(oq), statistics.mean(orr)
        out.append({
            "m": m,
            "eta": round(m / DOORS, 2),
            "instances": len(runs),
            "clean": len(clean),
            "nu_per_episode": round(statistics.mean([r["nu"] for r in runs]), 3),
            "ideal": round(statistics.mean([r["ideal"] for r in clean]), 1),
            "executed": round(statistics.mean([r["executed"] for r in clean]), 1),
            "penalty_pct": round(statistics.mean(pen), 2),
            "penalty_ci95": round(ci95(pen), 2),
            "omega_q_pct": round(q, 2),
            "omega_r_pct": round(rr, 2),
            "queue_share_pct": round(100 * q / (q + rr), 1) if (q + rr) else float("nan"),
        })
    return out


def paired_deltas(rows: list, job_rule: str, amr_rule: str) -> list:
    """Within-instance change in penalty from one fleet size to the next."""
    sel = [r for r in rows if r["job_rule"] == job_rule and r["rule"] == amr_rule]
    by_mi = {(r["amrs"], r["instance"]): r for r in sel if r["nu"] == 0}
    fleets = sorted({m for m, _ in by_mi})
    out = []
    for a, b in zip(fleets, fleets[1:]):
        d = [100 * (by_mi[(b, i)]["penalty"] - by_mi[(a, i)]["penalty"])
             for (m, i) in by_mi if m == a and (b, i) in by_mi]
        if not d:
            continue
        mean, half = statistics.mean(d), ci95(d)
        out.append({
            "from_m": a, "to_m": b, "n_paired": len(d),
            "delta_penalty_pct": round(mean, 2),
            "ci95": round(half, 2),
            # A CI excluding 0 is the whole point of pairing: it says the step
            # moved the penalty, rather than the instances happening to differ.
            "significant": "yes" if abs(mean) > half else "no",
        })
    return out


def marginal_fleet(rows: list, job_rule: str, amr_rule: str, lo: int, hi: int) -> dict:
    """Does buying more robots help? Paired, and asked of BOTH evaluators.

    The idealised model and the executor can disagree about the sign of this,
    which is a decision error rather than a value error -- the abstraction does
    not merely misprice the schedule, it recommends a fleet that does not pay.
    """
    sel = [r for r in rows if r["job_rule"] == job_rule and r["rule"] == amr_rule
           and r["nu"] == 0]
    by_mi = {(r["amrs"], r["instance"]): r for r in sel}
    inst = [i for (m, i) in by_mi if m == lo and (hi, i) in by_mi]
    if not inst:
        return {}
    d_ideal = [100 * (by_mi[(hi, i)]["ideal"] - by_mi[(lo, i)]["ideal"])
               / by_mi[(lo, i)]["ideal"] for i in inst]
    d_exec = [100 * (by_mi[(hi, i)]["executed"] - by_mi[(lo, i)]["executed"])
              / by_mi[(lo, i)]["executed"] for i in inst]
    return {
        "job_rule": job_rule, "from_m": lo, "to_m": hi, "n_paired": len(inst),
        "ideal_change_pct": round(statistics.mean(d_ideal), 2),
        "ideal_ci95": round(ci95(d_ideal), 2),
        "executed_change_pct": round(statistics.mean(d_exec), 2),
        "executed_ci95": round(ci95(d_exec), 2),
    }


def ranking_consistency(rows: list, m: int) -> dict:
    """Do the two evaluators order the rule combinations the same way?

    This is the decision error the paper cares about. A value error that
    preserved the ranking would be harmless for a policy that only ever has to
    pick the better of two schedules.
    """
    sel = [r for r in rows if r["amrs"] == m and r["nu"] == 0]
    by_combo = defaultdict(list)
    for r in sel:
        by_combo[(r["job_rule"], r["rule"])].append(r)
    combos = sorted(by_combo)
    ideal = {c: statistics.mean([r["ideal"] for r in by_combo[c]]) for c in combos}
    execd = {c: statistics.mean([r["executed"] for r in by_combo[c]]) for c in combos}

    conc = disc = 0
    for i, a in enumerate(combos):
        for b in combos[i + 1:]:
            s = (ideal[a] - ideal[b]) * (execd[a] - execd[b])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    tau = (conc - disc) / (conc + disc) if (conc + disc) else float("nan")
    best_i = min(combos, key=lambda c: ideal[c])
    best_e = min(combos, key=lambda c: execd[c])
    return {
        "m": m, "eta": round(m / DOORS, 2), "combos": len(combos),
        "concordant": conc, "discordant": disc, "kendall_tau": round(tau, 3),
        "best_by_ideal": "+".join(best_i), "best_by_executed": "+".join(best_e),
        "ideal_picks_the_executed_best": "yes" if best_i == best_e else "no",
    }


def selection_regret(rows: list, m: int) -> dict:
    """What does trusting the idealised model's favourite rule actually cost?

    Kendall tau alone overstates the damage: it weights every pair equally,
    including pairs deep in the ranking that nobody would ever choose between.
    The decision that matters is "pick the best combination", so this measures
    the executed makespan lost by taking the idealised model's argmin instead
    of the executor's, paired per instance.
    """
    sel = [r for r in rows if r["amrs"] == m and r["nu"] == 0]
    by_combo = defaultdict(list)
    for r in sel:
        by_combo[(r["job_rule"], r["rule"])].append(r)
    combos = sorted(by_combo)
    if not combos:
        return {}
    ideal = {c: statistics.mean([r["ideal"] for r in by_combo[c]]) for c in combos}
    execd = {c: statistics.mean([r["executed"] for r in by_combo[c]]) for c in combos}
    pick_i = min(combos, key=lambda c: ideal[c])
    pick_e = min(combos, key=lambda c: execd[c])

    a = {r["instance"]: r["executed"] for r in by_combo[pick_i]}
    b = {r["instance"]: r["executed"] for r in by_combo[pick_e]}
    common = sorted(set(a) & set(b))
    d = [100 * (a[i] - b[i]) / b[i] for i in common]
    mean, half = (statistics.mean(d), ci95(d)) if d else (float("nan"),) * 2
    return {
        "m": m, "eta": round(m / DOORS, 2),
        "ideal_pick": "+".join(pick_i), "executed_pick": "+".join(pick_e),
        "regret_pct": round(mean, 2), "regret_ci95": round(half, 2),
        "n_paired": len(d),
        "significant": "yes" if abs(mean) > half else "no",
    }


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fmt(rows: list, title: str) -> str:
    head = (f"{'m':>3} {'eta':>5} {'ideal':>8} {'exec':>8} {'Lambda':>16} "
            f"{'Om_q':>7} {'Om_r':>7} {'q-shr':>6} {'nu/ep':>6} {'clean':>8}")
    lines = [title, "-" * len(head), head]
    for r in rows:
        lines.append(
            f"{r['m']:>3} {r['eta']:>5.1f} {r['ideal']:>8.1f} {r['executed']:>8.1f} "
            f"{r['penalty_pct']:>9.2f}% +/-{r['penalty_ci95']:>4.2f} "
            f"{r['omega_q_pct']:>6.1f}% {r['omega_r_pct']:>6.1f}% "
            f"{r['queue_share_pct']:>5.0f}% {r['nu_per_episode']:>6.2f} "
            f"{str(r['clean']) + '/' + str(r['instances']):>8}")
    return "\n".join(lines)


def plot(all_curves: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figure")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    colors = {"milk_run": "#185FA5", "lpt": "#993C1D"}
    for jr, rows in all_curves.items():
        eta = [r["eta"] for r in rows]
        pen = [r["penalty_pct"] for r in rows]
        err = [r["penalty_ci95"] for r in rows]
        ax1.errorbar(eta, pen, yerr=err, marker="o", ms=4, lw=1.4, capsize=2.5,
                     color=colors.get(jr, "#444441"), label=jr)
        ax2.plot(eta, [r["omega_q_pct"] for r in rows], marker="o", ms=4, lw=1.4,
                 color=colors.get(jr, "#444441"), label=f"{jr} queueing")
        ax2.plot(eta, [r["omega_r_pct"] for r in rows], marker="s", ms=4, lw=1.4,
                 ls="--", color=colors.get(jr, "#444441"), label=f"{jr} routing")

    ax1.set_xlabel(r"contention ratio $\eta = m/|D^{\rm in}|$")
    ax1.set_ylabel(r"execution penalty $\Lambda$ (%)")
    ax1.set_title("Abstraction error vs contention", fontsize=9)
    ax1.legend(fontsize=7, frameon=False)
    ax1.grid(alpha=0.25, lw=0.5)

    ax2.set_xlabel(r"contention ratio $\eta$")
    ax2.set_ylabel("delay ratio (%)")
    ax2.set_title("Queueing vs routing component", fontsize=9)
    ax2.legend(fontsize=6.2, frameon=False, ncol=2)
    ax2.grid(alpha=0.25, lw=0.5)

    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {path} and {path.with_suffix('.png')}")


def main() -> None:
    global RAW, PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW),
                    help="raw jsonl to analyse (default: the original sweep)")
    ap.add_argument("--prefix", default="",
                    help="prepended to every output filename, so several raw "
                         "files can be analysed side by side without clobbering")
    args = ap.parse_args()
    RAW, PREFIX = Path(args.raw), args.prefix

    rows = load()
    job_rules = sorted({r["job_rule"] for r in rows})
    amr_rules = sorted({r["rule"] for r in rows})

    report, curves = [], {}
    for jr in job_rules:
        c = curve(rows, jr, REFERENCE_AMR)
        curves[jr] = c
        report.append(fmt(c, f"\njob rule: {jr}   AMR rule: {REFERENCE_AMR}   "
                             f"(headline; {c[0]['instances']} instances x 60 parcels)"))
        report.append("\npaired within-instance change in Lambda:")
        for d in paired_deltas(rows, jr, REFERENCE_AMR):
            report.append(f"  m {d['from_m']:>2} -> {d['to_m']:<2}  "
                          f"{d['delta_penalty_pct']:+6.2f} pp  +/-{d['ci95']:.2f}  "
                          f"(n={d['n_paired']}, moved: {d['significant']})")

    # Does a bigger fleet pay, and do the two evaluators agree that it does?
    fleets = sorted({r["amrs"] for r in rows})
    report.append("\n\nmarginal value of fleet, paired (negative = makespan improves):")
    report.append(f"  {'rule':>10} {'step':>10} {'idealised':>20} {'executed':>20}")
    marg = []
    for jr in job_rules:
        for lo, hi in zip(fleets, fleets[1:]):
            d = marginal_fleet(rows, jr, REFERENCE_AMR, lo, hi)
            if not d:
                continue
            marg.append(d)
            report.append(
                f"  {jr:>10} {f'{lo}->{hi}':>10} "
                f"{d['ideal_change_pct']:>13.2f}% +/-{d['ideal_ci95']:<4.2f} "
                f"{d['executed_change_pct']:>13.2f}% +/-{d['executed_ci95']:<4.2f}")

    report.append("\nranking consistency between the two evaluators "
                  f"({len(job_rules) * len(amr_rules)} rule combinations):")
    report.append(f"  {'m':>3} {'eta':>5} {'tau':>7} {'disc':>5} "
                  f"{'best by ideal':>34} {'best by executed':>34}")
    ranks = []
    for m in fleets:
        d = ranking_consistency(rows, m)
        ranks.append(d)
        report.append(f"  {d['m']:>3} {d['eta']:>5.1f} {d['kendall_tau']:>7.3f} "
                      f"{d['discordant']:>5} {d['best_by_ideal']:>34} "
                      f"{d['best_by_executed']:>34}")

    report.append("\nselection regret -- executed makespan lost by taking the "
                  "idealised model's favourite combination:")
    report.append(f"  {'m':>3} {'eta':>5} {'regret':>18} {'moved':>7}")
    regret = []
    for m in fleets:
        d = selection_regret(rows, m)
        regret.append(d)
        report.append(f"  {d['m']:>3} {d['eta']:>5.1f} "
                      f"{d['regret_pct']:>11.2f}% +/-{d['regret_ci95']:<4.2f} "
                      f"{d['significant']:>7}")

    write_csv(HERE / (PREFIX + "marginal_fleet.csv"), marg)
    write_csv(HERE / (PREFIX + "ranking_consistency.csv"), ranks)
    write_csv(HERE / (PREFIX + "selection_regret.csv"), regret)

    # Rule dependence: same curve under every AMR rule, so a reader can see
    # whether the shape is a property of the facility or of one heuristic.
    by_rule = []
    for jr in job_rules:
        for ar in amr_rules:
            for r in curve(rows, jr, ar):
                by_rule.append({"job_rule": jr, "amr_rule": ar, **r})

    write_csv(HERE / (PREFIX + "curve.csv"), [{"job_rule": jr, **r}
                                   for jr in job_rules for r in curves[jr]])
    write_csv(HERE / (PREFIX + "by_rule.csv"), by_rule)

    text = "\n".join(report)
    (HERE / (PREFIX + "summary.txt")).write_text(text + "\n")
    print(text)
    plot(curves, HERE / (PREFIX + "fig_penalty_curve.pdf"))


if __name__ == "__main__":
    main()
