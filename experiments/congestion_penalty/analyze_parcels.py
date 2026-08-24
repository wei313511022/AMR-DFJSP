"""Is Lambda a t=0 startup artefact, or steady-state congestion?

All parcels release at t=0, so at n=60 each of 16 robots handles only ~3.75 of
them and a large share of the makespan is the initial rush at the doors. If
Lambda is an artefact of that transient it must fall as n grows; if it is
steady-state contention for the doors it should hold.

Reads the m=16 slice of the fleet sweep (n=60) plus the two dedicated runs
(n=120, n=240), all on 100 clean test instances, and writes parcels.csv,
parcels_summary.txt and fig_parcel_scaling.pdf.

    python analyze_parcels.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = {
    60: HERE / "raw" / "final_fleet_test60.jsonl",
    120: HERE / "raw" / "parcels_120_m16.jsonl",
    240: HERE / "raw" / "parcels_240_m16.jsonl",
}
HEADLINE_M = 16
REFERENCE_AMR = "earliest_completion"
JOB_RULES = ("milk_run", "lpt")


def ci95(xs: list) -> float:
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def collect() -> list:
    out = []
    for job_rule in JOB_RULES:
        for n, path in SOURCES.items():
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            sel = [r for r in rows
                   if r["job_rule"] == job_rule
                   and r["rule"] == REFERENCE_AMR
                   and r["amrs"] == HEADLINE_M]
            clean = [r for r in sel if r["nu"] == 0]
            pen = [100 * r["penalty"] for r in clean]
            q = statistics.mean([100 * r["omega_q"] for r in clean])
            rr = statistics.mean([100 * r["omega_r"] for r in clean])
            out.append({
                "job_rule": job_rule,
                "parcels": n,
                "parcels_per_robot": round(n / HEADLINE_M, 2),
                "instances": len(sel),
                "clean": len(clean),
                "nu_per_episode": round(statistics.mean([r["nu"] for r in sel]), 3),
                "ideal": round(statistics.mean([r["ideal"] for r in clean]), 1),
                "executed": round(statistics.mean([r["executed"] for r in clean]), 1),
                "penalty_pct": round(statistics.mean(pen), 2),
                "penalty_ci95": round(ci95(pen), 2),
                "omega_q_pct": round(q, 2),
                "omega_r_pct": round(rr, 2),
                "queue_share_pct": round(100 * q / (q + rr), 1),
            })
    return out


def plot(rows: list, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"milk_run": "#185FA5", "lpt": "#993C1D"}
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for jr in JOB_RULES:
        sub = sorted([r for r in rows if r["job_rule"] == jr], key=lambda r: r["parcels"])
        ax.errorbar([r["parcels"] for r in sub], [r["penalty_pct"] for r in sub],
                    yerr=[r["penalty_ci95"] for r in sub], marker="o", ms=4, lw=1.4,
                    capsize=2.5, color=colors[jr], label=jr)
    ax.set_xscale("log", base=2)
    ax.set_xticks([60, 120, 240])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 30)
    ax.set_xlabel("parcels per instance")
    ax.set_ylabel(r"execution penalty $\Lambda$ (%)")
    ax.set_title(r"$\Lambda$ is flat in workload size ($m=16$)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> None:
    rows = collect()
    with (HERE / "parcels.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    head = (f"{'rule':>10} {'n':>5} {'n/robot':>8} {'ideal':>8} {'executed':>9} "
            f"{'Lambda':>16} {'Om_q':>7} {'Om_r':>7} {'q-shr':>6} {'clean':>9}")
    lines = [f"Parcel-count check at m={HEADLINE_M} (eta=3.2), AMR rule "
             f"{REFERENCE_AMR}, 100 instances each", "-" * len(head), head]
    for jr in JOB_RULES:
        for r in sorted([x for x in rows if x["job_rule"] == jr],
                        key=lambda x: x["parcels"]):
            lines.append(
                f"{r['job_rule']:>10} {r['parcels']:>5} {r['parcels_per_robot']:>8.2f} "
                f"{r['ideal']:>8.1f} {r['executed']:>9.1f} "
                f"{r['penalty_pct']:>9.2f}% +/-{r['penalty_ci95']:<4.2f} "
                f"{r['omega_q_pct']:>6.1f}% {r['omega_r_pct']:>6.1f}% "
                f"{r['queue_share_pct']:>5.0f}% "
                f"{str(r['clean']) + '/' + str(r['instances']):>9}")
        lines.append("")

    text = "\n".join(lines)
    (HERE / "parcels_summary.txt").write_text(text + "\n")
    print(text)
    plot(rows, HERE / "fig_parcel_scaling.pdf")


if __name__ == "__main__":
    main()
