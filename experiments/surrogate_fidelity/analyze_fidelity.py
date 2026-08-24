"""Value error, decision error and cost for C~ / Psi-hat / Phi.

Decision error is computed WITHIN an instance: the 12 candidate schedules of one instance
are ranked by each predictor and compared with the executor's ranking. Ranking pooled
across instances would mostly recover "more parcels take longer" and inflate tau for every
evaluator, including the idealised one.
"""

from __future__ import annotations

import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "policy_vs_rules"))
from analyze_policy import ci95, write_csv  # noqa: E402


def kendall_tau(pairs) -> float:
    conc = disc = 0
    for (a1, b1), (a2, b2) in itertools.combinations(pairs, 2):
        s = (a1 - a2) * (b1 - b2)
        if s > 0:
            conc += 1
        elif s < 0:
            disc += 1
    return (conc - disc) / (conc + disc) if conc + disc else float("nan")


def main() -> None:
    rows = [json.loads(l) for l in (HERE / "fidelity.jsonl").read_text().splitlines() if l.strip()]
    clean = [r for r in rows if r["nu"] == 0]
    by_inst = defaultdict(list)
    for r in clean:
        by_inst[(r["n_jobs"], r["instance"])].append(r)

    sizes = sorted({r["n_jobs"] for r in clean})
    out, csv_rows = [], []

    out.append("\nSURROGATE FIDELITY -- 12 candidate schedules per instance, m=16")
    out.append(f"dropped {len(rows)-len(clean)} of {len(rows)} schedules the executor could not route")
    out.append("=" * 104)
    out.append("\nVALUE ERROR  |predicted - executed| / executed")
    out.append(f"  {'n':>5}{'sched':>7}{'C~ MAPE':>10}{'C~ bias':>10}{'Psi MAPE':>11}{'Psi bias':>10}")
    for n in sizes:
        g = [r for r in clean if r["n_jobs"] == n]
        cm = statistics.mean([abs(r["c_tilde"]-r["executed"])/r["executed"] for r in g])
        cb = statistics.mean([(r["c_tilde"]-r["executed"])/r["executed"] for r in g])
        pm = statistics.mean([abs(r["psi_hat"]-r["executed"])/r["executed"] for r in g])
        pb = statistics.mean([(r["psi_hat"]-r["executed"])/r["executed"] for r in g])
        out.append(f"  {n:>5}{len(g):>7}{100*cm:>9.1f}%{100*cb:>9.1f}%{100*pm:>10.1f}%{100*pb:>9.1f}%")

    out.append("\nDECISION ERROR  within-instance ranking of the 12 candidates vs the executor")
    out.append(f"  {'n':>5}{'inst':>6}{'C~ tau':>10}{'Psi tau':>10}   "
               f"{'C~ regret':>11}{'Psi regret':>12}{'C~ best':>10}{'Psi best':>10}")
    for n in sizes:
        groups = [v for (m, _), v in by_inst.items() if m == n and len(v) >= 4]
        ct = [kendall_tau([(r["c_tilde"], r["executed"]) for r in g]) for g in groups]
        pt = [kendall_tau([(r["psi_hat"], r["executed"]) for r in g]) for g in groups]
        cr, pr, chit, phit = [], [], 0, 0
        for g in groups:
            best = min(r["executed"] for r in g)
            pick_c = min(g, key=lambda r: r["c_tilde"])["executed"]
            pick_p = min(g, key=lambda r: r["psi_hat"])["executed"]
            cr.append((pick_c-best)/best); pr.append((pick_p-best)/best)
            chit += pick_c == best; phit += pick_p == best
        out.append(f"  {n:>5}{len(groups):>6}{statistics.mean(ct):>10.3f}{statistics.mean(pt):>10.3f}   "
                   f"{100*statistics.mean(cr):>10.2f}%{100*statistics.mean(pr):>11.2f}%"
                   f"{f'{chit}/{len(groups)}':>10}{f'{phit}/{len(groups)}':>10}")
        csv_rows.append({
            "n_jobs": n, "instances": len(groups),
            "c_tilde_tau": round(statistics.mean(ct), 3),
            "psi_hat_tau": round(statistics.mean(pt), 3),
            "c_tilde_tau_ci95": round(ci95(ct), 3),
            "psi_hat_tau_ci95": round(ci95(pt), 3),
            "c_tilde_regret_pct": round(100*statistics.mean(cr), 2),
            "psi_hat_regret_pct": round(100*statistics.mean(pr), 2),
            "c_tilde_picks_best": chit, "psi_hat_picks_best": phit,
        })

    out.append("\nCOST  seconds to score ONE complete schedule")
    out.append(f"  {'n':>5}{'C~':>12}{'Psi-hat':>12}{'Phi':>12}   {'Phi/Psi':>9}{'Phi/C~':>9}")
    for i, n in enumerate(sizes):
        g = [r for r in clean if r["n_jobs"] == n]
        c = statistics.mean([r["ideal_s"] for r in g])
        p = statistics.mean([r["psi_s"] for r in g])
        f = statistics.mean([r["phi_s"] for r in g])
        out.append(f"  {n:>5}{c:>12.6f}{p:>12.6f}{f:>12.6f}   {f/p:>8.0f}x{f/c:>8.0f}x")
        csv_rows[i].update(c_tilde_s=round(c, 6), psi_hat_s=round(p, 6), phi_s=round(f, 6))

    write_csv(HERE / "fidelity.csv", csv_rows)
    text = "\n".join(out)
    (HERE / "fidelity_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
