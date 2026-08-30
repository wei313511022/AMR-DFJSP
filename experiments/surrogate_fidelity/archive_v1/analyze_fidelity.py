"""Value error, decision error and cost for C~ / Psi-hat / Phi.

Decision error is computed WITHIN an instance: the 12 candidate schedules of one instance
are ranked by each predictor and compared with the executor's ranking. Ranking pooled
across instances would mostly recover "more parcels take longer" and inflate tau for every
evaluator, including the idealised one.
"""

from __future__ import annotations

import argparse
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


def tau_b(pairs) -> float:
    """Kendall tau-b: unlike gamma above, ties enter the denominator.

    The two are not interchangeable when the predictors have different resolutions:
    C~ is integer-valued and ties often, Psi-hat is continuous and almost never does,
    so gamma silently drops a different number of pairs for each.
    """
    c = d = tx = ty = txy = 0
    for (a1, b1), (a2, b2) in itertools.combinations(pairs, 2):
        dx, dy = a1 - a2, b1 - b2
        s = dx * dy
        if s > 0:
            c += 1
        elif s < 0:
            d += 1
        elif dx == 0 and dy == 0:
            txy += 1
        elif dx == 0:
            tx += 1
        else:
            ty += 1
    den = ((c + d + tx + txy) * (c + d + ty + txy)) ** 0.5
    return (c - d) / den if den else float("nan")


def comparable_pairs(pairs) -> int:
    """P_i + D_i -- the pairs gamma actually uses. The rest are dropped in silence."""
    c = d = 0
    for (a1, b1), (a2, b2) in itertools.combinations(pairs, 2):
        s = (a1 - a2) * (b1 - b2)
        if s > 0:
            c += 1
        elif s < 0:
            d += 1
    return c + d


def ties_report(by_inst: dict, sizes: list) -> tuple:
    """Tie sensitivity of the selection regret.

    argmin over C~ is a SET, not a point: `min(g, key=...)` returns whichever member the
    candidate loop happened to emit first, so on a tied instance the reported regret is a
    property of the enumeration order, not of C~. This brackets it -- best, worst and
    mean over the minimiser set M_i -- and reports the tie statistics that say how much
    of the headline number is decided by that arbitrary choice.

    Ties are split by consequence, not by their existence:
      benign        every member of M_i executes to the same makespan; the tie-break
                    cannot move the regret. Usually two rule combos emitting one schedule.
      consequential members of M_i differ under Phi; here and only here does list order
                    decide the number.
    """
    out, csv_rows = [], []
    out.append("\nTIE SENSITIVITY OF THE SELECTION REGRET")
    out.append("=" * 104)
    out.append("\nWHERE THE TIES ARE  M_i = argmin C~ over the candidates of instance i")
    out.append(f"  {'n':>5}{'inst':>6}{'C~ tie':>9}{'Phi tie':>9}{'both':>7}"
               f"{'|M| avg':>9}{'benign':>9}{'conseq':>9}{'distinct/12':>13}")
    for n in sizes:
        gs = [v for (m, _), v in by_inst.items() if m == n and len(v) >= 4]
        it = et = bt = benign = conseq = 0
        M_sizes, distinct = [], []
        for g in gs:
            best = min(r["executed"] for r in g)
            cmin = min(r["c_tilde"] for r in g)
            M = [r for r in g if r["c_tilde"] == cmin]
            M_sizes.append(len(M))
            distinct.append(len({(r["c_tilde"], r["psi_hat"], r["executed"]) for r in g}))
            c_tie = len(M) > 1
            e_tie = sum(1 for r in g if r["executed"] == best) > 1
            it += c_tie
            et += e_tie
            bt += c_tie and e_tie
            if c_tie:
                if len({r["executed"] for r in M}) > 1:
                    conseq += 1
                else:
                    benign += 1
        k = len(gs)
        out.append(f"  {n:>5}{k:>6}{100*it/k:>8.1f}%{100*et/k:>8.1f}%{100*bt/k:>6.1f}%"
                   f"{statistics.mean(M_sizes):>9.2f}{f'{benign}/{k}':>9}{f'{conseq}/{k}':>9}"
                   f"{statistics.mean(distinct):>13.2f}")

    out.append("\nWHAT THE TIE-BREAK COSTS  C~ selection regret under three tie-break rules")
    out.append(f"  {'n':>5}{'best':>9}{'as-coded':>10}{'mean':>9}{'worst':>9}{'spread':>9}"
               f"   {'hit best':>9}{'hit as-coded':>13}{'hit pess':>9}")
    for n in sizes:
        gs = [v for (m, _), v in by_inst.items() if m == n and len(v) >= 4]
        rb, rc, rm, rw = [], [], [], []
        hb = hc = hp = 0
        for g in gs:
            best = min(r["executed"] for r in g)
            cmin = min(r["c_tilde"] for r in g)
            M = [r["executed"] for r in g if r["c_tilde"] == cmin]
            coded = min(g, key=lambda r: r["c_tilde"])["executed"]
            rb.append((min(M) - best) / best)
            rc.append((coded - best) / best)
            rm.append((statistics.mean(M) - best) / best)
            rw.append((max(M) - best) / best)
            hb += min(M) == best
            hc += coded == best
            hp += max(M) == best
        k = len(gs)
        m = statistics.mean
        out.append(f"  {n:>5}{100*m(rb):>8.2f}%{100*m(rc):>9.2f}%{100*m(rm):>8.2f}%"
                   f"{100*m(rw):>8.2f}%{100*(m(rw)-m(rb)):>8.2f}%   "
                   f"{f'{hb}/{k}':>9}{f'{hc}/{k}':>13}{f'{hp}/{k}':>9}")
        csv_rows.append({
            "n_jobs": n, "instances": k,
            "c_tilde_tie_rate": round(100 * sum(
                len([r for r in g if r["c_tilde"] == min(x["c_tilde"] for x in g)]) > 1
                for g in gs) / k, 1),
            "regret_best_pct": round(100 * m(rb), 2),
            "regret_as_coded_pct": round(100 * m(rc), 2),
            "regret_mean_pct": round(100 * m(rm), 2),
            "regret_worst_pct": round(100 * m(rw), 2),
            "picks_best_optimistic": hb, "picks_best_as_coded": hc, "picks_best_pessimistic": hp,
        })

    out.append("\nRANK CORRELATION UNDER TWO TIE CONVENTIONS")
    out.append("  gamma = (C-D)/(C+D), what the summary reports as 'tau'; tau_b keeps ties in the denominator")
    out.append(f"  {'n':>5}{'C~ gamma':>11}{'C~ tau_b':>10}{'Psi gamma':>11}{'Psi tau_b':>10}"
               f"   {'C~ P+D/66':>10}{'Psi P+D/66':>12}")
    for i, n in enumerate(sizes):
        gs = [v for (m_, _), v in by_inst.items() if m_ == n and len(v) >= 4]
        cp = [[(r["c_tilde"], r["executed"]) for r in g] for g in gs]
        pp = [[(r["psi_hat"], r["executed"]) for r in g] for g in gs]
        m = statistics.mean
        cg, cb = m([kendall_tau(p) for p in cp]), m([tau_b(p) for p in cp])
        pg, pb = m([kendall_tau(p) for p in pp]), m([tau_b(p) for p in pp])
        cpd, ppd = m([comparable_pairs(p) for p in cp]), m([comparable_pairs(p) for p in pp])
        out.append(f"  {n:>5}{cg:>11.3f}{cb:>10.3f}{pg:>11.3f}{pb:>10.3f}   {cpd:>10.1f}{ppd:>12.1f}")
        csv_rows[i].update(c_tilde_gamma=round(cg, 3), c_tilde_tau_b=round(cb, 3),
                           psi_hat_gamma=round(pg, 3), psi_hat_tau_b=round(pb, 3),
                           c_tilde_comparable_pairs=round(cpd, 1),
                           psi_hat_comparable_pairs=round(ppd, 1))
    return "\n".join(out), csv_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ties", action="store_true",
                    help="also bracket the selection regret over the C~ minimiser set")
    args = ap.parse_args()
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

    if args.ties:
        tie_text, tie_rows = ties_report(by_inst, sizes)
        write_csv(HERE / "tie_sensitivity.csv", tie_rows)
        (HERE / "tie_sensitivity.txt").write_text(tie_text + "\n")
        print(tie_text)


if __name__ == "__main__":
    main()
