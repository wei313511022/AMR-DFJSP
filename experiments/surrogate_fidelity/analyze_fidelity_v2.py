"""Value error, decision error and cost for C~ / Psi-hat / Phi, with ties handled explicitly.

Supersedes analyze_fidelity.py. Three changes, all of them about not letting an arbitrary
convention decide a reported number.

1. TAU_B IS THE HEADLINE, not Goodman-Kruskal gamma. gamma = (C-D)/(C+D) drops tied pairs
   from the denominator, and it drops a DIFFERENT number of them for each evaluator -- C~
   is integer-valued and ties often, Psi-hat is continuous and almost never does -- so the
   two columns were not computed on the same base. gamma is still printed for continuity
   with the published table, next to the pair counts that show the discrepancy.

2. THE SELECTION REGRET IS AN EXPECTATION OVER THE ARGMIN SET, not a list-order accident.
   argmin_c C~(c) is a SET M_i. `min(g, key=...)` returns whichever member the candidate
   loop emitted first, so on a tied instance the old number was a property of the
   enumeration order. The headline is now the mean over M_i -- what a uniformly random
   tie-break earns in expectation -- bracketed by the best and worst members. `as-coded`
   reproduces the old convention so the two tables can be reconciled.

3. INVERSION IS REPORTED UNDER BOTH CONVENTIONS. D/(C+D) excludes tied pairs; D/binom(k,2)
   counts every pair. They differ by several points wherever C~ ties.

Two slices are reported. `rules12` is the 12 rule-generated candidates alone and is
directly comparable with the published Table IV; `all19` adds the six policy cells and the
GA. The comparison is not cosmetic: cells A and B were trained to minimise C~ and E and F
to minimise Psi-hat, so all19 asks each evaluator to rank schedules built to exploit it.

Decision error is computed WITHIN an instance throughout. Ranking pooled across instances
would mostly recover "more parcels take longer" and inflate tau for every evaluator.

    python analyze_fidelity_v2.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "policy_vs_rules"))
from analyze_policy import ci95, write_csv  # noqa: E402

RULE_FAMILY = "rule"
PRED = {"c_tilde": "C~", "psi_hat": "Psi"}


# --- rank correlation ------------------------------------------------------------------
def _counts(pairs):
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
    return c, d, tx, ty, txy


def tau_b(pairs) -> float:
    """Kendall tau-b: ties enter the denominator, once per marginal."""
    c, d, tx, ty, txy = _counts(pairs)
    den = ((c + d + tx + txy) * (c + d + ty + txy)) ** 0.5
    return (c - d) / den if den else float("nan")


def gamma(pairs) -> float:
    """Goodman-Kruskal gamma. What the published table calls tau_b. Ties are dropped."""
    c, d, *_ = _counts(pairs)
    return (c - d) / (c + d) if c + d else float("nan")


def inversions(pairs):
    """(discordant/comparable, discordant/all-pairs). The first excludes ties."""
    c, d, tx, ty, txy = _counts(pairs)
    allp = c + d + tx + ty + txy
    return (d / (c + d) if c + d else float("nan"),
            d / allp if allp else float("nan"))


def comparable(pairs) -> int:
    c, d, *_ = _counts(pairs)
    return c + d


# --- selection under ties --------------------------------------------------------------
def selection(group, key):
    """Regret and hit rate of picking argmin_key, over the whole minimiser set M.

    Returns regret as (optimistic, expected, as_coded, pessimistic) and the probability
    that a uniformly random member of M is executor-optimal.
    """
    best = min(r["executed"] for r in group)
    kmin = min(r[key] for r in group)
    M = [r["executed"] for r in group if r[key] == kmin]
    coded = min(group, key=lambda r: r[key])["executed"]
    reg = lambda x: (x - best) / best
    return {
        "optimistic": reg(min(M)),
        "expected": reg(st.mean(M)),
        "as_coded": reg(coded),
        "pessimistic": reg(max(M)),
        "hit_expected": sum(1 for x in M if x == best) / len(M),
        "hit_optimistic": float(min(M) == best),
        "hit_as_coded": float(coded == best),
        "hit_pessimistic": float(max(M) == best),
        "tied": len(M) > 1,
        "consequential": len(M) > 1 and len(set(M)) > 1,
        "M": len(M),
    }


# --- report ----------------------------------------------------------------------------
def slice_report(clean, by_inst, sizes, label, out, csv_rows):
    out.append(f"\n\n{'='*108}\nSLICE: {label}\n{'='*108}")

    out.append("\nVALUE ERROR  |predicted - executed| / executed, over routable schedules")
    out.append(f"  {'n':>5}{'sched':>8}{'C~ MAPE':>10}{'C~ bias':>10}"
               f"{'Psi MAPE':>11}{'Psi bias':>10}")
    for n in sizes:
        g = [r for r in clean if r["n_jobs"] == n]
        f = lambda k, ab: st.mean([(abs(r[k]-r["executed"]) if ab else r[k]-r["executed"])
                                   / r["executed"] for r in g])
        out.append(f"  {n:>5}{len(g):>8}{100*f('c_tilde',1):>9.1f}%{100*f('c_tilde',0):>9.1f}%"
                   f"{100*f('psi_hat',1):>10.1f}%{100*f('psi_hat',0):>9.1f}%")

    out.append("\nDECISION ERROR  within-instance ranking of the candidates against the executor")
    out.append("  tau_b keeps tied pairs in the denominator; gamma drops them. P+D is the pairs")
    out.append("  gamma actually uses, out of the total shown -- the two evaluators do not share a base.")
    out.append(f"  {'n':>5}{'inst':>6}{'k':>4}{'pairs':>7}   "
               f"{'C~ tau_b':>10}{'C~ gamma':>10}{'C~ P+D':>9}{'C~ inv_c':>10}{'C~ inv_a':>10}   "
               f"{'Psi tau_b':>11}{'Psi gamma':>11}{'Psi P+D':>9}{'Psi inv_c':>11}{'Psi inv_a':>11}")
    stash = {}
    for n in sizes:
        groups = [v for (m, _), v in by_inst.items() if m == n and len(v) >= 4]
        k = st.mean([len(g) for g in groups])
        npairs = k * (k - 1) / 2
        row = {"n_jobs": n, "slice": label, "instances": len(groups), "candidates": round(k, 2)}
        cells = []
        for key in ("c_tilde", "psi_hat"):
            P = [[(r[key], r["executed"]) for r in g] for g in groups]
            tb = [tau_b(p) for p in P]
            gm = [gamma(p) for p in P]
            ic = [inversions(p)[0] for p in P]
            ia = [inversions(p)[1] for p in P]
            cp = st.mean([comparable(p) for p in P])
            cells.append((st.mean(tb), st.mean(gm), cp, st.mean(ic), st.mean(ia)))
            row.update({f"{key}_tau_b": round(st.mean(tb), 3),
                        f"{key}_tau_b_ci95": round(ci95(tb), 3),
                        f"{key}_gamma": round(st.mean(gm), 3),
                        f"{key}_comparable_pairs": round(cp, 1),
                        f"{key}_inv_comparable_pct": round(100*st.mean(ic), 2),
                        f"{key}_inv_allpairs_pct": round(100*st.mean(ia), 2)})
        (ct, cg, ccp, cic, cia), (pt, pg, pcp, pic, pia) = cells
        out.append(f"  {n:>5}{len(groups):>6}{k:>4.0f}{npairs:>7.0f}   "
                   f"{ct:>10.3f}{cg:>10.3f}{ccp:>9.1f}{100*cic:>9.1f}%{100*cia:>9.1f}%   "
                   f"{pt:>11.3f}{pg:>11.3f}{pcp:>9.1f}{100*pic:>10.1f}%{100*pia:>10.1f}%")
        stash[n] = (groups, row)

    out.append("\nSELECTION REGRET  makespan of the predicted-best candidate, over the executor's best")
    out.append("  argmin is a SET. 'expected' is a uniformly random tie-break; 'as-coded' is list order.")
    out.append(f"  {'n':>5}{'pred':>6}   {'optimistic':>11}{'expected':>10}{'as-coded':>10}"
               f"{'pessim.':>9}{'spread':>9}   {'hit exp':>9}{'hit coded':>11}"
               f"   {'tied':>7}{'conseq':>8}{'|M|':>6}")
    for n in sizes:
        groups, row = stash[n]
        for key in ("c_tilde", "psi_hat"):
            S = [selection(g, key) for g in groups]
            m = lambda f: st.mean([s[f] for s in S])
            k = len(groups)
            out.append(f"  {n:>5}{PRED[key]:>6}   {100*m('optimistic'):>10.2f}%{100*m('expected'):>9.2f}%"
                       f"{100*m('as_coded'):>9.2f}%{100*m('pessimistic'):>8.2f}%"
                       f"{100*(m('pessimistic')-m('optimistic')):>8.2f}%   "
                       f"{m('hit_expected')*k:>6.1f}/{k:<2d}{m('hit_as_coded')*k:>7.0f}/{k:<3d}"
                       f"   {100*m('tied'):>6.1f}%{100*m('consequential'):>7.1f}%{m('M'):>6.2f}")
            row.update({
                f"{key}_regret_optimistic_pct": round(100*m("optimistic"), 3),
                f"{key}_regret_expected_pct": round(100*m("expected"), 3),
                f"{key}_regret_as_coded_pct": round(100*m("as_coded"), 3),
                f"{key}_regret_pessimistic_pct": round(100*m("pessimistic"), 3),
                f"{key}_hit_expected": round(m("hit_expected")*k, 2),
                f"{key}_hit_as_coded": round(m("hit_as_coded")*k),
                f"{key}_tie_rate_pct": round(100*m("tied"), 1),
                f"{key}_consequential_tie_pct": round(100*m("consequential"), 1),
                f"{key}_argmin_set_size": round(m("M"), 2),
            })
        csv_rows.append(row)

    out.append("\nCOST  seconds to score ONE complete schedule")
    out.append(f"  {'n':>5}{'C~':>12}{'Psi-hat':>12}{'Phi':>12}   {'Phi/Psi':>9}{'Phi/C~':>9}")
    for n in sizes:
        g = [r for r in clean if r["n_jobs"] == n]
        c, p, f = (st.mean([r[k] for r in g]) for k in ("ideal_s", "psi_s", "phi_s"))
        out.append(f"  {n:>5}{c:>12.6f}{p:>12.6f}{f:>12.6f}   {f/p:>8.0f}x{f/c:>8.0f}x")
        for row in csv_rows:
            if row["n_jobs"] == n and row["slice"] == label:
                row.update(c_tilde_s=round(c, 6), psi_hat_s=round(p, 6), phi_s=round(f, 6))


def family_report(clean, sizes, out):
    """Value error by generator, which is the check the v1 README asked for and did not run.

    Cells A and B minimise C~ during training and E and F minimise Psi-hat, so each
    evaluator is being shown schedules built to exploit it. If an evaluator's error is
    materially worse on its own arm than on the rules, that is the self-consistency
    problem showing up, and it belongs in the table rather than in a caveat.
    """
    out.append(f"\n\n{'='*108}\nVALUE ERROR BY GENERATOR  does an evaluator degrade on schedules "
               f"optimised against it?\n{'='*108}")
    out.append("  A,B trained on C~   C,D trained on Phi   E,F trained on Psi-hat   "
               "rules and GA plan under Psi-hat")
    gens = sorted({(r["family"], r["method"]) for r in clean},
                  key=lambda t: (t[0] != "rule", t[1]))
    out.append(f"\n  {'generator':44s}{'sched':>7}{'C~ MAPE':>10}{'C~ bias':>10}"
               f"{'Psi MAPE':>11}{'Psi bias':>10}{'load/trip':>11}{'Phi mean':>10}")
    out.append("  " + "-" * 111)
    for fam, meth in gens:
        g = [r for r in clean if r["family"] == fam and r["method"] == meth]
        if not g:
            continue
        f = lambda k, ab: st.mean([(abs(r[k]-r["executed"]) if ab else r[k]-r["executed"])
                                   / r["executed"] for r in g])
        name = meth if fam != "policy" else f"policy {meth}"
        out.append(f"  {name:44s}{len(g):>7}{100*f('c_tilde',1):>9.1f}%{100*f('c_tilde',0):>9.1f}%"
                   f"{100*f('psi_hat',1):>10.1f}%{100*f('psi_hat',0):>9.1f}%"
                   f"{st.mean([r['load_per_trip'] for r in g]):>11.2f}"
                   f"{st.mean([r['executed'] for r in g]):>10.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(HERE / "fidelity_v2.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    clean = [r for r in rows if r["nu"] == 0]
    sizes = sorted({r["n_jobs"] for r in clean})

    out, csv_rows = [], []
    out.append(f"\nEVALUATOR FIDELITY -- {len({(r['family'], r['method']) for r in rows})} "
               f"candidate schedules per instance, m=16")
    out.append(f"{len({(r['n_jobs'], r['instance']) for r in rows})} instances at "
               f"n = {'/'.join(str(s) for s in sizes)}, "
               f"{len(rows)} schedules, {len(rows)-len(clean)} dropped as unroutable")

    slices = {
        "all19 -- 12 rules + 6 policy cells (A-F, seed 44, greedy) + GA": lambda r: True,
        "rules12 -- the 12 rule combinations alone (comparable with Table IV)":
            lambda r: r["family"] == RULE_FAMILY,
    }
    for label, keep in slices.items():
        sub = [r for r in clean if keep(r)]
        bi = defaultdict(list)
        for r in sub:
            bi[(r["n_jobs"], r["instance"])].append(r)
        slice_report(sub, bi, sizes, label, out, csv_rows)

    family_report(clean, sizes, out)

    write_csv(HERE / "fidelity_v2.csv", csv_rows)
    text = "\n".join(out)
    (HERE / "fidelity_v2_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
