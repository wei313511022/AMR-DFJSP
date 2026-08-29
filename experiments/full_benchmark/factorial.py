"""v10 A-F factorial table: training mask x training return, on identical instances.

Only seed 44 has all six cells finished at 4000 epochs, so s44 is the primary table.
D/E/F additionally have three finished seeds and get a seed-averaged table.
"""
from __future__ import annotations
import json, math, statistics, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SZ = (20, 40, 60, 80, 100)

CELLS = [  # cell, method key, mask, return
    ("A", "v10_A_s44",    "L3",   "idealised  C~"),
    ("B", "v10_B_s44",    "none", "idealised  C~"),
    ("E", "v10_E_s44",    "L3",   "surrogate  Psi"),
    ("F", "v10_F_s44",    "none", "surrogate  Psi"),
    ("C", "v10_C_s44",    "L3",   "executor   Phi"),
    ("D", "v9_only60_s44","none", "executor   Phi"),
]
VAL = {"A": 792.50, "B": 541.11, "C": 465.05, "D": 437.53, "E": 488.46, "F": 432.74}
SEEDS = {"D": "v9_only60_s{}", "E": "v10_E_s{}", "F": "v10_F_s{}"}
BESTRULE = {20: "earliest_completion_job+material_match",
            40: "earliest_completion_job+material_match",
            60: "milk_run+least_loaded",
            80: "milk_run+earliest_completion",
            100: "milk_run+earliest_available"}


def load():
    rows = []
    for f in ("raw/rows.jsonl", "raw/rows_ef.jsonl"):
        p = HERE / f
        if p.exists():
            rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def main():
    rows = load()
    ok = defaultdict(dict)      # (method, mode, n) -> {instance: executed}
    rule = defaultdict(dict)
    for r in rows:
        if r["nu"]:
            continue
        n = r["n_jobs"]
        if r["family"] == "rule" and r["method"] == BESTRULE[n]:
            rule[n][r["instance"]] = r["executed"]
        if r["family"] == "policy":
            ok[(r["method"], r["mode"], n)][r["instance"]] = r["executed"]

    out = []
    P = out.append
    P("=" * 104)
    P("v10 FACTORIAL A-F  |  training mask x training return")
    P("test_case/v3/trend/full_{n}.jsonl, 100 instances/size, m=16, seed 44, all cells 4000 epochs")
    P("Executed makespan under the collision-aware executor. Lower is better.")
    P("=" * 104)

    for mode, label in (("greedy", "GREEDY (1 deterministic rollout)"),
                        ("best8", "BEST-OF-8 (executor picks among 8 samples)")):
        P("")
        P(label)
        P("-" * 104)
        P(f"{'cell':<5}{'mask':<6}{'training return':<17}{'val':>8}" +
          "".join(f"{'n='+str(n):>10}" for n in SZ) + f"{'mean':>9}")
        P("-" * 104)
        best_per_n = {}
        for n in SZ:
            vals = [statistics.mean(ok[(m, mode, n)].values())
                    for _, m, _, _ in CELLS if ok[(m, mode, n)]]
            best_per_n[n] = min(vals) if vals else None
        for cell, m, mask, ret in CELLS:
            cells, ms = [], []
            for n in SZ:
                d = ok[(m, mode, n)]
                if not d:
                    cells.append(f"{'--':>10}"); continue
                v = statistics.mean(d.values()); ms.append(v)
                mark = "*" if best_per_n[n] and abs(v - best_per_n[n]) < 1e-9 else " "
                cells.append(f"{v:>9.1f}{mark}")
            avg = f"{statistics.mean(ms):>9.1f}" if ms else f"{'--':>9}"
            P(f"{cell:<5}{mask:<6}{ret:<17}{VAL[cell]:>8.1f}" + "".join(cells) + avg)

    # ---- effect sizes at each size, greedy
    P("")
    P("")
    P("EFFECT SIZES (greedy, seed 44) -- makespan cost of each factor, holding the other fixed")
    P("-" * 104)
    P(f"{'contrast':<44}" + "".join(f"{'n='+str(n):>10}" for n in SZ))
    P("-" * 104)
    def delta(a, b, n):
        A, B = ok[(a, "greedy", n)], ok[(b, "greedy", n)]
        ks = [i for i in A if i in B]
        return statistics.mean(A[i] - B[i] for i in ks) if ks else float("nan")
    contrasts = [
        ("return C~ -> Psi   (mask L3)   A - E", "v10_A_s44", "v10_E_s44"),
        ("return Psi -> Phi  (mask L3)   E - C", "v10_E_s44", "v10_C_s44"),
        ("return C~ -> Psi   (no mask)   B - F", "v10_B_s44", "v10_F_s44"),
        ("return Psi -> Phi  (no mask)   F - D", "v10_F_s44", "v9_only60_s44"),
        ("mask cost under C~             A - B", "v10_A_s44", "v10_B_s44"),
        ("mask cost under Psi            E - F", "v10_E_s44", "v10_F_s44"),
        ("mask cost under Phi            C - D", "v10_C_s44", "v9_only60_s44"),
    ]
    for lbl, a, b in contrasts:
        P(f"{lbl:<44}" + "".join(f"{delta(a,b,n):>+10.1f}" for n in SZ))

    # ---- vs the best rule
    P("")
    P("")
    P("VS THE STRONGEST DISPATCHING RULE AT EACH SIZE (paired per instance, greedy / best-of-8)")
    P("-" * 104)
    P(f"{'cell':<5}" + "".join(f"{'n='+str(n):>19}" for n in SZ))
    P("-" * 104)
    for cell, m, _, _ in CELLS:
        cs = []
        for n in SZ:
            g, b = ok[(m, "greedy", n)], ok[(m, "best8", n)]
            r = rule[n]
            def pc(d):
                ks = [i for i in d if i in r]
                if not ks: return None
                dd = [d[i] - r[i] for i in ks]
                return 100 * statistics.mean(dd) / statistics.mean(r[i] for i in ks)
            pg, pb = pc(g), pc(b)
            cs.append(f"{pg:>+8.1f}%/{pb:>+7.1f}%" if pg is not None and pb is not None else f"{'--':>19}")
        P(f"{cell:<5}" + "".join(cs))

    # ---- seed-averaged for the three cells with three finished seeds
    P("")
    P("")
    P("SEED-AVERAGED (D, E, F only -- the three cells with all three seeds at 4000 epochs)")
    P("-" * 104)
    for mode in ("greedy", "best8"):
        P(f"  {mode}")
        P(f"    {'cell':<5}" + "".join(f"{'n='+str(n):>12}" for n in SZ) + f"{'spread':>9}")
        for cell, tmpl in SEEDS.items():
            cs, spreads = [], []
            for n in SZ:
                per = [statistics.mean(ok[(tmpl.format(s), mode, n)].values())
                       for s in (42, 43, 44) if ok[(tmpl.format(s), mode, n)]]
                if len(per) < 3:
                    cs.append(f"{'--':>12}"); continue
                cs.append(f"{statistics.mean(per):>12.1f}")
                spreads.append(max(per) - min(per))
            sp = f"{statistics.mean(spreads):>9.1f}" if spreads else f"{'--':>9}"
            P(f"    {cell:<5}" + "".join(cs) + sp)
        P("")

    text = "\n".join(out)
    (HERE / "factorial_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
