"""F vs D across search budgets: greedy / best-of-8 / best-of-16.

Quality is seed-averaged WITHIN instance over seeds 42/43/44 before averaging across
instances, then paired against the strongest dispatching rule at each size. Compute time
comes from the serial pass (raw/timing_k.jsonl), never from the parallel sweep.
"""
import json, math, statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SZ = (20, 40, 60, 80, 100)
BR = {20: "earliest_completion_job+material_match", 40: "earliest_completion_job+material_match",
      60: "milk_run+least_loaded", 80: "milk_run+earliest_completion",
      100: "milk_run+earliest_available"}
ARMS = [("F", "v10_F_s{}"), ("D", "v9_only60_s{}")]
MODES = ["greedy", "best8", "best16"]
LBL = {"greedy": "greedy", "best8": "best-of-8", "best16": "best-of-16"}

ok, rule = defaultdict(dict), defaultdict(dict)
for f in ("raw/rows.jsonl", "raw/rows_ef.jsonl", "raw/rows_f16.jsonl"):
    p = HERE / f
    if not p.exists():
        continue
    for l in p.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r["nu"]:
            continue
        n = r["n_jobs"]
        if r["family"] == "rule" and r["method"] == BR[n]:
            rule[n][r["instance"]] = r["executed"]
        if r["family"] == "policy":
            ok[(r["method"], r["mode"], n)][r["instance"]] = r["executed"]

timing = defaultdict(list)
tp = HERE / "raw/timing_k.jsonl"
if tp.exists():
    for l in tp.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            timing[(r["method"], r["mode"], r["n_jobs"])].append(r["total_s"])

def seed_avg(tmpl, mode, n):
    seeds = [tmpl.format(s) for s in (42, 43, 44)]
    ks = [i for i in rule[n] if all(i in ok[(m, mode, n)] for m in seeds)]
    if not ks:
        return None
    pol = [statistics.mean(ok[(m, mode, n)][i] for m in seeds) for i in ks]
    d = [p - rule[n][i] for p, i in zip(pol, ks)]
    base = statistics.mean(rule[n][i] for i in ks)
    return (statistics.mean(pol), statistics.mean(d),
            1.96 * statistics.stdev(d) / math.sqrt(len(d)), 100 * statistics.mean(d) / base,
            100 * 1.96 * statistics.stdev(d) / math.sqrt(len(d)) / base, len(ks))

out = []
P = out.append
P("=" * 100)
P("SEARCH BUDGET: F (full features, Psi-hat reward) vs D (full features, executor reward)")
P("test_case/v3/trend/full_{n}.jsonl, 100 instances/size, m=16, seeds 42/43/44 averaged within instance")
P("=" * 100)
P("")
P("MAKESPAN (executed) and paired delta vs the strongest rule at each size")
P("-" * 100)
P(f"{'arm':<4}{'budget':<13}" + "".join(f"{'n='+str(n):>17}" for n in SZ))
P("-" * 100)
for cell, tmpl in ARMS:
    for mode in MODES:
        cells = []
        for n in SZ:
            r = seed_avg(tmpl, mode, n)
            cells.append(f"{r[0]:>8.1f}{r[3]:>+7.2f}%" if r else f"{'--':>17}")
        P(f"{cell:<4}{LBL[mode]:<13}" + "".join(cells))
    P("")
P("")
P("DELTA vs rule, with 95% CI  (negative = policy wins)")
P("-" * 100)
P(f"{'arm':<4}{'budget':<13}" + "".join(f"{'n='+str(n):>17}" for n in SZ))
P("-" * 100)
for cell, tmpl in ARMS:
    for mode in MODES:
        cells = []
        for n in SZ:
            r = seed_avg(tmpl, mode, n)
            cells.append(f"{r[3]:>+9.2f}% ±{r[4]:<5.2f}" if r else f"{'--':>17}")
        P(f"{cell:<4}{LBL[mode]:<13}" + "".join(cells))
    P("")
P("")
P("F minus D, paired per instance, seed-averaged  (negative = F wins)")
P("-" * 100)
P(f"{'budget':<13}" + "".join(f"{'n='+str(n):>17}" for n in SZ))
P("-" * 100)
for mode in MODES:
    cells = []
    for n in SZ:
        seeds_f = [f"v10_F_s{s}" for s in (42, 43, 44)]
        seeds_d = [f"v9_only60_s{s}" for s in (42, 43, 44)]
        ks = [i for i in rule[n] if all(i in ok[(m, mode, n)] for m in seeds_f + seeds_d)]
        if not ks:
            cells.append(f"{'--':>17}"); continue
        d = [statistics.mean(ok[(m, mode, n)][i] for m in seeds_f)
             - statistics.mean(ok[(m, mode, n)][i] for m in seeds_d) for i in ks]
        cells.append(f"{statistics.mean(d):>+9.1f} ±{1.96*statistics.stdev(d)/math.sqrt(len(d)):<5.1f}")
    P(f"{LBL[mode]:<13}" + "".join(cells))
P("")
P("")
P("COMPUTE TIME  seconds per instance, single process, no contention (seed 44)")
P("-" * 100)
P(f"{'arm':<4}{'budget':<13}" + "".join(f"{'n='+str(n):>12}" for n in SZ) + f"{'vs greedy':>11}")
P("-" * 100)
for cell, tmpl in ARMS:
    base = {}
    for mode in MODES:
        cells, vals = [], []
        for n in SZ:
            t = timing.get((tmpl.format(44), mode, n))
            if not t:
                cells.append(f"{'--':>12}"); continue
            v = statistics.mean(t); vals.append(v)
            cells.append(f"{v:>12.3f}")
        if mode == "greedy" and vals:
            base[cell] = statistics.mean(vals)
        ratio = f"{statistics.mean(vals)/base[cell]:>10.1f}x" if vals and cell in base else f"{'--':>11}"
        P(f"{cell:<4}{LBL[mode]:<13}" + "".join(cells) + ratio)
    P("")

text = "\n".join(out)
(HERE / "kbudget_summary.txt").write_text(text + "\n")
print(text)
