"""Job-rule x AMR-rule grid at the v3 facility -- chunked, appends rows.

Used to pick the counterfactual baseline for REINFORCE training. The baseline
has to be strong (a weak reference makes every advantage positive and the
gradient uninformative) AND it has to be able to express the behaviour the
policy is supposed to beat.

    python rule_grid.py --job_rules milk_run,lpt --count 3 --out ../grid.jsonl
    python rule_grid.py --summarise --out ../grid.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

STATIC_DIR = os.path.abspath(os.path.dirname(__file__))
if STATIC_DIR not in sys.path:
    sys.path.insert(0, STATIC_DIR)

import scenario_v3 as sc  # noqa: E402
import ideal_evaluator as ie  # noqa: E402
from GA.GA import load_dispatch_events  # noqa: E402
from reinforce_baseline import complete_with_dispatch_rule  # noqa: E402
from dispatching_rules.dispatching_rules import JOB_RULES, AMR_RULES  # noqa: E402


def summarise(path: Path) -> None:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[(r["job_rule"], r["amr_rule"])].append(r)
    jrs = sorted({r["job_rule"] for r in rows})
    ars = sorted({r["amr_rule"] for r in rows})

    print(f"{'job rule':<24}" + "".join(f"{a[:11]:>12}" for a in ars) + f"{'best':>10}")
    best_overall = (None, float("inf"))
    for jr in jrs:
        cells = []
        for ar in ars:
            g = by.get((jr, ar))
            cells.append(ie.aggregate(g)["executed"] if g else float("nan"))
        b = min(c for c in cells if c == c)
        if b < best_overall[1]:
            best_overall = (f"{jr}+{ars[cells.index(b)]}", b)
        print(f"{jr:<24}" + "".join(f"{c:>12.1f}" for c in cells) + f"{b:>10.1f}")
    print(f"\nbest combination: {best_overall[0]} at {best_overall[1]:.1f}")

    print("\nper-combination detail for the leaders:")
    ranked = sorted(((ie.aggregate(v)["executed"], k) for k, v in by.items()))[:6]
    print(f"  {'combination':<40}{'exec':>8}{'ideal':>8}{'Lambda':>9}{'nu/ep':>8}")
    for val, k in ranked:
        a = ie.aggregate(by[k])
        print(f"  {k[0]+'+'+k[1]:<40}{a['executed']:>8.1f}{a['ideal']:>8.1f}"
              f"{100*a['penalty']:>8.1f}%{a['nu_per_episode']:>8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", type=str, default="../test_case/v3/instances_60.jsonl")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--job_rules", type=str, default=",".join(JOB_RULES))
    ap.add_argument("--amr_rules", type=str, default=",".join(AMR_RULES))
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--amrs", type=int, default=16)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.summarise:
        summarise(out)
        return

    sc.apply_layout(num_amrs=args.amrs)
    events = load_dispatch_events(Path(args.inbox))[args.start: args.start + args.count]
    jrs = [r.strip() for r in args.job_rules.split(",") if r.strip()]
    ars = [r.strip() for r in args.amr_rules.split(",") if r.strip()]

    with out.open("a", encoding="utf-8") as fh:
        for ev in events:
            jobs = ev["jobs"]
            for jr in jrs:
                for ar in ars:
                    ind = complete_with_dispatch_rule(
                        jobs, [], {}, baseline_rule=f"{jr}+{ar}", seed=42)
                    row = ie.evaluate(ind, jobs)
                    row.update({"instance": ev["index"], "job_rule": jr, "amr_rule": ar})
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
    print(f"appended {len(events)} x {len(jrs)} x {len(ars)} rows -> {out}")


if __name__ == "__main__":
    main()
