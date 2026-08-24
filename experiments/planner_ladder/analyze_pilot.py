"""Planner-fidelity ladder -- paired analysis of the 10-instance pilot.

Every arm plans the same instances with the same rules and is executed by the
same collision-aware executor. Rows pair on (instance, job_rule, rule), so the
delta between arms is a within-schedule-slot comparison: the only thing that
differs is what the planner believed while building the schedule.

Rows with nu > 0 (the executor could not route some leg) are excluded from
makespan means and counted separately -- a failed schedule has no makespan to
average, and silently dropping it would move the denominator.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
ARMS = [
    ("ideal", "raw Manhattan + no queue estimate"),
    ("qblind", "calibrated travel + no queue estimate"),
    ("full", "calibrated travel + calibrated queue penalty"),
]
KEY = ("instance", "job_rule", "rule")


def load(arm: str) -> dict:
    path = RAW / f"ladder_{arm}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {tuple(r[k] for k in KEY): r for r in rows}


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return 1.96 * statistics.stdev(values) / len(values) ** 0.5


def clustered_ci95(pairs: list[tuple[int, float]]) -> tuple[float, float, int]:
    """CI over instance means, not over slots.

    The 12 rule combinations reuse the same 10 instances, so slots are not
    independent; treating them as such understates the interval by roughly
    sqrt(12). Averaging within instance first makes the effective sample size
    the number of instances, which is what the sweep actually varied.
    """
    by_inst: dict[int, list[float]] = {}
    for inst, value in pairs:
        by_inst.setdefault(inst, []).append(value)
    means = [statistics.mean(v) for v in by_inst.values()]
    return statistics.mean(means), ci95(means), len(means)


def mean_over_clean(rows: list[dict], field: str) -> float:
    vals = [r[field] for r in rows if r["nu"] == 0]
    return statistics.mean(vals) if vals else float("nan")


def main() -> None:
    data = {arm: load(arm) for arm, _ in ARMS}
    slots = sorted(set.intersection(*(set(d) for d in data.values())))
    print(f"{len(slots)} paired slots (instance x job_rule x AMR rule), m=16, test_60\n")

    print("PER-ARM MEANS (nu>0 excluded from makespan; failures counted separately)")
    print(f"{'arm':<8} {'exec':>8} {'ideal':>8} {'Lambda':>8} {'Om_q':>7} {'Om_r':>7} {'nu>0':>6}")
    for arm, _ in ARMS:
        rows = [data[arm][s] for s in slots]
        fails = sum(1 for r in rows if r["nu"] > 0)
        print(f"{arm:<8} {mean_over_clean(rows,'executed'):>8.1f} "
              f"{mean_over_clean(rows,'ideal'):>8.1f} "
              f"{100*mean_over_clean(rows,'penalty'):>7.1f}% "
              f"{100*mean_over_clean(rows,'omega_q'):>6.1f}% "
              f"{100*mean_over_clean(rows,'omega_r'):>6.1f}% "
              f"{fails:>4}/{len(rows)}")

    print("\nPAIRED COST OF PLANNING BLIND (executed makespan, vs the 'full' arm)")
    print("positive = the blind planner's schedule is slower once actually executed")
    base = "full"

    def paired(arm_a: str, arm_b: str):
        deltas, pcts, dropped = [], [], 0
        for s in slots:
            a, b = data[arm_a][s], data[arm_b][s]
            if a["nu"] > 0 or b["nu"] > 0:
                dropped += 1
                continue
            deltas.append((s[0], a["executed"] - b["executed"]))
            pcts.append((s[0], 100 * (a["executed"] - b["executed"]) / b["executed"]))
        return deltas, pcts, dropped

    for arm, desc in ARMS:
        if arm == base:
            continue
        deltas, pcts, dropped = paired(arm, base)
        d_mean, d_ci, n_inst = clustered_ci95(deltas)
        p_mean, p_ci, _ = clustered_ci95(pcts)
        wins = sum(1 for _, d in deltas if d > 0)
        print(f"\n  {arm:<8} ({desc})")
        print(f"    delta   {d_mean:+7.2f} +/-{d_ci:.2f}  ({p_mean:+.2f}% +/-{p_ci:.2f})"
              f"   [CI over {n_inst} instances]")
        print(f"    slower in {wins}/{len(deltas)} paired slots"
              f"{f'; {dropped} dropped for nu>0' if dropped else ''}")

    print("\n  ideal vs qblind (isolates the travel term alone)")
    deltas, _, _ = paired("ideal", "qblind")
    d_mean, d_ci, n_inst = clustered_ci95(deltas)
    print(f"    delta   {d_mean:+7.2f} +/-{d_ci:.2f}   [CI over {n_inst} instances]")

    print("\nWHICH DIRECTION DOES EACH EVALUATOR POINT? (blind arm vs full arm)")
    print("  a planner that is rewarded by one metric and punished by the other is")
    print("  exploiting the model, not solving the problem")
    for arm, _ in ARMS:
        if arm == base:
            continue
        for field, label in (("ideal", "idealised score"), ("executed", "executed makespan")):
            pairs = [(s[0], data[arm][s][field] - data[base][s][field])
                     for s in slots if data[arm][s]["nu"] == 0 and data[base][s]["nu"] == 0]
            m, c, _ = clustered_ci95(pairs)
            verdict = "BETTER" if m < 0 else "worse"
            print(f"    {arm:<8} {label:<18} {m:+7.2f} +/-{c:5.2f}   -> {verdict}")

    print("\nBY JOB RULE (executed makespan)")
    print(f"{'job_rule':<10} " + " ".join(f"{a:>9}" for a, _ in ARMS))
    for jr in sorted({s[1] for s in slots}):
        cells = []
        for arm, _ in ARMS:
            rows = [data[arm][s] for s in slots if s[1] == jr]
            cells.append(f"{mean_over_clean(rows,'executed'):>9.1f}")
        print(f"{jr:<10} " + " ".join(cells))

    print("\nBEST AMR RULE WITHIN EACH ARM (executed, averaged over instances+job rules)")
    for arm, _ in ARMS:
        by_rule = {}
        for s in slots:
            r = data[arm][s]
            if r["nu"] == 0:
                by_rule.setdefault(s[2], []).append(r["executed"])
        ranked = sorted(((statistics.mean(v), k) for k, v in by_rule.items()))
        best = ", ".join(f"{k} {v:.1f}" for v, k in ranked[:3])
        print(f"  {arm:<8} {best}")


if __name__ == "__main__":
    main()
