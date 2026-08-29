"""How much of a dispatching rule's behaviour is decided by an arbitrary identifier?

Every rule scores a legal action with a lexicographic tuple

    (job_rule_score, amr_rule_score, unload_first_flag, action.job_id, action.amr)

and BOTH inner scores already end with an identifier of their own: every branch of
`_job_rule_score` ends with `job.idx`, every branch of `_amr_rule_score` ends with
`action.amr`. Those trailing elements make the order total, so a rollout is reproducible
-- but "AMR1 before AMR2" is not a scheduling argument. Wherever the substantive keys tie,
the decision is settled by naming.

Measured per rule combination and workload:

  AMR BY NAME    among the actions still alive after the job keys, do the substantive AMR
                 keys tie, leaving the alphabetical AMR name to choose? None of the five
                 AMR rules intends `action.amr` as anything but a tie-break, so this share
                 is arbitrary behaviour outright.
  JOB BY INDEX   same question for `job.idx` over the substantive job keys. Read this one
                 with care: for `fifo` the index IS the rule (arrival order), so a high
                 share there is intent, not artifact. For spt/lpt/milk_run it is a
                 tie-break between equal-duration jobs.
  PERTURBATION   rerun with the identifiers REVERSED -- AMR name only, then both -- and
                 compare executed makespan. The rules' intent is untouched; only the
                 arbitrary part flips. Movement here means the tie-break has consequences
                 and belongs in a limitation.

    python rule_tie_depth.py --sizes 20,60,100 --instances 10
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO / "Static_alogorithm", REPO / "Static_alogorithm" / "extend_GNN"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ideal_evaluator as ie                                 # noqa: E402
import scenario_v3 as sc                                     # noqa: E402
from dispatching_rules import dispatching_rules as dr        # noqa: E402
from GA.GA import load_dispatch_events                       # noqa: E402
from reinforce_baseline import complete_with_dispatch_rule   # noqa: E402

STATS = defaultdict(int)
AMR_TIE_SIZE = []
JOB_TIE_SIZE = []


def make_choose(reverse_amr, reverse_job, record):
    """choose_operation, instrumented, with the identifier keys optionally flipped.

    With reverse_* off the emitted schedule is identical to the stock implementation --
    the score tuple is rebuilt from the same private scorers, split only so the trailing
    identifier can be counted and inverted.
    """

    def choose(jobs, state, job_rule, amr_rule, rng):
        actions = dr.legal_actions(
            jobs, state.picked_jobs, state.completed_jobs, state.carrier_map, state.inventory,
        )
        if not actions:
            raise RuntimeError("no legal action")
        workloads = dr.active_station_workload(jobs, state.completed_jobs)
        est = [
            dr.estimate_action(
                a, jobs, state.amr_positions, state.amr_availabilities,
                state.station_availabilities, state.inventory, state.assigned_count,
                workloads, dock_service_events=state.dock_service_events,
            )
            for a in actions
        ]
        js = [dr._job_rule_score(e.action, e, jobs, state, job_rule, rng) for e in est]
        as_ = [dr._amr_rule_score(e.action, e, jobs, state, amr_rule, rng) for e in est]
        # every branch of both scorers ends with its identifier; split it off
        job_sub, job_id = [s[:-1] for s in js], [s[-1] for s in js]
        amr_sub, amr_id = [s[:-1] for s in as_], [s[-1] for s in as_]

        if record:
            STATS["decisions"] += 1
            STATS["actions"] += len(est)
            n = range(len(est))

            best = min(job_sub[i] for i in n)
            s1 = [i for i in n if job_sub[i] == best]
            jobs_tied = {est[i].action.job_id for i in s1}
            if len(jobs_tied) > 1:
                STATS["job_by_index"] += 1
                JOB_TIE_SIZE.append(len(jobs_tied))

            best = min((job_sub[i], job_id[i]) for i in n)
            s2 = [i for i in n if (job_sub[i], job_id[i]) == best]
            best = min(amr_sub[i] for i in s2)
            s3 = [i for i in s2 if amr_sub[i] == best]
            amrs_tied = {est[i].action.amr for i in s3}
            if len(amrs_tied) > 1:
                STATS["amr_by_name"] += 1
                AMR_TIE_SIZE.append(len(amrs_tied))
            if len(s3) == 1:
                STATS["unique_before_amr_name"] += 1

        jsign = -1 if reverse_job else 1
        def key(i):
            a = est[i].action
            jid = job_id[i]
            aid = -a.amr_idx if reverse_amr else amr_id[i]
            return (
                (*job_sub[i], jsign * jid if isinstance(jid, (int, float)) else jid),
                (*amr_sub[i], aid),
                0 if a.kind == dr.UNLOAD else 1,
                jsign * a.job_id,
                -a.amr_idx if reverse_amr else a.amr,
            )

        best_i = min(range(len(est)), key=key)
        return est[best_i].action, est[best_i]

    return choose


def run_combo(jobs, job_rule, amr_rule, reverse_amr=False, reverse_job=False, record=False):
    original = dr.choose_operation
    dr.choose_operation = make_choose(reverse_amr, reverse_job, record)
    try:
        return complete_with_dispatch_rule(
            jobs, [], {}, baseline_rule=f"{job_rule}+{amr_rule}", seed=42,
        )
    finally:
        dr.choose_operation = original


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="20,60,100")
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--amrs", type=int, default=16)
    ap.add_argument("--job-rules", default="fifo,spt,lpt,milk_run,material_match,earliest_completion_job")
    ap.add_argument("--amr-rules", default="earliest_available,earliest_completion,least_loaded,nearest_amr")
    args = ap.parse_args()

    sc.apply_layout(num_amrs=args.amrs)
    sizes = [int(s) for s in args.sizes.split(",")]
    rows = []

    for n in sizes:
        events = load_dispatch_events(
            REPO / "test_case" / "v3" / "trend" / f"full_{n}.jsonl")[: args.instances]
        for jr in args.job_rules.split(","):
            for ar in args.amr_rules.split(","):
                STATS.clear()
                AMR_TIE_SIZE.clear()
                JOB_TIE_SIZE.clear()
                base, d_amr, d_both, moved_amr, moved_both = [], [], [], 0, 0
                for ev in events:
                    jobs = list(ev["jobs"])
                    m0 = ie.evaluate(run_combo(jobs, jr, ar, record=True), jobs)["executed"]
                    m1 = ie.evaluate(run_combo(jobs, jr, ar, reverse_amr=True), jobs)["executed"]
                    m2 = ie.evaluate(run_combo(jobs, jr, ar, reverse_amr=True, reverse_job=True), jobs)["executed"]
                    base.append(m0)
                    d_amr.append(100 * (m1 - m0) / m0)
                    d_both.append(100 * (m2 - m0) / m0)
                    moved_amr += m1 != m0
                    moved_both += m2 != m0
                d = STATS["decisions"]
                k = len(events)
                rows.append({
                    "n_jobs": n, "job_rule": jr, "amr_rule": ar,
                    "instances": k, "decisions": d,
                    "legal_actions_avg": round(STATS["actions"] / d, 1),
                    "amr_by_name_pct": round(100 * STATS["amr_by_name"] / d, 1),
                    "amr_tie_size_avg": round(statistics.mean(AMR_TIE_SIZE), 2) if AMR_TIE_SIZE else 0.0,
                    "job_by_index_pct": round(100 * STATS["job_by_index"] / d, 1),
                    "job_tie_size_avg": round(statistics.mean(JOB_TIE_SIZE), 2) if JOB_TIE_SIZE else 0.0,
                    "rev_amr_moves_pct": round(100 * moved_amr / k, 1),
                    "rev_amr_mean_delta_pct": round(statistics.mean(d_amr), 2),
                    "rev_amr_meanabs_pct": round(statistics.mean(abs(x) for x in d_amr), 2),
                    "rev_amr_absmax_pct": round(max(abs(x) for x in d_amr), 2),
                    "rev_both_moves_pct": round(100 * moved_both / k, 1),
                    "rev_both_mean_delta_pct": round(statistics.mean(d_both), 2),
                    "rev_both_meanabs_pct": round(statistics.mean(abs(x) for x in d_both), 2),
                    "rev_both_absmax_pct": round(max(abs(x) for x in d_both), 2),
                    "deltas_amr_pct": [round(x, 3) for x in d_amr],
                    "deltas_both_pct": [round(x, 3) for x in d_both],
                    "makespan_mean": round(statistics.mean(base), 1),
                })
                r = rows[-1]
                print(f"  n={n:>3} {jr}+{ar:<20} amr-by-name {r['amr_by_name_pct']:>5.1f}%"
                      f"  job-by-index {r['job_by_index_pct']:>5.1f}%"
                      f"  | rev-amr moves {r['rev_amr_moves_pct']:>5.1f}% mean|d| {r['rev_amr_meanabs_pct']:>5.2f}%"
                      f"  rev-both moves {r['rev_both_moves_pct']:>5.1f}% mean|d| {r['rev_both_meanabs_pct']:>5.2f}%",
                      flush=True)

    (HERE / "rule_tie_depth.json").write_text(json.dumps(rows, indent=1))
    flat = [{k: v for k, v in r.items() if not k.startswith("deltas_")} for r in rows]
    with (HERE / "rule_tie_depth.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    print(f"\nrows -> {HERE / 'rule_tie_depth.csv'}")


if __name__ == "__main__":
    main()
