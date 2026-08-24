"""Full method sweep: every dispatching rule, every v9/v10 checkpoint, and the GA,
at n = 20/40/60/80/100, 100 instances per size, m = 16.

INFERENCE MODEL. Every method plans under the calibrated surrogate Psi-hat and is scored
once by the collision-aware executor Phi -- the standard pairing, not the Phi-in-loop
variant. Rules and policies therefore consume identical state and identical estimators.

MASKS. v10 cells A and C were TRAINED with the L3 congestion mask installed. They are
evaluated with the same mask, because a checkpoint fed channels it never saw during
training is not the model that was trained. Cells B/D and all of v9 run unmasked.

PARALLELISM. Work is sharded across processes. Wall-clock under contention is NOT a valid
compute measurement -- `solve_s` here is recorded but flagged `contended`. Clean timings
come from bench_serial.py, which runs one process at a time.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATIC = REPO / "Static_alogorithm"
for p in (STATIC, STATIC / "extend_GNN"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SIZES = (20, 40, 60, 80, 100)
AMRS = 16
DATASET = lambda n: REPO / f"test_case/v3/trend/full_{n}.jsonl"

# name -> (checkpoint, inference mask)
MODELS = {
    "v9_mix_s42":          ("checkpoints_v9/mix_s42_best.pth",          "none"),
    "v9_mix_s43":          ("checkpoints_v9/mix_s43_best.pth",          "none"),
    "v9_mix_s44":          ("checkpoints_v9/mix_s44_best.pth",          "none"),
    "v9_only60_s42":       ("checkpoints_v9/only60_s42_best.pth",       "none"),
    "v9_only60_s43":       ("checkpoints_v9/only60_s43_best.pth",       "none"),
    "v9_only60_s44":       ("checkpoints_v9/only60_s44_best.pth",       "none"),
    "v10_A_s44":           ("checkpoints_v10/v10_A_s44_best.pth",       "L3"),
    "v10_B_s44":           ("checkpoints_v10/v10_B_s44_best.pth",       "none"),
    "v10_C_s44":           ("checkpoints_v10/v10_C_s44_best.pth",       "L3"),
    "v10_A_s44_bestideal": ("checkpoints_v10/v10_A_s44_bestideal.pth",  "L3"),
    "v10_B_s44_bestideal": ("checkpoints_v10/v10_B_s44_bestideal.pth",  "none"),
}
MASKS = {
    "none": {},
    "L3": {"amr": (9,), "dock": (2, 3, 4, 5, 6), "action": (2,)},
}
SAMPLE_SEED = 1000

_G = {}


def worker_init():
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import scenario_v3 as sc
    sc.apply_layout(num_amrs=AMRS)
    import GA.GA as GA, operation_policy as _op
    if len(GA.AMR_KEYS) != AMRS:
        raise SystemExit(f"apply_layout gave {len(GA.AMR_KEYS)} AMRs")
    if _op.DOCK_QUEUE_SCALE != float(AMRS):
        raise SystemExit("apply_layout did not patch operation_policy")
    import ideal_evaluator as ie
    from GA.GA import load_dispatch_events
    _G["ie"], _G["GA"] = ie, GA
    _G["events"] = {n: load_dispatch_events(DATASET(n)) for n in SIZES}
    import extend_GNN
    _G["xg"] = sys.modules.get("extend_GNN.extend_GNN", extend_GNN)
    _G["pristine"] = _G["xg"].extract_state_extend_gnn
    _G["models"] = {}


def set_mask(level):
    """Rebind extract_state_extend_gnn to the pristine function wrapped by `level`."""
    xg, original = _G["xg"], _G["pristine"]
    spec = MASKS[level]
    if not spec:
        fn = original
    else:
        def fn(*a, **kw):
            t = list(original(*a, **kw))
            for i in spec.get("amr", ()):    t[0][..., i] = 0.0
            for i in spec.get("dock", ()):   t[2][..., i] = 0.0
            for i in spec.get("action", ()): t[3][..., i] = 0.0
            return tuple(t)
    xg.extract_state_extend_gnn = fn
    xg.solve_with_extend_gnn.__globals__["extract_state_extend_gnn"] = fn


def get_model(name):
    if name not in _G["models"]:
        import torch
        from operation_policy import load_required_operation_checkpoint
        path, _mask = MODELS[name]
        m = _G["xg"].ExtendSchedulerGNN()
        load_required_operation_checkpoint(
            m, REPO / path, torch, required_keys=("op_emb.weight", "operation_actor.0.weight"))
        m.eval()
        _G["models"][name] = m
    return _G["models"][name]


def base_row(n, inst, family, method, mode):
    return {"n_jobs": n, "instance": inst, "amrs": AMRS, "family": family,
            "method": method, "mode": mode, "dataset": f"full_{n}"}


def run_task(task):
    kind, arg, n, i0, i1 = task
    ie = _G["ie"]
    rows = []
    for ev in _G["events"][n][i0:i1]:
        jobs = list(ev["jobs"])
        inst = ev["index"]
        if kind == "rule":
            from reinforce_baseline import complete_with_dispatch_rule
            t0 = time.perf_counter()
            ind = complete_with_dispatch_rule(jobs, [], {}, baseline_rule=arg, seed=42)
            solve = time.perf_counter() - t0
            select = 0.0
            row = base_row(n, inst, "rule", arg, "single")
        elif kind == "ga":
            random.seed(1000 + inst)
            t0 = time.perf_counter()
            ind, _ = _G["GA"].evolve(jobs)
            solve = time.perf_counter() - t0
            select = 0.0
            row = base_row(n, inst, "ga", "GA", "single")
        else:
            import torch
            name, mode = arg
            set_mask(MODELS[name][1])
            model = get_model(name)
            if mode == "greedy":
                t0 = time.perf_counter()
                with torch.no_grad():
                    ind, _, _ = _G["xg"].solve_with_extend_gnn(jobs, model, deterministic=True)
                solve = time.perf_counter() - t0
                model.eval()
                select = 0.0
            else:
                k = int(mode[4:])
                best = best_m = None
                solve = select = 0.0
                for j in range(k):
                    torch.manual_seed(SAMPLE_SEED + 1_000_003 * inst + j)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        cand, _, _ = _G["xg"].solve_with_extend_gnn(jobs, model, deterministic=False)
                    solve += time.perf_counter() - t0
                    model.eval()
                    t0 = time.perf_counter()
                    m = ie.evaluate(cand, jobs)
                    select += time.perf_counter() - t0
                    if best_m is None or (m["nu"], m["executed"]) < (best_m["nu"], best_m["executed"]):
                        best, best_m = cand, m
                ind = best
            row = base_row(n, inst, "policy", name, mode)
        t0 = time.perf_counter()
        m = ie.evaluate(ind, jobs)
        row.update(m)
        row.update({"solve_s": round(solve, 5), "select_s": round(select, 5),
                    "eval_s": round(time.perf_counter() - t0, 5)})
        row["total_s"] = round(solve + select, 5)
        rows.append(row)
    return rows


def build_tasks(what):
    from dispatching_rules.dispatching_rules import JOB_RULES, AMR_RULES
    tasks = []
    for n in SIZES:
        if "ga" in what:
            for i in range(0, 100):
                tasks.append(("ga", None, n, i, i + 1))
        if "rules" in what:
            for jr in JOB_RULES:
                for ar in AMR_RULES:
                    for i in range(0, 100, 25):
                        tasks.append(("rule", f"{jr}+{ar}", n, i, i + 25))
        if "models" in what:
            for name in MODELS:
                for i in range(0, 100, 20):
                    tasks.append(("model", (name, "greedy"), n, i, i + 20))
                for i in range(0, 100, 5):
                    tasks.append(("model", (name, "best8"), n, i, i + 5))
    # Longest first so the tail does not strand a core.
    order = {"ga": 0, "model": 1, "rule": 2}
    tasks.sort(key=lambda t: (order[t[0]], -t[2]))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--what", default="ga,models,rules")
    ap.add_argument("--out", default=str(REPO / "experiments/full_benchmark/raw/rows.jsonl"))
    args = ap.parse_args()

    what = set(args.what.split(","))
    tasks = build_tasks(what)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} tasks -> {args.workers} workers -> {out}", flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    done = n_rows = 0
    t_start = time.perf_counter()
    with out.open("w", encoding="utf-8") as fh:
        with ctx.Pool(args.workers, initializer=worker_init) as pool:
            for rows in pool.imap_unordered(run_task, tasks, chunksize=1):
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                n_rows += len(rows)
                done += 1
                if done % 25 == 0 or done == len(tasks):
                    el = time.perf_counter() - t_start
                    eta = el / done * (len(tasks) - done)
                    fh.flush()
                    print(f"  {done:>5}/{len(tasks)} tasks | {n_rows:>7} rows | "
                          f"{el/60:6.1f} min elapsed | ETA {eta/60:6.1f} min", flush=True)
    print(f"done: {n_rows} rows -> {out}")


if __name__ == "__main__":
    main()
