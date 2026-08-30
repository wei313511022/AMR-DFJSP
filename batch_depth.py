"""Batching depth, two definitions. Generation only -- no evaluator is called."""
import json, sys, time, os
from collections import defaultdict
from pathlib import Path
REPO = Path("/home/wei/Desktop/AMR-DFJSP")
for p in (REPO/"Static_alogorithm", REPO/"Static_alogorithm"/"extend_GNN"):
    sys.path.insert(0, str(p))
SIZES=(20,40,60,80,100); N=100
RULES=[f"{j}+{a}" for j in ("fifo","spt","lpt","milk_run","material_match","earliest_completion_job")
       for a in ("earliest_available","earliest_completion")]
POLICIES={"A":("checkpoints_v10/v10_A_s44_best.pth","L3"),"B":("checkpoints_v10/v10_B_s44_best.pth","none"),
          "C":("checkpoints_v10/v10_C_s44_best.pth","L3"),"D":("checkpoints_v9/only60_s44_best.pth","none"),
          "E":("checkpoints_v10/v10_E_s44_best.pth","L3"),"F":("checkpoints_v10/v10_F_s44_best.pth","none")}
MASKS={"none":{},"L3":{"amr":(9,),"dock":(2,3,4,5,6),"action":(2,)}}
_G={}
def init():
    import torch; torch.set_num_threads(1); os.environ.setdefault("OMP_NUM_THREADS","1")
    import scenario_v3 as sc; sc.apply_layout(num_amrs=16)
    import GA.GA as GA
    from GA.GA import load_dispatch_events
    import extend_GNN
    _G["GA"]=GA; _G["events"]={n:load_dispatch_events(REPO/f"test_case/v3/trend/full_{n}.jsonl") for n in SIZES}
    _G["xg"]=sys.modules.get("extend_GNN.extend_GNN",extend_GNN)
    _G["pristine"]=_G["xg"].extract_state_extend_gnn; _G["models"]={}
def set_mask(level):
    xg,orig=_G["xg"],_G["pristine"]; spec=MASKS[level]
    if not spec: fn=orig
    else:
        def fn(*a,**kw):
            t=list(orig(*a,**kw))
            for i in spec.get("amr",()): t[0][...,i]=0.0
            for i in spec.get("dock",()): t[2][...,i]=0.0
            for i in spec.get("action",()): t[3][...,i]=0.0
            return tuple(t)
    xg.extract_state_extend_gnn=fn
    xg.solve_with_extend_gnn.__globals__["extract_state_extend_gnn"]=fn
def get_model(c):
    if c not in _G["models"]:
        import torch
        from operation_policy import load_required_operation_checkpoint
        m=_G["xg"].ExtendSchedulerGNN()
        load_required_operation_checkpoint(m,REPO/POLICIES[c][0],torch,
            required_keys=("op_emb.weight","operation_actor.0.weight")); m.eval()
        _G["models"][c]=m
    return _G["models"][c]
def profile(ind, jobs):
    """A trip is a maximal run with the rack non-empty.

    peak_per_trip  parcels aboard AT ONCE, at the fullest point of the trip. Bounded by the
                   3-slot rack. This is 'how many does it collect before delivering'.
    moved_per_trip parcels picked up during the run. A robot that is never simultaneously
                   empty accumulates one long run, so this can exceed the rack -- it measures
                   'how long does it go without returning to empty', a different quantity.
    """
    GA=_G["GA"]
    ops=GA.repair_operation_order(ind.order,list(jobs))
    per=defaultdict(list)
    for o in ops: per[ind.amr_assignment[o.job_idx]].append(o)
    peaks=[]; moved=[]
    for seq in per.values():
        ob=0; pk=0; mv=0
        for o in seq:
            if o.kind==GA.PICKUP:
                ob+=1; mv+=1; pk=max(pk,ob)
            else:
                ob-=1
                if ob==0: peaks.append(pk); moved.append(mv); pk=0; mv=0
        if pk: peaks.append(pk); moved.append(mv)
    return (sum(peaks)/len(peaks) if peaks else float("nan"),
            sum(moved)/len(moved) if moved else float("nan"), max(peaks) if peaks else 0)
def task(t):
    kind,arg,n=t; rows=[]
    for ev in _G["events"][n][:N]:
        jobs=list(ev["jobs"])
        if kind=="rule":
            from reinforce_baseline import complete_with_dispatch_rule
            ind=complete_with_dispatch_rule(jobs,[],{},baseline_rule=arg,seed=42)
        else:
            import torch; set_mask(POLICIES[arg][1]); m=get_model(arg)
            with torch.no_grad(): ind,_,_=_G["xg"].solve_with_extend_gnn(jobs,m,deterministic=True)
            m.eval()
        pk,mv,mx=profile(ind,jobs)
        rows.append({"n_jobs":n,"instance":ev["index"],"kind":kind,"method":arg,
                     "peak_per_trip":round(pk,4),"moved_per_trip":round(mv,4),"max_onboard":mx})
    return rows
if __name__=="__main__":
    import multiprocessing as mp
    tasks=[("rule",r,n) for n in SIZES for r in RULES]+[("policy",c,n) for n in SIZES for c in POLICIES]
    out=Path("/tmp/claude-1001/-home-wei-Desktop-AMR-DFJSP/53d60f24-3f89-42ef-80b3-168fed5e6f0a/scratchpad/batch2.jsonl")
    with out.open("w") as fh, mp.get_context("spawn").Pool(20,initializer=init) as pool:
        for i,rows in enumerate(pool.imap_unordered(task,tasks),1):
            for r in rows: fh.write(json.dumps(r)+"\n")
            fh.flush()
    print("done", out)
