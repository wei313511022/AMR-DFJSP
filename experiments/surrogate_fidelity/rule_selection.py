"""Why Table IV scores 12 rule combinations and not the full 60-combination grid.

Two claims have to be true for the subset to be defensible, and both are measured here
rather than asserted:

  PERFORMANCE  the 12 are the better half of the grid. Their rank is taken from
               ../full_benchmark/results.csv -- the same 5 sizes x 100 instances the main
               benchmark reports -- so this is not a separate experiment with its own
               sampling error.

  BATCHING     the 12 span the transport policy that actually distinguishes a schedule in
               this problem: how many parcels a robot accumulates before it delivers.
               Measured as pickups per trip from the committed order in fidelity_v2.jsonl.
               A trip is a maximal run from an empty rack back to empty, so a rule that
               fetches one parcel and returns scores 1.00.

The excluded half matters for a reason beyond weakness. `random`, `nearest_station` and
`least_congested_station` execute 2-4x worse than the field; a candidate that bad is ranked
correctly by every evaluator, so adding it raises tau for C~ and Psi-hat alike without
telling us anything about which is the better ranker. Reporting them would inflate the
headline, not stress it.

    python rule_selection.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FB = HERE.parent / "full_benchmark" / "results.csv"

JOB_RULES = ("fifo", "spt", "lpt", "milk_run", "material_match", "earliest_completion_job")
AMR_RULES = ("earliest_available", "earliest_completion")
CHOSEN = [f"{j}+{a}" for j in JOB_RULES for a in AMR_RULES]


def main() -> None:
    rows = [r for r in csv.DictReader(FB.open()) if r["family"] == "rule"]
    by = defaultdict(dict)
    for r in rows:
        by[int(r["n_jobs"])][r["method"]] = float(r["makespan_mean"])
    sizes = sorted(by)

    rank = defaultdict(list)
    for n in sizes:
        for i, k in enumerate(sorted(by[n], key=lambda k: by[n][k]), 1):
            rank[k].append(i)
    mean_rank = {k: st.mean(v) for k, v in rank.items()}
    order = sorted(mean_rank, key=lambda k: mean_rank[k])
    pos = {k: i + 1 for i, k in enumerate(order)}

    # Batching depth, from the fidelity run if it exists.
    load = defaultdict(lambda: defaultdict(list))
    fj = HERE / "fidelity_v2.jsonl"
    if fj.exists():
        for line in fj.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            load[r["method"]][r["n_jobs"]].append(r["load_per_trip"])

    out = []
    out.append("WHY THESE 12 RULE COMBINATIONS")
    out.append("=" * 108)
    out.append(f"\nRank is over the {len(mean_rank)}-combination grid in {FB.relative_to(HERE.parent.parent)},")
    out.append(f"{len(sizes)} sizes x 100 instances. load/trip is parcels picked up per trip, from fidelity_v2.jsonl.")
    hdr = (f"\n  {'combination':44s}{'rank':>6}{'mean rank':>11}   "
           + "".join(f"{n:>7d}" for n in sizes) + f"{'load/trip':>11}")
    out.append(hdr)
    out.append("  " + "-" * 104)
    for k in sorted(CHOSEN, key=lambda k: mean_rank[k]):
        ld = [x for n in sizes for x in load[k][n]]
        lds = f"{st.mean(ld):.2f}" if ld else "--"
        out.append(f"  {k:44s}{pos[k]:>6}{mean_rank[k]:>11.1f}   "
                   + "".join(f"{by[n][k]:7.1f}" for n in sizes) + f"{lds:>11}")
    sel = [pos[k] for k in CHOSEN]
    out.append(f"\n  all 12 fall in ranks {min(sel)}-{max(sel)} of {len(mean_rank)}; "
               f"{sum(1 for p in sel if p <= 13)} are in the top 13.")

    out.append("\n\nWHAT WAS EXCLUDED  worst 10 of the grid, for contrast")
    out.append(f"  {'combination':44s}{'rank':>6}{'mean rank':>11}   " + "".join(f"{n:>7d}" for n in sizes))
    out.append("  " + "-" * 93)
    for k in order[-10:]:
        out.append(f"  {k:44s}{pos[k]:>6}{mean_rank[k]:>11.1f}   "
                   + "".join(f"{by[n][k]:7.1f}" for n in sizes))
    worst = order[-1]
    out.append(f"\n  {worst} is {by[sizes[-1]][worst]/min(by[sizes[-1]].values()):.1f}x "
               f"the best combination at n={sizes[-1]}.")

    if load:
        out.append("\n\nBATCHING DEPTH BY SIZE  parcels per trip, over the 12 chosen combinations")
        out.append(f"  {'combination':44s}" + "".join(f"{n:>9d}" for n in sizes))
        out.append("  " + "-" * 89)
        for k in sorted(CHOSEN, key=lambda k: -st.mean([x for n in sizes for x in load[k][n]] or [0])):
            out.append(f"  {k:44s}" + "".join(
                f"{st.mean(load[k][n]):9.2f}" if load[k][n] else f"{'--':>9}" for n in sizes))
        out.append("\n  Other generators in the same candidate field, for range:")
        for k in sorted(set(load) - set(CHOSEN), key=lambda k: -st.mean(
                [x for n in sizes for x in load[k][n]] or [0])):
            out.append(f"  {k:44s}" + "".join(
                f"{st.mean(load[k][n]):9.2f}" if load[k][n] else f"{'--':>9}" for n in sizes))

    text = "\n".join(out)
    (HERE / "rule_selection.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
