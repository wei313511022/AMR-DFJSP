# FJSP Simple Dispatching Rules (Standalone)

Each DPR script reads one instance from the JSONL dataset in the repository
root: `fjssp_training_dataset.jsonl`.

The DPR scripts now use the route-aware shared dispatcher in
`fjssp_dispatch.py`. Travel distances are calculated with `calcu_dist.py` using
the grid, station, material, AMR, and barrier settings in `map.py`.

## Usage

Run a specific rule on one instance (default index = 0):

```bash
python DPR-1.py
```

Select a different instance and seed:

```bash
python DPR-2.py --index 5 --seed 123
```

Override input and output paths:

```bash
python DPR-3.py --data ..\fjssp_training_dataset.jsonl --out dpr-3_out.jsonl --fig-out dpr-3_out.png
```

Skip PNG output:

```bash
python DPR-4.py --no-fig
```

## Route timing

- Machine ids `0..5` map to station ids `1..6` in `map.JSON_STATION_MAPPING`.
- The first operation starts from the job material node in
  `map.TYPE_TO_MATERIAL_NODE`.
- Later operations pick up from the previous operation's station node.
- Routes avoid `map.BARRIER_NODES`.
- AMR paths may overlap; no collision or edge-capacity constraint is added.
- AMR availability is still respected: one AMR handles one scheduled operation
  timeline at a time.

## Output format

Each line in the output JSONL represents a scheduled operation with keys:

- `job`, `op_index`, `machine`
- `amr`, `material`
- `processing`, `start`, `end`
- `pickup_node`, `delivery_node`, `amr_prev_node`
- `amr_ready`, `depart`, `arrival`
- `to_pick_travel`, `transport`, `travel_time`, `machine_wait`
- `route_nodes`, `route_legs`
- `rule`, `instance_index`, `makespan`

Each run also writes a PNG timeline by default, for example
`dpr-1_schedule.png`.
