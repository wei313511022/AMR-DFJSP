"""Scenario v2: open-floor AMR cross-dock with nested slot racks and outbound deadlines.

Centralises every constant that changed between the original 10x10 / 3-AMR setting
and the design in paper/problem_formulation.tex, so the old scenario stays runnable
and the two can be compared.

Key changes vs v1
-----------------
  layout    20x19 open floor, 5 inbound doors, 5 outbound stations, no obstacles
  fleet     12 AMRs parked in a wall bank at x=2  (contention ratio eta = 12/5 = 2.4)
  capacity  3x3 slot rack with DOWNWARD SUBSTITUTION -> nested constraint
            n_C <= 3,  n_B + n_C <= 6,  n_A + n_B + n_C <= 9
  time      all releases at t=0; time structure carried by shipment deadlines

Usage
-----
    import scenario_v2 as sc
    sc.apply_layout()          # mutate GA globals in place, before building jobs

`apply_layout` mutates the module-level containers in GA.GA in place (clear/update)
rather than rebinding them, because every other module does `from GA.GA import ...`
and holds references to those exact objects.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import GA.GA as GA

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

GRID_W = 19          # x in [0, GRID_W]
GRID_H = 19          # y in [1, GRID_H]
NUM_DOORS = 5        # per side
# eta = 16/5 = 3.2. Chosen from the contention sweep, not by convention: it is the
# lowest fleet at which the congestion tax reaches the level the paper's premise
# needs (24.2% vs 2.5% at eta=2.4), and where consolidation's advantage over
# single-trip dispatch has decayed from 26% to 5.5% -- i.e. where the choice of
# strategy is genuinely open and a policy can beat a committed rule.
NUM_AMRS = 16
# Leftmost charging-bank column. MUST clear the dock waiting areas: `dock_waiting_slots`
# fans out WAIT_LINE_DEPTH (=3) cells inward from each door, so queue cells reach x=3.
# With the bank at x=2 the bays sat *inside* the queues, and a robot assigned a waiting
# slot that happened to be another robot's parked bay could never arrive -- the failures
# were all `inbound_line` legs. x=5 leaves a two-cell margin.
DEPOT_X = 5
DEPOT_ROWS_PER_COL = 4   # bank is a block: 4 bays per column, so it never walls off
DEPOT_PODS = True        # 2x2 charging pods beside each door row
# Bays in dedicated rows below the operating floor. Intuitive, but MEASURED WORSE:
# only grid_w - 2*(WAIT_LINE_DEPTH+1) = 12 bays fit per row, so 16+ AMRs need two rows,
# and the fully-packed row y=0 becomes a horizontal wall that the y=-1 robots must
# cross. Same fault as the original single-column bank, rotated 90 degrees.
#   m=16: unroutable 0.44 (vs 0.00 for the x=5 block),  tax  9.5% (vs 4.1%)
#   m=24: unroutable 3.72 (vs 0.17),                    tax 30.8% (vs 5.7%)
# Worth revisiting only with staggered bays (alternate rows on alternate columns) so
# every column keeps a clear vertical channel.
DEPOT_BOTTOM_ROW = False

# Door rows, inset one cell from each wall so that EVERY service point has a full
# waiting set on both sides. `dock_waiting_slots` queues in the rows adjacent to a
# door; a door on row 1 or 19 loses one of those rows to the boundary and gets half
# the queue capacity of the others, which makes the five servers non-exchangeable
# for reasons unrelated to the scheduling problem.
#   y=2  -> waits on rows 1 and 3      y=14 -> rows 13 and 15
#   y=6  -> rows 5 and 7               y=18 -> rows 17 and 19
#   y=10 -> rows 9 and 11
DOOR_ROWS: Tuple[int, ...] = (2, 6, 10, 14, 18)


def contention_ratio(num_amrs: int = NUM_AMRS, num_doors: int = NUM_DOORS) -> float:
    """eta = robots per inbound service point. The governing density parameter."""
    return num_amrs / float(num_doors)


def apply_layout(
    grid_w: int = GRID_W,
    grid_h: int = GRID_H,
    num_amrs: int = NUM_AMRS,
    door_rows: Sequence[int] = DOOR_ROWS,
    depot_x: int = DEPOT_X,
) -> None:
    """Install the v2 facility into the GA module globals (in place)."""
    rows = list(door_rows)
    docks = {f"dock{i + 1}": (0, rows[i]) for i in range(len(rows))}
    stations = {f"station{i + 1}": (grid_w, rows[i]) for i in range(len(rows))}

    GA.INBOUND_DOCK_LOCATIONS.clear()
    GA.INBOUND_DOCK_LOCATIONS.update(docks)
    GA.INBOUND_DOCK_KEYS.clear()
    GA.INBOUND_DOCK_KEYS.extend(docks.keys())

    GA.STATIONS.clear()
    GA.STATIONS.update(stations)

    GA.SUPPLY_LOCATIONS.clear()
    GA.SUPPLY_LOCATIONS.update(docks)

    GA.DOCK_SERVICE_CELLS.clear()
    GA.DOCK_SERVICE_CELLS.update(set(docks.values()) | set(stations.values()))

    # Open floor: cross-docking holds no storage, so there is nothing to route around.
    GA.OBSTACLES.clear()

    if DEPOT_PODS:
        # 2x2 charging pods at x = DEPOT_X, DEPOT_X+1, one pod beside each door row,
        # extending away from the mid-row so the central lane stays clear. Sparse in
        # both axes, so no column and no row is ever cut.
        mid = (1 + grid_h) / 2.0
        pods = []
        for d in sorted(rows, key=lambda r: abs(r - mid), reverse=True):
            second = d + 1 if d < mid else d - 1
            if 1 <= second <= grid_h:
                pods.append((d, second))
        if num_amrs > 4 * len(pods):
            raise ValueError(
                f"{num_amrs} AMRs need {-(-num_amrs // 4)} pods but only {len(pods)} exist "
                f"({4 * len(pods)} bays). Add pods at non-door rows or widen them; "
                f"cycling would assign two AMRs the same bay, which permanently blocks it."
            )
        starts = {}
        for i in range(num_amrs):
            pod = pods[i // 4]
            y = pod[(i % 4) // 2]
            x = depot_x + (i % 2)
            starts[f"AMR{i + 1}"] = (x, y)
        GA.AMR_STARTS.clear(); GA.AMR_STARTS.update(starts)
        GA.AMR_KEYS.clear(); GA.AMR_KEYS.extend(starts.keys())
        GA.BASES.clear(); GA.BASES.extend(starts.values())
        GA.GRID_MIN_X, GA.GRID_MAX_X = 0, grid_w
        GA.GRID_MIN_Y, GA.GRID_MAX_Y = 1, grid_h
        for cached in (GA.dock_waiting_slots, GA.shortest_path):
            try: cached.cache_clear()
            except AttributeError: pass
        try:
            import operation_policy as _op
            _op.DOCK_KEYS[:] = list(docks.keys()) + list(stations.keys())
            _op.DOCK_QUEUE_SCALE = float(max(num_amrs, 1))
        except Exception:
            pass
        return

    if DEPOT_BOTTOM_ROW:
        # Charging bays live in dedicated rows BELOW the operating floor (y <= 0).
        # Doors and stations occupy y in [1, grid_h], so a parked AMR can never sit on
        # a travel lane or inside a dock waiting area -- the two geometry faults that
        # produced every routing failure. The depot rows are still traversable, so
        # they double as a bypass lane along the bottom edge.
        # Inset from both walls by WAIT_LINE_DEPTH+1: the queue fans at the y=1 doors
        # reach around the corner into the depot row, and a bay inside a queue is the
        # exact fault we are removing.
        margin = GA.WAIT_LINE_DEPTH + 1
        lo, hi = margin, grid_w - margin
        per_row = max(1, hi - lo + 1)
        starts = {}
        for i in range(num_amrs):
            row = -(i // per_row)                 # y = 0, -1, -2, ...
            slot = i % per_row
            starts[f"AMR{i + 1}"] = (lo + slot, row)
        GA.AMR_STARTS.clear(); GA.AMR_STARTS.update(starts)
        GA.AMR_KEYS.clear(); GA.AMR_KEYS.extend(starts.keys())
        GA.BASES.clear(); GA.BASES.extend(starts.values())
        GA.GRID_MIN_X, GA.GRID_MAX_X = 0, grid_w
        GA.GRID_MIN_Y = min(p[1] for p in starts.values())
        GA.GRID_MAX_Y = grid_h
        for cached in (GA.dock_waiting_slots, GA.shortest_path):
            try: cached.cache_clear()
            except AttributeError: pass
        try:
            import operation_policy as _op
            _op.DOCK_KEYS[:] = list(docks.keys()) + list(stations.keys())
            _op.DOCK_QUEUE_SCALE = float(max(num_amrs, 1))
        except Exception:
            pass
        return

    # Charging bank. Laid out as a BLOCK (several columns), never a single column.
    # A one-column bank spanning the full height becomes a wall: idle AMRs hold their
    # cells in the reservation table permanently, and since docks sit at x=0 and
    # stations at x=grid_w, every trip has to cross the bank. Measured free rows in a
    # single column x=2: 7 at m=12, 3 at m=16, 0 at m=20 -- i.e. the floor is
    # partitioned outright once the fleet parks, which showed up as "unroutable"
    # parcels rather than as congestion.
    starts = {}
    per_col = max(1, DEPOT_ROWS_PER_COL)
    for i in range(num_amrs):
        col = depot_x + (i // per_col)
        slot = i % per_col
        y = 1 + round((grid_h - 1) * slot / max(1, per_col - 1))
        starts[f"AMR{i + 1}"] = (col, y)
    GA.AMR_STARTS.clear()
    GA.AMR_STARTS.update(starts)
    GA.AMR_KEYS.clear()
    GA.AMR_KEYS.extend(starts.keys())
    GA.BASES.clear()
    GA.BASES.extend(starts.values())

    GA.GRID_MIN_X, GA.GRID_MAX_X = 0, grid_w
    GA.GRID_MIN_Y, GA.GRID_MAX_Y = 1, grid_h

    for cached in (GA.dock_waiting_slots, GA.shortest_path):
        try:
            cached.cache_clear()
        except AttributeError:
            pass

    # operation_policy builds DOCK_KEYS at import time from the pre-v2 dicts.
    try:
        import operation_policy as _op

        _op.DOCK_KEYS[:] = list(docks.keys()) + list(stations.keys())
        _op.DOCK_QUEUE_SCALE = float(max(num_amrs, 1))
    except Exception:
        pass


# --------------------------------------------------------------------------
# Nested slot capacity
# --------------------------------------------------------------------------
# Size classes ordered smallest -> largest. A parcel of class s occupies one slot
# of class s or larger, so the binding constraints are the suffix sums.

SIZE_ORDER: Tuple[str, ...] = ("A", "B", "C")
SLOT_CAPACITY: Dict[str, int] = {"A": 3, "B": 3, "C": 3}
TOTAL_SLOTS = sum(SLOT_CAPACITY.values())

# Suffix capacity: SUFFIX_CAP[s] = number of slots able to hold class s or larger.
SUFFIX_CAP: Dict[str, int] = {
    s: sum(SLOT_CAPACITY[t] for t in SIZE_ORDER[i:]) for i, s in enumerate(SIZE_ORDER)
}


def _counts_of(inventory_entry: Mapping[str, object]) -> Dict[str, int]:
    """Accept either {class: int} or {class: [job ids]} inventory representations."""
    out = {}
    for s in SIZE_ORDER:
        v = inventory_entry.get(s, 0) if inventory_entry else 0
        out[s] = len(v) if isinstance(v, (list, tuple, set)) else int(v or 0)
    return out


def nested_capacity_ok(counts: Mapping[str, int]) -> bool:
    """True iff `counts` satisfies every suffix-sum constraint."""
    for i, s in enumerate(SIZE_ORDER):
        if sum(counts.get(t, 0) for t in SIZE_ORDER[i:]) > SUFFIX_CAP[s]:
            return False
    return True


def can_load(inventory_entry: Mapping[str, object], size_class: str) -> bool:
    """True iff one more parcel of `size_class` fits in the rack."""
    counts = _counts_of(inventory_entry)
    counts[size_class] = counts.get(size_class, 0) + 1
    return nested_capacity_ok(counts)


def slack_slots(inventory_entry: Mapping[str, object], size_class: str) -> int:
    """Remaining slots usable by `size_class` (the tightest binding suffix)."""
    counts = _counts_of(inventory_entry)
    idx = SIZE_ORDER.index(size_class)
    room = TOTAL_SLOTS
    for i, s in enumerate(SIZE_ORDER[: idx + 1]):
        used = sum(counts.get(t, 0) for t in SIZE_ORDER[i:])
        room = min(room, SUFFIX_CAP[s] - used)
    return max(0, room)


# --------------------------------------------------------------------------
# Size-class mix
# --------------------------------------------------------------------------
# Real parcel mixes are skewed toward small. The skew is deliberate: it is both more
# faithful than a uniform draw and it makes the small-slot constraint bind more often,
# which is what gives the consolidation claim empirical support.

# Uniform. A skewed (60/30/10) mix is more faithful to real parcel streams, but
# measured on identical layouts it cuts the congestion tax from 15.9% to 5.8%:
# a small-parcel-heavy mix has a mean service time of 7.5 vs 10.0, so the doors
# are occupied less and queue less. Signal strength wins over that particular
# realism argument; note the choice as a limitation.
SIZE_MIX: Dict[str, float] = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}


def sample_size_class(rng) -> str:
    r = rng.random()
    acc = 0.0
    for s in SIZE_ORDER:
        acc += SIZE_MIX[s]
        if r < acc:
            return s
    return SIZE_ORDER[-1]


# --------------------------------------------------------------------------
# Shipments and deadlines
# --------------------------------------------------------------------------

SHIPMENT_MIN = 6
SHIPMENT_MAX = 10
DEFAULT_GAMMA = 1.0          # deadline tightness; sweep {0.8, 1.0, 1.2}


def partition_shipments(
    jobs: Sequence["GA.Job"],
    rng,
    lo: int = SHIPMENT_MIN,
    hi: int = SHIPMENT_MAX,
) -> Dict[int, List[int]]:
    """Group parcels sharing an outbound station into trailer-sized shipments.

    Returns {shipment_id: [job idx, ...]}. Shipment ids are globally unique and
    ordered by station then by departure position, so id order is departure order.
    """
    by_station: Dict[str, List[int]] = {}
    for job in jobs:
        by_station.setdefault(job.station, []).append(job.idx)

    shipments: Dict[int, List[int]] = {}
    sid = 0
    for station in sorted(by_station):
        members = by_station[station]
        rng.shuffle(members)
        i = 0
        while i < len(members):
            size = rng.randint(lo, hi)
            chunk = members[i : i + size]
            # Absorb a runt tail into the previous chunk rather than emitting it.
            if 0 < len(members) - (i + size) < lo:
                chunk = members[i:]
                i = len(members)
            else:
                i += size
            shipments[sid] = chunk
            sid += 1
    return shipments


def assign_deadlines(
    shipments: Mapping[int, Sequence[int]],
    jobs: Sequence["GA.Job"],
    reference_makespan: float,
    gamma: float = DEFAULT_GAMMA,
) -> Dict[int, float]:
    """Staggered departures: D_g = gamma * C_ref * (i + 1) / K, per station.

    Trailers leave across the shift rather than all at the horizon end, which is
    what creates the "which shipment first" sequencing pressure.
    """
    station_of = {j.idx: j.station for j in jobs}
    by_station: Dict[str, List[int]] = {}
    for sid, members in shipments.items():
        if not members:
            continue
        by_station.setdefault(station_of[members[0]], []).append(sid)

    deadlines: Dict[int, float] = {}
    for station, sids in by_station.items():
        sids = sorted(sids)
        k = len(sids)
        for i, sid in enumerate(sids):
            deadlines[sid] = gamma * reference_makespan * (i + 1) / k
    return deadlines


def shipment_completions(
    jobs: Sequence["GA.Job"], completion_of: Mapping[int, float]
) -> Dict[int, float]:
    """C_g = max over parcels in g of delivery completion."""
    out: Dict[int, float] = {}
    for job in jobs:
        sid = getattr(job, "shipment_id", -1)
        if sid < 0:
            continue
        c = completion_of.get(job.idx)
        if c is None:
            continue
        out[sid] = max(out.get(sid, 0.0), float(c))
    return out


def total_tardiness(
    jobs: Sequence["GA.Job"], completion_of: Mapping[int, float]
) -> Tuple[float, int]:
    """Return (sum of T_g, number of late shipments)."""
    deadline_of = {}
    for job in jobs:
        sid = getattr(job, "shipment_id", -1)
        if sid >= 0:
            deadline_of[sid] = float(getattr(job, "deadline", 0.0))

    total, late = 0.0, 0
    for sid, c_g in shipment_completions(jobs, completion_of).items():
        t = max(0.0, c_g - deadline_of.get(sid, 0.0))
        total += t
        if t > 0:
            late += 1
    return total, late
