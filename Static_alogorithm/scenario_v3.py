"""Scenario v3: capacitated multi-robot pickup and delivery with exclusive doors.

Supersedes scenario_v2.py. Source of truth for the design is SCENARIO_SPEC_v3.md;
the formal statement is paper/problem_formulation.tex.

What this module is NOT
----------------------
It is no longer the place where the facility is defined. GA.GA now owns the v3
layout as its module default, so every entry point gets the right scenario
whether or not it remembers to call `apply_layout`. That was the single largest
source of drift in the previous version: only three scripts called it, and the
whole benchmark and training path silently ran the v1 10x10 / 3-AMR facility.

What this module IS
-------------------
  * `apply_layout(num_amrs=...)` -- rebuild GA's facility in place for a
    DIFFERENT fleet size, for the contention sweep over eta = m / |D_in|.
  * the parcel size-class mix and its sampler
  * thin capacity helpers that delegate to GA's rack primitives

Changes from v2
---------------
  deadlines     removed. The objective is pure makespan under the executor
                (eq. 8). Shipments, departure times, tardiness and the EDD/ATC
                rules are gone: the tardiness gradient was sparse and
                flat-then-spiky, and the generator was degenerate (two distinct
                deadline values per instance).
  rack          v2 declared a SECOND, contradictory SLOT_CAPACITY here (3/3/3,
                i.e. 9 slots) alongside GA's 1/1/1. It was dead code, but it
                would have silently given a 9-slot rack to anything that
                imported it. There is now exactly one definition, in GA.
  fleet         16 AMRs in 2x2 pods, not 12 in a single-column bank.

Usage
-----
    import scenario_v3 as sc
    sc.apply_layout(num_amrs=24)   # only needed to depart from the default

`apply_layout` mutates the module-level containers in GA.GA in place
(clear/update) rather than rebinding them, because every other module does
`from GA.GA import ...` and holds references to those exact objects.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import GA.GA as GA

# --------------------------------------------------------------------------
# Layout -- re-exported from GA so there is one definition, not two
# --------------------------------------------------------------------------

GRID_W = GA.GRID_W
GRID_H = GA.GRID_H
NUM_DOORS = len(GA.DOOR_ROWS)
NUM_AMRS = GA.NUM_AMRS
DEPOT_X = GA.DEPOT_X
DOOR_ROWS: Tuple[int, ...] = GA.DOOR_ROWS


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
    """Reinstall the facility into the GA module globals (in place).

    Calling this with the defaults is a no-op in effect -- GA already holds this
    layout. Its real use is the fleet sweep: `apply_layout(num_amrs=24)`.
    """
    rows = list(door_rows)
    docks = {f"dock{i + 1}": (0, rows[i]) for i in range(len(rows))}
    stations = {f"station{i + 1}": (grid_w, rows[i]) for i in range(len(rows))}

    GA.DOOR_ROWS = tuple(rows)
    GA.NUM_AMRS = num_amrs
    GA.GRID_W, GA.GRID_H = grid_w, grid_h

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

    GA.OBSTACLES.clear()

    starts = GA.build_pod_depot(
        num_amrs=num_amrs, door_rows=tuple(rows), depot_x=depot_x, grid_h=grid_h
    )
    # Validate here rather than inside build_pod_depot: the checks need the
    # docks, stations and dock_waiting_slots, which GA defines AFTER the depot
    # at import time. By this point they are installed.
    GA.dock_waiting_slots.cache_clear()
    GA.validate_depot(starts, door_rows=tuple(rows), grid_w=grid_w, grid_h=grid_h)

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

    # operation_policy builds DOCK_KEYS at import time from GA's dicts.
    try:
        import operation_policy as _op

        _op.DOCK_KEYS[:] = list(docks.keys()) + list(stations.keys())
        _op.DOCK_QUEUE_SCALE = float(max(num_amrs, 1))
    except Exception:
        pass


# --------------------------------------------------------------------------
# Nested slot capacity -- delegates to GA, which owns the primitive
# --------------------------------------------------------------------------

SIZE_ORDER: Tuple[str, ...] = GA.SIZE_ORDER
SLOT_CAPACITY: Dict[str, int] = GA.SLOT_CAPACITY
SUFFIX_CAP: Dict[str, int] = GA.SUFFIX_CAP
TOTAL_SLOTS = GA.TOTAL_SLOTS_PER_AMR


def nested_capacity_ok(counts: Mapping[str, int]) -> bool:
    """True iff `counts` satisfies every suffix-sum constraint of eq. (1)."""
    for i, s in enumerate(SIZE_ORDER):
        if sum(counts.get(t, 0) for t in SIZE_ORDER[i:]) > SUFFIX_CAP[s]:
            return False
    return True


def can_load(inventory_entry: Mapping[str, object], size_class: str) -> bool:
    """True iff one more parcel of `size_class` fits in the rack."""
    return GA.rack_can_load(inventory_entry, size_class)


def slack_slots(inventory_entry: Mapping[str, object], size_class: str) -> int:
    """Remaining slots usable by `size_class` (the tightest binding suffix)."""
    counts = GA._slot_counts(inventory_entry)
    idx = SIZE_ORDER.index(size_class)
    room = TOTAL_SLOTS
    for i, s in enumerate(SIZE_ORDER[: idx + 1]):
        used = sum(counts.get(t, 0) for t in SIZE_ORDER[i:])
        room = min(room, SUFFIX_CAP[s] - used)
    return max(0, room)


# --------------------------------------------------------------------------
# Size-class mix
# --------------------------------------------------------------------------
# Uniform. A skewed (60/30/10) mix is more faithful to real parcel streams, but
# measured on identical layouts it cuts the execution penalty from 15.9% to
# 5.8%: a small-parcel-heavy mix has a mean service time of 7.5 vs 10.0, so the
# doors are occupied less and queue less. Signal strength wins over that
# particular realism argument; the choice is recorded as a limitation.
SIZE_MIX: Dict[str, float] = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}


def sample_size_class(rng) -> str:
    r = rng.random()
    acc = 0.0
    for s in SIZE_ORDER:
        acc += SIZE_MIX[s]
        if r < acc:
            return s
    return SIZE_ORDER[-1]
