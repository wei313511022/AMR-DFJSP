from __future__ import annotations
from typing import Dict, Set

TIME_LIMIT = None

# Grid
GRID_SIZE: int = 12
BARRIER_NODES: Set[int] = {39, 43, 47, 50, 51, 54, 55, 58, 59, 87, 91, 95, 98, 99, 102, 103, 106, 107}
EXIT_NODE = 138

# Material
TYPE_TO_MATERIAL_NODE: Dict[str, int] = {"A": 10, "B": 6, "C": 2}
MATERIAL_PICK_QTY: int = 3
P_NODES = set(TYPE_TO_MATERIAL_NODE.values())

# Stations (station id -> delivery node)
JSON_STATION_MAPPING: Dict[int, int] = {1:46, 2:42, 3:38, 4:94, 5:90, 6:86}

# AMRs
M_SET = range(1, 4)
S_m: Dict[int, int] = {1: 10, 2: 6, 3: 2}

# Files
INBOX = "test_inbox.jsonl"
SCHEDULE_OUTBOX = "Random_Job_Arrivals/schedule_outbox.jsonl"


def validate_fixed_nodes() -> None:
    fixed_nodes_to_check = set(S_m.values()) | set(
        JSON_STATION_MAPPING.values()
    )
    bad_fixed = sorted(int(n) for n in fixed_nodes_to_check if int(n) in BARRIER_NODES)
    if bad_fixed:
        raise ValueError(
            f"Barrier nodes overlap with fixed start/pickup/delivery nodes: {bad_fixed}"
        )


validate_fixed_nodes()


