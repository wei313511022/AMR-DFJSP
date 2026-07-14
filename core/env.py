import json
import math
from collections import defaultdict, deque
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from core.data_io import record_to_jobs

Coord = Tuple[int, int]

DEFAULT_ENV_SPEC_PATH = Path(__file__).resolve().parents[1] / "configs" / "env_spec.json"


class MachineBuffer:
    """Single-slot machine buffer state.

    A part occupies its machine from the drop-off, through processing, until
    an AMR picks it up. All mutations go through occupy()/schedule_pickup()
    so the invariants (release is None ⟺ an occupant is present with no
    scheduled pickup; a pickup only frees the slot when the picker really is
    the occupant's successor) hold structurally.
    """

    __slots__ = ("process_end", "release", "occupant")

    def __init__(self) -> None:
        self.process_end = 0.0
        # Time the slot is/was freed by the occupant's pickup;
        # None = occupied and the pickup is not scheduled yet.
        self.release: Optional[float] = 0.0
        self.occupant: Optional[Tuple[int, int]] = None  # (jid, op_index)

    @property
    def blocked(self) -> bool:
        return self.release is None

    def occupy(self, jid: int, op_index: int, process_end: float) -> None:
        self.process_end = float(process_end)
        self.release = None
        self.occupant = (int(jid), int(op_index))

    def schedule_pickup(self, jid: int, op_index: int, pickup_t: float) -> bool:
        """Free the slot at pickup_t if (jid, op_index) is the occupant."""
        if self.occupant == (int(jid), int(op_index)):
            self.release = float(pickup_t)
            self.occupant = None
            return True
        return False


def load_env_spec(env_spec: Union[dict, str, Path, None] = None) -> dict:
    """Load the Phase III environment spec (contract section 2) from dict or JSON file."""
    if isinstance(env_spec, dict):
        return env_spec
    path = Path(env_spec) if env_spec is not None else DEFAULT_ENV_SPEC_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TaskSchedulingEnv:
    """
    Event-driven dispatch with dock/station mutual exclusion and AMR collision avoidance.

    Environment constants follow the Phase III I/O contract
    (docs/Phase3_Model_IO_Contract.md section 2) and are loaded from
    configs/env_spec.json (current field: 12x12 grid, 5 AMRs, stations
    S1..S6, material docks MA/MB/MC, output point T).

    FJSSP semantics (data/Generate_training_data.py instances):
      - a job is an ordered list of operations; operation k+1 is released
        only when operation k finishes (precedence is never reordered)
      - each operation lists its feasible machines with processing times;
        dispatching a task = choosing one machine for that operation
      - each machine has a SINGLE buffer slot: the part occupies it from the
        drop-off, through processing, until an AMR picks it up (next
        operation / delivery). A new part can only be dropped after that
        pickup, so an AMR may hold material at the machine while waiting;
        once dropped, processing starts immediately and the AMR is free
        (haul another job, fetch material, or idle — policy's choice).
        Deliveries to a machine whose occupant has no scheduled pickup yet
        are masked out of the action space (schedule the removal first)
      - op 0 additionally requires raw material picked up at the job's
        material dock (pickup service = material duration, dock is mutex);
        later ops fetch the in-process part at the previous station
      - after the last operation an AMR carries the finished part to the
        output point "T"; the job completes on arrival, so makespan = the
        last job's arrival at T (deliver_finished_to_output toggles this;
        the contract inference path turns it off)

    Decision point:
      - at least one idle AMR at current time
      - AND there exists at least one available task (released operations)
    Action:
      - choose which task (index in available_tasks) to assign to current_robot
        (+ how many material units to batch-pick at the dock for op-0 tasks)
    Objective:
      - minimize makespan (+ small AMR load-balance term)
    """

    def __init__(self, env_spec: Union[dict, str, Path, None] = None):
        spec = load_env_spec(env_spec)
        self.env_spec = spec

        grid = spec.get("grid", {})
        self.W = int(grid.get("width", 10))
        self.H = int(grid.get("height", 10))
        self.obstacles = {(int(x), int(y)) for x, y in grid.get("obstacles", [])}

        # Inbound docks: each job specifies which dock its material comes from.
        self.dock_locs: Dict[str, Coord] = {
            str(k): (int(v[0]), int(v[1])) for k, v in spec.get("docks", {}).items()
        }
        self._dock_by_coord: Dict[Coord, str] = {v: k for k, v in self.dock_locs.items()}

        # Materials: type -> duration (pickup time == process time == duration).
        self.material_durations: Dict[str, float] = {
            str(k).upper(): float(v) for k, v in spec.get("materials", {}).items()
        }
        self.material_types = sorted(self.material_durations.keys())

        # Material -> dock mapping. Preferred: explicit "material_dock_map" in
        # the spec (Route_Map field: A->MA, B->MB, C->MC); fallback spreads
        # materials across the dock column.
        mdm = spec.get("material_dock_map")
        if mdm:
            self.material_dock_map: Dict[str, str] = {
                str(m).upper(): str(d) for m, d in mdm.items()
            }
        else:
            dock_keys = sorted(self.dock_locs.keys())
            self.material_dock_map = {
                m: dock_keys[min(i * 2, len(dock_keys) - 1)]
                for i, m in enumerate(self.material_types)
            }
        self.source_locs: Dict[str, Coord] = {
            m: self.dock_locs[d] for m, d in self.material_dock_map.items()
        }

        # Output/delivery point(s) (finished-goods dock "T"). When
        # deliver_finished_to_output is on, a job's last operation releases a
        # final delivery task: an AMR fetches the finished part at the last
        # station and carries it to the output point; the job only counts as
        # complete (and extends makespan) when it arrives there.
        self.output_locs: Dict[str, Coord] = {
            str(k): (int(v[0]), int(v[1])) for k, v in spec.get("output", {}).items()
        }
        # Disabled by the contract inference path (the plan schema has no
        # delivery operation).
        self.deliver_finished_to_output = True
        # Single-slot machine buffer: a part occupies its machine from the
        # drop-off until an AMR PICKS IT UP (next operation / delivery), not
        # merely until processing ends. A new part can only be dropped after
        # that pickup, so an AMR may stand at the machine holding material.
        # While the occupant's pickup is not scheduled yet (its successor is
        # not dispatched), deliveries to that machine are masked out of the
        # action space (see blocked_task_indices). Disabled by the contract
        # inference path (contract stations are wait-until-process-end).
        self.enable_buffer_blocking = True

        self.station_locs: Dict[str, Coord] = {
            str(k): (int(v[0]), int(v[1])) for k, v in spec.get("stations", {}).items()
        }
        self._station_by_coord: Dict[Coord, str] = {v: k for k, v in self.station_locs.items()}

        amr_cfg = spec.get("amrs", {})
        self.num_robots = int(amr_cfg.get("count", 5))
        self.initial_robot_positions: List[Coord] = [
            (int(p[0]), int(p[1])) for p in amr_cfg.get("start_positions", [])
        ]
        if len(self.initial_robot_positions) != self.num_robots:
            raise ValueError("env_spec: amrs.count must match len(amrs.start_positions)")
        self.capacity_per_type = int(amr_cfg.get("capacity_per_type", 3))

        # Batch pickup / proactive replenishment: dock operations may pick up
        # extra units of the job's material (up to capacity_per_type) so later
        # same-material jobs can be served from onboard stock without a dock
        # trip. Score(i) = Q(i) + cover/load/wait bonuses (see features.py).
        self.allow_proactive_replenish = True
        self.proactive_replenish_bias_weight = 1.5
        self.proactive_full_load_bias_weight = 0.8
        self.proactive_waiting_replenish_bias_weight = 1.2
        # If False, stock provided via init_state is shown as zero and cannot be
        # consumed (inference: the integrator replays every job's own pickup).
        self.consume_initial_inventory = True
        # Contract objective: makespan + w * sum(per-AMR finish times).
        # The dense reward mirrors it so the summed episode reward equals the
        # negative objective (secondary term balances load across AMRs).
        self.objective_load_balance_weight = 0.001
        # If True, AMRs use time-aware collision avoidance; if False, route overlap is allowed.
        self.enable_collision_avoidance = True

        self.dist_cache: Dict[Tuple[Coord, Coord], int] = {}
        self.path_cache: Dict[Tuple[Coord, Coord], List[Coord]] = {}
        self._reservation_cache: Dict[Tuple[int, float, int, int], dict] = {}
        self.reset([])

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H

    def _passable(self, x: int, y: int) -> bool:
        return (x, y) not in self.obstacles

    def _bfs_distance(self, start: Coord, goal: Coord) -> int:
        if start == goal:
            return 0
        if (start, goal) in self.dist_cache:
            return self.dist_cache[(start, goal)]
        _ = self._bfs_path(start, goal)
        return self.dist_cache.get((start, goal), 10**9)

    def _bfs_path(self, start: Coord, goal: Coord) -> List[Coord]:
        if start == goal:
            return [start]
        if (start, goal) in self.path_cache:
            return self.path_cache[(start, goal)]

        sx, sy = start
        visited = {start}
        parent: Dict[Coord, Coord] = {}
        dq = deque([(sx, sy)])

        found = False
        while dq:
            x, y = dq.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)
                if not self._in_bounds(nx, ny) or not self._passable(nx, ny):
                    continue
                if nxt in visited:
                    continue
                visited.add(nxt)
                parent[nxt] = (x, y)
                if nxt == goal:
                    found = True
                    dq.clear()
                    break
                dq.append(nxt)

        if not found:
            self.path_cache[(start, goal)] = [start]
            self.path_cache[(goal, start)] = [goal]
            self.dist_cache[(start, goal)] = 10**9
            self.dist_cache[(goal, start)] = 10**9
            return self.path_cache[(start, goal)]

        path: List[Coord] = [goal]
        cur = goal
        while cur != start:
            cur = parent[cur]
            path.append(cur)
        path.reverse()

        rev = list(reversed(path))
        self.path_cache[(start, goal)] = path
        self.path_cache[(goal, start)] = rev
        d = len(path) - 1
        self.dist_cache[(start, goal)] = d
        self.dist_cache[(goal, start)] = d
        return path

    def _dist(self, a: Coord, b: Coord) -> int:
        return self._bfs_distance(a, b)

    def _path(self, a: Coord, b: Coord) -> List[Coord]:
        return self._bfs_path(a, b)

    @staticmethod
    def _time_key(t: float) -> float:
        return round(float(t), 6)

    @staticmethod
    def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
        return max(a0, b0) < min(a1, b1) - 1e-9

    @staticmethod
    def _to_coord(p: Union[List[int], Tuple[int, int], Coord]) -> Coord:
        return (int(p[0]), int(p[1]))

    def _neighbors_with_wait(self, c: Coord) -> List[Coord]:
        x, y = c
        out: List[Coord] = [c]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if self._in_bounds(nx, ny) and self._passable(nx, ny):
                out.append((nx, ny))
        return out

    def _build_dynamic_reservations(
        self,
        exclude_robot: int,
        from_time: float,
        horizon_end: float,
        future_work_after_dispatch: bool,
    ) -> dict:
        key = (
            exclude_robot,
            self._time_key(from_time),
            self._dispatch_counter,
            1 if future_work_after_dispatch else 0,
        )
        cached = self._reservation_cache.get(key)
        if cached is not None and float(cached["horizon_end"]) >= horizon_end - 1e-9:
            return cached["data"]

        points: Dict[float, set] = defaultdict(set)
        intervals: Dict[Coord, List[Tuple[float, float]]] = defaultdict(list)
        edges: Dict[Tuple[Coord, Coord], List[Tuple[float, float]]] = defaultdict(list)

        def add_point(t: float, c: Coord) -> None:
            if t < from_time - 1e-9 or t > horizon_end + 1e-9:
                return
            points[self._time_key(t)].add(self._to_coord(c))

        def add_interval(c: Coord, t0: float, t1: float) -> None:
            if t1 <= t0 + 1e-9:
                return
            if t1 < from_time - 1e-9 or t0 > horizon_end + 1e-9:
                return
            a = max(float(t0), float(from_time))
            b = min(float(t1), float(horizon_end))
            if b <= a + 1e-9:
                return
            cell = self._to_coord(c)
            intervals[cell].append((a, b))
            # Keep interval endpoints as occupied points as well to prevent
            # same-timestamp overlap at hand-off boundaries.
            add_point(a, cell)
            add_point(b, cell)

        def add_edge(a: Coord, b: Coord, t0: float, t1: float) -> None:
            if t1 <= t0 + 1e-9:
                return
            if t1 < from_time - 1e-9 or t0 > horizon_end + 1e-9:
                return
            x0 = max(float(t0), float(from_time))
            x1 = min(float(t1), float(horizon_end))
            if x1 <= x0 + 1e-9:
                return
            k = (self._to_coord(a), self._to_coord(b))
            edges[k].append((x0, x1))

        actions_by_robot: Dict[int, List[dict]] = {rid: [] for rid in range(self.num_robots)}
        next_release_t = self._next_release_time()

        for item in self.trace:
            rid = int(item.get("robot", -1))
            if rid < 0 or rid == exclude_robot:
                continue

            segs: Dict[str, dict] = {}
            for seg in item.get("segments", []):
                kind = seg.get("kind")
                if kind:
                    segs[kind] = seg

            raw_path = item.get("transport_path", [])
            path = [self._to_coord(p) for p in raw_path] if raw_path else []
            if not path:
                drop = item.get("drop", (0, 0))
                path = [self._to_coord(drop)]

            end_cell = path[-1]
            seg_t = segs.get("transport")
            if seg_t is not None:
                ts = float(seg_t.get("start", from_time))
                te = float(seg_t.get("end", ts))
                if len(path) == 1 or te <= ts + 1e-9:
                    add_point(ts, path[0])
                    add_point(te, path[-1])
                else:
                    dt = (te - ts) / float(len(path) - 1)
                    for i, c in enumerate(path):
                        add_point(ts + dt * i, c)
                    for i in range(len(path) - 1):
                        c0 = path[i]
                        c1 = path[i + 1]
                        t0 = ts + dt * i
                        t1 = t0 + dt
                        if c0 == c1:
                            add_interval(c0, t0, t1)
                        else:
                            add_edge(c0, c1, t0, t1)

            # Drop-off semantics: the robot occupies cells only along its
            # transport path (incl. dock-service hold steps); the "wait"
            # (part queued at the machine) and "process" (machine working)
            # intervals involve no robot, so they reserve nothing. The robot
            # idles at post_pos from the drop-off (transport end) onward.
            seg_transport = segs.get("transport")
            t_start = float(seg_transport.get("start", 0.0)) if seg_transport else 0.0
            t_end = float(seg_transport.get("end", t_start)) if seg_transport else t_start

            actions_by_robot[rid].append(
                {
                    "start": t_start,
                    "end": t_end,
                    "idle_cell": self._to_coord(item.get("post_pos", end_cell)),
                }
            )

        # Reserve known idle intervals between committed actions and after
        # the last committed action within planning horizon.
        for rid in range(self.num_robots):
            if rid == exclude_robot:
                continue

            robot_actions = actions_by_robot[rid]
            robot_actions.sort(key=lambda a: float(a["start"]))

            start_positions = getattr(
                self, "_episode_start_positions", self.initial_robot_positions
            )
            idle_pos = self._to_coord(start_positions[rid])
            idle_from = 0.0
            for act in robot_actions:
                act_start = float(act["start"])
                if act_start > idle_from + 1e-9:
                    add_interval(idle_pos, idle_from, act_start)
                idle_pos = self._to_coord(act["idle_cell"])
                idle_from = max(idle_from, float(act["end"]))

            if future_work_after_dispatch:
                # Post-action tail is uncertain (other robots may get new
                # dispatches), so reserve only immediate occupancy.
                if math.isfinite(next_release_t):
                    tail_end = min(horizon_end, max(idle_from, next_release_t))
                else:
                    tail_end = min(horizon_end, idle_from + 1.0)
            else:
                # No further work is expected after this dispatch, keep tail
                # occupied to avoid final-stage overlap.
                tail_end = horizon_end

            add_interval(idle_pos, idle_from, tail_end)
            add_point(idle_from, idle_pos)

        data = {
            "points": {k: set(v) for k, v in points.items()},
            "intervals": {k: sorted(v) for k, v in intervals.items()},
            "edges": {k: sorted(v) for k, v in edges.items()},
        }
        self._reservation_cache[key] = {"horizon_end": float(horizon_end), "data": data}
        return data

    def _point_conflict(self, reservations: dict, c: Coord, t: float) -> bool:
        cell = self._to_coord(c)
        pts = reservations["points"].get(self._time_key(t), set())
        if cell in pts:
            return True
        for a, b in reservations["intervals"].get(cell, []):
            if a - 1e-9 <= t < b - 1e-9:
                return True
        return False

    def _transition_conflict(
        self,
        reservations: dict,
        c0: Coord,
        c1: Coord,
        t0: float,
        t1: float,
        ignore_source_point_at_t0: bool = False,
    ) -> bool:
        a = self._to_coord(c0)
        b = self._to_coord(c1)

        if (not ignore_source_point_at_t0) and self._point_conflict(reservations, a, t0):
            return True
        if self._point_conflict(reservations, b, t1):
            return True

        if a == b:
            for x0, x1 in reservations["intervals"].get(a, []):
                if self._interval_overlap(t0, t1, x0, x1):
                    return True
            return False

        for x0, x1 in reservations["edges"].get((a, b), []):
            if self._interval_overlap(t0, t1, x0, x1):
                return True
        for x0, x1 in reservations["edges"].get((b, a), []):
            if self._interval_overlap(t0, t1, x0, x1):
                return True
        return False

    def _plan_path_time_aware(
        self,
        start: Coord,
        goal: Coord,
        start_time: float,
        reservations: dict,
        min_arrival_time: float = 0.0,
        block_goal_before_min: bool = False,
        max_steps: int = 200,
    ) -> List[Coord]:
        s = self._to_coord(start)
        g = self._to_coord(goal)

        if max_steps <= 0:
            return []
        if not self._in_bounds(s[0], s[1]) or not self._passable(s[0], s[1]):
            return []
        if not self._in_bounds(g[0], g[1]) or not self._passable(g[0], g[1]):
            return []

        if (
            s == g
            and (not block_goal_before_min or start_time + 1e-9 >= min_arrival_time)
            and not self._point_conflict(reservations, s, start_time)
        ):
            return [s]

        heap: List[Tuple[float, int, Coord]] = []
        h0 = float(self._dist(s, g))
        heappush(heap, (h0, 0, s))

        parent: Dict[Tuple[Coord, int], Optional[Tuple[Coord, int]]] = {(s, 0): None}
        seen = {(s, 0)}

        while heap:
            _f, step, cur = heappop(heap)
            t_cur = start_time + float(step)
            state = (cur, step)

            if cur == g and (not block_goal_before_min or t_cur + 1e-9 >= min_arrival_time):
                out: List[Coord] = []
                p: Optional[Tuple[Coord, int]] = state
                while p is not None:
                    out.append(p[0])
                    p = parent[p]
                out.reverse()
                return out

            if step >= max_steps:
                continue

            for nxt in self._neighbors_with_wait(cur):
                t_nxt = t_cur + 1.0
                if block_goal_before_min and nxt == g and t_nxt + 1e-9 < min_arrival_time:
                    continue
                ignore_t0 = step == 0
                if self._transition_conflict(
                    reservations,
                    cur,
                    nxt,
                    t_cur,
                    t_nxt,
                    ignore_source_point_at_t0=ignore_t0,
                ):
                    continue

                nxt_state = (nxt, step + 1)
                if nxt_state in seen:
                    continue
                seen.add(nxt_state)
                parent[nxt_state] = state

                h = float(self._dist(nxt, g))
                lb = float(step + 1) + h
                if block_goal_before_min:
                    est_t = start_time + lb
                    if est_t < min_arrival_time:
                        lb += min_arrival_time - est_t
                heappush(heap, (lb, step + 1, nxt))

        return []

    def _delay_before_goal(self, path: List[Coord], wait_steps: int) -> List[Coord]:
        if wait_steps <= 0 or not path:
            return list(path)
        if len(path) >= 2:
            hold = path[-2]
            return list(path[:-1]) + [hold] * wait_steps + [path[-1]]

        start = path[0]
        for nxt in self._neighbors_with_wait(start):
            if nxt != start:
                return [start, nxt] + [nxt] * wait_steps + [start]
        return list(path) + [start] * wait_steps

    def _post_process_candidates(self, transport_path: List[Coord], drop: Coord) -> List[Coord]:
        d = self._to_coord(drop)
        rev_distinct: List[Coord] = []
        for i in range(len(transport_path) - 2, -1, -1):
            c = self._to_coord(transport_path[i])
            if c == d:
                continue
            if not rev_distinct or rev_distinct[-1] != c:
                rev_distinct.append(c)

        candidates: List[Coord] = []
        if len(rev_distinct) >= 2:
            candidates.append(rev_distinct[1])
        if len(rev_distinct) >= 1:
            candidates.append(rev_distinct[0])
        for c in rev_distinct[2:]:
            candidates.append(c)

        for nxt in self._neighbors_with_wait(d):
            if nxt != d:
                candidates.append(self._to_coord(nxt))

        candidates.append(d)

        out: List[Coord] = []
        seen = set()
        for c in candidates:
            cc = self._to_coord(c)
            if cc in seen:
                continue
            seen.add(cc)
            out.append(cc)
        return out

    def _post_process_position(self, transport_path: List[Coord], drop: Coord) -> Coord:
        cands = self._post_process_candidates(transport_path, drop)
        return cands[0] if cands else self._to_coord(drop)

    def _is_machine(self, station: str) -> bool:
        """True for processing machines (S1..); False for the output point
        "T" and any other non-machine drop target."""
        return station in self.station_buffer

    def _station_drop_free_time(self, station: str) -> float:
        """Earliest time a new part can be dropped at `station` (single-slot
        buffer: when the current occupant is picked up). If the occupant's
        pickup is not scheduled yet, fall back to its process end — callers
        normally mask such tasks (blocked_task_indices); the fallback only
        feeds the stall-break path. Output point / unknown keys -> 0."""
        buf = self.station_buffer.get(station)
        if buf is None:
            return 0.0
        if not self.enable_buffer_blocking or buf.release is None:
            return float(buf.process_end)
        return max(float(buf.release), 0.0)

    def blocked_task_indices(self) -> set:
        """Indices in available_tasks whose drop target is a machine that is
        occupied with an unscheduled pickup (single-slot buffer full)."""
        if not self.enable_buffer_blocking:
            return set()
        out = set()
        for i, t in enumerate(self.available_tasks):
            if t.get("is_delivery"):
                continue
            buf = self.station_buffer.get(str(t.get("station", "")))
            if buf is not None and buf.blocked:
                if buf.occupant == (
                    int(t.get("jid", -1)),
                    int(t.get("op_index", 0)) - 1,
                ):
                    # The occupant is this task's own predecessor — its
                    # pickup removes it, so the drop slot frees itself.
                    continue
                out.add(i)
        return out

    def _normalize_replenish_plan(
        self, task: dict, replenish: Union[int, Dict[str, int], None]
    ) -> Dict[str, int]:
        jtype = str(task.get("type", "")).upper()
        plan: Dict[str, int] = {t: 0 for t in self.material_types}
        if isinstance(replenish, dict):
            for t, v in replenish.items():
                key = str(t).upper()
                if key in plan:
                    plan[key] = max(0, int(v))
        elif replenish is not None:
            if jtype in plan:
                plan[jtype] = max(0, int(replenish))
        return plan

    def _estimate_action_plan(
        self,
        robot_id: int,
        task: dict,
        replenish: Union[int, Dict[str, int], None],
        for_execution: bool = True,
    ) -> dict:
        # Batch-pickup timeline (drop-off semantics — the AMR never waits at
        # the station; the part queues there on its own):
        #   dock op, add>=1 : travel to dock -> (wait for dock) -> pick `add`
        #                     units (add x unit duration) -> travel to station
        #                     -> drop off; part waits -> machine processes
        #   dock op, add==0 : deliver from onboard stock, straight to station
        #   transfer op     : fetch the part at the previous station (0 service)
        # Dock waiting and pickup service are embedded into transport_path as
        # hold steps so collision reservations naturally keep the dock exclusive.
        pos = self._to_coord(self.robot_positions[robot_id])
        dock = self._to_coord(task["pickup"])
        dock_key = str(task.get("dock", self._dock_by_coord.get(dock, str(dock))))
        drop = self._to_coord(task["drop"])
        station = str(task["station"])
        jtype = str(task["type"]).upper()
        proc = float(task["proc_time"])

        op_index = int(task.get("op_index", 0))
        replenish_plan = self._normalize_replenish_plan(task, replenish)
        add_units = int(replenish_plan.get(jtype, 0)) if op_index == 0 else 0
        need_pickup = (op_index > 0) or (add_units > 0)
        unit_service = float(task.get("pickup_service", proc))
        pickup_service = float(add_units) * unit_service if op_index == 0 else unit_service
        pickup_steps = int(math.ceil(pickup_service - 1e-9))

        start_t = max(self.t, self.robot_free_times[robot_id])
        # Single-slot buffer: the drop must wait for the occupant's pickup
        # (see _station_drop_free_time). Delivery legs target the output
        # point, which has no busy window.
        station_free = self._station_drop_free_time(station)
        dock_free = float(self.dock_busy_until.get(dock_key, 0.0)) if need_pickup else 0.0

        pickup_types = [jtype] if need_pickup else []

        if not self.enable_collision_avoidance:
            if not for_execution:
                # Feature-estimation fast path: same timeline as the
                # materialized branch below, computed analytically — no
                # multi-thousand-cell hold paths are allocated per candidate
                # action. `travel` covers movement + dock queue/service;
                # `wait` is the machine-buffer hold at the drop target.
                if need_pickup:
                    d1 = float(self._dist(pos, dock))
                    arrive_dock_t = start_t + d1
                    dock_wait = float(
                        max(0, int(math.ceil(max(0.0, dock_free - arrive_dock_t) - 1e-9)))
                    )
                    pickup_start_t = arrive_dock_t + dock_wait
                    pickup_end_t = pickup_start_t + float(pickup_steps)
                    arrive_t = pickup_end_t + float(self._dist(dock, drop))
                else:
                    arrive_t = start_t + float(self._dist(pos, drop))
                    pickup_start_t = start_t
                    pickup_end_t = start_t
                    dock_wait = 0.0
                process_start_t = max(arrive_t, station_free)
                return {
                    "start_t": float(start_t),
                    "travel": float(arrive_t - start_t),
                    "wait": float(max(0.0, process_start_t - arrive_t)),
                    "proc": float(proc),
                    "arrive_t": float(arrive_t),
                    "process_start_t": float(process_start_t),
                    "transport_path": [],
                    "need_pickup": bool(need_pickup),
                    "replenish_plan": dict(replenish_plan),
                    "pickup_types": list(pickup_types),
                    "dock": dock_key,
                    "dock_wait": dock_wait,
                    "pickup_start_t": float(pickup_start_t),
                    "pickup_end_t": float(pickup_end_t),
                }

            if need_pickup:
                leg1 = [self._to_coord(p) for p in self._path(pos, dock)]
                if (not leg1) or (leg1[-1] != dock):
                    raise RuntimeError(
                        f"No path for AMR{robot_id+1}: {pos}->{dock} at t={start_t:.3f}"
                    )
                arrive_dock_t = start_t + float(max(0, len(leg1) - 1))
                dock_wait_steps = int(
                    math.ceil(max(0.0, dock_free - arrive_dock_t) - 1e-9)
                )
                if dock_wait_steps > 0:
                    leg1 = self._delay_before_goal(leg1, dock_wait_steps)
                pickup_start_t = start_t + float(max(0, len(leg1) - 1))
                pickup_end_t = pickup_start_t + float(pickup_steps)

                leg2 = [self._to_coord(p) for p in self._path(dock, drop)]
                if (not leg2) or (leg2[-1] != drop):
                    raise RuntimeError(
                        f"No path for AMR{robot_id+1}: {dock}->{drop} at t={start_t:.3f}"
                    )
                transport_path: List[Coord] = (
                    list(leg1) + [dock] * pickup_steps + leg2[1:]
                )
                dock_wait = float(max(0.0, pickup_start_t - arrive_dock_t))
            else:
                # Deliver from onboard stock: straight to the station.
                transport_path = [self._to_coord(p) for p in self._path(pos, drop)]
                if (not transport_path) or (transport_path[-1] != drop):
                    raise RuntimeError(
                        f"No path for AMR{robot_id+1}: {pos}->{drop} at t={start_t:.3f}"
                    )
                pickup_start_t = start_t
                pickup_end_t = start_t
                dock_wait = 0.0

            travel = float(max(0, len(transport_path) - 1))
            arrive_t = start_t + travel
            # Single-slot buffer: the AMR holds the part at the machine until
            # the occupant is picked up (station_free), then drops it and the
            # machine starts immediately. The hold is embedded as path hold
            # steps so the AMR is genuinely busy during the wait.
            process_start_t = max(arrive_t, station_free)
            if process_start_t > arrive_t + 1e-9:
                add_steps = int(math.ceil(process_start_t - arrive_t - 1e-9))
                transport_path = self._delay_before_goal(transport_path, add_steps)
                travel = float(max(0, len(transport_path) - 1))
                arrive_t = start_t + travel
                process_start_t = max(arrive_t, station_free)
            wait = max(0.0, process_start_t - arrive_t)
            return {
                "start_t": float(start_t),
                "travel": float(travel),
                "wait": float(wait),
                "proc": float(proc),
                "arrive_t": float(arrive_t),
                "process_start_t": float(process_start_t),
                "transport_path": [self._to_coord(p) for p in transport_path],
                "need_pickup": bool(need_pickup),
                "replenish_plan": dict(replenish_plan),
                "pickup_types": list(pickup_types),
                "dock": dock_key,
                "dock_wait": dock_wait,
                "pickup_start_t": float(pickup_start_t),
                "pickup_end_t": float(pickup_end_t),
            }

        if need_pickup:
            base_dist = (
                float(self._dist(pos, dock))
                + float(pickup_steps)
                + float(self._dist(dock, drop))
            )
        else:
            base_dist = float(self._dist(pos, drop))

        future_work_after_dispatch = (len(self.available_tasks) > 1) or math.isfinite(
            self._next_release_time()
        )
        # Both the dock service and the buffer wait can delay the AMR.
        slack = max(0.0, station_free - start_t) + max(0.0, dock_free - start_t)

        def leg_candidates_steps(
            a: Coord,
            b: Coord,
            t0: float,
            min_arr: float,
            horizon: float,
        ) -> List[int]:
            leg_dist = float(self._dist(a, b))
            extra = max(0.0, min_arr - t0)
            horizon_cap = int(max(1, math.floor(horizon - t0 + 1e-9)))
            base = [
                int(max(30, math.ceil(leg_dist + extra + 20.0))),
                int(max(60, math.ceil(leg_dist + extra + 80.0))),
                int(max(120, math.ceil(leg_dist + extra + 220.0))),
                horizon_cap,
            ]
            out: List[int] = []
            for x in base:
                x = int(min(max(1, x), horizon_cap))
                if x not in out:
                    out.append(x)
            return out

        def plan_leg(
            a: Coord,
            b: Coord,
            t0: float,
            min_arr: float,
            block_goal: bool,
            reservations: dict,
            horizon: float,
        ) -> List[Coord]:
            for max_steps in leg_candidates_steps(a, b, t0, min_arr, horizon):
                out = self._plan_path_time_aware(
                    a,
                    b,
                    t0,
                    reservations,
                    min_arrival_time=min_arr,
                    block_goal_before_min=block_goal,
                    max_steps=max_steps,
                )
                if out:
                    return out
            return []

        base_budget = float(max(80.0, math.ceil(base_dist + slack + 80.0)))
        budget_scales = [1.0, 1.8, 3.0, 5.0, 8.0, 12.0, 18.0]
        transport_path: List[Coord] = []
        pickup_start_t = start_t
        pickup_end_t = start_t

        tail_modes = [future_work_after_dispatch]
        if not future_work_after_dispatch:
            tail_modes.append(True)

        for tail_mode in tail_modes:
            for scale in budget_scales:
                search_budget = int(max(80, math.ceil(base_budget * scale)))
                horizon_end = start_t + float(search_budget + 5)
                reservations = self._build_dynamic_reservations(
                    robot_id,
                    start_t,
                    horizon_end,
                    future_work_after_dispatch=tail_mode,
                )

                if need_pickup:
                    # Leg 1: to the pickup point; block arrival until it is free.
                    leg1 = plan_leg(
                        pos,
                        dock,
                        start_t,
                        dock_free,
                        True,
                        reservations,
                        horizon_end,
                    )
                    if not leg1:
                        continue
                    candidate: List[Coord] = [self._to_coord(p) for p in leg1]
                    t_arr_dock = start_t + float(max(0, len(leg1) - 1))
                    # Pickup service: hold at the dock for add x unit duration.
                    candidate.extend([dock] * pickup_steps)
                    t_after_pickup = t_arr_dock + float(pickup_steps)

                    # Leg 2: dock -> station; the AMR may only enter the drop
                    # cell once the buffer frees (occupant picked up).
                    tail = plan_leg(
                        dock,
                        drop,
                        t_after_pickup,
                        station_free,
                        True,
                        reservations,
                        horizon_end,
                    )
                    if not tail:
                        continue
                    candidate.extend([self._to_coord(p) for p in tail[1:]])
                    transport_path = candidate
                    pickup_start_t = t_arr_dock
                    pickup_end_t = t_after_pickup
                else:
                    # Deliver from onboard stock: straight to the station;
                    # enter the drop cell only once the buffer frees.
                    candidate = plan_leg(
                        pos,
                        drop,
                        start_t,
                        station_free,
                        True,
                        reservations,
                        horizon_end,
                    )
                    if not candidate:
                        continue
                    transport_path = [self._to_coord(p) for p in candidate]
                    pickup_start_t = start_t
                    pickup_end_t = start_t
                break

            if transport_path:
                break

        if not transport_path:
            raise RuntimeError(
                f"No collision-free path for AMR{robot_id+1}: "
                f"{pos}->{dock}->{drop} at t={start_t:.3f}"
            )

        travel = float(max(0, len(transport_path) - 1))
        arrive_t = start_t + travel
        # Single-slot buffer hold (see the collision-free branch above).
        process_start_t = max(arrive_t, station_free)
        if process_start_t > arrive_t + 1e-9:
            add_steps = int(math.ceil(process_start_t - arrive_t - 1e-9))
            transport_path = self._delay_before_goal(transport_path, add_steps)
            travel = float(max(0, len(transport_path) - 1))
            arrive_t = start_t + travel
            process_start_t = max(arrive_t, station_free)
        wait = max(0.0, process_start_t - arrive_t)
        return {
            "start_t": float(start_t),
            "travel": float(travel),
            "wait": float(wait),
            "proc": float(proc),
            "arrive_t": float(arrive_t),
            "process_start_t": float(process_start_t),
            "transport_path": [self._to_coord(p) for p in transport_path],
            "need_pickup": bool(need_pickup),
            "replenish_plan": dict(replenish_plan),
            "pickup_types": list(pickup_types),
            "dock": dock_key,
            "dock_wait": (
                float(max(0.0, pickup_start_t - (start_t + float(self._dist(pos, dock)))))
                if need_pickup
                else 0.0
            ),
            "pickup_start_t": float(pickup_start_t),
            "pickup_end_t": float(pickup_end_t),
        }

    def _normalize_release_times(
        self, release_events: List[Tuple[float, List[dict]]]
    ) -> List[Tuple[float, List[dict]]]:
        if not release_events:
            return release_events
        t0 = min(t for t, _ in release_events)
        return [(t - t0, jobs) for (t, jobs) in release_events]

    def _resolve_dock(self, j: dict, jtype: str) -> Tuple[str, Coord]:
        """Resolve a job's dock key + coordinate. Accepts "dock" (1..5 / "D3")
        or "dock_xy" ([x, y]); legacy jobs without a dock fall back to the
        material's default dock."""
        if j.get("dock_xy") is not None:
            coord = self._to_coord(j["dock_xy"])
            key = self._dock_by_coord.get(coord)
            if key is None:
                raise ValueError(f"Unknown dock_xy: {j['dock_xy']}")
            return key, coord
        if j.get("dock") is not None:
            d_key = str(j["dock"]).upper()
            if d_key in self.dock_locs:
                return d_key, self.dock_locs[d_key]
            if "D" + d_key in self.dock_locs:
                return "D" + d_key, self.dock_locs["D" + d_key]
            # Field version without numbered docks (e.g. material docks MA/MB/MC):
            # fall back to the job material's dock.
        dock_key = self.material_dock_map[jtype]
        return dock_key, self.dock_locs[dock_key]

    def _next_op_uid(self) -> int:
        self._op_uid_counter += 1
        return self._op_uid_counter

    def _make_op_tasks(
        self,
        jid: int,
        op_index: int,
        release_time: float,
        prev_station: Optional[str] = None,
    ) -> List[dict]:
        """
        Create one available_task per feasible machine for job `jid`'s
        operation `op_index` (FJSSP flexibility: choosing a task = choosing the
        machine). Siblings share op_uid; dispatching one removes the others.

        Operation 0 picks raw material at the job's material dock (service time
        = material duration). Later operations pick the part up at the previous
        operation's station (service time 0, no dock mutex).
        """
        job_def = self._job_defs[jid]
        ops = job_def["operations"]
        op = ops[op_index]
        jtype = job_def["type"]
        uid = self._next_op_uid()

        if op_index == 0:
            dock_key = self.material_dock_map[jtype]
            pickup = self.dock_locs[dock_key]
            pickup_service = float(self.material_durations.get(jtype, 0.0))
        else:
            if prev_station is None or prev_station not in self.station_locs:
                raise ValueError(f"job {jid} op {op_index}: unknown previous station")
            dock_key = str(prev_station)
            pickup = self.station_locs[prev_station]
            pickup_service = 0.0

        tasks: List[dict] = []
        for choice in op:
            m = int(choice["machine"])
            s_key = f"S{m + 1}"
            if s_key not in self.station_locs:
                raise ValueError(
                    f"job {jid} op {op_index}: machine {m} has no station {s_key}"
                )
            tasks.append(
                {
                    "jid": jid,
                    "type": jtype,
                    "proc_time": float(choice["processing"]),
                    "op_index": int(op_index),
                    "num_ops": len(ops),
                    "op_uid": uid,
                    "dock": dock_key,
                    "pickup": pickup,
                    "pickup_service": pickup_service,
                    "drop": self.station_locs[s_key],
                    "station": s_key,
                    "release_time": float(release_time),
                }
            )
        return tasks

    def _make_delivery_task(self, task: dict, station: str, release_time: float) -> dict:
        """Final leg of a finished job: fetch the part at its last station
        (service 0) and carry it to the output point. No machine choice, no
        processing; arrival at the output completes the job."""
        out_key, out_loc = sorted(self.output_locs.items())[0]
        return {
            "jid": int(task["jid"]),
            "type": str(task["type"]),
            "proc_time": 0.0,
            "op_index": int(task.get("op_index", 0)) + 1,
            "num_ops": int(task.get("num_ops", 1)),
            "op_uid": self._next_op_uid(),
            "dock": str(station),
            "pickup": self.station_locs[station],
            "pickup_service": 0.0,
            "drop": out_loc,
            "station": out_key,
            "release_time": float(release_time),
            "is_delivery": True,
        }

    def _jobs_to_tasks(self, jobs: List[dict], release_time: float) -> List[dict]:
        tasks = []
        for j in jobs:
            if isinstance(j, list):
                # Raw FJSSP row (data/data_README.md format, e.g.
                # sample_abz5.json): the job IS its operation list, with no
                # material field.
                j = {"operations": j}
            jid = j.get("jid")
            if jid is None or int(jid) < 0:
                jid = self._next_job_id
                self._next_job_id += 1
            jid = int(jid)

            jtype = str(j.get("type", j.get("material", ""))).upper()
            if not jtype and "operations" in j:
                # No material given (raw FJSSP instance): assign one
                # deterministically (round-robin by job id) so op 0 has a
                # dock to pick raw material from.
                jtype = self.material_types[jid % len(self.material_types)]
            if jtype not in self.material_durations:
                raise ValueError(f"Unknown job type: {jtype} (expect A/B/C)")

            if "operations" in j:
                # FJSSP-style multi-operation job (scripts/Generate_training_data.py
                # format): each operation lists feasible machines with processing
                # times; operation k+1 is released when operation k completes.
                self._job_defs[jid] = {
                    "type": jtype,
                    "operations": list(j["operations"]),
                }
                tasks.extend(self._make_op_tasks(jid, 0, release_time))
                continue

            proc_time = float(
                j.get("proc_time", j.get("duration", self.material_durations[jtype]))
            )

            st = j.get("station")
            s_key = str(st)
            if not s_key.startswith("S"):
                s_key = "S" + s_key
            if s_key not in self.station_locs:
                raise ValueError(
                    f"Unknown station: {st} (expect an index or S1..S{len(self.station_locs)})"
                )

            dock_key, dock_coord = self._resolve_dock(j, jtype)

            tasks.append(
                {
                    "jid": jid,
                    "type": jtype,
                    "proc_time": proc_time,
                    "op_index": 0,
                    "num_ops": 1,
                    "op_uid": self._next_op_uid(),
                    "dock": dock_key,
                    "pickup": dock_coord,
                    # Per-unit pickup time is a property of the material, so
                    # batched units for other jobs cost the same regardless of
                    # this job's processing time (contract jobs: identical).
                    "pickup_service": float(self.material_durations[jtype]),
                    "drop": self.station_locs[s_key],
                    "station": s_key,
                    "release_time": float(release_time),
                }
            )
        return tasks

    def reset(
        self,
        scenario: Union[dict, List[dict]],
        init_state: Optional[dict] = None,
    ) -> List[float]:
        """
        scenario can be:
          - record dict: {"dispatch_time":..., "jobs":[...]}
          - records list: [{"dispatch_time":..., "jobs":[...]}, ...]
          - jobs list: [{"type","station",...}, ...]  (single batch, release at 0)

        init_state (optional) warm-starts the fleet mid-run (contract "mid state"
        scenes; times are relative to t=0):
          {
            "positions":  [[x, y], ...],          # len == num_robots
            "free_times": [float, ...],           # when each AMR can take new work
            "inventory":  [{"A":int,"B":int,"C":int}, ...],
          }
        """
        release_events: List[Tuple[float, List[dict]]] = []

        if isinstance(scenario, dict) and "jobs" in scenario:
            dt, jobs = record_to_jobs(scenario)
            release_events.append((dt, jobs))
        elif isinstance(scenario, list):
            if len(scenario) == 0:
                release_events = []
            elif isinstance(scenario[0], dict) and "jobs" in scenario[0]:
                for rec in scenario:
                    dt, jobs = record_to_jobs(rec)
                    release_events.append((dt, jobs))
            else:
                release_events.append((0.0, scenario))
        else:
            raise ValueError("Unsupported scenario format")

        release_events.sort(key=lambda x: x[0])
        self.release_events = self._normalize_release_times(release_events)
        self.release_idx = 0

        max_jid = -1
        for _t, jobs in release_events:
            for j in jobs:
                if "jid" in j:
                    try:
                        max_jid = max(max_jid, int(j["jid"]))
                    except Exception:
                        pass
        self._next_job_id = max_jid + 1

        self.t = 0.0
        self.available_tasks: List[dict] = []

        self.robot_positions = self.initial_robot_positions.copy()
        self.robot_free_times = [0.0] * self.num_robots
        self.robot_inventory = [
            {t: 0 for t in self.material_types} for _ in range(self.num_robots)
        ]

        if init_state:
            positions = init_state.get("positions")
            if positions is not None:
                if len(positions) != self.num_robots:
                    raise ValueError("init_state.positions length must equal num_robots")
                self.robot_positions = [self._to_coord(p) for p in positions]
            free_times = init_state.get("free_times")
            if free_times is not None:
                if len(free_times) != self.num_robots:
                    raise ValueError("init_state.free_times length must equal num_robots")
                self.robot_free_times = [max(0.0, float(x)) for x in free_times]
            inventory = init_state.get("inventory")
            if inventory is not None:
                if len(inventory) != self.num_robots:
                    raise ValueError("init_state.inventory length must equal num_robots")
                self.robot_inventory = [
                    {t: int(inv.get(t, 0)) for t in self.material_types}
                    for inv in inventory
                ]

        # Onboard-stock unit events (FIFO per robot per material). Each unit
        # remembers the dock-visit sub-interval that produced it, so the job
        # that eventually consumes it can attribute its "pickup" to that visit
        # (needed for the contract plan's order).
        if not self.consume_initial_inventory:
            self.robot_inventory = [
                {t: 0 for t in self.material_types} for _ in range(self.num_robots)
            ]
        self._stock_events: List[Dict[str, deque]] = [
            {t: deque() for t in self.material_types} for _ in range(self.num_robots)
        ]
        for rid in range(self.num_robots):
            for t in self.material_types:
                for _ in range(int(self.robot_inventory[rid].get(t, 0))):
                    self._stock_events[rid][t].append(
                        {"pickup_start": 0.0, "pickup_end": 0.0, "seq": -1}
                    )

        self._episode_start_positions = [self._to_coord(p) for p in self.robot_positions]

        # One single-slot buffer per machine (see MachineBuffer).
        self.station_buffer: Dict[str, MachineBuffer] = {
            k: MachineBuffer() for k in self.station_locs.keys()
        }
        # Count of forced drops onto a machine whose release was unscheduled
        # (stall-break fallback approximates release = process end).
        self.buffer_override_count = 0
        self.dock_busy_until: Dict[str, float] = {
            k: 0.0 for k in self.dock_locs.keys()
        }

        self.trace: List[dict] = []
        self._dispatch_counter = 0
        self._reservation_cache = {}
        self._max_completion = 0.0

        # Multi-operation (FJSSP) bookkeeping: full job definitions and
        # successor operations waiting for their predecessor to finish.
        self._job_defs: Dict[int, dict] = {}
        self._pending_ops: List[Tuple[float, List[dict]]] = []
        self._op_uid_counter = 0

        self._advance_to_decision_point()
        return self._get_state()

#push the new task into the available_task
    def _release_until(self, t: float) -> None:
        while self.release_idx < len(self.release_events):
            rt, jobs = self.release_events[self.release_idx]
            if rt <= t + 1e-9:
                self.available_tasks.extend(self._jobs_to_tasks(jobs, release_time=rt))
                self.release_idx += 1
            else:
                break
        while self._pending_ops and self._pending_ops[0][0] <= t + 1e-9:
            _rt, op_tasks = self._pending_ops.pop(0)
            self.available_tasks.extend(op_tasks)

    def _next_release_time(self) -> float:
        t = float("inf")
        if self.release_idx < len(self.release_events):
            t = float(self.release_events[self.release_idx][0])
        if self._pending_ops:
            t = min(t, float(self._pending_ops[0][0]))
        return t

    def enqueue_jobs(self, jobs: List[dict], dispatch_time: Optional[float] = None) -> None:
        if dispatch_time is None:
            dispatch_time = self.t
        dispatch_time = float(dispatch_time)
        if dispatch_time < self.t:
            dispatch_time = self.t
        if self.release_events:
            last_t = self.release_events[-1][0]
            if dispatch_time < last_t:
                dispatch_time = last_t

        jobs_copy = []
        for j in jobs:
            item = dict(j)
            if item.get("jid", None) is None:
                item["jid"] = self._next_job_id
                self._next_job_id += 1
            jobs_copy.append(item)

        self.release_events.append((dispatch_time, jobs_copy))

#switch the time slot to the model decision point (event-driven) 
    def _advance_to_decision_point(self) -> None:
        while True:
            self._release_until(self.t)
            
            #the AMR free time
            idle = [
                i for i in range(self.num_robots) if self.robot_free_times[i] <= self.t + 1e-9
            ]

            #into the decision point
            if idle and self.available_tasks:
                if (
                    self.enable_buffer_blocking
                    and len(self.blocked_task_indices()) == len(self.available_tasks)
                ):
                    # Every dispatchable task targets a full machine whose
                    # occupant has no scheduled pickup. Advance to the next
                    # operation release (a processing part will finish and
                    # its successor can unblock the chain) instead of
                    # dispatching a blocked task. Only a true swap-deadlock
                    # (no pending releases) falls through to the stall-break.
                    next_rel = self._next_release_time()
                    if math.isfinite(next_rel) and next_rel > self.t + 1e-9:
                        self.t = max(self.t, next_rel)
                        continue
                self.current_robot = min(idle, key=lambda i: (self.robot_free_times[i], i)) #the decision point ==> no consider the amr location
                self.current_time = self.t
                return

            #no available amr can choose
            if not self.available_tasks:
                next_rel = self._next_release_time()
                if math.isfinite(next_rel):
                    self.t = max(self.t, next_rel)
                    continue
                self.current_robot = int(np.argmin(self.robot_free_times))
                self.current_time = self.t
                return

            self.t = max(self.t, min(self.robot_free_times))

    def done(self) -> bool:
        self._release_until(self.t)
        return (
            (len(self.available_tasks) == 0)
            and (self.release_idx >= len(self.release_events))
            and (not self._pending_ops)
        )

    def makespan(self) -> float:
        """Completion time of the last finished operation — including the
        final delivery to the output point when deliver_finished_to_output
        is on. AMR hand-over times do not extend it."""
        return float(self._max_completion)

    def _get_state(self) -> List[float]:
        """
        State (times are relative to episode start; num_robots=M, stations=S, docks=D):
          - current_robot one-hot (M)
          - available task count (1)
          - current time t (1)
          - for each robot: free_time, x, y, invA, invB, invC (6*M)
          - station busy_until, sorted station keys (S)
          - dock busy_until, sorted dock keys (D)
        total = M + 2 + 6*M + S + D  (current field M=5, S=6, D=3 -> 46)
        """
        st: List[float] = []
        onehot = [0.0] * self.num_robots
        onehot[self.current_robot] = 1.0
        st.extend(onehot)

        st.append(float(len(self.available_tasks)))
        st.append(float(self.t))

        for rid in range(self.num_robots):
            st.append(float(self.robot_free_times[rid]))
            x, y = self.robot_positions[rid]
            st.append(float(x))
            st.append(float(y))
            inv = self.robot_inventory[rid]
            for tkey in self.material_types:
                st.append(float(inv[tkey]))

        for sname in sorted(self.station_locs.keys()):
            # Time from which the machine can accept a new part — the same
            # quantity the planner uses (_station_drop_free_time: occupant's
            # scheduled pickup, else process end as a lower bound; plain
            # process end when buffer blocking is disabled).
            st.append(
                max(
                    float(self.station_buffer[sname].process_end),
                    self._station_drop_free_time(sname),
                )
            )
        for dname in sorted(self.dock_locs.keys()):
            st.append(float(self.dock_busy_until[dname]))
        return st

    def action_features(
        self, robot_id: int, task: dict, replenish: Union[int, Dict[str, int], None]
    ) -> Tuple[float, float, float, float]:
        """
        Return (travel_time, wait_time, proc_time, replenish_total).
        travel_time includes dock waiting + pickup service ticks; wait_time
        is the machine-buffer hold at the drop target; replenish_total is the
        number of units picked up at the dock (0 when delivering from onboard
        stock or transferring a part).
        """
        plan = self._estimate_action_plan(robot_id, task, replenish, for_execution=False)
        rep_total = float(sum(int(v) for v in plan.get("replenish_plan", {}).values()))
        return plan["travel"], plan["wait"], plan["proc"], rep_total

    def step(self, action: Tuple[int, Union[int, Dict[str, int], None]]):
        """
        Timeline:
          start_t = max(env.t, robot_free_time)
          transport [start_t, start_t+travel]
          wait      [arrive, arrive+wait]
          process   [arrive+wait, arrive+wait+proc_time]
        """
        if isinstance(action, tuple):
            action_index, replenish = action
        else:
            action_index = int(action)
            replenish = 0

        if action_index < 0 or action_index >= len(self.available_tasks):
            raise ValueError(
                f"Invalid action_index {action_index}, available={len(self.available_tasks)}"
            )
        prev_makespan = self.makespan()
        prev_finish_sum = float(sum(self.robot_free_times))
        rid = self.current_robot
        task = self.available_tasks[action_index]

        jtype = task["type"]
        station = task["station"]
        jid = task["jid"]
        op_index = int(task.get("op_index", 0))

        replenish_plan = self._normalize_replenish_plan(task, replenish)
        add_units = int(replenish_plan.get(jtype, 0)) if op_index == 0 else 0
        inv = int(self.robot_inventory[rid].get(jtype, 0))
        if op_index == 0:
            if add_units < 0 or inv + add_units > self.capacity_per_type:
                raise ValueError(
                    f"Invalid replenish {add_units} for {jtype} "
                    f"(inv={inv}, cap={self.capacity_per_type})"
                )
            if inv == 0 and add_units == 0:
                raise ValueError(
                    "Invalid action: dock operation with empty stock requires add >= 1"
                )

        plan = self._estimate_action_plan(rid, task, replenish_plan)
        start_t = float(plan["start_t"])
        travel = float(plan["travel"])
        wait = float(plan["wait"])
        proc = float(plan["proc"])
        need_pickup = bool(plan["need_pickup"])
        transport_path = [self._to_coord(p) for p in plan["transport_path"]]
        pickup_types = list(plan.get("pickup_types", []))
        dock_key = str(plan.get("dock", task.get("dock", "")))
        pickup_start_t = float(plan.get("pickup_start_t", start_t))
        pickup_end_t = float(plan.get("pickup_end_t", start_t))

        self.available_tasks.pop(action_index)
        # FJSSP flexibility: dispatching one machine-choice removes its
        # sibling tasks (same operation on other feasible machines).
        op_uid = task.get("op_uid")
        if op_uid is not None:
            self.available_tasks = [
                tk for tk in self.available_tasks if tk.get("op_uid") != op_uid
            ]

        # Stock bookkeeping + pickup attribution. A dock visit picking `add`
        # units creates one unit event per add x unit-duration sub-interval;
        # the job consumes the oldest unit (FIFO) and its "pickup" is
        # attributed to that unit's dock-visit sub-interval so the contract
        # plan's order reproduces the batching.
        attributed_start = pickup_start_t
        attributed_end = pickup_end_t
        if op_index == 0:
            fifo = self._stock_events[rid][jtype]
            unit_dur = float(task.get("pickup_service", proc))
            for k in range(add_units):
                fifo.append(
                    {
                        "pickup_start": pickup_start_t + k * unit_dur,
                        "pickup_end": pickup_start_t + (k + 1) * unit_dur,
                        "seq": self._dispatch_counter,
                    }
                )
            self.robot_inventory[rid][jtype] = inv + add_units
            if not fifo:
                raise RuntimeError("stock accounting error: no unit to consume")
            unit = fifo.popleft()
            self.robot_inventory[rid][jtype] -= 1
            attributed_start = float(unit["pickup_start"])
            attributed_end = float(unit["pickup_end"])

        t_travel_end = float(plan["arrive_t"])
        t_wait_end = float(plan["process_start_t"])
        t_proc_end = t_wait_end + proc

        # Single-slot buffer bookkeeping:
        #  - this dispatch's pickup at a station (op>0 transfer / delivery)
        #    removes the occupant there -> that buffer's release is now
        #    scheduled at the pickup time;
        #  - dropping onto a machine occupies its buffer until some later
        #    dispatch picks the part up (release=None until then). The AMR
        #    held the part during any buffer wait (embedded in transport).
        # Delivery legs target the output point: no occupancy, completion =
        # arrival. Machine processing starts right at the drop.
        is_delivery_leg = bool(task.get("is_delivery", False))
        if op_index > 0 or is_delivery_leg:
            prev_buf = self.station_buffer.get(dock_key)
            if prev_buf is not None:
                prev_buf.schedule_pickup(jid, op_index - 1, pickup_start_t)
        buf = self.station_buffer.get(station)
        if buf is not None:
            if self.enable_buffer_blocking and buf.blocked:
                # Stall-break fallback dropped onto a machine whose pickup
                # was not scheduled yet (release approximated; counted).
                self.buffer_override_count += 1
            buf.occupy(jid, op_index, t_proc_end)
        self._max_completion = max(self._max_completion, t_proc_end)
        if need_pickup and dock_key in self.dock_busy_until:
            self.dock_busy_until[dock_key] = max(
                self.dock_busy_until[dock_key], pickup_end_t
            )

        # Multi-operation job: release the next operation when this one
        # completes; the part then waits at this station for its next pickup.
        # After the LAST operation, release a delivery task to the output
        # point (dock "T") instead — the job completes on arrival there.
        op_index = int(task.get("op_index", 0))
        num_ops = int(task.get("num_ops", 1))
        is_delivery = bool(task.get("is_delivery", False))
        successors: Optional[List[dict]] = None
        if not is_delivery:
            if op_index + 1 < num_ops:
                successors = self._make_op_tasks(
                    task["jid"], op_index + 1, t_proc_end, prev_station=station
                )
            elif self.deliver_finished_to_output and self.output_locs:
                successors = [self._make_delivery_task(task, station, t_proc_end)]
        if successors:
            self._pending_ops.append((float(t_proc_end), successors))
            self._pending_ops.sort(key=lambda x: x[0])

        if self.enable_collision_avoidance:
            # The robot steps aside right after the drop-off (t_travel_end).
            post_pos = self._post_process_position(transport_path, self._to_coord(task["drop"]))
            post_cands = self._post_process_candidates(transport_path, self._to_coord(task["drop"]))
            post_res = self._build_dynamic_reservations(
                rid,
                t_travel_end,
                t_travel_end + 3.0,
                future_work_after_dispatch=True,
            )
            for cand in post_cands:
                if self._point_conflict(post_res, cand, t_travel_end):
                    continue
                if self._transition_conflict(
                    post_res,
                    cand,
                    cand,
                    t_travel_end,
                    t_travel_end + 1.0,
                    ignore_source_point_at_t0=True,
                ):
                    continue
                post_pos = self._to_coord(cand)
                break
        else:
            # Keep robot at the workstation after the drop-off when collision
            # avoidance is disabled. This removes artificial travel between
            # consecutive jobs at the same station.
            post_pos = self._to_coord(task["drop"])

        # AMR is free as soon as the part is dropped off at the machine.
        self.robot_free_times[rid] = t_travel_end
        self.robot_positions[rid] = post_pos

        self.trace.append(
            {
                "seq": self._dispatch_counter,
                "robot": rid,
                "jid": jid,
                "op_index": int(op_index),
                "num_ops": int(num_ops),
                "is_delivery": is_delivery,
                "replenish": int(add_units),
                "replenish_main": int(add_units),
                "replenish_plan": {
                    t: int(replenish_plan.get(t, 0)) for t in self.material_types
                },
                "type": jtype,
                "src": dock_key,
                "dst": station,
                "proc_time": proc,
                "transport_path": transport_path,
                "need_pickup": bool(need_pickup),
                "pickup_types": list(pickup_types),
                "dock": dock_key,
                "pickup": task["pickup"],
                # Attributed pickup interval (which dock-visit sub-interval
                # produced this job's material) -> used for the contract order.
                "pickup_start": attributed_start,
                "pickup_end": attributed_end,
                # Physical dock-visit interval of THIS dispatch (empty stock
                # delivery has no visit: start == end == dispatch start).
                "dock_visit_start": pickup_start_t,
                "dock_visit_end": pickup_end_t,
                "drop": task["drop"],
                "post_pos": post_pos,
                "segments": [
                    {"kind": "transport", "start": start_t, "end": t_travel_end},
                    {"kind": "wait", "start": t_travel_end, "end": t_wait_end},
                    {"kind": "process", "start": t_wait_end, "end": t_proc_end},
                ],
            }
        )
        self._dispatch_counter += 1
        self._reservation_cache = {}

        next_release = self._next_release_time()
        next_robot_free = min(self.robot_free_times) if self.robot_free_times else float("inf")
        next_event_t = min(next_release, next_robot_free)
        if not math.isfinite(next_event_t):
            next_event_t = self.makespan()
        # Keep simulation time monotonic to avoid timeline rollback.
        self.t = max(self.t, float(next_event_t))

        self._advance_to_decision_point()

        # Dense reward aligned with the contract objective
        # makespan + w * sum(per-AMR finish times):
        #   sum_t reward_t = -(final_makespan + w * sum(final finish times))
        new_makespan = self.makespan()
        new_finish_sum = float(sum(self.robot_free_times))
        reward = -(new_makespan - prev_makespan) - self.objective_load_balance_weight * (
            new_finish_sum - prev_finish_sum
        )
        if self.done():
            return None, reward, True
        return self._get_state(), reward, False
