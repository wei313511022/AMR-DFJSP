import sys
import unittest
from pathlib import Path
from unittest import mock


STATIC_DIR = Path(__file__).resolve().parents[1]
if str(STATIC_DIR) not in sys.path:
    sys.path.insert(0, str(STATIC_DIR))

import GA.GA as GA  # noqa: E402


class DockWaitingLineTests(unittest.TestCase):
    def make_jobs(self, count=5, inbound_dock="dock1", station="station1", duration=5.0):
        return [
            GA.Job(
                idx=i,
                type_="A",
                duration=duration,
                station=station,
                inbound_dock=inbound_dock,
                arrival_time=0.0,
            )
            for i in range(count)
        ]

    def make_individual(self, jobs, include_unload=False):
        order = [GA.Operation(job.idx, GA.PICKUP) for job in jobs]
        if include_unload:
            order.extend(GA.Operation(job.idx, GA.UNLOAD) for job in jobs)
        return GA.Individual(
            order=order,
            amr_assignment=[GA.AMR_KEYS[i % len(GA.AMR_KEYS)] for i in range(len(jobs))],
        )

    def test_waiting_line_geometry_matches_configuration(self):
        """Derived from the live door positions, so it survives layout changes."""
        self.assertEqual(GA.WAIT_LINE_DEPTH, 3)
        for door in (GA.INBOUND_DOCK_LOCATIONS["dock3"], GA.STATIONS["station3"]):
            dx, dy = GA._dock_inward_direction(door)
            slots = GA.dock_waiting_slots(door)
            lateral = ((0, 1), (0, -1)) if dx else ((1, 0), (-1, 0))
            if GA.QUEUE_GEOMETRY == "inward":
                self.assertEqual(
                    slots,
                    tuple((door[0] + dx * d, door[1] + dy * d)
                          for d in range(1, GA.WAIT_LINE_DEPTH + 1)),
                    "inward queue must be a single file leading away from the door")
                for lx, ly in lateral:
                    self.assertNotIn(
                        (door[0] + lx, door[1] + ly), slots,
                        "the cells flanking a wall-mounted door are its only "
                        "escape routes and must stay out of the queue")
            else:
                for d in range(1, GA.WAIT_LINE_DEPTH + 1):
                    self.assertNotIn((door[0] + dx * d, door[1] + dy * d), slots,
                                     "lateral queue must leave the centreline clear")

    def test_boundary_docks_only_use_valid_grid_slots(self):
        for dock_pos in (GA.INBOUND_DOCK_LOCATIONS["dock1"], GA.INBOUND_DOCK_LOCATIONS["dock5"], GA.STATIONS["station1"], GA.STATIONS["station5"]):
            for slot in GA.dock_waiting_slots(dock_pos):
                self.assertTrue(GA._is_within_bounds(slot))
                self.assertNotIn(slot, GA.DOCK_SERVICE_CELLS)

    def test_no_waiting_slot_is_claimed_by_two_doors(self):
        """A cell in two queues at once is a reservation conflict waiting to happen."""
        if GA.QUEUE_GEOMETRY != "inward":
            self.skipTest("lateral lines of nearby doors may legitimately overlap")
        owner = {}
        for name, door in {**GA.INBOUND_DOCK_LOCATIONS, **GA.STATIONS}.items():
            for slot in GA.dock_waiting_slots(door):
                self.assertNotIn(
                    slot, owner,
                    f"{slot} is a waiting slot for both {owner.get(slot)} and {name}")
                owner[slot] = name

    def test_bays_are_never_inside_a_waiting_area(self):
        """A robot sent to wait in another robot's parked bay can never arrive."""
        queue_cells = set()
        for door in list(GA.INBOUND_DOCK_LOCATIONS.values()) + list(GA.STATIONS.values()):
            queue_cells |= set(GA.dock_waiting_slots(door))
        self.assertFalse(set(GA.AMR_STARTS.values()) & queue_cells)

    def test_inbound_pickup_uses_job_duration_and_fifo_service(self):
        jobs = self.make_jobs(count=5, inbound_dock="dock1", duration=5.0)
        individual = self.make_individual(jobs)

        _, timeline, _, _, invalid_count = GA.decode_schedule(individual, jobs, need_log=True, check_collision=False)

        self.assertEqual(invalid_count, 0)
        load_events = [event for event in timeline if event[3] == "load_inbound"]
        self.assertEqual(len(load_events), 5)
        for previous, current in zip(load_events, load_events[1:]):
            self.assertGreaterEqual(current[1], previous[2])
        self.assertTrue(all(event[2] - event[1] == 5.0 for event in load_events))

    def test_outbound_dock_uses_fifo_service(self):
        jobs = self.make_jobs(count=5, inbound_dock="dock1", station="station1", duration=5.0)
        individual = self.make_individual(jobs, include_unload=True)

        _, timeline, _, _, invalid_count = GA.decode_schedule(individual, jobs, need_log=True, check_collision=False)

        self.assertEqual(invalid_count, 0)
        process_events = [event for event in timeline if event[3] == "process_A"]
        self.assertEqual(len(process_events), 5)
        for previous, current in zip(process_events, process_events[1:]):
            self.assertGreaterEqual(current[1], previous[2])

    def test_queue_full_holds_amr_upstream(self):
        jobs = self.make_jobs(count=2, inbound_dock="dock1", duration=20.0)
        individual = self.make_individual(jobs)

        with mock.patch.object(GA, "dock_waiting_slots", return_value=tuple()):
            _, timeline, _, _, invalid_count = GA.decode_schedule(individual, jobs, need_log=True, check_collision=False)

        self.assertEqual(invalid_count, 0)
        self.assertTrue(any(event[3] == "hold_upstream" for event in timeline))


if __name__ == "__main__":
    unittest.main()
