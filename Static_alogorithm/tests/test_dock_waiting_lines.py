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

    def test_waiting_line_depth_generates_inward_slots(self):
        self.assertEqual(GA.WAIT_LINE_DEPTH, 3)
        self.assertEqual(
            GA.dock_waiting_slots(GA.INBOUND_DOCK_LOCATIONS["dock3"]),
            ((0, 6), (0, 4), (1, 6), (1, 4), (2, 6), (2, 4), (3, 6), (3, 4)),
        )
        self.assertEqual(
            GA.dock_waiting_slots(GA.STATIONS["station3"]),
            ((9, 6), (9, 4), (8, 6), (8, 4), (7, 6), (7, 4), (6, 6), (6, 4)),
        )

    def test_boundary_docks_only_use_valid_grid_slots(self):
        for dock_pos in (GA.INBOUND_DOCK_LOCATIONS["dock1"], GA.INBOUND_DOCK_LOCATIONS["dock5"], GA.STATIONS["station1"], GA.STATIONS["station5"]):
            for slot in GA.dock_waiting_slots(dock_pos):
                self.assertTrue(GA._is_within_bounds(slot))
                self.assertNotIn(slot, GA.DOCK_SERVICE_CELLS)

    def test_adjacent_docks_share_candidate_slot_for_global_reservation(self):
        dock1_slots = GA.dock_waiting_slots(GA.INBOUND_DOCK_LOCATIONS["dock1"])
        dock2_slots = GA.dock_waiting_slots(GA.INBOUND_DOCK_LOCATIONS["dock2"])
        self.assertIn((0, 8), dock1_slots)
        self.assertIn((0, 8), dock2_slots)

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
