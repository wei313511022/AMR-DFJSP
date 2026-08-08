"""Door clearance and the one-cell following gap.

The bug: a door was free the instant service ended, so the next AMR reserved
the cell while the incumbent was still standing on it. The incumbent was then
evicted with nowhere legal to go and its leg failed.

The fix keeps the safety gap untouched and gives the incumbent a departure
grace period (DOCK_CLEARANCE_TICKS) instead.
"""
import sys
import unittest
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parents[1]
if str(STATIC_DIR) not in sys.path:
    sys.path.insert(0, str(STATIC_DIR))

from GA import GA as G  # noqa: E402


class TestFollowingGap(unittest.TestCase):
    """The one-cell headway must survive the fix, unchanged."""

    def test_head_on_swap_is_blocked(self):
        # B occupies (5,6) at t=10 and moves into (5,5) at t=11.
        # A at (5,5) must not take (5,6) at t=11 -- they would pass through.
        # Routing around is fine; only the direct exchange must be forbidden.
        res = {((5, 6), 10): "B", ((5, 5), 11): "B"}
        path = G.find_dynamic_path((5, 5), (5, 6), 10, res, {}, "A")
        self.assertNotEqual(path[1], (5, 6), "a genuine head-on swap must stay blocked")

    def test_following_too_closely_is_blocked(self):
        # B vacates (5,6) at t=11 (it is at (5,7) then). A may NOT step straight
        # in behind it: that closes the one-cell gap.
        res = {((5, 6), 10): "B", ((5, 7), 11): "B"}
        path = G.find_dynamic_path((5, 5), (5, 6), 10, res, {}, "A")
        self.assertNotEqual(path[1], (5, 6),
                            "stepping in at t=11 closes the gap and must stay blocked")

    def test_following_after_waiting_is_allowed(self):
        # Same departure, but A waits a tick first. By t=12 the gap is two
        # cells, so the move is legal -- this is the behaviour the clearance
        # period exists to make reachable.
        res = {((5, 6), 10): "B", ((5, 7), 11): "B", ((5, 8), 12): "B"}
        path = G.find_dynamic_path((5, 5), (5, 6), 10, res, {}, "A")
        self.assertEqual(path[:3], [(5, 5), (5, 5), (5, 6)],
                         "A should wait one tick, then follow with a full two-cell gap")


class TestSelfReservationExemption(unittest.TestCase):
    """A robot's own reservations mark where it is, not an obstacle."""

    def test_own_reservation_does_not_block_waiting(self):
        # A holds (5,5) through t=14 (its own door clearance) and everything
        # around it is briefly busy. It must still be able to stand there.
        res = {((5, 5), t): "A" for t in range(10, 15)}
        res[((5, 6), 11)] = "B"
        path = G.find_dynamic_path((5, 5), (5, 9), 10, res, {}, "A")
        self.assertGreater(len(path), 1, "a robot must not be blocked by its own reservation")

    def test_other_robots_reservation_still_blocks(self):
        res = {((5, 6), t): "B" for t in range(10, 40)}
        path = G.find_dynamic_path((5, 5), (5, 6), 10, res, {}, "A")
        self.assertNotEqual(path[1], (5, 6),
                            "another robot's reservation must still block entry")


class TestDockClearance(unittest.TestCase):
    def test_clearance_constant_is_configured(self):
        self.assertGreaterEqual(G.DOCK_CLEARANCE_TICKS, 0)

    def test_incumbent_keeps_standing_room(self):
        """The reported failure, reduced.

        AMR5 finishes at a door at t=12. AMR7 books the same cell. With no
        clearance AMR7 arrives at t=14 and AMR5 -- whose neighbours are all
        taken at that instant -- is stuck. With clearance the door stays held,
        AMR7 cannot arrive that early, and AMR5 leaves with full headway.
        """
        door = (0, 10)
        res = {}
        for t in range(10, 13):                      # AMR5 in service
            res[(door, t)] = "AMR5"
        for t in range(13, 16):                      # neighbours briefly busy
            res[((0, 9), t)] = "AMR15"
            res[((0, 11), t)] = "AMR7"
            res[((1, 10), t)] = "AMR6"
        no_clearance = dict(res)
        no_clearance[(door, 14)] = "AMR7"            # next robot evicts the incumbent
        self.assertEqual(
            G.find_dynamic_path(door, (5, 10), 12, no_clearance, {}, "AMR5"), [door],
            "without clearance the evicted incumbent has no legal move")

        with_clearance = dict(res)
        for t in range(13, 15):                      # door held by AMR5 instead
            with_clearance[(door, t)] = "AMR5"
        self.assertGreater(
            len(G.find_dynamic_path(door, (5, 10), 12, with_clearance, {}, "AMR5")), 1,
            "with clearance the incumbent can wait and then depart")


if __name__ == "__main__":
    unittest.main(verbosity=2)
