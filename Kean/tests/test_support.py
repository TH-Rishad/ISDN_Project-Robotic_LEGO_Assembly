import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import unittest

from Kean.memory.scan import ScanSchedule
from Kean.memory.store import Store
from Kean.support.grip_adjust import GripAdjust
from Kean.support.hold import NoSupport, support_target
from Kean.support.sequence import Handoff, NotSettled, Zone, ZoneBusy


class SupportTest(unittest.TestCase):
    def test_consumes_brace_pose(self):
        target = support_target({
            "feasible": True, "brace_pose": [0.0, -0.04, 0.02], "brace_tilt_rotvec": [0.3, 0, 0],
            "brace_axis": [1, 0, 0], "brace_force_N": 20.0, "target_brick_id": "b1",
        })
        np.testing.assert_allclose(target["pose"], [0, -0.04, 0.02])
        self.assertAlmostEqual(target["yaw"], np.pi / 2)
        self.assertEqual(target["brick_id"], "b1")

    def test_rejects_missing_brace(self):
        with self.assertRaises(NoSupport):
            support_target({"feasible": False})

    def test_b_waits_for_a(self):
        zone = Zone(np.zeros(3), 0.05)
        hand = Handoff(zone)
        with self.assertRaises(NotSettled):
            hand.begin("B", np.zeros(3), np.array([0.4, 0, 0.2]))
        hand.begin("A", np.zeros(3), np.array([0.4, 0, 0.2]))
        hand.settled("A")
        hand.begin("B", np.zeros(3), np.zeros(3))
        self.assertEqual(hand.b, "moving")

    def test_one_arm_moves_in_the_zone(self):
        zone = Zone(np.zeros(3), 0.05)
        hand = Handoff(zone)
        hand.a = "holding"
        hand.begin("B", np.zeros(3), np.zeros(3))
        with self.assertRaises(ZoneBusy):
            hand.begin("A", np.zeros(3), np.zeros(3))

    def test_grip_adjust_rescans_then_freezes(self):
        store = Store()
        q = np.array([0.0, 0.0, 0.0, 1.0])
        store.freeze("B", "b0", np.zeros(3), q, np.array([0, 0, 0.1]), q)
        schedule = ScanSchedule()
        zone = Zone(np.zeros(3), 0.05)
        hand = Handoff(zone)
        hand.a = "holding"
        adj = GripAdjust("b0", [0.0, 0.0, 0.12])
        outside = np.array([0.5, 0.0, 0.2])
        adj.start(hand, np.zeros(3), outside)
        adj.arrived("A", store, schedule, handoff=hand)
        self.assertEqual(adj.arrived("B", store, schedule)["do"], "scan")
        self.assertNotIn("B", store.grasps)
        adj.scanned(store, schedule, [{"id": "b0", "xyz": [0.01, 0, 0.1], "yaw": 0.0}], hand, np.zeros(3), outside)
        np.testing.assert_allclose(store.pose("b0")[0], [0.01, 0, 0.1])
        self.assertNotIn("B", store.grasps)
        new_hand = np.array([0.01, 0.0, 0.12])
        adj.arrived("B", store, schedule, new_hand, q, hand, outside, new_hand)
        inferred, _ = store.infer("B", new_hand, q)
        np.testing.assert_allclose(inferred, [0.01, 0, 0.1], atol=1e-6)
        adj.arrived("A", store, schedule, handoff=hand)
        self.assertEqual(adj.phase, "done")
        self.assertEqual(hand.a, "holding")


if __name__ == "__main__":
    unittest.main()
