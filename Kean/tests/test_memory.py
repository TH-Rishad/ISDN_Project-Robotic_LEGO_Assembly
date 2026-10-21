import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import unittest

from Kean.memory.scan import ScanRefused, ScanSchedule, lifted
from Kean.memory.store import Store, compose, qrot, relative


class MemoryTest(unittest.TestCase):
    def test_freeze_round_trip(self):
        store = Store()
        hand = np.array([0.1, 0.2, 0.3])
        hq = np.array([0.0, 0.0, 0.0, 1.0])
        brick = np.array([0.1, 0.2, 0.28])
        bq = np.array([0.0, 0.0, np.sin(0.2), np.cos(0.2)])
        store.freeze("B", "b0", hand, hq, brick, bq)
        moved = hand + np.array([0.0, 0.0, 0.01])
        xyz, quat = store.infer("B", moved, hq)
        np.testing.assert_allclose(xyz, brick + np.array([0.0, 0.0, 0.01]), atol=1e-6)
        np.testing.assert_allclose(quat, bq, atol=1e-6)

    def test_held_brick_is_not_overwritten(self):
        store = Store()
        store.freeze("B", "b0", np.zeros(3), np.array([0, 0, 0, 1.0]), np.array([0, 0, 0.1]), np.array([0, 0, 0, 1.0]))
        schedule = ScanSchedule()
        schedule.allow_step_done()
        schedule.apply(store, [{"id": "b0", "xyz": [1, 1, 1], "yaw": 0.0}, {"id": "b1", "xyz": [0.2, 0, 0], "yaw": 0.0}])
        np.testing.assert_allclose(store.pose("b0")[0], [0, 0, 0.1])
        np.testing.assert_allclose(store.pose("b1")[0], [0.2, 0, 0])

    def test_scan_refused_until_a_reason(self):
        schedule = ScanSchedule()
        with self.assertRaises(ScanRefused):
            schedule.apply(Store(), [])

    def test_init_is_one_shot(self):
        schedule = ScanSchedule()
        schedule.allow_init()
        schedule.apply(Store(), [{"id": "b0", "xyz": [0, 0, 0], "yaw": 0}])
        with self.assertRaises(ScanRefused):
            schedule.apply(Store(), [])

    def test_grip_adjust_updates_only_that_brick(self):
        store = Store()
        store.freeze("A", "held", np.zeros(3), np.array([0, 0, 0, 1.0]), np.ones(3), np.array([0, 0, 0, 1.0]))
        store.freeze("B", "b0", np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3), np.array([0, 0, 0, 1.0]))
        store.release("B")
        schedule = ScanSchedule()
        schedule.begin_adjust("b0")
        schedule.apply(store, [{"id": "b0", "xyz": [0.4, 0, 0], "yaw": 0.3}, {"id": "held", "xyz": [9, 9, 9], "yaw": 0}])
        np.testing.assert_allclose(store.pose("b0")[0], [0.4, 0, 0])
        np.testing.assert_allclose(store.pose("held")[0], np.ones(3))

    def test_lift_freezes_once(self):
        store = Store()
        q = np.array([0.0, 0.0, 0.0, 1.0])
        hand, brick = np.zeros(3), np.array([0.0, 0.0, 0.1])
        self.assertFalse(lifted(store, "descend", "grasp", "B", "b0", hand, q, brick, q))
        self.assertTrue(lifted(store, "grasp", "lift", "B", "b0", hand, q, brick, q))
        self.assertFalse(lifted(store, "grasp", "lift", "B", "b0", hand, q, brick + 1, q))
        np.testing.assert_allclose(store.infer("B", hand, q)[0], brick)

    def test_relative_matches_compose(self):
        hand = np.array([0.2, -0.1, 0.4])
        hq = np.array([0.0, 0.0, np.sin(0.4), np.cos(0.4)])
        off = np.array([0.0, 0.01, -0.02])
        oq = np.array([0.0, 0.0, 0.0, 1.0])
        brick, bq = compose(hand, hq, off, oq)
        back, bq2 = relative(hand, hq, brick, bq)
        np.testing.assert_allclose(back, off, atol=1e-6)
        np.testing.assert_allclose(qrot(hq, off) + hand, brick, atol=1e-6)
        np.testing.assert_allclose(bq2, oq, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
