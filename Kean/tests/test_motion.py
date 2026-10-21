import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import unittest

from Kean.motion.contact_segment import CONTACT_M, STEP_M, sample, split
from Kean.motion.plan_curobo import GraphCaptured, NotWarmed, Planner, PlannerUnavailable, Request, ZoneBusy, pre_insert, same_start


class Backend:
    def __init__(self, captures=0):
        self.captures = captures
        self.warmed = False

    def warmup(self, guard):
        self.warmed = True
        guard.note(0)

    def plan(self, request, handoff):
        q = np.asarray(request.q, float)[:7]
        return {"joint_pos": np.vstack([q, q + 0.1]), "captures": self.captures, "dt": 0.02}


class MotionTest(unittest.TestCase):
    def test_last_30_mm_is_straight(self):
        start = np.zeros(3)
        goal = np.array([0.0, 0.0, 0.10])
        handoff, end = split(start, goal)
        np.testing.assert_allclose(end, goal)
        np.testing.assert_allclose(np.linalg.norm(goal - handoff), CONTACT_M)
        pts = sample(handoff, goal)
        self.assertLessEqual(np.diff(np.linalg.norm(pts - pts[0], axis=1)).max(), STEP_M + 1e-9)
        step = np.diff(pts, axis=0)
        step = step / np.linalg.norm(step, axis=1)[:, None]
        np.testing.assert_allclose(step, np.repeat(step[:1], len(step), axis=0), atol=1e-6)

    def test_short_move_is_all_cartesian(self):
        start = np.zeros(3)
        goal = np.array([0.0, 0.0, 0.02])
        handoff, end = split(start, goal)
        np.testing.assert_allclose(handoff, start)
        np.testing.assert_allclose(end, goal)

    def test_pre_insert_is_25_mm(self):
        np.testing.assert_allclose(pre_insert([0.1, 0.2, 0.03]), [0.1, 0.2, 0.055])

    def test_refuses_before_warmup_and_after_capture(self):
        planner = Planner(Backend())
        request = Request("B", np.zeros(7), np.zeros(3), np.array([0.2, 0, 0.2]), np.array([0, 0, 0, 1.0]))
        with self.assertRaises(NotWarmed):
            planner.plan(request)
        planner.warmup()
        out = planner.plan(request)
        self.assertIsNotNone(out["joint_pos"])
        self.assertAlmostEqual(out["dt"], 0.04)
        planner.close()
        late = Planner(Backend(captures=1))
        late.warmup()
        with self.assertRaises(GraphCaptured):
            late.plan(request)
        late.close()

    def test_cartesian_only_when_inside_30_mm(self):
        planner = Planner(Backend())
        planner.warmup()
        request = Request("B", np.zeros(7), np.zeros(3), np.array([0, 0, 0.02]), np.array([0, 0, 0, 1.0]))
        out = planner.plan(request)
        self.assertIsNone(out["joint_pos"])
        self.assertGreater(len(out["cartesian"]), 1)
        planner.close()

    def test_a_flipped_seed_is_not_the_same_start(self):
        q = np.zeros(7)
        self.assertTrue(same_start(q, q + 0.1))
        flipped = q.copy()
        flipped[0] = 3.0
        self.assertFalse(same_start(flipped, q))

    def test_shared_motion_coexists_with_kean(self):
        import importlib
        from Kean.shared import add_cell_path
        import Kean.motion as kean_motion
        add_cell_path()
        vendor = importlib.import_module("motion.vendor")
        self.assertIn("brickassembly", vendor.__file__)
        probe = Path(vendor.__file__).resolve().parents[1] / "scripts" / "12_curobo_probe.py"
        if probe.is_file() and "add_overlay" in probe.read_text():
            self.assertTrue(hasattr(vendor, "add_overlay"))
        self.assertIs(importlib.import_module("Kean.motion"), kean_motion)

    def test_zone_and_thread(self):
        planner = Planner(Backend())
        planner.warmup()
        request = Request("B", np.zeros(7), np.zeros(3), [0.2, 0, 0.2], [0, 0, 0, 1], zone_clear=False)
        with self.assertRaises(ZoneBusy):
            planner.plan(request)
        ok = Request("A", np.zeros(7), np.zeros(3), [0.2, 0, 0.2], [0, 0, 0, 1])
        future = planner.submit(ok)
        self.assertIn("cartesian", future.result(timeout=2))
        planner.close()


if __name__ == "__main__":
    unittest.main()
