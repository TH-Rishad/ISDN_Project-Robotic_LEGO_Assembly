import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import unittest

from Kean.motion.gate import apply_brace_all, count_events, fold, jobs, judge, placement_count, stamp
from Kean.motion.world import brick_boxes, held_brick, yaw_of


class GateTest(unittest.TestCase):
    def test_job_list_is_the_m05_grid(self):
        rows = jobs()
        self.assertEqual(len(rows), 23)
        self.assertEqual(placement_count(rows), 235)
        self.assertEqual(sum(row["stress"] for row in rows), 3)
        self.assertEqual(sum(row["strategy"] == "weakest_joint" for row in rows), 20)

    def test_brace_all_rewrites_only_feasible_steps(self):
        plan = {"sequence": [
            {"step": 0, "requires_brace": False, "brace": None},
            {"step": 1, "requires_brace": False, "brace": None},
        ], "validation": {}}
        apply_brace_all(plan, lambda step: {"feasible": True, "brace_pose": [0, 0, 0]} if step["step"] else None)
        self.assertFalse(plan["sequence"][0]["requires_brace"])
        self.assertTrue(plan["sequence"][1]["requires_brace"])
        self.assertEqual(plan["validation"]["bracing_strategy"], "brace_all")
        self.assertEqual(plan["validation"]["brace_required_count"], 1)

    def test_arm_arm_is_split_and_rub_is_not_counted(self):
        events = [
            {"class": "arm_arm", "braced": True},
            {"class": "arm_arm", "braced": False},
            {"class": "neighbour_rub"},
            {"class": "unintended"},
        ]
        counted = count_events(events)
        self.assertEqual(counted["arm_arm_braced"], 1)
        self.assertEqual(counted["arm_arm_unbraced"], 1)
        self.assertEqual(counted["neighbour_rub"], 1)
        self.assertEqual(counted["unintended"], 1)
        raw = [{"class": "arm_arm"}]
        stamp(raw, 3, True)
        self.assertTrue(raw[0]["braced"])
        stamp(raw, 4, False)
        self.assertEqual(raw[0]["step"], 3)

    def test_clean_gate_passes_and_a_collision_fails(self):
        runs = [{"shape": "cube", "seed": 0, "stress": False, "placed": 8, "events": []}]
        clean = fold(runs, {"plan_failures": 0, "captures": 0, "min_clearance_m": 0.004})
        self.assertTrue(judge(clean)["ok"])
        dirty = dict(clean, arm_arm_unbraced=1)
        self.assertIn("arm_arm_unbraced", judge(dirty)["reasons"])

    def test_held_brick_is_left_out_of_the_world(self):
        self.assertIsNone(held_brick("B", "grasp", "b0"))
        self.assertEqual(held_brick("B", "transport", "b0"), "b0")
        self.assertIsNone(held_brick("A", "hold", "b0"))
        self.assertAlmostEqual(yaw_of([0, 0, 0, 1]), 0.0)
        from unittest.mock import Mock
        box_of = Mock(return_value={"shared": True})
        boxes = brick_boxes(np.array([[0.1, 0.2, 0.0, 0, 0, 0, 1.0]]), {"b0": 0}, {"b0": (0.032, 0.016)}, box_of=box_of)
        self.assertEqual(boxes, [{"shared": True}])
        args, kwargs = box_of.call_args
        self.assertEqual(kwargs, {"name": "b0"})
        np.testing.assert_allclose(args[1]["b0"]["bottom"], [0.1, 0.2, 0.0])
        self.assertEqual(args[1]["b0"]["L"], 0.032)


if __name__ == "__main__":
    unittest.main()
