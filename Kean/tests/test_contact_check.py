import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest

from Kean.motion.contact_check import rescore


def _row(name, brick, finger_sphere, finger_box, arm=1.0, events=None):
    return {
        "file": name,
        "brick": brick,
        "b_step": 0,
        "min_clearance_mm": {"finger": finger_sphere, "arm": arm, "arm_arm": 5.0, "held_brick": 2.0},
        "finger_box_min_mm": finger_box,
        "monitor_events": events or [],
        "monitor_events_known": True,
    }


class ContactCheckTest(unittest.TestCase):
    def test_boxes_clear_a_sphere_false_positive(self):
        summ = {"segments": 1, "pads_mm": [0.0, 0.25], "rows": [
            _row("cube_grasp_lp_r0.npz", "b_000", finger_sphere=-0.11, finger_box=0.18)]}
        out = rescore(summ, {"cube_grasp_lp_r0.npz": set()})
        self.assertEqual(out["pad0"]["false_positives"], 0)
        self.assertTrue(out["ok"])
        self.assertGreaterEqual(out["min_clean_segment_clearance_mm"], 0.18)

    def test_declared_graze_event_is_not_a_miss(self):
        summ = {"segments": 1, "pads_mm": [0.0], "rows": [
            _row("cube_grasp_lp_r0.npz", "b_005", finger_sphere=-0.03, finger_box=0.14, events=["finger"])]}
        out = rescore(summ, {"cube_grasp_lp_r0.npz": {"b_005"}})
        self.assertEqual(out["pad0"]["missed_events"], 0)
        self.assertEqual(out["pad0"]["false_positives"], 0)
        self.assertEqual(out["declared_graze_events"], 1)
        self.assertTrue(out["ok"])

    def test_non_graze_event_the_check_misses_fails(self):
        summ = {"segments": 1, "pads_mm": [0.0], "rows": [
            _row("cube_grasp_lp_r0.npz", "b_000", finger_sphere=1.0, finger_box=1.0, events=["arm"])]}
        out = rescore(summ, {"cube_grasp_lp_r0.npz": set()})
        self.assertEqual(out["pad0"]["missed_events"], 1)
        self.assertFalse(out["ok"])


if __name__ == "__main__":
    unittest.main()
