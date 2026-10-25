import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest

from Kean.acceptance import generate, judge, plate_z, preview, preview_ok


class AcceptanceTest(unittest.TestCase):
    def test_plate_matches_the_cell(self):
        self.assertAlmostEqual(plate_z(), 0.0032)

    def test_arch_and_s3_are_generated(self):
        payload = preview()
        self.assertEqual(preview_ok(payload), [])
        by = {row["structure"]: row for row in payload["structures"]}
        arch, s3 = by["arch"], by["S3"]
        self.assertEqual(arch["bricks"], 11)
        self.assertEqual(s3["bricks"], 16)
        self.assertGreaterEqual(arch["brace_required_count"], 1)
        self.assertGreaterEqual(s3["brace_required_count"], 1)
        self.assertTrue(s3["single_arm_fails"])
        self.assertEqual(arch["strategy"], "weakest_joint")
        self.assertFalse(payload["image_input"])
        crown = arch["steps"][-1]
        self.assertTrue(crown["requires_brace"])
        self.assertIsNotNone(crown["support_pose"])
        self.assertTrue(any(s["requires_brace"] and s["support_pose"] for s in s3["steps"]))

    def test_none_is_refused(self):
        with self.assertRaises(ValueError):
            generate("S3", "structure", "none")

    def test_judge_requires_each_generated_support_held(self):
        arch = {
            "shape": "arch", "placed": 11, "total": 11, "failure": None,
            "required_supports": [10], "supports": [{"step": 10, "executed": False}],
        }
        s3 = {
            "shape": "S3", "placed": 16, "total": 16, "failure": None,
            "required_supports": [11], "supports": [{"step": 11, "executed": True}],
        }
        self.assertIn("unheld:arch:10", judge([arch, s3]))


if __name__ == "__main__":
    unittest.main()
