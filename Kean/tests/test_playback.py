import tempfile
import unittest
from pathlib import Path

import numpy as np

from Kean.motion.playback import apply_recorded, plan_file, recording_file


class PlaybackTest(unittest.TestCase):
    def test_apply_writes_arm_joints_and_matching_bricks(self):
        joints = np.zeros(20)
        bodies = np.zeros((4, 7))
        updated_q, updated_b = apply_recorded(
            joints, bodies, ["b_001", "b_000"], {"b_000": 3, "b_001": 1},
            np.arange(18), np.array([[1, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 1]]))
        np.testing.assert_allclose(updated_q[:18], np.arange(18))
        self.assertEqual(updated_q[18], 0.0)
        np.testing.assert_allclose(updated_b[1, 0], 1.0)
        np.testing.assert_allclose(updated_b[3, 0], 2.0)
        self.assertEqual(joints[0], 0.0)

    def test_missing_brick_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_recorded(np.zeros(18), np.zeros((1, 7)), ["b_000"], {}, np.zeros(18), np.zeros((1, 7)))

    def test_paths_follow_the_saved_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "plans").mkdir()
            (run / "plans" / "arch_weakest_joint.json").write_text("{}")
            (run / "arch_weakest_joint_s0.npz").write_bytes(b"")
            self.assertEqual(recording_file(run, "arch").name, "arch_weakest_joint_s0.npz")
            self.assertEqual(plan_file(run, "arch").name, "arch_weakest_joint.json")


if __name__ == "__main__":
    unittest.main()
