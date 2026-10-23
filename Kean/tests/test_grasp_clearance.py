import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Kean.motion.grasp_clearance import choose_offset


class GraspClearanceTest(unittest.TestCase):
    def test_clear_grasp_stays_centred(self):
        np.testing.assert_allclose(choose_offset(lambda o: 0.002, [1, 0, 0]), 0)

    def test_shift_clears_neighbour_with_margin(self):
        offset = choose_offset(lambda o: o[0] - 0.00075, [1, 0, 0])
        self.assertAlmostEqual(offset[0], 0.00125)
        np.testing.assert_allclose(offset[1:], 0)

    def test_checks_both_directions(self):
        offset = choose_offset(lambda o: -o[1] - 0.00075, [0, 1, 0])
        self.assertAlmostEqual(offset[1], -0.00125)

    def test_blocked_grasp_is_not_silently_accepted(self):
        with self.assertRaises(RuntimeError):
            choose_offset(lambda o: -0.001, [1, 0, 0])
