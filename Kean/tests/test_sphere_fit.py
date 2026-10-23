import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import unittest

from Kean.motion.sphere_fit import cover_radius, inward_sphere, keep_within, needs_repair, shrunk_radius


class SphereFitTest(unittest.TestCase):
    def test_already_inside_limits_is_not_repaired(self):
        out = {"vertex_max_mm": 1.9, "protrusion_max_mm": 7.2}
        self.assertFalse(needs_repair(out, 2.0, 8.0))

    def test_hand_vertex_miss_needs_repair(self):
        self.assertTrue(needs_repair({"vertex_max_mm": 2.47, "protrusion_max_mm": 7.19}, 2.0, 8.0))

    def test_fat_finger_needs_repair(self):
        self.assertTrue(needs_repair({"vertex_max_mm": 0.44, "protrusion_max_mm": 1.34}, 0.5, 0.5))

    def test_inward_sphere_covers_the_point(self):
        point = np.array([0.1, -0.2, 0.3])
        center, radius = inward_sphere(point, [0.0, 0.0, 2.0], 0.002)
        self.assertAlmostEqual(np.linalg.norm(point - center), radius)
        self.assertLess(center[2], point[2])

    def test_drops_only_spheres_outside_the_protrusion_limit(self):
        self.assertEqual(keep_within([0.0004, 0.00134, 0.0005], 0.0005), [True, False, True])

    def test_shrink_keeps_a_legal_sphere_and_pulls_a_fat_one_in(self):
        self.assertAlmostEqual(shrunk_radius(0.01, 0.004, 0.008), 0.01)
        self.assertLess(shrunk_radius(0.01, 0.00134, 0.0005), 0.01 - 0.0008)

    def test_cover_radius_stays_inside_both_limits(self):
        self.assertAlmostEqual(cover_radius(0.002, 0.008), 0.002)
        self.assertLessEqual(cover_radius(0.0005, 0.0005), 0.0005)
        self.assertGreater(cover_radius(0.0005, 0.0005), 0.0)


if __name__ == "__main__":
    unittest.main()
