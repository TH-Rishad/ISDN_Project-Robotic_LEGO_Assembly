import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Kean.motion.clearance import Clearance


class Probe:
    FINGER_LOCK = 0.0075
    def link_slices(self, model):
        return {'fr3_leftfinger': [0], 'fr3_rightfinger': [1]}
    def arm_world_spheres(self, name, q, model):
        return np.array([[[q[0, 0], 0, 0, .001], [q[0, 0], 0, 0, .001]]])
    def tcp_poses(self, name, q):
        return None, np.eye(3)[None]
    def clearance_boxes(self, own, boxes):
        self.own = own
        return .004, 'table'
    def clearance_spheres(self, own, other):
        return -.002


class ClearanceTest(unittest.TestCase):
    def test_measured_joints_fingers_and_other_arm_collision(self):
        probe = Probe()
        monitor = Clearance(probe, lambda ex, name: ([], None))
        q = np.zeros(18)
        q[9], q[16], q[17] = .25, .02, .03
        ex = SimpleNamespace(state_0=SimpleNamespace(joint_q=SimpleNamespace(numpy=lambda: q)))
        self.assertEqual(monitor(ex, 'B', np.ones(7)*99), -.002)
        np.testing.assert_allclose(probe.own[:, 0], .25)
        np.testing.assert_allclose(probe.own[:, 1], [.0125, -.0225])
