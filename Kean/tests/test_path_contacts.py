"""Support exemptions apply only to selected finger/object pairs."""
from types import SimpleNamespace
import unittest

import numpy as np

from Kean.motion.path_check import PathCollision, validate


class Probe:
    FINGER_LOCK = .0075

    def link_slices(self, model):
        return {'fr3_leftfinger': [1], 'fr3_rightfinger': [2]}

    def arm_world_spheres(self, name, q, model):
        offset = 10 if name == 'B' else 0
        return np.array([[[offset, 0, 0, .001], [offset + 1, 0, 0, .001],
                          [offset + 2, 0, 0, .001]]])

    def tcp_poses(self, name, q):
        return None, np.eye(3)[None]

    def clearance_boxes(self, spheres, boxes):
        for box in boxes:
            if any(abs(s[0] - box['x']) < .01 for s in spheres):
                return -.001, box['name']
        return 1., 'none'

    def clearance_spheres(self, own, other):
        return 1.


class PathContactTest(unittest.TestCase):
    def request(self, x, name='selected'):
        return SimpleNamespace(arm='A', other_q=np.zeros(9), grip=.0075,
                               boxes=[dict(name=name, x=x)], carried=None,
                               allowed_finger_contacts=('selected',), clearance=.001)

    def test_selected_finger_contact_is_allowed(self):
        validate(Probe(), self.request(1), np.zeros((1, 7)))

    def test_unselected_brick_still_blocks_fingers(self):
        with self.assertRaisesRegex(PathCollision, 'neighbour'):
            validate(Probe(), self.request(1, 'neighbour'), np.zeros((1, 7)))

    def test_selected_brick_still_blocks_wrist(self):
        with self.assertRaisesRegex(PathCollision, 'selected'):
            validate(Probe(), self.request(0), np.zeros((1, 7)))

    def test_extra_preflight_clearance_is_enforced(self):
        probe = Probe()
        probe.clearance_boxes = lambda *args: (.0011, 'neighbour')
        request = self.request(5)
        validate(probe, request, np.zeros((1, 7)))
        request.clearance = .0012
        with self.assertRaisesRegex(PathCollision, 'neighbour'):
            validate(probe, request, np.zeros((1, 7)))


if __name__ == '__main__':
    unittest.main()
