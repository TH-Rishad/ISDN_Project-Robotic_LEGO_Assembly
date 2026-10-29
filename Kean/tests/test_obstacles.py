import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from Kean.motion.obstacles import other_arm_boxes
from Kean.tests.test_clearance import Probe


class ObstaclesTest(unittest.TestCase):
    def test_strict_backend_uses_measured_obstacles(self):
        from Kean.motion.plan_curobo import CuroboBackend, Request
        probe = Probe()
        probe.tool_pose_w = lambda *args: (np.zeros(3), [1, 0, 0, 0])
        probe.scene_for = lambda arm, boxes: boxes
        probe.js7 = lambda q: q
        probe.goal_from = lambda *args: None
        probe.plan_ok = lambda result: True
        probe.plan_traj = lambda *args: (np.zeros((2, 7)), None, .02)
        probe.guard = lambda: SimpleNamespace(since_mark=lambda: {})
        planner = SimpleNamespace(update_world=Mock(), plan_pose=Mock(return_value=object()))
        backend = CuroboBackend()
        backend.probe = probe
        backend.rig = SimpleNamespace(pl={'B': planner}, set_world=Mock())
        request = Request('B', np.zeros(7), np.zeros(3), np.ones(3), [0, 0, 0, 1],
                          boxes=[], other_q=np.r_[np.zeros(7), .02, .03],
                          full_path=True, grip=.01)
        with patch('Kean.motion.path_check.validate') as validate:
            backend.plan(request, request.goal)
        backend.rig.set_world.assert_not_called()
        boxes = planner.update_world.call_args.args[0]
        np.testing.assert_allclose([b['c'][1] for b in boxes], [.0125, -.0225])
        self.assertEqual(validate.call_count, 2)

    def test_hover_retry_validates_each_candidate_and_detaches(self):
        from Kean.motion.plan_curobo import CuroboBackend, Request
        probe = Probe()
        probe.tool_pose_w = lambda arm, goal, yaw: (goal, [1, 0, 0, 0])
        probe.scene_for = lambda arm, boxes: boxes
        probe.js7 = lambda q: q
        probe.goal_from = lambda pl, pos, quat: pos
        probe.plan_ok = lambda result: result is not None
        probe.plan_traj = lambda *args: (np.zeros((2, 7)), None, .02)
        probe.guard = lambda: SimpleNamespace(since_mark=lambda: {})
        planner = SimpleNamespace(update_world=Mock(),
                                  plan_pose=Mock(side_effect=[None, object(), object()]),
                                  attachment_manager=SimpleNamespace(update=Mock()))
        backend = CuroboBackend()
        backend.probe = probe
        backend.rig = SimpleNamespace(pl={'B': planner}, detach_brick=Mock())
        request = Request('B', np.zeros(7), np.zeros(3), [0, 0, .1], [0, 0, 0, 1],
                          boxes=[], other_q=np.zeros(9), full_path=True, grip=.01,
                          carried={'dims': [.01]*3},
                          alternate_goals=[[0, 0, .18], [0, 0, .26]])
        with patch('Kean.motion.brick_spheres.as_link_spheres', return_value=object()), \
             patch('Kean.motion.path_check.validate', side_effect=[None, ValueError('blocked'), None]) as check:
            result = backend.plan(request, request.goal)
        np.testing.assert_allclose(result['goal'], [0, 0, .26])
        self.assertEqual(check.call_count, 3)
        backend.rig.detach_brick.assert_called_once_with('B')
        copied = Request(**request.snapshot())
        np.testing.assert_allclose(copied.alternate_goals, request.alternate_goals)

    def test_measured_finger_openings(self):
        q = np.r_[np.zeros(7), .02, .03]
        boxes = other_arm_boxes(Probe(), 'A', q)
        np.testing.assert_allclose([b['c'][1] for b in boxes], [.0125, -.0225])
        self.assertEqual(len({b['name'] for b in boxes}), len(boxes))

    def test_split_preserves_spheres_and_leaves_concavity_open(self):
        probe = Probe()
        spheres = np.array([[x, y, 0., .001] for x, y in
                            [(-.1, 0), (-.1, .1), (0, .1), (.1, .1), (.1, 0)]])
        probe.link_slices = lambda model: {'fr3_hand': list(range(5)),
                                          'fr3_leftfinger': [], 'fr3_rightfinger': []}
        probe.arm_world_spheres = lambda *args: spheres[None]
        boxes = other_arm_boxes(probe, 'A', np.r_[np.zeros(7), .0075, .0075])
        for sphere in spheres:
            self.assertTrue(any(np.all(np.abs((sphere[:3] - b['c']) @ b['R']) + sphere[3]
                                       <= b['dims'] / 2 + 1e-12) for b in boxes))
        self.assertFalse(any(np.all(np.abs((-b['c']) @ b['R']) <= b['dims']/2) for b in boxes))


if __name__ == '__main__':
    unittest.main()
