import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np

from Kean.motion.plan_curobo import CuroboBackend, PlannerUnavailable, Request
from Kean.scripts.replay_motion import replay_guard
from Kean.tests.test_clearance import Probe


class WorkaroundTest(unittest.TestCase):
    def test_replay_paths_do_not_depend_on_launcher_working_directory(self):
        from Kean.scripts.replay_motion import ROOT, repository_path
        with patch('pathlib.Path.cwd', return_value=ROOT / 'Sakib/brickassembly'):
            self.assertEqual(repository_path(Path('Kean/results/acceptance/summary.json')),
                             ROOT / 'Kean/results/acceptance/summary.json')
        absolute = Path('/private/tmp/saved-summary.json')
        self.assertEqual(repository_path(absolute), absolute.resolve())

    def backend(self):
        probe = Probe()
        probe.tool_pose_w = lambda arm, goal, yaw: (goal, [1, 0, 0, 0])
        probe.scene_for = lambda arm, boxes: boxes
        probe.js7 = lambda q: q
        probe.goal_from = lambda pl, pos, quat: pos
        probe.plan_ok = lambda result: result is not None
        probe.plan_traj = lambda *args: (np.zeros((2, 7)), None, .02)
        probe.guard = lambda: SimpleNamespace(since_mark=lambda: {})
        planner = SimpleNamespace(update_world=Mock(), plan_pose=Mock())
        backend = CuroboBackend()
        backend.probe = probe
        backend.rig = SimpleNamespace(pl={'B': planner})
        world = [dict(name='table', c=[0, 0, -1], dims=[1, 1, .1])]
        request = Request('B', np.zeros(7), np.zeros(3), [0, 0, .1], [0, 0, 0, 1],
                          boxes=world, other_q=np.zeros(9), full_path=True, grip=.01)
        return backend, planner, request

    def test_rejected_sphere_is_fed_back_to_planner(self):
        from Kean.motion.path_check import PathCollision
        backend, planner, request = self.backend()
        planner.plan_pose.side_effect = [None]*3 + [object()]*2
        sphere = np.array([.1, .2, .3, .01])
        with patch('Kean.motion.path_check.validate',
                   side_effect=[None, PathCollision('near A', sphere), None]) as check:
            result = backend.plan(request, request.goal)
        self.assertEqual(result['proposal_mode'], 'full-sphere-filter')
        self.assertEqual(check.call_count, 3)
        self.assertIs(check.call_args.args[1], request)
        scenes = [c.args[0] for c in planner.update_world.call_args_list]
        self.assertEqual(scenes[1], request.boxes)
        self.assertGreater(len(scenes[0]), len(request.boxes))
        cut = scenes[2][-1]
        np.testing.assert_allclose(cut['c'], sphere[:3])
        np.testing.assert_allclose(cut['dims'], [.023]*3)
        self.assertIs(scenes[-1], scenes[0])

    def test_feedback_search_is_bounded_when_collision_persists(self):
        from Kean.motion.path_check import PathCollision
        backend, planner, request = self.backend()
        planner.plan_pose.side_effect = [None]*3 + [object()]*11
        violation = PathCollision('still blocked', np.array([0, 0, .1, .01]))
        with patch('Kean.motion.path_check.validate', side_effect=[None] + [violation]*11):
            with self.assertRaisesRegex(PlannerUnavailable, 'still blocked'):
                backend.plan(request, request.goal)
        self.assertEqual(planner.plan_pose.call_count, 14)
        scenes = [c.args[0] for c in planner.update_world.call_args_list]
        self.assertEqual(len(scenes[-2]), len(request.boxes) + 8)
        self.assertIs(scenes[-1], scenes[0])

    def test_invalid_measured_start_does_not_try_a_workaround(self):
        backend, planner, request = self.backend()
        with patch('Kean.motion.path_check.validate', side_effect=ValueError('collision')):
            with self.assertRaisesRegex(PlannerUnavailable, 'measured start'):
                backend.plan(request, request.goal)
        planner.plan_pose.assert_not_called()

    def test_guard_replay_checks_intermediate_commands_repeatedly(self):
        probe = Probe()
        probe.clearance_spheres = Mock(return_value=.02)
        failure = dict(measured_q=[0.]*18, target_q=[.1]*18)
        self.assertEqual(replay_guard(probe, failure), [.02]*3)
        self.assertEqual(probe.clearance_spheres.call_count, 18)
        failure['target_q'][0] = float('nan')
        with self.assertRaises(ValueError):
            replay_guard(probe, failure)


if __name__ == '__main__':
    unittest.main()
