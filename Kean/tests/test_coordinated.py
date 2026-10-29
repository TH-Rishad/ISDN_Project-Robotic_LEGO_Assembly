import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

from Kean.tests.test_tick import Ex, Later, wrap, resample_trajectory
from Kean.motion.tick import Tick
from Kean.motion.coordinated import install
from Kean.motion.plan_curobo import Planner, Request
from Kean.motion.path_check import dense_joints, validate
from Kean.motion.world import held_brick
from Kean.acceptance import judge


class CoordinationTest(unittest.TestCase):
    def setup_tick(self):
        ex, planner = Ex(), Later()
        ex.fail = Mock()
        tick = Tick(planner, strict=True, wrap_angle=wrap,
                    resample_trajectory=resample_trajectory).attach(ex)
        ex.B.push('transport', [0.3, 0.3, 0.3], 0., .01, 1.)
        ex.B.update(.02, ex.B.ee)
        return ex, planner, tick

    def test_failure_holds_without_fallback(self):
        ex, planner, tick = self.setup_tick()
        planner.fut.set_exception(RuntimeError('blocked by A'))
        ex.B.update(.02, ex.B.ee)
        self.assertEqual(ex.B.t, 0.)
        self.assertEqual(tick.slots[ex.B].mode, 'stopped')
        ex.fail.assert_called_once_with('motion_plan_failed')
        ex.B.update(.02, ex.B.ee)
        self.assertEqual(ex.B.t, 0.)

    def test_endpoint_does_not_execute_cartesian_tail(self):
        ex, planner, tick = self.setup_tick()
        done = Mock()
        ex.B.cur['then'] = done
        self.assertTrue(planner.request.full_path)
        planner.fut.set_result(dict(joint_pos=np.zeros((2, 7)), dt=1/60,
                                   cartesian=np.array([[99, 99, 99]])))
        for _ in range(6):
            ex.B.update(.02, ex.B.ee)
        done.assert_called_once()
        self.assertIsNone(ex.B.cur)
        self.assertTrue(ex.B.idle())

    def test_new_move_is_planned_even_if_python_reuses_an_object_id(self):
        # Force the allocator-reuse case deterministically. Keeping only id(move)
        # skipped the second query and ran it through the shared Cartesian Arm.
        with patch('Kean.motion.tick.id', return_value=123, create=True):
            ex, planner, tick = self.setup_tick()
            planner.fut.set_result(dict(joint_pos=np.zeros((2, 7)), dt=1/60,
                                       cartesian=np.zeros((1, 3))))
            for _ in range(4):
                ex.B.update(.02, ex.B.ee)
            self.assertTrue(ex.B.idle())
            ex.B.push('to feeder', [.5, .1, .3], 0., .01, 1.)
            ex.B.update(.02, ex.B.ee)
            self.assertEqual(planner.calls, 2)
            self.assertEqual(tick.slots[ex.B].mode, 'waiting')
            self.assertEqual(ex.B.t, 0.)

    def test_alternate_goal_is_used_at_handoff(self):
        ex, planner, tick = self.setup_tick()
        goal = np.array([.3, .3, .46])
        planner.fut.set_result(dict(joint_pos=np.zeros((2, 7)), dt=1/60,
                                   cartesian=np.zeros((1, 3)), goal=goal))
        for _ in range(4):
            ex.B.update(.02, ex.B.ee)
        np.testing.assert_allclose(ex.B.cmd[0], goal)

    def test_retracted_signal_waits_for_escape_and_B_waits_for_A(self):
        ex = Ex()
        ex.legacy = False
        ex.parks = {ex.B: np.array([.3, 0., .2])}
        released = Mock()
        ex.queue_place = lambda step: ex.B.push('retract', [0, 0, .2], 0., .02, .8, then=released)
        install(ex)
        ex.queue_place({'requires_brace': True, 'brace': {}})
        retract, escape, wait = ex.B.moves
        self.assertIsNone(retract['then'])
        self.assertEqual(escape['phase'], 'clearance')
        escape['then']()
        released.assert_called_once()
        ex.A.push('park', [0, 0, .2], 0, 0, 1)
        self.assertFalse(wait['wait']())
        ex.A.moves.clear()
        self.assertTrue(wait['wait']())

    def test_braced_transport_hovers_on_the_placer_side(self):
        ex = Ex()
        ex.legacy = False
        ex.A.base = ((-0.45, 0.0, 0.0), 0.0)
        ex.B.base = ((0.45, 0.0, 0.0), 3.14)
        ex.parks = {ex.B: np.array([.3, 0., .2])}
        def original(step):
            ex.B.push('transport', [0.0, 0.0, 0.13], 0., .01, 1.5)
            ex.B.push('retract', [0, 0, .2], 0., .02, .8)
        ex.queue_place = original
        install(ex)
        ex.queue_place({'requires_brace': True, 'brace': {'feasible': True}})
        transport = next(m for m in ex.B.moves if m['phase'] == 'transport')
        self.assertGreater(transport['pos'][0], 0.12)
        self.assertAlmostEqual(transport['pos'][1], 0.0)
        self.assertAlmostEqual(transport['pos'][2], 0.13)

    def test_released_brick_is_obstacle_on_escape(self):
        for phase in ('release', 'retract', 'clearance'):
            self.assertIsNone(held_brick('B', phase, 'brick'))

    def test_full_path_reaches_goal_even_for_short_move(self):
        backend = SimpleNamespace(warmup=lambda guard: None,
                                  plan=Mock(return_value=dict(joint_pos=np.zeros((2, 7)), dt=.02)))
        with Planner(backend) as planner:
            planner.warmup()
            r = Request('B', np.zeros(7), np.zeros(3), np.array([0, 0, .01]),
                        [0, 0, 0, 1], full_path=True)
            planner.plan(r)
        np.testing.assert_allclose(backend.plan.call_args.args[1], r.goal)

    def test_dense_path_checks_between_waypoints(self):
        path = dense_joints(np.array([np.zeros(7), np.ones(7) * .1]))
        self.assertGreater(len(path), 2)
        self.assertLessEqual(np.max(np.abs(np.diff(path, axis=0))), .02000001)
        with self.assertRaises(ValueError):
            dense_joints([[float('nan')] * 7])

    def test_full_model_rejects_collision_between_path_endpoints(self):
        from Kean.tests.test_clearance import Probe
        probe = Probe()
        # Own sphere x follows q[0]; the obstacle occupies the path midpoint.
        probe.clearance_spheres = lambda own, other: abs(own[0, 0] - .05) - .015
        r = Request('B', np.zeros(7), np.zeros(3), np.ones(3), [0, 0, 0, 1],
                    boxes=[], other_q=np.zeros(9), grip=.01, full_path=True)
        with self.assertRaisesRegex(ValueError, 'other arm'):
            validate(probe, r, [np.zeros(7), np.ones(7) * .1])

    def test_carried_brick_is_checked_even_if_robot_clears(self):
        from Kean.tests.test_clearance import Probe
        probe = Probe()
        probe.tcp_poses = lambda name, q: (np.zeros((1, 3)), np.eye(3)[None])
        probe.clearance_spheres = lambda own, other: .02
        probe.clearance_boxes = lambda own, boxes: ((-.01, 'build') if len(own) == 1 else (.02, 'build'))
        r = Request('B', np.zeros(7), np.zeros(3), np.ones(3), [0, 0, 0, 1],
                    boxes=[], other_q=np.zeros(9), grip=.01, full_path=True,
                    carried=dict(center=[0, 0, 0], dims=[.048, .016, .012]))
        with self.assertRaisesRegex(ValueError, 'carried brick'):
            validate(probe, r, [np.zeros(7)])

    def test_command_guard_stops_before_issuing_arm_collision(self):
        ex, planner, tick = self.setup_tick()
        tick.clearance = SimpleNamespace(spheres=lambda name, q: q,
                                         probe=SimpleNamespace(clearance_spheres=lambda a, b: -.003))
        ex.control.joint_target_pos.assign(np.ones(18) * .1)
        ex.fail.assert_called_with('arm_clearance')
        np.testing.assert_allclose(ex.control.joint_target_pos.data, 0)
        np.testing.assert_allclose(ex.control.joint_target_vel.data, 0)

    def test_collision_and_unfinished_park_fail_acceptance(self):
        row = dict(shape='arch', placed=11, total=11, required_supports=[10],
                   supports=[dict(step=10, executed=True)], done=False,
                   events=[])
        row['events'] = [{'class': 'arm_arm'}]
        reasons = judge([row])
        self.assertIn('arm_collision:arch', reasons)
        self.assertIn('unfinished_motion:arch', reasons)


if __name__ == '__main__':
    unittest.main()
