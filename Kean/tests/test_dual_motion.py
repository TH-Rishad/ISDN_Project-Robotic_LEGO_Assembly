from types import SimpleNamespace
import unittest

import numpy as np

from Kean.motion.brace_hold import BraceHoldIK
from Kean.motion.reservations import Reservations, separated
from Kean.motion.world import box_gap


class Probe:
    FINGER_LOCK = .01

    def arm_world_spheres(self, name, q, model):
        out = np.zeros((len(q), 3, 4))
        out[..., 0] = q[:, 0, None] + (1 if name == 'B' else 0)
        out[..., 1] = [0, .02, -.02]
        out[..., 2:] = [.2, .01]
        return out

    def tcp_poses(self, name, q):
        p = np.zeros((len(q), 3))
        p[:, 0] = q[:, 0] + (1 if name == 'B' else 0)
        p[:, 2] = .2
        return p, np.tile(np.eye(3), (len(q), 1, 1))


class ReservationsTest(unittest.TestCase):
    def setUp(self):
        probe = Probe()
        self.checker = SimpleNamespace(probe=probe, slices={'arm': np.array([0]),
                    'fr3_leftfinger': np.array([1]), 'fr3_rightfinger': np.array([2])})
        self.checker.spheres = lambda name, q: probe.arm_world_spheres(name, np.asarray(q[:7])[None], 'full')[0]
        self.r = Reservations(self.checker)

    def request(self, start=0, carried=None):
        q = np.zeros(7)
        q[0] = start
        return SimpleNamespace(q=q, grip=.01, carried=carried)

    def test_disjoint_paths_execute_together(self):
        path = np.zeros((2, 7))
        self.assertTrue(self.r.admit('A', self.request(), path))
        self.assertTrue(self.r.admit('B', self.request(), path))
        self.r.step()
        self.assertEqual(self.r.parallel_frames, 1)

    def test_crossing_sweeps_are_rejected_even_with_distinct_endpoints(self):
        path = np.zeros((2, 7))
        path[1, 0] = 1.2
        self.assertTrue(self.r.admit('A', self.request(), path))
        self.assertFalse(self.r.admit('B', self.request(), np.zeros((2, 7))))
        self.assertEqual(self.r.waits, 1)

    def test_carried_geometry_is_part_of_reservation(self):
        path = np.zeros((2, 7))
        self.assertTrue(self.r.admit('A', self.request(carried=dict(center=[.5, 0, 0], dims=[1.1, .02, .02])), path))
        self.assertFalse(self.r.admit('B', self.request(), path))

    def test_other_arm_motion_must_stay_in_reserved_volume(self):
        path = np.zeros((2, 7))
        self.r.admit('A', self.request(), path)
        self.r.admit('B', self.request(), path)
        self.assertTrue(self.r.allows_other('A', 'B', np.zeros(9)))
        self.assertFalse(self.r.allows_other('A', 'B', np.ones(9)))

    def test_contact_reservation_excludes_other_arm(self):
        a, b = object(), object()
        slots = {a: SimpleNamespace(name='A'), b: SimpleNamespace(name='B')}
        tick = SimpleNamespace(slots=slots, other=lambda arm: b if arm is a else a, _moving=lambda arm: False)
        self.assertTrue(self.r.can_begin(tick, a, False))
        self.assertFalse(self.r.can_begin(tick, b, True))
        self.r.release('A')
        self.assertTrue(self.r.can_begin(tick, b, False))

    def test_clearance_margin_is_retained(self):
        a = (np.array([[0., 0., 0.]]), np.array([[1., 1., 1.]]))
        self.assertFalse(separated(a, (np.array([[1.004, 0., 0.]]), np.array([[2., 1., 1.]]))))

    def test_waiting_freezes_measured_joints_until_reservation_is_granted(self):
        from Kean.tests.test_tick import Ex, Later, wrap, resample_trajectory
        from Kean.motion.tick import Tick
        ex = Ex()
        allowed = [False]
        reservation = SimpleNamespace(can_begin=lambda *a: allowed[0])
        tick = Tick(Later(), strict=True, wrap_angle=wrap, resample_trajectory=resample_trajectory,
                    reservations=reservation).attach(ex)
        ex.B.push('to feeder', [.5, 0, .3], 0., .01, 1.)
        ex.B.update(.02, ex.B.ee)
        slot = tick.slots[ex.B]
        self.assertTrue(slot.paused)
        np.testing.assert_array_equal(slot.q, ex.state_0.joint_q.numpy()[9:16])
        np.testing.assert_array_equal(slot.v, 0)
        allowed[0] = True
        ex.B.update(.02, ex.B.ee)
        self.assertFalse(slot.paused)
        self.assertEqual(slot.mode, 'waiting')

    def test_force_guard_cancels_playback_and_holds_measured_pose(self):
        from Kean.tests.test_tick import Ex, Later, wrap, resample_trajectory
        from Kean.motion.tick import Tick
        from unittest.mock import Mock
        ex = Ex()
        reservation = SimpleNamespace(release=Mock())
        tick = Tick(Later(), strict=True, wrap_angle=wrap, resample_trajectory=resample_trajectory,
                    reservations=reservation).attach(ex)
        slot = tick.slots[ex.B]
        slot.mode = 'playing'
        slot.q = np.ones(7)
        tick.cancel_at_measured(ex.B)
        self.assertEqual(slot.mode, 'idle')
        self.assertTrue(slot.paused)
        np.testing.assert_array_equal(slot.q, ex.state_0.joint_q.numpy()[9:16])
        reservation.release.assert_called_once_with('B')


class HoldBothRolesTest(unittest.TestCase):
    def test_B_can_be_the_measured_joint_supporter(self):
        from Kean.tests.test_brace_hold import Array, IK
        ex = SimpleNamespace(A=SimpleNamespace(phase='insert'), B=SimpleNamespace(phase='hold'),
                             ik=IK(), state_0=SimpleNamespace(joint_q=Array(np.arange(18))))
        wrapper = BraceHoldIK(ex)
        q = Array(np.zeros((2, 9)))
        wrapper.step(q, q)
        np.testing.assert_array_equal(q.numpy()[1, :7], np.arange(9, 16))
        np.testing.assert_array_equal(q.numpy()[0], 99)


class CarriedBoxTest(unittest.TestCase):
    def test_rotated_box_separation_and_overlap(self):
        a = dict(c=[0., 0., 0.], dims=[2., .2, .2])
        b = dict(c=[0., .3, 0.], dims=[2., .2, .2])
        self.assertAlmostEqual(box_gap(a, b), .1)
        b['yaw'] = np.pi / 2
        self.assertLess(box_gap(a, b), 0.)

    def test_carried_box_can_clear_table_when_circumsphere_would_not(self):
        table = dict(c=[0., 0., -.05], dims=[2., 2., .1])
        brick = dict(c=[0., 0., .009], dims=[.048, .016, .011])
        self.assertGreater(box_gap(table, brick), .003)


if __name__ == '__main__':
    unittest.main()
