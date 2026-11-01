import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from Kean.motion import newton_fit, wriggle
from Kean.tests.test_tick import Ex
from Kean.tests.test_wriggle import FakeP, arch_plan


class LawsTest(unittest.TestCase):
    def test_seat_gate_is_tighter_than_the_cell(self):
        self.assertTrue(newton_fit.seated(0.0002, 0.0002))
        self.assertFalse(newton_fit.seated(0.0019, 0.0))
        self.assertFalse(newton_fit.seated(0.0, 0.0010))

    def test_descent_is_capped_at_8_mm_s(self):
        self.assertAlmostEqual(newton_fit.cap_descent(0.020, 1.0), 0.008)
        self.assertAlmostEqual(newton_fit.cap_descent(0.001, 0.5), 0.001)

    def test_clutch_holds_insertion_and_breaks_harder_on_extraction(self):
        up, released = newton_fit.clutch_axial(4, 0.0010)
        self.assertAlmostEqual(up, 4 * 8.9)
        self.assertFalse(released)
        hold, still = newton_fit.clutch_axial(4, 0.0, extract_n=4 * 11.3)
        self.assertEqual(hold, 0.0)
        self.assertFalse(still)
        gone, broken = newton_fit.clutch_axial(4, 0.0, extract_n=4 * 11.3 + 0.1)
        self.assertEqual(gone, 0.0)
        self.assertTrue(broken)
        self.assertGreater(11.3, 8.9)

    def test_impedance_and_joint_torque(self):
        wrench = newton_fit.impedance([0.001, 0, 0], np.zeros(3), np.zeros(3), np.zeros(3))
        self.assertAlmostEqual(wrench[0], 1.2)
        self.assertAlmostEqual(newton_fit.impedance(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)).sum(), 0.0)
        J = np.zeros((6, 7))
        J[0, 0] = 2.0
        tau = newton_fit.joint_torque(J, wrench)
        self.assertAlmostEqual(tau[0], 2.4)
        self.assertEqual(tau.shape, (7,))

    def test_search_target_moves_from_the_measured_hand(self):
        measured = np.array([0.0, 0.0, 0.014])
        target = np.array([0.0, 0.0, 0.01])
        goal, tilt, phase, centre, t = newton_fit.closed_target(measured, target, 0.0, 0.05, None, "descend")
        self.assertEqual(phase, "search")
        self.assertLess(measured[2] - goal[2], 0.008 * 0.05 + 1e-12)
        goal, tilt, phase, centre, t = newton_fit.closed_target(goal, target, t, 0.05, centre, phase)
        self.assertGreater(abs(tilt[0]) + abs(tilt[1]), 0.0)
        self.assertGreater(t, 0.05)

    def test_transfer_refuses_until_the_probe_matches(self):
        self.assertTrue(newton_fit.transfer_ok(26, 68, 32, 68))
        self.assertFalse(newton_fit.transfer_ok(26, 68, 40, 68))
        self.assertFalse(newton_fit.transfer_ok(0, 0, 1, 1))
        with self.assertRaises(newton_fit.TransferRefused):
            newton_fit.consider_policy("missing.pt", 26, 68, 40, 68)
        self.assertEqual(newton_fit.consider_policy("missing.pt", 26, 68, 32, 68), "missing.pt")
        self.assertFalse(os.path.exists("missing.pt"))


class InstallTest(unittest.TestCase):
    def cell(self):
        ex = Ex()
        ex.plan = arch_plan()
        ex.b_step = 2

        def original(step):
            ex.B.push("insert", [0.0, 0.0, 0.06], 0.0, 0.01, 1.0, then=lambda: None)

        ex.queue_place = original
        return ex

    def test_default_install_is_a_no_op(self):
        ex = self.cell()
        with mock.patch.dict(os.environ, {"KEAN_NEWTON_FIT": "0"}):
            self.assertIsNone(newton_fit.install(ex, SimpleNamespace(P=FakeP())))
        self.assertFalse(hasattr(ex, "newton_fit"))
        ex.queue_place(ex.plan["sequence"][2])
        self.assertEqual([m["phase"] for m in ex.B.moves], ["insert"])

    def test_flag_collapses_the_waypoint_band(self):
        ex = self.cell()
        D = SimpleNamespace(P=FakeP())
        with mock.patch.dict(os.environ, {"KEAN_WRIGGLE": "1", "KEAN_NEWTON_FIT": "1"}):
            wriggle.install(ex, D)
            fit = newton_fit.install(ex, D)
        ex.queue_place(ex.plan["sequence"][2])
        inserts = [m for m in ex.B.moves if m["phase"] == "insert"]
        self.assertEqual(len(inserts), 2)
        self.assertLess(len(inserts), 1 + wriggle.WIGGLE_WAYPOINTS + 1)
        self.assertTrue(fit.armed)
        self.assertEqual(fit.n_studs, 12)
        np.testing.assert_allclose(inserts[-1]["pos"], [0.0, 0.0, 0.06])
        crown = fit.target.copy()
        ex.B.moves.clear()
        ex.queue_place(ex.plan["sequence"][0])
        self.assertEqual([m["phase"] for m in ex.B.moves], ["insert"])
        np.testing.assert_allclose(fit.target, crown)
        self.assertFalse(fit.armed)

    def test_override_tracks_the_hand_not_a_fixed_wiggle(self):
        ex = self.cell()
        D = SimpleNamespace(P=FakeP())
        with mock.patch.dict(os.environ, {"KEAN_NEWTON_FIT": "1", "KEAN_WRIGGLE": "0"}):
            fit = newton_fit.install(ex, D)
        ex.queue_place(ex.plan["sequence"][2])
        ex.B.phase = "insert"
        ee = np.array([0.001, 0.0, 0.062])
        cmd = (ee.copy(), 0.0, 0.01, np.zeros(3))
        goal, _, _, tilt = fit.override(0.05, ee, cmd)
        self.assertFalse(np.allclose(goal, ee))
        self.assertLessEqual(ee[2] - goal[2], newton_fit.PRESS_V * 0.05 + 1e-12)
        self.assertEqual(tilt.shape, (3,))


if __name__ == "__main__":
    unittest.main()
