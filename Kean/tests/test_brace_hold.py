import unittest
from types import SimpleNamespace

import numpy as np

from Kean.motion.brace_hold import install


class Array:
    def __init__(self, data):
        self.data = np.array(data, dtype=float)

    def numpy(self):
        return self.data.copy()

    def assign(self, data):
        self.data[:] = data


class IK:
    def step(self, q_in, q_out, **kwargs):
        q_out.assign(np.full((2, 9), 99.0))


class BraceHoldTest(unittest.TestCase):
    def test_hold_latches_measured_arm_only_and_releases_for_next_brace(self):
        measured = Array(np.arange(18))
        ex = SimpleNamespace(A=SimpleNamespace(phase='close'), ik=IK(),
                             state_0=SimpleNamespace(joint_q=measured), graph_ik=object())
        install(ex)
        wrapper = ex.ik
        install(ex)
        self.assertIs(ex.ik, wrapper)
        self.assertIsNone(ex.graph_ik)
        q = Array(np.zeros((2, 9)))
        ex.ik.step(q, q, iterations=24)
        np.testing.assert_array_equal(q.numpy(), 99)
        ex.A.phase = 'hold'
        ex.ik.step(q, q)
        np.testing.assert_array_equal(q.numpy()[0, :7], np.arange(7))
        np.testing.assert_array_equal(q.numpy()[0, 7:], 99)
        np.testing.assert_array_equal(q.numpy()[1], 99)
        measured.assign(np.arange(18) + 10)
        ex.ik.step(q, q)
        np.testing.assert_array_equal(q.numpy()[0, :7], np.arange(7))
        ex.A.phase = 'open'
        ex.ik.step(q, q)
        np.testing.assert_array_equal(q.numpy(), 99)
        ex.A.phase = 'hold'
        ex.ik.step(q, q)
        np.testing.assert_array_equal(q.numpy()[0, :7], np.arange(7) + 10)


if __name__ == '__main__':
    unittest.main()
