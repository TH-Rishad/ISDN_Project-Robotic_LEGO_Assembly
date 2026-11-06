import unittest

from Kean.dual_coordinator import Coordinator


class CoordinatorTest(unittest.TestCase):
    def test_independent_steps_start_together(self):
        c = Coordinator({'A': [False], 'B': [False]})
        self.assertEqual(c.dispatch(dict(A=True, B=True), 0), [('A', None), ('B', None)])

    def test_helper_finishes_current_step_before_grant(self):
        c = Coordinator({'A': [False, True], 'B': [False, False]})
        c.dispatch(dict(A=True, B=True), 0)
        c.complete('A', 1)
        self.assertEqual(c.dispatch(dict(A=True, B=False), 1), [])
        self.assertEqual(c.pending, {'A': 1})
        c.complete('B', 2)
        self.assertEqual(c.dispatch(dict(A=True, B=True), 2), [('A', 'B')])
        self.assertEqual(c.builds['B'].step, 1)

    def test_tie_A_then_retained_B_without_starvation(self):
        c = Coordinator({'A': [True, True], 'B': [True]})
        self.assertEqual(c.dispatch(dict(A=True, B=True), 0), [('A', 'B')])
        self.assertFalse(c.builds['B'].busy)  # B never picks up its pending brick.
        self.assertEqual(c.dispatch(dict(A=False, B=False), 1), [])
        c.complete('A', 2)
        self.assertEqual(c.events[-1]['event'], 'support_released')
        self.assertEqual(c.events[-1]['step'], 0)
        self.assertEqual(c.dispatch(dict(A=True, B=True), 2), [('B', 'A')])
        c.complete('B', 3)
        self.assertEqual(c.dispatch(dict(A=True, B=True), 3), [('A', 'B')])

    def test_completed_arm_remains_available_as_helper(self):
        c = Coordinator({'A': [], 'B': [True]})
        self.assertEqual(c.dispatch(dict(A=True, B=True), 0), [('B', 'A')])
        c.complete('B', 1)
        self.assertTrue(c.done)

    def test_active_support_is_not_preempted(self):
        c = Coordinator({'A': [False, True], 'B': [True]})
        # Pending B reserves A before A can start another ordinary step.
        self.assertEqual(c.dispatch(dict(A=True, B=True), 0), [('B', 'A')])
        self.assertEqual(c.dispatch(dict(A=True, B=True), 1), [])


if __name__ == '__main__':
    unittest.main()
