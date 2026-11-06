import unittest
from types import SimpleNamespace as NS

import numpy as np

from Kean.motion.stud_fit import StudFit


class Array:
    def __init__(self, a):
        self.a = np.array(a, dtype=object)

    def numpy(self):
        return self.a.copy()

    def assign(self, a):
        self.a = np.array(a, dtype=object)


class FakeWp:
    """Translation-only transforms: enough to check the hold's relpose."""
    vec3 = staticmethod(lambda *v: np.array(v, float))
    quat = staticmethod(lambda *v: None)
    transform = staticmethod(lambda p, q: np.asarray(p, float))
    transform_identity = staticmethod(lambda: np.zeros(3))
    transform_inverse = staticmethod(lambda t: -t)
    transform_multiply = staticmethod(lambda a, b: a + b)


def cell(z, push):
    planned = 'planned'
    step = dict(step=0, brick_id='b', mating_studs=[[None, 2]])     # 2 studs: 17.8 N to push through
    snapped = []
    view = NS(active=True, b_step=0, plan=dict(sequence=[step], bricks=[dict(id='b', type='2x4', yaw_index=0)]), snap=lambda s: snapped.append(s['brick_id']))
    c = NS(views={'A': view}, welds={'b': [[0, None, planned, False]]}, body={'b': 0},
           target={'b': (np.zeros(3), 0., False)}, sim_time=0.,
           model=NS(equality_constraint_relpose=Array([planned]), equality_constraint_enabled=Array([False])),
           solver=NS(notify_model_changed=lambda flags: None))
    c.state_0 = NS(body_q=NS(numpy=lambda: np.array([[0., 0., z[0], 0, 0, 0, 1]])))
    return c, view, step, snapped


class StudFitTest(unittest.TestCase):
    def test_hold_on_stud_tops_until_pushed_through(self):
        z, push = [.010], [0.]
        c, view, step, _ = cell(z, push)
        fit = StudFit(c, self.D(c, push))
        fit.update()                                   # 10 mm above the seat: free
        self.assertFalse(fit.held)
        z[0] = .0018                                   # 1.8 mm: inside the 1.7 + 0.3 mm band
        fit.update()
        self.assertIn('b', fit.held)
        self.assertTrue(c.model.equality_constraint_enabled.numpy()[0])
        self.assertAlmostEqual(float(c.model.equality_constraint_relpose.numpy()[0][2]), .0018)
        push[0] = 17.
        fit.update()
        self.assertIn('b', fit.held)                   # 17 N < 2 x 8.9 N
        push[0] = 18.
        fit.update()
        self.assertNotIn('b', fit.held)
        self.assertFalse(c.model.equality_constraint_enabled.numpy()[0])
        self.assertEqual(c.model.equality_constraint_relpose.numpy()[0], 'planned')
        fit.update()                                   # pushed through once: never held again
        self.assertNotIn('b', fit.held)
        self.assertEqual([e['event'] for e in fit.events], ['held', 'pushed_through'])

    def test_brick_inside_band_within_a_frame_is_lifted_to_the_stud_tops(self):
        z, push = [.0002], [0.]
        c, *_ = cell(z, push)
        fit = StudFit(c, self.D(c, push))
        fit.update()
        self.assertAlmostEqual(float(c.model.equality_constraint_relpose.numpy()[0][2]), .0017)

    def test_first_course_needs_its_whole_footprint_pushed_through(self):
        from Kean.motion.stud_fit import needed
        foot = lambda t, yaw: (2, 4)
        self.assertAlmostEqual(needed(dict(mating_studs=[]), dict(type='2x4', yaw_index=0), foot), 8 * 8.9)
        self.assertAlmostEqual(needed(dict(mating_studs=[['x', 2], ['y', 1]]), dict(type='2x4', yaw_index=0), foot), 3 * 8.9)

    def test_snap_while_held_releases_first_and_still_snaps(self):
        z, push = [.0017], [5.]
        c, view, step, snapped = cell(z, push)
        fit = StudFit(c, self.D(c, push))
        fit.update()
        view.snap(step)
        self.assertEqual(snapped, ['b'])
        self.assertEqual(fit.events[-1]['event'], 'not_pushed_through')
        self.assertFalse(c.model.equality_constraint_enabled.numpy()[0])

    @staticmethod
    def D(c, push):
        return NS(wp=FakeWp, P=NS(footprint=lambda t, yaw: (2, 4)), newton=NS(solvers=NS(SolverNotifyFlags=NS(CONSTRAINT_PROPERTIES=1))),
                  weld_rows=lambda solver: {0: None},
                  brick_wrench=lambda solver, model, eq, row, body: (np.array([0., 0., push[0]]), None))


if __name__ == '__main__':
    unittest.main()
