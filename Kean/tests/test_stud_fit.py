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


def cell(z, push, tcp=None):
    tcp = [1.] if tcp is None else tcp
    planned = 'planned'
    step = dict(step=0, brick_id='b', mating_studs=[[None, 2]], grasp=dict(tcp_offset_m=[0., 0.]))     # 2 studs: 17.8 N to push through
    snapped = []
    arm = NS(phase='insert', cur=None, t=0., bias=np.array([.0005, .0002, 0.]), moves=[dict(phase='press', pos=np.array([0., 0., .02])),
                                             dict(phase='press', pos=np.zeros(3), dwell=1.5, wait=None)])
    view = NS(active=True, b_step=0, owner='A', B=arm, plan=dict(sequence=[step], bricks=[dict(id='b', type='2x4', yaw_index=0, grid_pos=[0, 0, 0])]), snap=lambda s: snapped.append(s['brick_id']))
    c = NS(views={'A': view}, welds={'b': [[0, None, planned, False]]}, body={'b': 0},
           target={'b': (np.zeros(3), 0., False)}, sim_time=0., n_arm_bodies=0, frame=0,
           model=NS(equality_constraint_relpose=Array([planned]), equality_constraint_enabled=Array([False])),
           solver=NS(notify_model_changed=lambda flags: None))
    c.state_0 = NS(body_q=NS(numpy=lambda: np.array([[.001, 0., z[0], 0, 0, 0, 1], [0., 0., tcp[0], 0, 0, 0, 1]])))
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
        fit.update()
        # Press centred on the held brick (x 1 mm), less the hand's in-hand bias.
        np.testing.assert_allclose(view.B.moves[0]['pos'], [.0005, -.0002, .02])
        dwell = view.B.moves[1]['wait']
        self.assertFalse(dwell())                      # held: keep pressing
        push[0] = 17.
        fit.update()
        self.assertIn('b', fit.held)                   # 17 N < 2 x 8.9 N
        push[0] = 18.
        fit.update()
        self.assertNotIn('b', fit.held)
        # Through: the weld guides it straight down to the seat, keeping its x, y.
        self.assertTrue(c.model.equality_constraint_enabled.numpy()[0])
        np.testing.assert_allclose(c.model.equality_constraint_relpose.numpy()[0], [.001, 0., 0.])
        self.assertFalse(dwell())                      # the slide and spring-back have not been simulated
        c.sim_time += .25
        self.assertTrue(dwell())                       # snap now
        view.snap(step)                                # the gate's clutch welds take over
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

    def test_hold_uses_every_support_weld_and_sums_their_push(self):
        z, push = [.0017], [0.]
        c, view, step, _ = cell(z, push)
        c.welds['b'].append([1, None, 'planned2', False])
        c.model.equality_constraint_relpose = Array(['planned', 'planned2'])
        c.model.equality_constraint_enabled = Array([False, False])
        D = self.D(c, push)
        D.weld_rows = lambda solver: {0: None, 1: None}
        fit = StudFit(c, D)
        fit.update()
        self.assertEqual(list(c.model.equality_constraint_enabled.numpy()), [True, True])
        push[0] = 9.                                   # 9 N on each weld: 18 N >= 2 x 8.9 N
        fit.update()
        self.assertNotIn('b', fit.held)
        view.snap(step)
        self.assertEqual(list(c.model.equality_constraint_relpose.numpy()), ['planned', 'planned2'])
        self.assertEqual(list(c.model.equality_constraint_enabled.numpy()), [False, False])

    def test_press_point_is_the_planners_unless_the_grip_fell_back(self):
        from Kean.motion.stud_fit import press_offset
        foot = lambda t, yaw: tuple(map(int, t.split('x')))[::-1] if yaw else tuple(map(int, t.split('x')))
        corbel = dict(type='2x6', yaw_index=1, grid_pos=[0, 0, 5])          # 6 x 2 cells: i 0-5, j 0-1
        below = [dict(type='2x4', yaw_index=1, grid_pos=[0, 0, 4])]         # i 0-3: 8 of 12 studs
        planned = dict(tcp_offset_m=[-.0105, 0.])                           # the planner's press point
        np.testing.assert_allclose(press_offset(planned, corbel, below + [corbel], foot, .008), [-.0105, 0.])
        edge = dict(tcp_offset_m=[0., .007], grip_fallback='edge')            # where the pads fit, not a press point
        np.testing.assert_allclose(press_offset(edge, corbel, below + [corbel], foot, .008), [-.008, 0.])
        np.testing.assert_allclose(press_offset(edge, dict(corbel, grid_pos=[0, 0, 0]), [], foot, .008), [0., 0.])

    def test_overhang_anywhere_under_the_brick_forces_a_support(self):
        from Kean.motion.stud_fit import overhanging
        foot = {'2x2': (2, 2), '2x6': (2, 6), '2x4': (2, 4)}

        def cells(b):
            nx, ny = foot[b[1]]
            return {(b[2] + a, b[3] + c) for a in range(nx) for c in range(ny)}, b[4]

        def supports(b, placed):
            own, k = cells(b)
            return {p[0]: len(own & cells(p)[0]) for p in placed if cells(p)[1] == k - 1 and own & cells(p)[0]}
        P = NS(cells=cells, supports=supports)
        pier = [('p0', '2x2', 0, 0, 0, 0), ('p1', '2x2', 0, 0, 1, 0)]
        corbel = ('c', '2x6', 0, 0, 2, 0)                  # j 0-5 on a 2-wide pier: overhangs
        on_corbel = ('t', '2x2', 0, 0, 3, 0)               # fully supported, but over the corbel
        lone = [('q0', '2x2', 0, 8, 0, 0)]
        self.assertFalse(overhanging(P, pier[1], pier[:1]))
        self.assertTrue(overhanging(P, corbel, pier))
        self.assertTrue(overhanging(P, on_corbel, pier + [corbel]))
        self.assertFalse(overhanging(P, ('r', '2x2', 0, 8, 1, 0), lone))

    def test_support_nearest_the_press_among_safe_grips(self):
        from Kean.motion.stud_fit import near_press_assign
        util = {-.20: .0, -.18: .1, -.17: .5, -.13: .0}           # grip y -> predicted utilisation
        r0 = NS(s=1., util={('a', 'b'): 1.})
        BR = NS(ST=NS(SAFETY_FACTOR=3.), GRIP_N=14.3,
                predicted=lambda brick, placed, br=None: r0 if br is None else NS(s=util[br]),
                candidates=lambda brick, placed, clearance: [(y, .05) for y in util],
                gripped=lambda placed, y, z: ('a',), _brace_obj=lambda placed, y, z, g, mu: y,
                _record=lambda brick, placed, choice, r0, strategy, why: (choice and choice[0], why))
        P = NS(brick_pose=lambda b: (0., -.16, .05), cells=lambda b: ({(0, 0)}, 0), supports=lambda b, p: {})
        assign = near_press_assign(P, BR, original=None)
        # -0.17 is nearest but unsafe (0.5 x 3 >= 1); -0.18 is the nearest safe grip.
        self.assertEqual(assign(('x',), [('a',)], 'weakest_joint'), (-.18, 'near_press'))
        r0.s = .1                                                   # safe unsupported, no overhang: none
        self.assertIsNone(assign(('x',), [('a',)], 'weakest_joint'))

    def test_support_grips_the_bridge_not_the_nearer_pier(self):
        from Kean.motion.stud_fit import bridge_bricks, near_press_assign
        foot = {'2x2': (2, 2), '2x4': (2, 4)}

        def cells(b):
            nx, ny = foot[b[1]]
            return {(b[2] + a, b[3] + c) for a in range(nx) for c in range(ny)}, b[4]

        def supports(b, placed):
            own, k = cells(b)
            return {p[0]: len(own & cells(p)[0]) for p in placed if cells(p)[1] == k - 1 and own & cells(p)[0]}

        P = NS(cells=cells, supports=supports, brick_pose=lambda b: (0., -.16, .05))
        corbel = ('cor', '2x4', 0, 3, 6, 0)          # j 3-6
        column = ('col', '2x2', 0, 8, 6, 0)          # j 8-9
        bridge = ('br', '2x4', 0, 6, 7, 0)           # j 6-9, nothing under j 7
        crown = ('top', '2x4', 0, 6, 8, 0)
        placed = [corbel, column, bridge]
        self.assertEqual(bridge_bricks(P, crown, placed), {'br'})
        util = {-.20: .0, -.04: .2}                  # pier is nearer the press and safer
        r0 = NS(s=1., util={('col', None): 1.})
        BR = NS(ST=NS(SAFETY_FACTOR=3.), GRIP_N=14.3,
                predicted=lambda brick, placed, br=None: r0 if br is None else NS(s=util[br]),
                candidates=lambda brick, placed, clearance: [(-.20, .05), (-.04, .05)],
                gripped=lambda placed, y, z: ('col',) if y < -.1 else ('br',),
                _brace_obj=lambda placed, y, z, g, mu: y,
                _record=lambda brick, placed, choice, r0, strategy, why: (choice and choice[0], why))
        assign = near_press_assign(P, BR, original=None)
        self.assertEqual(assign(crown, placed, 'weakest_joint'), (-.04, 'near_press_bridge'))

    def test_retry_must_keep_the_upper_bridge_brick(self):
        from Kean.motion.stud_fit import bridge_bricks, corridor_blocked, finger_blocked, required_bridge
        foot = {'2x2': (2, 2), '2x4': (2, 4)}

        def cells(b):
            nx, ny = foot[b[1]]
            return {(b[2] + a, b[3] + c) for a in range(nx) for c in range(ny)}, b[4]

        def supports(b, placed):
            own, k = cells(b)
            return {p[0]: len(own & cells(p)[0]) for p in placed if cells(p)[1] == k - 1 and own & cells(p)[0]}

        P = NS(cells=cells, supports=supports)
        base = ('base', '2x2', 0, 3, 5, 0)          # j 3-4, so the corbel below is itself a bridge
        lower = ('lo', '2x4', 0, 3, 6, 0)           # j 3-6
        pier = ('pi', '2x2', 0, 8, 6, 0)            # j 8-9
        upper = ('up', '2x4', 0, 6, 7, 0)           # j 6-9, nothing under j 7
        crown = ('top', '2x4', 0, 6, 8, 0)
        placed = [base, lower, pier, upper]
        self.assertEqual(bridge_bricks(P, crown, placed), {'lo', 'up'})
        self.assertEqual(required_bridge(P, crown, placed, {'lo', 'up'}), {'up'})
        self.assertFalse({'lo', 'pi'} >= required_bridge(P, crown, placed, {'lo', 'up'}))
        self.assertFalse(corridor_blocked(-0.00007, True))
        self.assertTrue(corridor_blocked(-0.0011, True))
        self.assertTrue(corridor_blocked(-0.00007, False))
        self.assertFalse(finger_blocked(0.00094, True))
        self.assertFalse(finger_blocked(0.0, True))
        self.assertTrue(finger_blocked(-0.0001, True))
        self.assertTrue(finger_blocked(0.00094, False))

    def test_press_stops_at_the_grip_when_the_margin_would_overload_it(self):
        from Kean.motion.stud_fit import press_cap
        self.assertAlmostEqual(press_cap(17.8, None), 1.3 * 17.8)
        self.assertAlmostEqual(press_cap(17.8, {'brace_force_N': 28.6, 'brace_mu': 1.}), 1.3 * 17.8)
        # 10 N a finger: pads hold 20 N, between the insertion force and the 1.3 margin.
        self.assertAlmostEqual(press_cap(17.8, {'brace_force_N': 10., 'brace_mu': 1.}), 20.)
        # Pads hold less than the insertion force: still press that, or the studs never release.
        self.assertAlmostEqual(press_cap(17.8, {'brace_force_N': 5., 'brace_mu': 1.}), 17.8)

    def test_snap_gate_follows_the_bricks_underneath(self):
        z, push = [-.00163], [0.]
        c, view, step, _ = cell(z, push)
        step['mating_studs'] = [['s', 2]]
        c.body['s'] = 2
        c.target['s'] = (np.array([0., 0., .0]), 0., False)
        q = np.zeros((3, 7))
        q[:, 6] = 1
        q[0, 2] = -.00163
        q[2, 2] = -.00160
        c.state_0 = NS(body_q=NS(numpy=lambda: q))
        view.gate = lambda bid: (q[0], 0.1, -1.63, 0.4, 2.9, False, False)
        StudFit(c, self.D(c, push))
        _, lateral, dz, _, _, ok, screen = view.gate('b')
        self.assertAlmostEqual(dz, -0.03, places=2)
        self.assertLess(lateral, 2.0)
        self.assertTrue(ok)
        self.assertTrue(screen)
        q[0, 2] = 0.                               # brick did not follow the sag
        _, _, dz, _, _, ok, _ = view.gate('b')
        self.assertAlmostEqual(dz, 1.6, places=2)
        self.assertFalse(ok)

    def test_first_course_needs_its_whole_footprint_pushed_through(self):
        from Kean.motion.stud_fit import needed
        foot = lambda t, yaw: (2, 4)
        self.assertAlmostEqual(needed(dict(mating_studs=[]), dict(type='2x4', yaw_index=0), foot), 8 * 8.9)
        self.assertAlmostEqual(needed(dict(mating_studs=[['x', 2], ['y', 1]]), dict(type='2x4', yaw_index=0), foot), 3 * 8.9)

    def test_closed_tips_on_the_held_brick_ramp_the_press_then_stop_when_through(self):
        z, push, tcp = [.0017], [0.], [.05]
        c, view, step, _ = cell(z, push, tcp)
        c.press_w = Array([[0.] * 6] * 2)
        fit = StudFit(c, self.D(c, push))
        fit.update()                                   # held; the hand is still high
        view.B.phase, view.B.cur = 'press', dict(phase='press', contact=True, pos=np.zeros(3))
        fit.update()
        self.assertEqual(float(c.press_w.numpy()[0][2]), 0.)    # pressing, but the tips are still 5 cm up
        tcp[0] = .0017 + .0096 + .0017 + .0089        # tips on the held brick's stud tops
        view.B.cur = dict(phase='press', contact=False, pos=np.zeros(3))
        fit.update()
        self.assertEqual(float(c.press_w.numpy()[0][2]), 0.)    # a hover or close move never presses
        view.B.cur = dict(phase='press', contact=True, pos=np.zeros(3))
        fit.update()                                   # t = 0: the ramp starts
        c.sim_time = .25
        fit.update()
        self.assertAlmostEqual(float(c.press_w.numpy()[0][2]), -.5 * 1.3 * 17.8, places=4)
        c.sim_time = 1.
        fit.update()
        self.assertAlmostEqual(float(c.press_w.numpy()[0][2]), -1.3 * 17.8, places=4)
        push[0] = 20.                                  # through: no press force left on the hand
        fit.update()
        self.assertNotIn('b', fit.held)
        self.assertEqual(float(c.press_w.numpy()[0][2]), 0.)

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
        return NS(wp=FakeWp, P=NS(footprint=lambda t, yaw: (2, 4), BRICK_H=.0096, PITCH=.008), ex=NS(STUD_HEIGHT=.0017),
                  EE=1, TIP_BELOW_TCP=.0089, newton=NS(solvers=NS(SolverNotifyFlags=NS(CONSTRAINT_PROPERTIES=1))),
                  weld_rows=lambda solver: {0: None},
                  brick_wrench=lambda solver, model, eq, row, body: (np.array([0., 0., push[0]]), None))


if __name__ == '__main__':
    unittest.main()
