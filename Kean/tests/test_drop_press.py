import unittest

import numpy as np

from Kean.motion.contact_policy import stud_mating
from Kean.motion.grasp_clearance import grip_candidates
from Kean.motion.drop_press import placement_neighbours, splice

GEOM = dict(grasp_dz=.013, press=.001, tip_below_tcp=.0089, brick_h=.0096, stud_h=.0017)


def queue():
    snap, placed = object(), object()
    m = lambda phase, z, grip, then=None: dict(phase=phase, pos=np.array([.1, .2, z]), yaw=.3, grip=grip,
                                              dur=1., wait=None, then=then, contact=False,
                                              tilt=np.zeros(3), rel=None)
    return [m('transport', .2, .004), m('pre-insert', .05, .004), m('insert', .02, .004),
            m('insert', .012, .004, snap), m('release', .014, .008, placed), m('retract', .2, .008),
            m('clearance', .3, .01)], snap, placed


class DropPressTest(unittest.TestCase):
    def test_splice_releases_above_neighbours_then_presses_with_closed_tips(self):
        moves, snap, placed = queue()
        target = np.array([.1, .2, .005])
        splice(moves, target, .0087, grasp_offset=[.002, 0.], **GEOM)
        phases = [m['phase'] for m in moves]
        self.assertEqual(phases, ['transport', 'pre-insert', 'insert', 'insert', 'release',
                                  'press', 'press', 'press', 'press', 'retract', 'clearance'])
        drop, release = moves[3], moves[4]
        # Released at the clear height with the fingers opening, not at the seat.
        self.assertAlmostEqual(drop['pos'][2] - target[2], .013 - .001 + .0087)
        self.assertEqual(release['grip'], .008)
        self.assertIsNone(release['then'])
        rise, close, seat, up = moves[5:9]
        # Fingers close only once the tips clear a brick resting on the lower studs.
        tips = rise['pos'][2] - GEOM['tip_below_tcp'] - target[2]
        self.assertGreater(tips, GEOM['brick_h'] + 2 * GEOM['stud_h'])
        self.assertEqual((close['grip'], seat['grip']), (0., 0.))
        # Closed tips reach 1 mm past the seated stud tops, centred on the brick.
        self.assertAlmostEqual(seat['pos'][2] - GEOM['tip_below_tcp'] - target[2], .0096 + .0017 - .001)
        np.testing.assert_allclose(seat['pos'][:2], [.098, .2])
        self.assertTrue(seat['contact'])
        self.assertIs(seat['then'], snap)
        self.assertIs(up['then'], placed)

    def test_press_phase_stud_engagement_is_intentional(self):
        self.assertTrue(stud_mating('press', 'A_base', ['A_base'], -.0012, 6, 8, .01035))
        self.assertFalse(stud_mating('press', 'A_other', ['A_base'], -.0012, 6, 8, .01035))

    def test_corner_touching_bricks_of_a_filled_layer_are_neighbours(self):
        # Pyramid layer 0: the 1x1 at (4,4) touches the 2x4 at i 2-3, j 0-3 only at a corner.
        foot = lambda t, yaw: tuple(map(int, t.split('x')))[::-1] if yaw else tuple(map(int, t.split('x')))
        bricks = [dict(id='c', type='1x1', yaw_index=0, grid_pos=[4, 4, 0]),
                  dict(id='m', type='2x4', yaw_index=0, grid_pos=[2, 0, 0]),
                  dict(id='far', type='1x1', yaw_index=0, grid_pos=[0, 0, 0]),
                  dict(id='up', type='1x1', yaw_index=0, grid_pos=[4, 4, 1])]
        n = placement_neighbours(bricks, foot)
        self.assertEqual(n['m'], {'c'})
        self.assertEqual(n['c'], {'m'})
        self.assertEqual(n['far'], set())

    def test_grip_fallback_order(self):
        planner = dict(yaw_offset_deg=0, grip_width_m=.0068, tcp_offset_m=[0., 0.])
        out = grip_candidates(planner, (.032, .008))      # a 1x4 along x
        kinds = [k for k, _ in out]
        self.assertEqual(kinds[:2], ['planner', 'other_axis'])
        self.assertIs(out[0][1], planner)
        # The other axis pinches the 32 mm ends, centred.
        self.assertEqual(out[1][1], dict(yaw_offset_deg=90, grip_width_m=.0308, tcp_offset_m=[0., 0.]))
        edges = [g for k, g in out if k == 'edge']
        offsets = [abs(g['tcp_offset_m'][0]) for g in edges]
        self.assertEqual(offsets, sorted(offsets))             # nearest first
        self.assertAlmostEqual(max(offsets), .006)             # 16 - 8.5 - 1 mm: pads stay on the brick
        self.assertTrue(all(g['yaw_offset_deg'] == 0 for g in edges))   # no room along the 8 mm side
        self.assertEqual(kinds[2:], ['edge'] * len(edges))
        # A 1x1 has no room for edge grips: planner, then the other axis only.
        self.assertEqual([k for k, _ in grip_candidates(planner, (.008, .008))], ['planner', 'other_axis'])


if __name__ == '__main__':
    unittest.main()
