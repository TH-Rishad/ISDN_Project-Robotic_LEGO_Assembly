import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from Kean.motion.brace_search import install, check_approach, candidates
from Kean.motion.plan_curobo import PlannerUnavailable
from Kean.tests.test_tick import Ex


class BraceSearchTest(unittest.TestCase):
    def test_generated_candidates_keep_active_frame_and_support_capacity(self):
        from Kean.acceptance import bricks_for, _planner, plate_z
        P = _planner()
        bricks_for('arch', 'shape')
        with patch.object(P, 'VOXEL_ORIGIN', (-.008, -.04, plate_z())):
            plan = P.build_plan('arch', 'weakest_joint', brace_model='grasp_lp',
                                brace_grip_N=14.3, brace_mu=1.)
            step = next(s for s in plan['sequence'] if s['requires_brace'])
            ex = SimpleNamespace(name='arch', plan=plan)
            picks = list(candidates(ex, step['step'], step['brace'], SimpleNamespace(P=P)))
            self.assertEqual(len(picks), 8)
            # Physical retry budget must reach the opposite pier early.
            self.assertGreater(picks[1]['brace_pose'][1], 0.)
            self.assertEqual(len({round(p['brace_pose'][1], 4) for p in picks[:4]}), 4)
            self.assertGreater(len({tuple(p['brace_tilt_rotvec']) for p in picks}), 1)
            for pick in picks:
                self.assertAlmostEqual(pick['brace_pose'][0], 0.)
                self.assertLessEqual(pick['predicted_util_braced'],
                                     step['brace']['predicted_util_braced'] + 1e-4)
                self.assertEqual(pick['brace_force_N'], 14.3)

    def setup_search(self, choices):
        ex = Ex()
        ex.legacy = False
        ex.parks = {ex.A: np.array([.3, 0, .2])}
        ex.plan = {'sequence': [{'brace': choices[0]}]}
        ex.fail = Mock()
        self.closed = Mock()
        self.rebased = Mock()
        def queue(n, brace):
            ex.A.push('brace-guard', [0, 0, .1], 0, .01, 1, then=self.rebased)
            ex.A.push('close', [0, 0, .1], 0, .0065, 1, then=self.closed)
        ex.queue_brace = queue
        D = SimpleNamespace(rotate=lambda q, v: v, qrot=lambda v: v, BRACE_BACK=.05)
        install(ex, None, None, None, D)
        return ex

    def test_rejection_retracts_then_parks_before_next_candidate(self):
        picks = [dict(brace_pose=[0, 0, .02], brace_tilt_rotvec=[x, 0, 0]) for x in (.8, 1.)]
        ex = self.setup_search(picks)
        with patch('Kean.motion.brace_search.candidates', return_value=iter(picks)), \
             patch('Kean.motion.brace_search.check_approach', side_effect=[PlannerUnavailable('blocked'), None]):
            ex.queue_brace(0, picks[0])
            guard = ex.A.moves.pop(0)
            guard['then']()
            self.closed.assert_not_called()
            self.assertEqual([m['phase'] for m in ex.A.moves], ['retract', 'park'])
            park = ex.A.moves[-1]
            ex.A.moves.clear()
            park['then']()
            self.assertEqual(ex.plan['sequence'][0]['brace'], picks[1])
            ex.A.moves[0]['then']()
        self.assertTrue(ex.brace_search_report[-1]['accepted'])
        self.assertEqual([m['phase'] for m in ex.A.moves], ['brace-guard', 'close'])
        ex.fail.assert_not_called()

    def test_exhaustion_fails_without_closing(self):
        pick = dict(brace_pose=[0, 0, .02], brace_tilt_rotvec=[1, 0, 0])
        ex = self.setup_search([pick])
        with patch('Kean.motion.brace_search.candidates', return_value=iter([pick])), \
             patch('Kean.motion.brace_search.check_approach', side_effect=PlannerUnavailable('blocked')):
            ex.queue_brace(0, pick)
            ex.A.moves.pop(0)['then']()
            park = ex.A.moves[-1]
            ex.A.moves.clear()
            park['then']()
        ex.fail.assert_called_once_with('brace_search_exhausted')
        self.closed.assert_not_called()

    def test_preinsert_failure_backtracks_to_another_hover(self):
        ex = Ex()
        ex.B.push('transport', [.1, 0, .2], 0, .01, 1)
        ex.B.moves[0]['alternate_goals'] = [[.2, 0, .3]]
        ex.B.push('pre-insert', [0, 0, .1], 0, .01, 1)
        staging = ex.state_0.joint_q.numpy()[9:16].copy()
        requests = []
        def plan(request):
            requests.append(request)
            if len(requests) == 2:
                raise PlannerUnavailable('first hover is a dead end')
            return dict(joint_pos=np.full((2, 7), .1 * len(requests)), goal=request.goal)
        planner = SimpleNamespace(plan=plan, backend=SimpleNamespace(probe=object()))
        D = SimpleNamespace(BRACE_HALF_W=.008, BRACE_SQUEEZE=.0015)
        with patch('Kean.motion.path_check.validate') as validate:
            check_approach(ex, planner, lambda *a: ([], np.zeros(9)), lambda *a: {}, D)
        self.assertEqual(len(requests), 4)
        np.testing.assert_allclose(requests[2].q, staging)
        np.testing.assert_allclose(ex.B.moves[0]['pos'], [.2, 0, .3])
        np.testing.assert_allclose(ex.B.moves[1]['pos'], [0, 0, .1])
        self.assertEqual(validate.call_count, 3)
        self.assertNotIn('alternate_goals', ex.B.moves[0])

    def test_failed_preinsert_does_not_commit_a_new_hover(self):
        ex = Ex()
        ex.B.push('transport', [.1, 0, .2], 0, .01, 1)
        ex.B.push('pre-insert', [0, 0, .1], 0, .01, 1)
        planner = SimpleNamespace(plan=Mock(side_effect=[dict(joint_pos=np.zeros((2, 7)), goal=[.2, 0, .3]),
                                                          PlannerUnavailable('preinsert blocked')]),
                                  backend=SimpleNamespace(probe=object()))
        D = SimpleNamespace(BRACE_HALF_W=.008, BRACE_SQUEEZE=.0015)
        with patch('Kean.motion.path_check.validate'):
            with self.assertRaises(PlannerUnavailable):
                check_approach(ex, planner, lambda *a: ([], np.zeros(9)), lambda *a: {}, D)
        np.testing.assert_allclose(ex.B.moves[0]['pos'], [.1, 0, .2])
        self.assertEqual(planner.plan.call_count, 2)


if __name__ == '__main__':
    unittest.main()
