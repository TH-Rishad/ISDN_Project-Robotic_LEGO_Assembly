import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Kean.verify_dual import verify_run


class VerifyDualTest(unittest.TestCase):
    def fixture(self, run, support=False):
        manifest = dict(plans={a: a+'.json' for a in 'AB'}, recording='recording.npz', starts=dict(A=0, B=0),
                        step_limit=1, diagnostic=True)
        (run/'manifest.json').write_text(json.dumps(manifest))
        for a in 'AB':
            (run/(a+'.json')).write_text(json.dumps(dict(bricks=[dict(id=a+'_b0')], sequence=[dict(
                step=0, brick_id=a+'_b0', requires_brace=support and a == 'A')])) )
        summary = dict(ok=True, done=True, builds={a: dict(completed_steps=1, supports=[]) for a in 'AB'}, contacts={})
        if support:
            summary['builds']['A']['supports'] = [dict(step=0, realised_grip_N=[10., 10.])]
        (run/'summary.json').write_text(json.dumps(summary))
        events = [dict(event='support_requested', arm='A', step=0, time=0.),
                  dict(event='support_granted', arm='A', step=0, time=.001),
                  dict(event='support_released', arm='A', step=0, time=.05)] if support else []
        (run/'events.json').write_text(json.dumps(events))
        n = 6
        q = np.tile(np.arange(n)[:, None] * .01, (1, 18))
        own = np.zeros((n, 2), int)
        own[-1] = 1
        poses = np.zeros((n, 2, 7))
        poses[..., 6] = 1
        fields = dict(frame=np.arange(1, n+1), sim_time=np.arange(n)/60, q_exec=q, q_cmd=q,
                      brick_q=poses, flag_ab=np.zeros(n, bool), break_frame=np.array([], int),
                      failure=np.array(''), phase_names=np.array(['to feeder', 'insert', 'clearance', 'hold']),
                      a_phase=np.array([0, 1, 1, 2, 2, 2]), b_phase=np.array([0, 1, 1, 2, 2, 2]),
                      own_steps=own, roles=np.full((n, 2), 'builder', dtype='U10'),
                      task_builds=np.tile(['A', 'B'], (n, 1)), brick_ids=np.array(['A_b0', 'B_b0']),
                      fire_frame_by_brick=np.array([2, 2]))
        np.savez(run/'recording.npz', **fields)
        return fields

    def test_valid_independent_recording_and_actual_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            self.fixture(p)
            report=verify_run(p)
            self.assertTrue(report['ok'], report['errors'])
            self.assertGreater(report['recorded_parallel_frames'], 0)
            self.assertFalse(report['full_build'])

    def test_saved_ok_does_not_hide_arm_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            fields=self.fixture(p)
            fields['flag_ab'][2] = True
            np.savez(p/'recording.npz', **fields)
            self.assertIn('recorded arm-arm contact', verify_run(p)['errors'])

    def test_support_requires_recorded_helper_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            self.fixture(p, support=True)
            self.assertTrue(any('helper not holding' in e for e in verify_run(p)['errors']))

    def test_simultaneous_pending_requests_reject_B_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            self.fixture(p)
            events=[dict(event='support_requested', arm=a, step=0, time=0.) for a in 'AB']
            events.append(dict(event='support_granted', arm='B', step=0, time=1.))
            (p/'events.json').write_text(json.dumps(events))
            self.assertTrue(any('priority' in e for e in verify_run(p)['errors']))


if __name__ == '__main__':
    unittest.main()
