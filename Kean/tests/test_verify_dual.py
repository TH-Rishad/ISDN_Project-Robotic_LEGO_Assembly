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

    def test_unproven_stud_contact_label_does_not_hide_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            self.fixture(p)
            summary = json.loads((p/'summary.json').read_text())
            summary['contacts']['stud_mating'] = dict(count=1)
            (p/'summary.json').write_text(json.dumps(summary))
            (p/'contacts.json').write_text(json.dumps([dict(
                **{'class': 'stud_mating'}, a='brick:A_b0#1', b='brick:B_b0#2',
                phase=dict(A='insert', B='insert'), max_pen_mm=-1.2)]))
            self.assertIn('invalid stud-mating contact evidence', verify_run(p)['errors'])

    def test_planned_stud_contact_requires_matching_geometry_and_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            fields = self.fixture(p)
            manifest = json.loads((p/'manifest.json').read_text())
            manifest['starts']['A'] = 1
            manifest['contacts'] = 'contacts.json'
            (p/'manifest.json').write_text(json.dumps(manifest))
            (p/'A.json').write_text(json.dumps(dict(
                bricks=[dict(id='A_base', type='2x2'), dict(id='A_b0', type='2x2')], sequence=[
                    dict(step=0, brick_id='A_base', requires_brace=False),
                    dict(step=1, brick_id='A_b0', requires_brace=False, mating_studs=[['A_base',4]])])))
            fields['own_steps'][:,0] += 1
            fields['brick_ids'] = np.array(['A_base','A_b0','B_b0'])
            fields['brick_q'] = np.zeros((6,3,7))
            fields['brick_q'][...,6] = 1
            fields['fire_frame_by_brick'] = np.array([0,2,2])
            np.savez(p/'recording.npz', **fields)
            summary = json.loads((p/'summary.json').read_text())
            summary['builds']['A']['completed_steps'] = 2
            summary['contacts']['stud_mating'] = dict(count=1)
            (p/'summary.json').write_text(json.dumps(summary))
            event = dict(**{'class':'stud_mating'}, a='brick:A_base#28', b='brick:A_b0#29',
                         phase=dict(A='insert',B='park'), max_pen_mm=-1.2,
                         geometry=dict(upper='A_b0',lower='A_base',stud_type=6,upper_type=8,stud_local_z=.01035))
            (p/'contacts.json').write_text(json.dumps([event]))
            self.assertTrue(verify_run(p)['ok'], verify_run(p)['errors'])
            for field, value in (('stud_type',7), ('stud_local_z',.004)):
                bad = json.loads(json.dumps(event))
                bad['geometry'][field] = value
                (p/'contacts.json').write_text(json.dumps([bad]))
                self.assertIn('invalid stud-mating contact evidence', verify_run(p)['errors'])
            event['max_pen_mm'] = -1.71
            (p/'contacts.json').write_text(json.dumps([event]))
            self.assertIn('invalid stud-mating contact evidence', verify_run(p)['errors'])
            event['max_pen_mm'] = -1.2
            event['geometry'].update(stud_type=8, stud_local_z=0., contact_frame=2,
                                     contact_local=[-.0065,-.004,.0107],
                                     contact_world=[-.0065,-.004,.0107], lower_dims=[.016,.016])
            (p/'contacts.json').write_text(json.dumps([event]))
            self.assertTrue(verify_run(p)['ok'], verify_run(p)['errors'])
            event['geometry']['contact_world'][2] = .005
            event['geometry']['contact_local'][2] = .005
            (p/'contacts.json').write_text(json.dumps([event]))
            self.assertIn('invalid stud-mating contact evidence', verify_run(p)['errors'])


if __name__ == '__main__':
    unittest.main()
