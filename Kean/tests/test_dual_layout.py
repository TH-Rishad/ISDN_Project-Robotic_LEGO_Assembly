"""Isolate Newton imports from the CPU-only verifier regression."""
import subprocess
import sys
import unittest


class LayoutTest(unittest.TestCase):
    def test_layout_frames_ids_and_feeders_in_isolated_interpreter(self):
        result = subprocess.run([sys.executable, '-c', CODE], capture_output=True, text=True)
        if "No module named 'newton'" in result.stderr or "No module named 'warp'" in result.stderr:
            self.skipTest('shared Newton environment required for scene layout checks')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


CODE = """
import numpy as np
from Kean.dual_build import D, plans_for, feeders_for, pad_for
saved = tuple(D.P.VOXEL_ORIGIN)
plans, connections = plans_for(dict(A='arch', B='S3'))
assert tuple(D.P.VOXEL_ORIGIN) == saved
for a, expected in (('A', [10]), ('B', [11,13,14])):
    p = plans[a]
    assert [s['step'] for s in p['sequence'] if s['requires_brace']] == expected
    assert all(b['id'].startswith(a + '_') for b in p['bricks'])
    assert all(m[0].startswith(a + '_') for s in p['sequence'] for m in s['mating_studs'])
assert all(c.upper[0] == c.lower[0] for c in connections if c.lower is not None)
pads = [pad_for(plans[a]) for a in 'AB']
feeders = feeders_for(plans, pads)
assert len(feeders) == 27
boxes = []
for p in pads:
    center, half = np.array(p['c'][:2]), np.array(p['dims'][:2])/2
    boxes.append((center-half, center+half))
for p in feeders.values():
    lo, hi = np.array(p[:2])-.03, np.array(p[:2])+.03
    assert not any(np.all(lo < b) and np.all(a < hi) for a,b in boxes)
    boxes.append((lo,hi))
try:
    plans_for(dict(A='arch',B='S3'),.1)
except ValueError:
    pass
else:
    raise AssertionError('accepted an overlapping layout')

from types import SimpleNamespace
from Kean.dual_build import Build
view = Build.__new__(Build)
view.cell = SimpleNamespace(sim_time=0.)
view.A = SimpleNamespace(phase='hold')
view.close_t = 0.
view.braced = set()
info = dict(executed=True, grip_samples=[])
view.brace_info = lambda n: info
failures = []
view.fail = failures.append
view.confirm_brace(10)
assert not info['executed']
for _ in range(4):
    info['grip_samples'].append([10.,10.])
    view.check_support()
assert 10 not in view.braced
info['grip_samples'].append([10.,10.])
view.check_support()
assert 10 in view.braced and info['executed']
for _ in range(10):
    info['grip_samples'].append([0.,0.])
    view.check_support()
assert failures == ['support_lost']
view.braced.clear()
view.confirm_brace(11)
info['grip_samples'].clear()
view.cell.sim_time = 3.1
view.check_support()
assert failures[-1] == 'support_not_established'

from Kean.dual_build import DualExample
own = SimpleNamespace(active=True, b_step=0, placed=set(), plan=dict(sequence=[dict(brick_id='A_b0')]))
other = SimpleNamespace(active=False)
ex = SimpleNamespace(views=dict(A=own, B=other), arms=dict(A=SimpleNamespace(phase='stage'), B=SimpleNamespace(phase='park')))
assert DualExample.held_objects(ex) == {'A': 'A_b0'}
own.placed.add(0)
assert DualExample.held_objects(ex) == {}

builder = D.Arm('A builder', D.ARM_A, [-.15, 0., .35])
staged = set()
view = Build.__new__(Build)
view.B = builder
view.drop_press = {}
view.owner = 'A'
step = dict(step=0, brick_id='A_b0', requires_brace=True)
view.plan = dict(sequence=[step], voxel_origin=D.P.VOXEL_ORIGIN)
view.cell = SimpleNamespace(feeder={'A_b0': np.zeros(3)}, parks={builder: np.array([-.15, 0., .35])}, stud_fit=None)
view.pickup_clearance = lambda step: np.zeros(3)
def queue(step):
    builder.push('lift', [0., 0., .17], 0., .0065, 1., then=lambda: staged.add(0))
    builder.push('stage', [0., 0., .17], 0., .0065, .2)
    builder.push('transport', [0., 0., .17], 0., .0065, 1.)
    builder.push('pre-insert', [0., 0., .065], 0., .0065, 1.)
    builder.push('insert', [0., 0., .04], 0., .0065, 1., then=lambda: None)
    builder.push('release', [0., 0., .04], 0., .01, .4)
view.queue_place = queue
view.start(0)
assert [m['phase'] for m in builder.moves] == ['lift','transport','stage','transport','pre-insert','insert','insert','insert','release']
assert builder.moves[0]['then'] is None and not staged
builder.moves[1]['then']()
assert staged == {0}
np.testing.assert_allclose(builder.moves[2]['pos'], view.cell.parks[builder])
assert np.isclose(builder.moves[-1]['pos'][2], .042)
np.testing.assert_allclose(builder.moves[5]['pos'], builder.moves[4]['pos'])
assert builder.moves[5]['dur'] == 1.5 and builder.moves[-2]['dur'] == .4
builder.moves.clear()
step['requires_brace'] = False
view.start(0)
assert sum(m['phase'] == 'insert' for m in builder.moves) == 1
assert next(m for m in builder.moves if m['phase'] == 'insert')['dur'] == 1.
assert builder.moves[-1]['phase'] == 'clearance'

# A valid stud contact must not conceal a deep wall contact on the same pair.
from cell.contacts import ContactMonitor
from types import MethodType
class Buffer:
    def __init__(self, value): self.value = np.asarray(value)
    def numpy(self): return self.value
ex = SimpleNamespace(
    plan=dict(sequence=[dict(brick_id='A_top')]), b_step=0,
    A=SimpleNamespace(phase='insert'), B=SimpleNamespace(phase='park'),
    arms=dict(A=SimpleNamespace(phase='insert'), B=SimpleNamespace(phase='park')),
    state_0=SimpleNamespace(body_q=Buffer(np.zeros((30,7)))),
    contact_shape_types=np.array([6,8,7,8]),
    dims={'A_base':(.016,.016), 'A_top':(.016,.016)},
    contact_shape_poses=np.array([[0,0,.01035],[0,0,0],[0,0,.005],[0,0,0]]))
ex.held_objects = lambda: {'A':'A_top'}
ex.classify_contact = lambda *args: 'neighbour_rub'
ex.penetration_kind = MethodType(DualExample.penetration_kind, ex)
ex.solver = SimpleNamespace(
    mjw_data=SimpleNamespace(nacon=Buffer([2]), contact=SimpleNamespace(
        dist=Buffer([-.0012,-.0011]), geom=Buffer([[0,1],[2,3]]), worldid=Buffer([0,0]),
        pos=Buffer([[0.,0.,.0104],[0.,0.,.005]]))),
    mjc_geom_to_newton_shape=Buffer([[0,1,2,3]]), update_contacts=lambda *args: None)
monitor = ContactMonitor.__new__(ContactMonitor)
monitor.n_arm, monitor.finger, monitor.ground_shape = 14, {12,13}, 999
monitor.sb = np.array([28,29,28,29])
monitor.body_brick = {28:'A_base',29:'A_top'}
monitor.supports, monitor.neighbours = {'A_top':['A_base']}, {}
monitor.brace_phases, monitor.scale = (), 1.
monitor.rest, monitor.raw, monitor._active, monitor.frame = {}, [], {}, 0
monitor.contacts = SimpleNamespace(force=Buffer([[0.,0.,1.],[0.,0.,1.]]))
monitor.step(ex)
monitor.finalize()
assert monitor.counts['stud_mating'] == 1 and monitor.counts['penetration'] == 1
"""

if __name__ == '__main__':
    unittest.main()
