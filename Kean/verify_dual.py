"""CPU verification of dual-builder plans, recordings, and support arbitration."""
import argparse
import json
from pathlib import Path

import numpy as np


def mesh_stud_evidence(geometry, brick, record, ids):
    """Recompute the mesh feature location from recorded body pose and contact."""
    try:
        nx, ny = map(int, brick['type'].split('x'))
        dims = np.array([max(nx, ny), min(nx, ny)]) * .008
        frame = int(geometry['contact_frame']) - 1
        if not 0 <= frame < len(record['frame']):
            return False
        pose = record['brick_q'][frame, ids.index(geometry['lower'])]
        q = np.asarray(pose[3:], float)
        q /= np.linalg.norm(q)
        x, y, z, w = q
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        local = rotation.T @ (np.asarray(geometry['contact_world']) - pose[:3])
        if not np.allclose(local, geometry['contact_local'], atol=1e-6, rtol=0):
            return False
        if not np.allclose(dims, geometry['lower_dims'], atol=1e-6, rtol=0):
            return False
        centres = np.array([((i+.5)*.008-dims[0]/2, (j+.5)*.008-dims[1]/2)
                            for i in range(round(dims[0]/.008)) for j in range(round(dims[1]/.008))])
        return bool(.0098 <= local[2] <= .0115 and np.linalg.norm(centres-local[:2], axis=1).min() <= .0027)
    except (KeyError, ValueError, TypeError, IndexError):
        return False


def verify_run(run):
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    events = json.loads((run / 'events.json').read_text())
    plans = {a: json.loads((run / manifest['plans'][a]).read_text()) for a in 'AB'}
    errors = []
    with np.load(run / manifest['recording']) as z:
        record = {k: z[k] for k in z.files}
    n = len(record['frame'])
    if not n or not np.array_equal(record['frame'], np.arange(1, n + 1)):
        errors.append('invalid frame sequence')
    for field, shape in (('q_exec', (n, 18)), ('q_cmd', (n, 18)), ('own_steps', (n, 2)),
                         ('roles', (n, 2)), ('task_builds', (n, 2))):
        if record[field].shape != shape:
            errors.append('invalid ' + field + ' shape')
    if errors:
        return dict(ok=False, errors=errors)
    if not np.isfinite(record['q_exec']).all() or not np.isfinite(record['brick_q']).all():
        errors.append('non-finite recorded pose')
    if record['flag_ab'].any():
        errors.append('recorded arm-arm contact')
    if len(record['break_frame']):
        errors.append('recorded broken joint')
    if str(record['failure']):
        errors.append('recorded execution failure: ' + str(record['failure']))
    phases = [record['phase_names'][record[k]] for k in ('a_phase', 'b_phase')]
    ids = list(map(str, record['brick_ids']))
    expected_ids = [s['brick_id'] for a in 'AB' for s in plans[a]['sequence']]
    if ids != expected_ids:
        errors.append('brick IDs/order differ from saved plans')
    starts = manifest.get('starts', dict(A=0, B=0))
    for i, a in enumerate('AB'):
        steps = plans[a]['sequence']
        stop = min(len(steps), starts[a] + manifest['step_limit']) if manifest.get('step_limit') else len(steps)
        if np.any(np.diff(record['own_steps'][:, i]) < 0) or summary['builds'][a]['completed_steps'] != stop:
            errors.append(a + ': incomplete/nonmonotonic own-build progress')
        for step in steps[starts[a]:stop]:
            sn = step['step']
            if step['brick_id'] not in ids or record['fire_frame_by_brick'][ids.index(step['brick_id'])] < 0:
                errors.append('%s step %d: no recorded mating' % (a, sn))
            insertion = (record['own_steps'][:, i] == sn) & (phases[i] == 'insert') & (record['roles'][:, i] == 'builder')
            if not insertion.any():
                errors.append('%s step %d: no insertion frames' % (a, sn))
            if not step['requires_brace']:
                continue
            helper = 1 - i
            supported = ((phases[helper] == 'hold') & (record['roles'][:, helper] == 'supporter') &
                         (record['task_builds'][:, helper] == a))
            if not np.all(supported[insertion]):
                errors.append('%s step %d: helper not holding throughout insertion' % (a, sn))
            hold_q = record['q_exec'][insertion, 9*helper:9*helper+7]
            if len(hold_q) and np.max(np.ptp(hold_q, axis=0)) > .02:
                errors.append('%s step %d: helper moved during brace hold' % (a, sn))
            brace = next((s for s in summary['builds'][a]['supports'] if s['step'] == sn), None)
            force = None if brace is None else brace.get('realised_grip_N')
            if force is None or min(force) <= .5:
                errors.append('%s step %d: no measured two-finger support force' % (a, sn))
            requested = [e for e in events if e['event'] == 'support_requested' and e['arm'] == a and e['step'] == sn]
            pickup = (record['own_steps'][:, i] == sn) & (phases[i] == 'to feeder') & (record['roles'][:, i] == 'builder')
            if not requested or not pickup.any() or requested[0]['time'] > record['sim_time'][pickup][0] + 1e-6:
                errors.append('%s step %d: support was not requested before pickup' % (a, sn))
    pending, active, next_request = {}, None, None
    grants = []
    for e in events:
        kind, arm = e['event'], e['arm']
        if kind == 'support_requested':
            pending[arm] = e['step']
        elif kind == 'support_granted':
            expected = next_request if next_request in pending else min(pending, default='')
            if active is not None or arm != expected or pending.get(arm) != e['step']:
                errors.append('support grant violates A priority or retained-request order')
            active = arm
            pending.pop(arm, None)
            next_request = None
            grants.append(dict(arm=arm, step=e['step'], time=e['time']))
        elif kind == 'support_released':
            if arm != active:
                errors.append('support released out of order')
            active = None
            helper = 'B' if arm == 'A' else 'A'
            next_request = helper if helper in pending else None
    if pending or active is not None:
        errors.append('unserved support request')
    measured = np.abs(np.diff(record['q_exec'], axis=0))
    commanded = np.abs(np.diff(record['q_cmd'], axis=0))
    moving = [((measured[:, 9*i:9*i+7].max(axis=1) > .0001) &
               (commanded[:, 9*i:9*i+7].max(axis=1) > .0001)) for i in (0, 1)] if n > 1 else [np.zeros(0, bool)] * 2
    parallel = int((moving[0] & moving[1] & np.all(record['roles'][1:] == 'builder', axis=1)).sum())
    if not any(starts.values()) and not parallel:
        errors.append('no recorded simultaneous building motion')
    for cls in ('arm_arm', 'unintended', 'penetration', 'displaced'):
        if summary.get('contacts', {}).get(cls, {}).get('count', 0):
            errors.append('contact monitor: ' + cls)
    stud_count = summary.get('contacts', {}).get('stud_mating', {}).get('count', 0)
    if manifest.get('contacts') or stud_count:
        contact_file = run / manifest.get('contacts', 'contacts.json')
        if not contact_file.is_file():
            errors.append('missing detailed contact evidence')
        else:
            contacts = json.loads(contact_file.read_text())
            studs = [e for e in contacts if e['class'] == 'stud_mating']
            if len(studs) != stud_count:
                errors.append('stud contact count differs from detailed evidence')
            steps_by_id = {s['brick_id']: s for p in plans.values() for s in p['sequence']}
            bricks_by_id = {b['id']: b for p in plans.values() for b in p['bricks']}
            for event in contacts:
                if event['class'] in ('arm_arm', 'unintended', 'penetration', 'displaced'):
                    errors.append('detailed contact evidence: ' + event['class'])
            for event in studs:
                geometry = event.get('geometry', {})
                upper, lower = geometry.get('upper'), geometry.get('lower')
                step = steps_by_id.get(upper, {})
                owner = upper.split('_', 1)[0] if upper else ''
                pair = {event.get(k, '').split('#', 1)[0] for k in ('a', 'b')}
                feature_valid = ((geometry.get('stud_type') == 6 and
                                  abs(geometry.get('stud_local_z', 0.) - .01035) < 1e-6)
                                 or (geometry.get('stud_type') == 8 and mesh_stud_evidence(
                                     geometry, bricks_by_id.get(lower, {}), record, ids)))
                valid = (owner in ('A', 'B') and lower in steps_by_id
                         and lower.startswith(owner + '_')
                         and lower in [m[0] for m in step.get('mating_studs', [])]
                         and pair == {'brick:' + upper, 'brick:' + lower}
                         and event.get('phase', {}).get(owner) in ('insert', 'release', 'press')
                         and -1.7 <= event.get('max_pen_mm', -999.) < 0.
                         and geometry.get('upper_type') in (7, 8) and feature_valid)
                if not valid:
                    errors.append('invalid stud-mating contact evidence')
    if not summary.get('done') or summary.get('failure') or summary.get('error'):
        errors.append('execution did not finish cleanly')
    return dict(ok=not errors, full_build=not errors and not manifest.get('diagnostic'),
                errors=errors, recorded_parallel_frames=parallel, grants=grants, frames=n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = verify_run(args.run)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
