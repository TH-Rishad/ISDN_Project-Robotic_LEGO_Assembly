"""Test the placer's approach before closing each automatically generated brace."""
from copy import deepcopy
from types import MethodType

import numpy as np

from Kean.motion.plan_curobo import Request, PlannerUnavailable
from Kean.motion.tick import planner_yaw, resolve_wrap


def candidates(ex, n, original, D, limit=8):
    """Reuse the support generator and force model; preserve the required brace."""
    from Kean.shared import add_cell_path
    add_cell_path()
    import bracing
    P = D.P
    if bracing.P is not P:
        raise RuntimeError('brace search and cell must share one planner module')
    if not np.allclose(P.VOXEL_ORIGIN, ex.plan['voxel_origin']):
        raise RuntimeError('brace search planner frame differs from the active build')
    order = P.sequence(P.STRUCTURES[ex.name])
    brick, placed = order[n], order[:n]
    pose = original['brace_pose']
    positions = bracing.candidates(brick, placed)
    # Cover different columns before spending physical trials on nearby heights.
    # The old target-first ordering exhausted all eight trials at y=-36 mm.
    ys = sorted({round(y, 8) for y, _ in positions},
                key=lambda y: (-abs(y - pose[1]), y))
    columns = [sorted([(y, z) for y, z in positions if round(y, 8) == column],
                      key=lambda yz: abs(yz[1] - pose[2])) for column in ys]
    positions = [column[i] for i in range(max(map(len, columns), default=0))
                 for column in columns if i < len(column)]
    yield deepcopy(original)
    seen = {tuple(np.round(pose + original['brace_tilt_rotvec'], 5))}
    count = 1
    # Try 45/60/75 degrees at each column's nearest height before moving
    # to another height, interleaving columns within each tilt round.
    width = max(1, len(columns))
    schedule = [(y, z, lean) for i in range(0, len(positions), width)
                for lean in (45., 60., 75.) for y, z in positions[i:i + width]]
    for y, z, lean in schedule:
        sign = np.sign(bracing.lean_for(brick, y)[0]) or 1
        pick = bracing.brace_at(brick, placed, y, z, sign * lean,
                                grip_N=original['brace_force_N'], mu=original.get('brace_mu', 1.),
                                strategy=original.get('strategy', 'weakest_joint'))
        if not pick.get('feasible'):
            continue
        # Do not trade the support's load capacity for motion clearance.
        if pick['predicted_util_braced'] > original['predicted_util_braced'] + 1e-4:
            continue
        key = tuple(np.round(pick['brace_pose'] + pick['brace_tilt_rotvec'], 5))
        if key in seen:
            continue
        seen.add(key)
        yield pick
        count += 1
        if count >= limit:
            return


def check_approach(ex, planner, world, carried, D):
    """Check staged B -> hover -> pre-insert with A's measured open-hand pose."""
    joints = ex.state_0.joint_q.numpy()
    other = joints[:9].copy()
    # Include both current open fingers and the eventual closed configuration:
    # the open-hand check is conservative for passage outside the hand.
    boxes, _ = world(ex, 'B')
    brick = carried(ex, 'B')
    if brick is None:
        raise PlannerUnavailable('brace search requires the staged carried brick')
    moves = [m for m in ex.B.moves if m['phase'] in ('transport', 'pre-insert')]
    if [m['phase'] for m in moves] != ['transport', 'pre-insert']:
        raise PlannerUnavailable('brace search expected transport and pre-insert queue')
    # A reachable hover is not necessarily connected to the insertion pose.
    # Backtrack across whole two-segment routes, always from measured staging q.
    failures = []
    hovers = [moves[0]['pos']] + moves[0].get('alternate_goals', [])
    for hover in hovers:
        q, ee = joints[9:16].copy(), np.asarray(ex.B.ee).copy()
        selected = []
        try:
            for move, goal in zip(moves, (hover, moves[1]['pos'])):
                request = Request('B', q, ee, np.asarray(goal) + ex.B.bias, [0, 0, 0, 1],
                                  yaw=planner_yaw(ex, ex.B, move['yaw'], resolve_wrap(D)),
                                  boxes=boxes, other_q=other, full_path=True,
                                  grip=move['grip'], carried=brick)
                result = planner.plan(request)
                q = np.asarray(result['joint_pos'][-1]).copy()
                ee = np.asarray(result.get('goal') if result.get('goal') is not None else request.goal)
                from Kean.motion.path_check import validate
                request.other_q = np.r_[other[:7], [D.BRACE_HALF_W - D.BRACE_SQUEEZE]*2]
                try:
                    validate(planner.backend.probe, request,
                             np.vstack([request.q[:7], result['joint_pos']]))
                except ValueError as exc:
                    raise PlannerUnavailable('closed brace: %s' % exc) from exc
                selected.append((move, ee - ex.B.bias))
        except PlannerUnavailable as exc:
            failures.append('hover %s: %s' % (np.round(hover, 4).tolist(), exc))
            continue
        # Commit only after BOTH segments passed. Execution still revalidates.
        for move, goal in selected:
            move['pos'] = goal.copy()
            move.pop('alternate_goals', None)
        return
    raise PlannerUnavailable('; '.join(failures))


def install(ex, planner, world, carried, D):
    original = ex.queue_brace
    ex.brace_search_report = []

    def queue_brace(self, n, brace):
        if self.legacy or not brace.get('feasible', True):
            return original(n, brace)
        choices = iter(candidates(self, n, brace, D))
        initial = deepcopy(brace)

        def attempt():
            try:
                pick = next(choices)
            except StopIteration:
                self.plan['sequence'][n]['brace'] = initial
                self.fail('brace_search_exhausted')
                return
            self.B.waited = 0.  # each bounded physical trial has its own timeout
            self.plan['sequence'][n]['brace'] = pick
            record = dict(step=n, pose=pick['brace_pose'], tilt=pick['brace_tilt_rotvec'], accepted=False)
            self.brace_search_report.append(record)
            print('brace search step %d: pose %s tilt %s' % (n, record['pose'], record['tilt']), flush=True)
            original(n, pick)
            guard = next(m for m in self.A.moves if m['phase'] == 'brace-guard')
            rebase = guard['then']

            def checked():
                rebase()
                try:
                    check_approach(self, planner, world, carried, D)
                except PlannerUnavailable as exc:
                    record['error'] = str(exc)
                    print('brace search rejected step %d: %s' % (n, exc), flush=True)
                    # A has not closed. Discard close/hold, retreat with the
                    # shared controller, and try again only after parking.
                    self.A.moves.clear()
                    p, yaw, grip, tilt = self.A.cmd
                    tool = D.rotate(D.qrot(tilt), np.array([0., 0., -1.]))
                    self.A.push('retract', np.asarray(p) - D.BRACE_BACK * tool,
                                yaw, grip, .8, tilt=tilt)
                    self.A.push('park', self.parks[self.A], 0., 0., 1.5, then=attempt)
                    return
                record['accepted'] = True
                print('brace search accepted step %d before closing' % n, flush=True)

            guard['then'] = checked

        attempt()

    ex.queue_brace = MethodType(queue_brace, ex)
