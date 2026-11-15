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
    lateral = getattr(ex, 'brace_lateral_offsets', ())
    if lateral:
        positions = [(y + dy, z) for y, z in positions for dy in (0., *lateral)
                     if bracing.feasible(brick, placed, y + dy, z)]
    offsets = getattr(ex, 'brace_height_offsets', ())
    if offsets:
        positions = [(y, z + dz) for y, z in positions for dz in (0., *offsets)
                     if bracing.feasible(brick, placed, y, z + dz)]
    # Cover different columns before spending physical trials on nearby heights.
    # The old target-first ordering exhausted all eight trials at y=-36 mm.
    ys = sorted({round(y, 8) for y, _ in positions},
                key=lambda y: (-abs(y - pose[1]), y))
    if getattr(ex, 'brace_near_first', False):
        # --stud-fit: the planned column first (its other leans and heights), then
        # far-first as before. Far-first sent S3 step 15's blocked bridge grip to the
        # pier, 66 mm away, and the bridge joint broke (dual_studfit_diag_021); all
        # columns nearest-first gave step 14 a grip above its brick that slipped (_023).
        ys.sort(key=lambda y: abs(y - pose[1]) > .0025)
    height_key = (lambda yz: -yz[1]) if getattr(ex, 'brace_high_first', False) else (lambda yz: abs(yz[1] - pose[2]))
    columns = [sorted([(y, z) for y, z in positions if round(y, 8) == column],
                      key=height_key) for column in ys]
    positions = [column[i] for i in range(max(map(len, columns), default=0))
                 for column in columns if i < len(column)]
    keep_bridge = set()
    if getattr(ex, 'brace_near_first', False):
        from Kean.motion.stud_fit import required_bridge
        keep_bridge = required_bridge(P, brick, placed, original.get('gripped_bricks'))
    yield deepcopy(original)
    seen = {tuple(np.round(pose + original['brace_tilt_rotvec'], 5))}
    count = 1
    # Try leans at each column's nearest height before moving to another
    # height, interleaving columns within each tilt round.
    # Opposite lean / early yaw-flip of the original burned the budget and
    # pushed step 14 off the planned grip (dual_studfit_diag_038/039).
    # Reordering to mild-first for the crown broke step 13 (diag_048/049).
    width = max(1, len(columns))
    schedule = [(y, z, lean) for i in range(0, len(positions), width)
                for lean in getattr(ex, 'brace_leans', (45., 60., 75.)) for y, z in positions[i:i + width]]
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
        # --stud-fit: a retry must still hold the upper bridge brick. Keeping only
        # a lower brick on the same path sent S3 step 15 to the pier (_diag_027).
        if keep_bridge and not keep_bridge <= set(pick.get('gripped_bricks') or []):
            continue
        key = tuple(np.round(pick['brace_pose'] + pick['brace_tilt_rotvec'], 5))
        if key in seen:
            continue
        seen.add(key)
        yield pick
        count += 1
        if count >= limit:
            return
        if getattr(ex, 'brace_yaw_alternatives', False):
            # Parallel jaws can grip along the same axis with the wrist
            # turned 180 degrees. The nearer wrist choice can be unreachable
            # for a strongly tilted helper, so test the other orientation too.
            alternate = deepcopy(pick)
            ax = pick.get('brace_axis') or [1., 0., 0.]
            yaw = np.arctan2(ax[0], ax[1])
            base = ex.A.base[1]
            alternate['helper_yaw'] = float(base + resolve_wrap(D)(yaw - base, np.pi) + np.pi)
            yield alternate
            count += 1
            if count >= limit:
                return


def check_approach(ex, planner, world, carried, D):
    """Check staged B -> hover -> pre-insert with A's measured open-hand pose."""
    joints = ex.state_0.joint_q.numpy()
    builder = ex.B.name[0]
    bi = 0 if builder == 'A' else 9
    hi = 9 - bi
    other = joints[hi:hi+9].copy()
    # Include both current open fingers and the eventual closed configuration:
    # the open-hand check is conservative for passage outside the hand.
    boxes, _ = world(ex, builder)
    brick = carried(ex, builder)
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
        q, ee = joints[bi:bi+7].copy(), np.asarray(ex.B.ee).copy()
        selected = []
        try:
            for move, goal in zip(moves, (hover, moves[1]['pos'])):
                request = Request(builder, q, ee, np.asarray(goal) + ex.B.bias, [0, 0, 0, 1],
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


def _retreat_helper(ex, D, then, *, open_first):
    """Park the helper and continue the candidate loop. B stays at stage."""
    # Drop a waiting hold/close (studfit_011: cur stayed on hold's retracted-wait,
    # so the queued open never started and B hit wait_timeout:stage).
    ex.A.moves.clear()
    ex.A.cur = None
    ex.A.waited = 0.
    # Each physical retry is a new trial; B has already been staged through the
    # failed close/settle and must not keep its prior wait clock.
    ex.B.waited = 0.
    p, yaw, grip, tilt = ex.A.cmd
    tool = D.rotate(D.qrot(tilt), np.array([0., 0., -1.]))
    back = np.asarray(p) - D.BRACE_BACK * tool
    if open_first:
        g_open = D.BRACE_HALF_W + D.BRACE_CLEAR
        ex.A.push('open', np.asarray(p), yaw, g_open, .4, tilt=tilt, contact=True)
        grip = g_open
    ex.A.push('retract', back, yaw, grip, .8 if not open_first else .6, tilt=tilt)
    ex.A.push('park', ex.parks[ex.A], 0., 0., 1.5, then=then)


def install(ex, planner, world, carried, D):
    original = ex.queue_brace
    ex.brace_search_report = []
    ex.before_braced = None
    ex.on_support_failed = None

    def queue_brace(self, n, brace):
        if self.legacy or not brace.get('feasible', True):
            return original(n, brace)
        choices = iter(candidates(self, n, brace, D, limit=getattr(self, 'brace_candidate_limit', 8)))
        initial = deepcopy(brace)
        # Grip failures burn the TCP pose: a 180° yaw flip hits the same brick
        # face (studfit_014 step 13 → support_lost on the yaw twin).
        skip_poses = set()

        def pose_key(pick):
            return tuple(np.round(pick['brace_pose'], 5))

        def attempt():
            self.before_braced = None
            self.on_support_failed = None
            while True:
                try:
                    pick = next(choices)
                except StopIteration:
                    self.plan['sequence'][n]['brace'] = initial
                    self.fail('brace_search_exhausted')
                    return
                if pose_key(pick) in skip_poses:
                    print('brace search skip same pose step %d: %s' % (n, pick['brace_pose']), flush=True)
                    continue
                break
            self.B.waited = 0.  # each bounded physical trial has its own timeout
            self.plan['sequence'][n]['brace'] = pick
            record = dict(step=n, pose=pick['brace_pose'], tilt=pick['brace_tilt_rotvec'],
                          helper_yaw=pick.get('helper_yaw'), accepted=False)
            self.brace_search_report.append(record)
            print('brace search step %d: pose %s tilt %s' % (n, record['pose'], record['tilt']), flush=True)
            original(n, pick)
            if hasattr(self, 'check_helper_approach'):
                try:
                    self.check_helper_approach(self.A.moves)
                except PlannerUnavailable as exc:
                    record['error'] = 'helper approach: ' + str(exc)
                    self.A.moves.clear()
                    print('brace search rejected helper approach step %d: %s' % (n, str(exc)[-300:]), flush=True)
                    attempt()
                    return
            guard = next(m for m in self.A.moves if m['phase'] == 'brace-guard')
            rebase = guard['then']

            def abandon_hold(reason):
                """Closed (or closing) helper cannot support — try the next pose."""
                record['accepted'] = False
                record['error'] = reason
                print('brace search rejected %s step %d' % (reason, n), flush=True)
                if reason.startswith('support_'):
                    skip_poses.add(pose_key(pick))
                self.before_braced = None
                self.on_support_failed = None
                self.pending_hold = None
                _retreat_helper(self, D, attempt, open_first=True)
                return True

            def checked():
                rebase()
                try:
                    check_approach(self, planner, world, carried, D)
                    if hasattr(self, 'check_insertion_clearance'):
                        self.check_insertion_clearance()
                except PlannerUnavailable as exc:
                    record['error'] = str(exc)
                    print('brace search rejected step %d: %s' % (n, exc), flush=True)
                    # A has not closed. Discard close/hold, retreat with the
                    # shared controller, and try again only after parking.
                    _retreat_helper(self, D, attempt, open_first=False)
                    return
                record['accepted'] = True
                print('brace search accepted step %d before closing' % n, flush=True)

                # Pre-close check uses A's brace-guard joints. After close the
                # measured hold can sit deeper into B's pre-insert corridor
                # (studfit_007–009). Revalidate with the closed helper before
                # releasing B from stage; refuse and try the next pose if not.
                def before_braced(_n):
                    try:
                        check_approach(self, planner, world, carried, D)
                    except PlannerUnavailable as exc:
                        abandon_hold('held approach: ' + str(exc))
                        return False  # do not braced.add; retreat already queued
                    self.before_braced = None
                    print('brace search held approach ok step %d' % n, flush=True)
                    return True

                # Guard-stop close can land one finger only (studfit_010 step 13).
                # That used to end the whole dual build; retry the next candidate.
                def on_support_failed(_n):
                    return abandon_hold('support_not_established')

                self.before_braced = before_braced
                self.on_support_failed = on_support_failed

            guard['then'] = checked

        attempt()

    ex.queue_brace = MethodType(queue_brace, ex)
