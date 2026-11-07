"""Two build contexts sharing Kean's controller and one Newton physics world."""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
from types import MethodType

import numpy as np

from Kean.shared import add_cell_path
add_cell_path()
import dual_arm_sim as D

from Kean.acceptance import bricks_for
from Kean.dual_coordinator import Coordinator
from Kean.motion.world import brick_boxes, plate_box, box_gap


def plans_for(names, separation=.32):
    """Generate plans in independent frames; keep grid/force data together."""
    if not np.isfinite(separation) or separation < .16 or separation > .6:
        raise ValueError('pad separation must be between 0.16 and 0.6 m')
    plans, connections = {}, []
    saved = D.P.VOXEL_ORIGIN
    try:
        for arm, sign in (('A', 1), ('B', -1)):
            name = names[arm]
            kind = 'shape' if name in D.B.SAMPLES else 'structure'
            raw = bricks_for(name, kind)
            internal = arm + '_' + name
            bricks = [(arm + '_' + b[0], *b[1:]) for b in raw]
            D.P.STRUCTURES[internal] = bricks
            ni = max(b[2] + D.P.footprint(b[1], b[5])[0] for b in bricks)
            nj = max(b[3] + D.P.footprint(b[1], b[5])[1] for b in bricks)
            D.P.VOXEL_ORIGIN = (-ni * D.P.PITCH / 2, sign * separation / 2 - nj * D.P.PITCH / 2, D.Z0)
            plan = D.P.build_plan(internal, 'weakest_joint', brace_model='grasp_lp', brace_grip_N=D.BRACE_GRIP_N, brace_mu=1.)
            order = small_first(plan, bricks)
            if order:
                sequence, D.P.sequence = D.P.sequence, lambda _: order
                try:
                    plan = D.P.build_plan(internal, 'weakest_joint', brace_model='grasp_lp',
                                          brace_grip_N=D.BRACE_GRIP_N, brace_mu=1.)
                finally:
                    D.P.sequence = sequence
            plan.update(structure_id=name, context_name=internal, owner=arm)
            for step in plan['sequence']:
                step['placer_arm'] = arm
                if step['requires_brace'] and not step['brace'].get('feasible', True):
                    raise ValueError('%s step %s has no feasible generated support' % (arm, step['step']))
            plans[arm] = plan
            connections.extend(D.ST.connections(bricks))
    finally:
        D.P.VOXEL_ORIGIN = saved
    # The pads must remain physically distinct, including their four-stud borders.
    pads = [pad_for(plans[a]) for a in 'AB']
    if abs(pads[0]['c'][1] - pads[1]['c'][1]) <= (pads[0]['dims'][1] + pads[1]['dims'][1]) / 2 + .02:
        raise ValueError('build pads overlap; increase --separation')
    return plans, connections


def small_first(plan, bricks):
    """Reorder courses where a 1x1 has no clear pinch, fewest studs first.

    A dropped 1x1 rolls off the single stud below and tips (dual_pyramid_s3_004
    step 6), so it must be side-gripped: placed before the larger bricks that
    would box it in, beside at most an in-line neighbour that a short-face edge
    grip slides off. Larger bricks then go last with drop-and-press. Other
    courses keep the planner's order (its docstring: a per-layer search pushed
    the arch crown into a tight slot). None when nothing changes.
    """
    by_id = {b[0]: b for b in bricks}
    order = [by_id[s['brick_id']] for s in plan['sequence']]
    studs = lambda b: np.prod(D.P.BRICKS[b[1]])
    tight = {b[4] for b, s in zip(order, plan['sequence']) if studs(b) == 1 and not s['grasp']['accessible']}
    new = sorted(order, key=lambda b: (b[4], studs(b) if b[4] in tight else 0))
    return new if new != order else None


def pad_for(plan):
    ni = max(b['grid_pos'][0] + D.P.footprint(b['type'], b['yaw_index'])[0] for b in plan['bricks'])
    nj = max(b['grid_pos'][1] + D.P.footprint(b['type'], b['yaw_index'])[1] for b in plan['bricks'])
    box = plate_box(plan['voxel_origin'], ni, nj, D.P.PITCH, D.Z0)
    box['name'] = 'plate_' + plan['owner']
    return box


def merged_plan(plans):
    merged = deepcopy(plans['A'])
    merged.update(structure_id='dual', bricks=[], sequence=[])
    for arm in 'AB':
        plan = plans[arm]
        shift = np.asarray(plan['voxel_origin']) - np.asarray(merged['voxel_origin'])
        for brick in deepcopy(plan['bricks']):
            # Used by neighbour classification only; physics uses world target poses.
            brick['grid_pos'][:2] = (np.asarray(brick['grid_pos'][:2]) + np.rint(shift[:2] / D.P.PITCH).astype(int)).tolist()
            merged['bricks'].append(brick)
        for step in deepcopy(plan['sequence']):
            step.update(local_step=step['step'], owner=arm, step=len(merged['sequence']))
            merged['sequence'].append(step)
    merged['validation']['brace_required_count'] = sum(s['requires_brace'] for s in merged['sequence'])
    return merged


def feeders_for(plans, pads):
    """Choose feeder cells clear of both build pads and the other feeder."""
    feeders, occupied = {}, []
    for pad in pads:
        c, h = np.asarray(pad['c'][:2]), np.asarray(pad['dims'][:2]) / 2
        occupied.append((c - h, c + h))
    for arm in 'AB':
        candidates = [np.asarray(xy) * (-1 if arm == 'A' else 1) for xy in D.feeder_grid()]
        for step in plans[arm]['sequence']:
            for xy in candidates:
                lo, hi = xy - .03, xy + .03
                if any(np.all(lo < ohi + .004) and np.all(olo < hi + .004) for olo, ohi in occupied):
                    continue
                feeders[step['brick_id']] = np.r_[xy, 0.].tolist()
                occupied.append((lo, hi))
                break
            else:
                raise ValueError('not enough non-overlapping feeder cells for arm ' + arm)
    return feeders


class Build:
    """A view onto shared physics, with the original cell's per-build state."""
    def __init__(self, cell, arm, plan):
        self.cell, self.owner, self.plan = cell, arm, plan
        self.name = plan['context_name']
        self.B = cell.arms[arm]                       # role: builder
        self.A = cell.arms['B' if arm == 'A' else 'A']  # role: helper
        self.b_step = self.a_step = self.start_step = self.sham_s = None
        self.next_step = 0
        self.staged, self.braced, self.retracted, self.placed, self.queued = (set() for _ in range(5))
        self.brace_by_step, self.vis, self.place_t = {}, {}, {}
        self.episodes, self.snap_gates = [], []
        self.last_gate = {}
        self.hand_off = self.fk = self.yaw_shift = self.close_t = self.acontacts = None
        self.i_xy = np.zeros(2)
        self.guard_n = 0
        self.brace_contact_phases = (*D.BRACE_PHASES, 'brace-in')
        self.guard_phases = ('brace-in', 'brace-guard')
        self.brace_candidate_limit = 32
        self.brace_high_first = True
        self.brace_leans = (75., 60., 45.)
        self.brace_yaw_alternatives = True
        self.brace_height_offsets = (.002, .0012, -.0012)
        self.brace_lateral_offsets = (.006, -.006, .004, -.004, .002, -.002)
        self.wiggle_amp = .0001
        self.wiggle_tilt = math.radians(.1)
        self.wiggle_press_dur = 1.2
        self.pending_hold = None
        self.drop_press = {}  # step -> release height (m) when no side pinch can reach the seat
        self.support_loss_frames = 0
        self.active = False
        self.spec = True  # Write a single combined episode log in the dual runner.
        self.log_path = cell.out / 'episodes.jsonl'
        for method in ('queue_place', 'queue_brace', 'brace_info', 'a_contacts', 'brace_readout',
                       'aim_brick', 'gate', 'snap', 'brace_row'):
            setattr(self, method, MethodType(getattr(D.Example, method), self))
        from Kean.motion.coordinated import install as coordination
        from Kean.motion.wriggle import install as wriggle
        coordination(self)
        wriggle(self, D)

    def __getattr__(self, name):
        return getattr(self.cell, name)

    @contextmanager
    def planner_frame(self):
        saved = D.P.VOXEL_ORIGIN
        D.P.VOXEL_ORIGIN = self.plan['voxel_origin']
        try:
            yield
        finally:
            D.P.VOXEL_ORIGIN = saved

    def set_group(self, shapes, group=None):
        self.cell.set_group(shapes, (-2 if self.owner == 'A' else -3) if group is None else group)

    def fail(self, reason):
        self.cell.fail(self.owner + ':' + reason)

    def start(self, n):
        self.b_step, self.active = n, True
        with self.planner_frame():
            step = self.plan['sequence'][n]
            bid = step['brick_id']
            slot = self.feeder[bid]
            offset = self.pickup_clearance(step)
            self.feeder[bid] = slot + offset
            try:
                self.queue_place(step)
            finally:
                self.feeder[bid] = slot
        # Let the existing GT integral settle after joint playback before the
        # brick reaches the studs. A planned endpoint can retain ~1 mm of
        # tracking error even though the measured joints have arrived.
        if step['requires_brace']:
            pre = next(m for m in self.B.moves if m['phase'] == 'pre-insert')
            first_insert = next(i for i, m in enumerate(self.B.moves) if m['phase'] == 'insert')
            self.B.push('insert', pre['pos'], pre['yaw'], pre['grip'], 1., contact=True)
            self.B.moves.insert(first_insert, self.B.moves.pop())
            for move in self.B.moves:
                if move['phase'] == 'insert':
                    move['dur'] *= 1.5
            final_insert = max(i for i, m in enumerate(self.B.moves) if m['phase'] == 'insert')
            final = self.B.moves[final_insert]
            snap, final['then'] = final['then'], None
            self.B.push('insert', final['pos'], final['yaw'], final['grip'], .4, contact=True, then=snap)
            self.B.moves.insert(final_insert + 1, self.B.moves.pop())
        # Once the snap gate has engaged the clutch, open while lifting clear
        # of the lower course instead of spreading fingers at press depth.
        for move in self.B.moves:
            if move['phase'] == 'release':
                move['pos'] = move['pos'] + [0., 0., .002]
        if step['requires_brace']:
            # Feeder locations vary by brick. Stage the loaded builder at its
            # checked park pose so the helper has a consistent clear approach.
            lift_index = next(i for i, m in enumerate(self.B.moves) if m['phase'] == 'lift')
            lift = self.B.moves[lift_index]
            staged = lift['then']
            lift['then'] = None
            self.B.push('transport', self.parks[self.B], lift['yaw'], lift['grip'], 1.5, then=staged)
            self.B.moves.insert(lift_index + 1, self.B.moves.pop())
            self.B.moves[lift_index + 2]['pos'] = self.parks[self.B].copy()
        # An ordinary step also ends empty-handed at a checked safe park pose.
        if not self.plan['sequence'][n]['requires_brace']:
            self.B.push('clearance', self.parks[self.B], 0., .01, 1.5)
        if n in self.drop_press:
            if step['requires_brace']:
                # ponytail: no supported tight step exists yet; add when one does.
                raise RuntimeError('drop-and-press with a generated support is not implemented')
            from Kean.motion.drop_press import splice
            splice(self.B.moves, self.target[bid][0], self.drop_press[n], grasp_dz=D.GRASP_DZ, press=D.PRESS,
                   tip_below_tcp=D.TIP_BELOW_TCP, brick_h=D.P.BRICK_H, stud_h=D.ex.STUD_HEIGHT,
                   grasp_offset=step['grasp'].get('tcp_offset_m', [0., 0.]))
            print('drop-and-press %s: release %.2f mm above the seat' % (bid, self.drop_press[n] * 1000), flush=True)
        elif self.stud_fit is not None:
            from Kean.motion.drop_press import top_press
            top_press(self.B.moves, self.target[bid][0], press=D.PRESS, tip_below_tcp=D.TIP_BELOW_TCP,
                      brick_h=D.P.BRICK_H, stud_h=D.ex.STUD_HEIGHT,
                      grasp_offset=step['grasp'].get('tcp_offset_m', [0., 0.]))

    def finger_model(self):
        checker = self.cell.tick.clearance
        probe = checker.probe
        arm = self.B.name[0]
        i = 0 if arm == 'A' else 9
        q = self.state_0.joint_q.numpy()[i:i+9].copy()
        q[7:] = probe.FINGER_LOCK
        spheres = checker.spheres(arm, q)
        p, rotation = probe.tcp_poses(arm, q[:7][None])
        left = checker.slices['fr3_leftfinger']
        ids = np.r_[left, checker.slices['fr3_rightfinger']]
        return (spheres[ids, :3] - p[0]) @ rotation[0], spheres[ids, 3], len(left)

    def pickup_clearance(self, step):
        """Choose a grip whose fingers reach the seat, raising it up to 2 mm.

        Fallback order: the planner's grip, the other axis centred, edge grips
        nearest first (pads kept on the brick), else drop-and-press with the
        planner's grip. Each candidate uses the full finger-sphere check.
        """
        from scipy.spatial.transform import Rotation
        from Kean.motion.grasp_clearance import choose_offset, grip_candidates
        from Kean.motion.tick import planner_yaw, resolve_wrap
        local, radii, nl = self.finger_model()
        probe = self.cell.tick.clearance.probe
        boxes, _ = self.cell.world(self, self.owner)
        boxes = [b for b in boxes if b['name'] != step['brick_id']]
        brick = next(b for b in self.plan['bricks'] if b['id'] == step['brick_id'])
        extent = np.asarray(D.P.footprint(brick['type'], brick['yaw_index'])) * D.P.PITCH

        def gap_for(grasp):
            yaw = planner_yaw(self, self.B, math.radians(grasp['yaw_offset_deg']), resolve_wrap(D))
            rotation = Rotation.from_euler('z', yaw).as_matrix() @ Rotation.from_quat(D.Q_DOWN).as_matrix()
            tcp = self.target[step['brick_id']][0] + np.r_[grasp['tcp_offset_m'], 0.] + [0, 0, D.GRASP_DZ - D.PRESS]
            width = grasp['grip_width_m'] + .0012

            def gap_at(offset):
                if offset[2] < 0:
                    return -1.
                gaps = []
                for opening in (width / 2 - .0015, width / 2 + .002):
                    xyz = local.copy()
                    xyz[:nl, 1] += opening - probe.FINGER_LOCK
                    xyz[nl:, 1] -= opening - probe.FINGER_LOCK
                    fingers = np.c_[xyz @ rotation.T + tcp + offset, radii]
                    gaps.append(probe.clearance_boxes(fingers, boxes)[0])
                return min(gaps)
            return gap_at

        for kind, grasp in grip_candidates(step['grasp'], extent):
            try:
                offset = choose_offset(gap_for(grasp), [0., 0., 1.], margin=.0015)
            except RuntimeError:
                continue
            if kind != 'planner':
                step['grasp'].update(grasp, grip_fallback=kind)
                print('grip fallback %s: %s, yaw %d deg, offset %s mm' % (
                    step['brick_id'], kind, grasp['yaw_offset_deg'],
                    np.round(np.asarray(grasp['tcp_offset_m']) * 1000, 1).tolist()), flush=True)
            break
        else:
            # No side pinch reaches the seat beside flush neighbours: release higher, then press.
            self.drop_press[step['step']] = choose_offset(gap_for(step['grasp']), [0., 0., 1.],
                                                          max_offset=.015, margin=.0015)[2]
            offset = np.zeros(3)
        offset += np.r_[step['grasp'].get('tcp_offset_m', [0., 0.]), 0.]
        print('pickup clearance %s: %s mm' % (step['brick_id'], np.round(offset * 1000, 2).tolist()), flush=True)
        return offset

    def check_helper_approach(self, moves):
        """Reject generated braces with unsafe helper routes before moving."""
        from Kean.motion.plan_curobo import Request
        from Kean.motion.tick import planner_yaw, resolve_wrap
        helper = self.A.name[0]
        self.A.absolute_yaw = 'helper_yaw' in self.plan['sequence'][self.b_step]['brace']
        for move in moves:
            if move['phase'] == 'to brace':
                move['pos'][2] = max(move['pos'][2], .30)
        i = 0 if helper == 'A' else 9
        q = self.state_0.joint_q.numpy()[i:i+7].copy()
        ee = np.asarray(self.A.ee).copy()
        boxes, other_q = self.cell.world(self, helper)
        self.check_finger_sweep(moves, boxes)
        cached = []
        for move in moves:
            if move['phase'] not in ('to brace', 'approach', 'brace-in', 'brace-guard'):
                break
            allowed = self.cell.allowed_finger_contacts(helper, move['phase'])
            request = Request(helper, q, ee, move['pos'], [0, 0, 0, 1], boxes=boxes, other_q=other_q,
                              full_path=True, grip=move['grip'], tilt=move['tilt'], allowed_finger_contacts=allowed,
                              proposal_build=self.owner, clearance=.0012,
                              yaw=planner_yaw(self, self.A, move['yaw'], resolve_wrap(D)))
            result = self.cell.planner.plan(request)
            if move['phase'] != 'brace-guard':
                cached.append((move, result))
            q, ee = result['joint_pos'][-1], np.asarray(result['goal'])
        # Check the whole finger-closing sweep, including neighbouring bricks
        # outside the generator's selected grasp. A clear open hand is not a
        # certificate that it can close safely.
        from Kean.motion.path_check import validate
        for opening in np.linspace(D.BRACE_HALF_W + D.BRACE_CLEAR, D.BRACE_HALF_W - D.BRACE_SQUEEZE, 12):
            request.grip = float(opening)
            try:
                validate(self.cell.planner.backend.probe, request, np.asarray(q)[None])
            except ValueError as exc:
                from Kean.motion.plan_curobo import PlannerUnavailable
                raise PlannerUnavailable('closed helper at %.2f mm opening: %s' % (opening * 1000, exc)) from exc
        for move, result in cached:
            move['preplanned'] = result

    def check_finger_sweep(self, moves, boxes):
        """Reject impossible closing geometry before asking cuRobo for routes."""
        from scipy.spatial.transform import Rotation
        from Kean.motion.plan_curobo import PlannerUnavailable
        from Kean.motion.tick import planner_yaw, resolve_wrap
        checker = self.cell.tick.clearance
        probe = checker.probe
        helper = self.A.name[0]
        i = 0 if helper == 'A' else 9
        q = self.state_0.joint_q.numpy()[i:i+9].copy()
        q[7:] = probe.FINGER_LOCK
        sp = checker.spheres(helper, q)
        p, rotation = probe.tcp_poses(helper, q[:7][None])
        left = checker.slices['fr3_leftfinger']
        ids = np.r_[left, checker.slices['fr3_rightfinger']]
        local = (sp[ids, :3] - p[0]) @ rotation[0]
        hand_ids = np.r_[checker.slices['fr3_hand'], checker.slices['fr3_link7']]
        hand_local = (sp[hand_ids, :3] - p[0]) @ rotation[0]
        goal = next(m for m in moves if m['phase'] == 'brace-guard')
        yaw = planner_yaw(self, self.A, goal['yaw'], resolve_wrap(D))
        world = (Rotation.from_rotvec(goal['tilt']).as_matrix() @
                 Rotation.from_euler('z', yaw).as_matrix() @ Rotation.from_quat(D.Q_DOWN).as_matrix())
        allowed = self.cell.allowed_finger_contacts(helper, 'brace-guard')
        obstacles = [b for b in boxes if b['name'] not in allowed]
        insertion = self.insertion_box()
        hand = np.c_[hand_local @ world.T + goal['pos'], sp[hand_ids, 3]]
        for opening in np.linspace(D.BRACE_HALF_W + D.BRACE_CLEAR, D.BRACE_HALF_W - D.BRACE_SQUEEZE, 12):
            xyz = local.copy()
            xyz[:len(left), 1] += opening - probe.FINGER_LOCK
            xyz[len(left):, 1] -= opening - probe.FINGER_LOCK
            fingers = np.c_[xyz @ world.T + goal['pos'], sp[ids, 3]]
            gap, who = probe.clearance_boxes(fingers, obstacles)
            if gap < .0012:
                raise PlannerUnavailable('closed finger geometry %.2f mm at %s' % (gap * 1000, who))
            gap, _ = probe.clearance_boxes(np.vstack([hand, fingers]), [insertion])
            if gap < .0012:
                raise PlannerUnavailable('helper obstructs carried-brick insertion corridor: %.2f mm' % (gap * 1000))

    def insertion_box(self):
        """Enclose the seated brick and its guided/wriggle insertion sweep."""
        bid = self.plan['sequence'][self.b_step]['brick_id']
        target, yaw, _ = self.target[bid]
        length, width = self.dims[bid]
        height = D.P.BRICK_H + .0018
        return dict(name='insertion_' + bid, c=target + [0, 0, (height + .025)/2],
                    dims=[length + .004, width + .004, height + .029], yaw=yaw)

    def check_insertion_clearance(self):
        from Kean.motion.plan_curobo import PlannerUnavailable
        checker = self.cell.tick.clearance
        helper = self.A.name[0]
        i = 0 if helper == 'A' else 9
        q = self.state_0.joint_q.numpy()[i:i+9].copy()
        for opening in (D.BRACE_HALF_W + D.BRACE_CLEAR, D.BRACE_HALF_W - D.BRACE_SQUEEZE):
            q[7:] = opening
            spheres = checker.spheres(helper, q)
            gap, _ = checker.probe.clearance_boxes(spheres, [self.insertion_box()])
            if gap < .0012:
                raise PlannerUnavailable('measured helper obstructs insertion corridor: %.2f mm' % (gap * 1000))

    def brace_guard_stopped(self, arm):
        self.cell.tick.cancel_at_measured(arm)

    def confirm_brace(self, n):
        # The original close callback rebases the measured pose. Keep the
        # builder staged until the contact readout confirms both fingers.
        self.brace_info(n)['executed'] = False
        self.pending_hold = n
        self.support_loss_frames = 0

    def check_support(self):
        n = self.pending_hold
        if n is None or self.A.phase != 'hold':
            return
        samples = self.brace_info(n)['grip_samples']
        established = len(samples) >= 5 and min(np.mean(samples[-5:], axis=0)) > .5
        if n not in self.braced:
            if established:
                self.brace_info(n)['executed'] = True
                self.braced.add(n)
            elif self.sim_time - self.close_t > 3.:
                self.fail('support_not_established')
        else:
            self.support_loss_frames = 0 if established else self.support_loss_frames + 1
            if self.support_loss_frames >= 6:
                self.fail('support_lost')


class DualExample(D.Example):
    def __init__(self, viewer, args, plans, connections, out, replay=False, starts=None, step_limit=None, feeders=None,
                 stud_fit=False):
        self.plans, self.connections, self.out = plans, connections, Path(out)
        self.combined = merged_plan(plans)
        self.replay = replay
        self.starts = starts or dict(A=0, B=0)
        self.preplaced = {s['brick_id'] for a in 'AB' for s in plans[a]['sequence'][:self.starts[a]]}
        self.platforms = [pad_for(plans[a]) for a in 'AB']
        self.feeders = feeders if feeders is not None else feeders_for(plans, self.platforms)
        self.views = {}
        self.role_views = {}
        if stud_fit:
            # A feed-forward press on each hand (fr3_hand_tcp), allocated before
            # the base class captures the graph; the hand indices are set after.
            self.press_body = D.wp.zeros(2, dtype=D.wp.int32)
            self.press_w = D.wp.zeros(2, dtype=D.wp.spatial_vector)
        super().__init__(viewer, args)
        self.arms = {'A': self.A, 'B': self.B}
        self.A.name, self.B.name = 'A builder', 'B builder'
        self.coordinator = Coordinator({a: [s['requires_brace'] for s in plans[a]['sequence']] for a in 'AB'})
        for a in 'AB':
            self.coordinator.builds[a].step = self.starts[a]
            if step_limit is not None:
                self.coordinator.builds[a].supports = self.coordinator.builds[a].supports[:self.starts[a] + step_limit]
        self.views = {a: Build(self, a, plans[a]) for a in 'AB'}
        self.role_views = dict(self.views)
        self.roles = dict(A='builder', B='builder')
        self.task_builds = dict(A='A', B='B')
        for arm in 'AB':
            self.set_grip_role(arm, False)
            self.set_group(plans[arm]['_plate_shapes'], -2 if arm == 'A' else -3)
        for bid in self.preplaced:
            self.breaker.fire(bid)
            self.set_group(self.shapes_of[bid], -2 if bid.startswith('A_') else -3)
        self.log_path = self.out / 'episodes.jsonl'
        self.last_progress = 0
        self.last_events = 0
        self.held_reported = set()
        self.actual_parallel_frames = 0
        self.last_measured = self.state_0.joint_q.numpy()[:18].copy()
        self.contact_shape_types = self.model.shape_type.numpy()
        self.contact_shape_poses = self.model.shape_transform.numpy()
        self.stud_fit = None
        if stud_fit:
            from Kean.motion.stud_fit import StudFit
            self.press_body.assign(np.array([D.EE, D.EE + self.n_arm_bodies], dtype=np.int32))
            self.stud_fit = StudFit(self, D, self.out / 'stud_fit.json')

    def simulate(self):
        if getattr(self, 'press_w', None) is None:
            return super().simulate()
        # D.Example.simulate plus the --stud-fit press wrench (clear_forces() erases body_f every substep).
        self.pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.fx:
                D.wp.launch(D.add_fixture_wrench, dim=len(self.fx_body), inputs=[self.fx_body, self.fx_w, self.state_0.body_f])
            D.wp.launch(D.add_fixture_wrench, dim=2, inputs=[self.press_body, self.press_w, self.state_0.body_f])
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def plan_input(self, args):
        return self.combined, 'dual'

    def build_scene(self, plan, exe, n0):
        scene, c = D.build_cell(plan, exe, n0, self.substeps, False,
                               workspaces=[self.plans[a] for a in 'AB'], feeders=self.feeders,
                               placed_ids=self.preplaced, velocity_control=True)
        return scene, c

    def set_grip_role(self, arm, support):
        index = 0 if arm == 'A' else 9
        ke, kd = self.model.joint_target_ke.numpy(), self.model.joint_target_kd.numpy()
        gain = D.A_FINGER_KE if support else D.FINGER_KE
        ke[index+7:index+9] = gain
        kd[index+7:index+9] = 2 * math.sqrt(gain * .1)
        self.model.joint_target_ke.assign(ke)
        self.model.joint_target_kd.assign(kd)
        self.solver.notify_model_changed(D.newton.solvers.SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def configure(self, planner):
        from Kean.motion.clearance import Clearance
        from Kean.motion.brace_hold import install as brace_hold
        from Kean.motion.brace_search import install as brace_search
        from Kean.motion.reservations import Reservations
        from Kean.motion.tick import Tick, resolve_wrap
        brace_hold(self)
        self.planner = planner
        checker = Clearance(planner.backend.probe, self.world, cache=True)
        self.reservations = Reservations(checker)
        self.tick = Tick(planner, world=self.world, clearance=checker, wrap_angle=resolve_wrap(D),
                         resample_trajectory=planner.backend.probe.resample, strict=True,
                         carried=self.carried, reservations=self.reservations).attach(self)
        for view in self.views.values():
            brace_search(view, planner, self.world, self.carried, D)

    def settle_feeders(self):
        """Let the initial table contacts settle before measuring task motion."""
        before = self.state_0.body_q.numpy()
        for _ in range(60):
            if self.graph:
                D.wp.capture_launch(self.graph)
            else:
                self.simulate()
            self.sim_time += self.frame_dt
        after = self.state_0.body_q.numpy()
        drift = max(float(np.linalg.norm(after[b, :2] - before[b, :2])) for b in self.body.values())
        if drift > .002:
            raise RuntimeError('feeder layout unstable: %.2f mm horizontal settling' % (drift * 1000))
        self.initial_settling_xy_m = drift

    def schedule(self):
        if self.failure or self.replay:
            return
        c = self.coordinator
        for arm, view in self.views.items():
            if not view.active:
                continue
            cooperative = c.active == arm
            if cooperative and view.b_step in view.staged and view.b_step not in view.queued:
                view.queued.add(view.b_step)
                with view.planner_frame():
                    view.queue_brace(view.b_step, view.plan['sequence'][view.b_step]['brace'])
            if view.B.idle() and (not cooperative or view.A.idle()):
                if view.b_step not in view.placed:
                    self.fail(arm + ':unfinished_placement')
                    return
                c.complete(arm, self.sim_time)
                view.active = False
                if cooperative:
                    for a in 'AB':
                        self.roles[a], self.task_builds[a], self.role_views[a] = 'builder', a, self.views[a]
                        self.set_grip_role(a, False)
        ready = {a: self.arms[a].idle() for a in 'AB'}
        for arm, helper in c.dispatch(ready, self.sim_time):
            view = self.views[arm]
            if helper:
                self.roles[helper], self.task_builds[helper], self.role_views[helper] = 'supporter', arm, view
                self.set_grip_role(helper, True)
                c.event('helper_ready', arm, self.sim_time, helper=helper)
            view.start(c.builds[arm].step)
        for event in c.events[self.last_events:]:
            print('coordinator:', event, flush=True)
        self.last_events = len(c.events)
        self.done = c.done and all(a.idle() for a in self.arms.values())
        if self.done:
            self.done_t = self.sim_time

    def arm_command(self, arm, k, body_q):
        view = self.role_views.get(arm.name[0])
        ee = body_q[D.EE + k * self.n_arm_bodies][:3]
        if view is None:
            return arm.update(self.frame_dt, ee)
        pending = arm.cur if arm.cur is not None else (arm.moves[0] if arm.moves else None)
        arm.absolute_yaw = (self.roles[arm.name[0]] == 'supporter' and view.b_step is not None
                            and 'helper_yaw' in view.plan['sequence'][view.b_step]['brace']
                            and pending is not None and pending['phase'] != 'park')
        with view.planner_frame():
            if arm is view.B and view.active:
                # Anti-windup: a frozen arm cannot act on the brick-aim
                # integral, which would otherwise saturate into the next goal.
                tick = getattr(self, 'tick', None)
                # --stud-fit: after release the brick sits on the stud tops, held; aiming
                # it would only wind the hand's bias off the brick under the top press.
                pressing = self.stud_fit is not None and arm.phase == 'press'
                if (tick is None or not tick.slots[arm].paused) and not pressing:
                    view.aim_brick(ee, body_q)
            else:
                arm.bias[:] = 0.
            return arm.update(self.frame_dt, ee)

    def brace_readout(self):
        for view in self.views.values():
            if view.active and self.coordinator.active == view.owner:
                with view.planner_frame():
                    view.brace_readout()
                    view.check_support()

    def step(self):
        super().step()
        if self.stud_fit is not None:
            self.stud_fit.update()
        for a, view in self.views.items():
            key = (a, view.b_step)
            if view.b_step in view.braced and key not in self.held_reported:
                self.held_reported.add(key)
                self.coordinator.event('support_held', a, self.sim_time, helper=view.A.name[0])
        if hasattr(self, 'reservations'):
            self.reservations.step()
            measured = self.state_0.joint_q.numpy()[:18]
            moving = [self.tick.slots[self.arms[a]].mode == 'playing' and
                      np.max(np.abs(measured[9*k:9*k+7] - self.last_measured[9*k:9*k+7])) > .0001
                      for k, a in enumerate('AB')]
            self.actual_parallel_frames += int(all(moving) and self.coordinator.active is None)
            self.last_measured = measured.copy()
        if self.views:
            current = self.views['B']
            if not current.active:
                current = self.views['A']
            self.b_step = next((s['step'] for s in self.plan['sequence'] if s['owner'] == current.owner and
                                s['local_step'] == current.b_step), None)
            count = sum(len(v.episodes) for v in self.views.values())
            if count != self.last_progress:
                self.episodes = [row for v in self.views.values() for row in v.episodes]
                self.log_path.write_text(''.join(json.dumps(row) + '\n' for row in self.episodes))
                self.last_progress = count
            if self.frame % 120 == 0:
                progress = dict(frame=self.frame, sim_time=self.sim_time,
                                phases={a: self.arms[a].phase for a in 'AB'},
                                modes={a: self.tick.slots[self.arms[a]].mode for a in 'AB'} if hasattr(self, 'tick') else {},
                                steps={a: self.coordinator.builds[a].step for a in 'AB'},
                                pending=self.coordinator.pending)
                (self.out / 'progress.json').write_text(json.dumps(progress, indent=2))

    def held_objects(self):
        from cell.contacts import HELD_PHASES
        return {a: v.plan['sequence'][v.b_step]['brick_id'] for a, v in self.views.items()
                if v.active and v.b_step is not None and v.b_step not in v.placed
                and self.arms[a].phase in (*HELD_PHASES, 'press')}

    def measured_state(self):
        # Planning and command validation are synchronous: physics is fixed
        # throughout a frame. Copy GPU state once for its geometry queries.
        if getattr(self, '_snapshot_frame', None) != self.frame:
            self._snapshot_frame = self.frame
            self._body_snapshot = self.state_0.body_q.numpy()
            self._joint_snapshot = self.state_0.joint_q.numpy()
            from Kean.shared import load_probe
            self._brick_snapshot = brick_boxes(self._body_snapshot, self.body, self.dims,
                                                None, box_of=load_probe().brick_box)
        return self._body_snapshot, self._joint_snapshot

    def world(self, ex, arm):
        from Kean.shared import load_probe
        probe = load_probe()
        _, joints = self.measured_state()
        held = self.held_objects()
        # Picking/release contacts concern a real world obstacle; only an
        # actually carried brick is excluded from this arm's static scene.
        skip = held.get(arm) if self.arms[arm].phase not in ('descend', 'settle', 'grasp', 'release') else None
        boxes = [probe.TABLE, *self.platforms]
        boxes += [b for b in self._brick_snapshot if b['name'] != skip]
        index = 9 if arm == 'A' else 0
        return boxes, joints[index:index+9]

    def carried(self, ex, arm):
        held = self.held_objects().get(arm)
        if held is None or self.arms[arm].phase not in ('lift', 'stage', 'transport', 'pre-insert', 'insert'):
            return None
        from scipy.spatial.transform import Rotation
        q, _ = self.measured_state()
        i = 0 if arm == 'A' else self.n_arm_bodies
        tcp, brick = q[D.EE+i], q[self.body[held]]
        rt, rb = Rotation.from_quat(tcp[3:]).as_matrix(), Rotation.from_quat(brick[3:]).as_matrix()
        h = D.P.BRICK_H + .0018
        center = rt.T @ (brick[:3] + rb @ np.array([0., 0., h/2]) - tcp[:3])
        quat = Rotation.from_matrix(rt.T @ rb).as_quat()
        return dict(dims=[*self.dims[held], h], center=center.tolist(), quat_wxyz=quat[[3, 0, 1, 2]].tolist())

    def allowed_finger_contacts(self, arm, phase):
        if self.roles[arm] != 'supporter' or phase not in ('brace-in', 'brace-guard', 'close', 'hold', 'open', 'retract'):
            return ()
        view = self.role_views[arm]
        brace = view.plan['sequence'][view.b_step]['brace']
        return tuple(brace.get('gripped_bricks') or [brace['target_brick_id']])

    def neighbours_of(self, bid):
        if not hasattr(self, '_neighbours'):
            from Kean.motion.drop_press import placement_neighbours
            self._neighbours = placement_neighbours(self.plan['bricks'], D.P.footprint)
        return self._neighbours.get(bid, ())

    def classify_contact(self, monitor, cls_a, cls_b):
        pair = {cls_a, cls_b}
        for arm, bid in self.held_objects().items():
            phase = self.arms[arm].phase
            brick = 'brick:' + bid
            if pair == {arm + '_finger', brick}:
                return None
            if brick in pair and pair.intersection(('ground', 'plate')) and phase in ('descend', 'settle', 'grasp', 'lift'):
                return None
            if phase == 'press' and any(pair == {arm + '_finger', 'brick:' + n}
                                        for n in self.neighbours_of(bid)):
                return None  # closed tips pressing the course top beside the dropped brick
            if brick in pair and phase in ('insert', 'release', 'press'):
                other = next((c for c in pair if c != brick), '')
                oid = other[6:] if other.startswith('brick:') else None
                if oid in monitor.supports.get(bid, []) or (other == 'plate' and None in monitor.supports.get(bid, [])) or oid in self.neighbours_of(bid):
                    return 'neighbour_rub'
        for arm, view in self.views.items():
            if view.active and view.b_step is not None and self.arms[arm].phase == 'retract':
                if pair == {arm + '_finger', 'brick:' + view.plan['sequence'][view.b_step]['brick_id']}:
                    return None  # finger separation after the intended release
        owner = self.coordinator.active
        if owner is not None:
            view = self.views[owner]
            if view.A.phase in view.brace_contact_phases:
                brace = view.plan['sequence'][view.b_step]['brace']
                if any(pair == {view.A.name[0] + '_finger', 'brick:' + bid} for bid in brace.get('gripped_bricks', [brace['target_brick_id']])):
                    return None
        return 'unintended'

    def penetration_kind(self, monitor, cls_a, cls_b, shape_a, shape_b, depth, point=None):
        from Kean.motion.contact_policy import stud_mating
        for arm, bid in self.held_objects().items():
            upper = 'brick:' + bid
            if upper not in (cls_a, cls_b):
                continue
            other = cls_b if cls_a == upper else cls_a
            if not other.startswith('brick:'):
                continue
            lower = other[6:]
            stud, top = (shape_b, shape_a) if cls_a == upper else (shape_a, shape_b)
            st, tt = int(self.contact_shape_types[stud]), int(self.contact_shape_types[top])
            z = float(self.contact_shape_poses[stud, 2])
            local = None
            if st == 8 and point is not None:
                pose = monitor.body_poses[self.body[lower]]
                local = D.rotmat(pose[3:]).T @ (point - pose[:3])
            if stud_mating(self.arms[arm].phase, lower, monitor.supports.get(bid, ()),
                           depth, st, tt, z, D.P.BRICK_H, D.ex.STUD_HEIGHT, D.ex.COLLIDER_INSET,
                           contact_local=local, dims=self.dims[lower]):
                evidence = dict(upper=bid, lower=lower, stud_shape=int(stud),
                                upper_shape=int(top), stud_type=st, upper_type=tt, stud_local_z=z)
                if local is not None:
                    evidence.update(contact_frame=int(self.frame), contact_world=np.asarray(point).tolist(),
                                    contact_local=local.tolist(), lower_dims=list(self.dims[lower]))
                return 'stud_mating', evidence
        return 'penetration', None

    def command_clearance(self, checker, joints, spheres=None):
        """Full links versus both structures/platforms, including Cartesian phases."""
        minimum, label = float('inf'), ''
        predicted = {}
        for arm in 'AB':
            carried = self.carried(self, arm)
            if carried is None:
                continue
            i = 0 if arm == 'A' else 9
            p, rotation = checker.probe.tcp_poses(arm, joints[i:i+7][None])
            q = np.asarray(carried['quat_wxyz'])[[1, 2, 3, 0]]
            predicted[arm] = dict(name=self.held_objects()[arm], dims=carried['dims'],
                                 c=p[0] + rotation[0] @ carried['center'], R=rotation[0] @ D.rotmat(q))
        for arm in 'AB':
            i = 0 if arm == 'A' else 9
            own = spheres[arm] if spheres is not None else checker.spheres(arm, joints[i:i+9])
            finger_ids = np.r_[checker.slices['fr3_leftfinger'], checker.slices['fr3_rightfinger']]
            nonfinger = np.ones(len(own), bool)
            nonfinger[finger_ids] = False
            permitted = set()
            view = self.role_views[arm]
            if arm == view.owner and view.active and view.b_step is not None:
                active = view.plan['sequence'][view.b_step]['brick_id']
                permitted.add(active)
                if self.arms[arm].phase == 'press':
                    permitted.update(self.neighbours_of(active))
            elif self.roles[arm] == 'supporter' and self.arms[arm].phase in ('brace-in', 'brace-guard', 'close', 'hold', 'open', 'retract'):
                brace = view.plan['sequence'][view.b_step]['brace']
                permitted.update(brace.get('gripped_bricks') or [brace['target_brick_id']])
            boxes, _ = self.world(self, arm)
            for part, selected in ((own[nonfinger], boxes),
                                   (own[finger_ids], [b for b in boxes if b['name'] not in permitted])):
                gap, who = checker.probe.clearance_boxes(part, selected)
                if gap < minimum:
                    minimum, label = gap, arm + ':' + str(who)
        for arm, carried in predicted.items():
            other = 'B' if arm == 'A' else 'A'
            i = 0 if other == 'A' else 9
            other_spheres = spheres[other] if spheres is not None else checker.spheres(other, joints[i:i+9])
            gap, who = checker.probe.clearance_boxes(other_spheres, [carried])
            if gap < minimum:
                minimum, label = gap, arm + ':carried/arm_' + other
            view = self.views[arm]
            phase = self.arms[arm].phase
            allowed = {b['name'] for b in predicted.values()}
            if phase == 'lift':
                allowed.add('table')  # intended pick-up surface contact
            if phase == 'insert':
                step = view.plan['sequence'][view.b_step]
                allowed.update(m[0] for m in step['mating_studs'])
                allowed.update(self.neighbours_of(carried['name']))
                if not step['mating_studs']:
                    allowed.add('plate_' + arm)
            boxes, _ = self.world(self, arm)
            for box in boxes:
                if box['name'] in allowed:
                    continue
                gap = box_gap(carried, box)
                if gap < minimum:
                    minimum, label = gap, arm + ':carried/' + box['name']
        if len(predicted) == 2:
            gap = box_gap(predicted['A'], predicted['B'])
            if gap < minimum:
                minimum, label = gap, 'carried/carried'
        return minimum, label

    def gui(self, ui):
        ui.text('Two builders / reciprocal support')
        for arm in 'AB':
            state = self.coordinator.builds[arm]
            ui.text('%s: %s  %d/%d  %s' % (arm, self.plans[arm]['structure_id'], state.step,
                                          len(state.supports), self.roles[arm]))
            ui.text('  ' + self.arms[arm].phase)
        ui.text('Support waiting: ' + ', '.join(self.coordinator.pending))
        if self.failure:
            ui.text('STOP: ' + self.failure)

    def finish(self, error=None):
        if self.monitor:
            self.monitor.finalize()
            (self.out / 'contacts.json').write_text(json.dumps(self.monitor.events, indent=2))
        if self.recorder:
            self.recorder.save(self.args.record_all, self)
        (self.out / 'events.json').write_text(json.dumps(self.coordinator.events, indent=2))


def make_args(names):
    from Kean.scripts.m05 import _args
    return _args(dict(shape=names['B'], kind='shape', strategy='weakest_joint', seed=0), None)


def load_example(run, viewer):
    """Reconstruct the recorded layout, without replanning or stepping it."""
    manifest = json.loads((run / 'manifest.json').read_text())
    plans = {a: json.loads((run / manifest['plans'][a]).read_text()) for a in 'AB'}
    connections = []
    saved = D.P.VOXEL_ORIGIN
    try:
        for a, plan in plans.items():
            bricks = [(b['id'], b['type'], *b['grid_pos'], b['yaw_index']) for b in plan['bricks']]
            D.P.STRUCTURES[plan['context_name']] = bricks
            D.P.VOXEL_ORIGIN = plan['voxel_origin']
            connections.extend(D.ST.connections(bricks))
    finally:
        D.P.VOXEL_ORIGIN = saved
    args = make_args(manifest['names'])
    args.viewer = 'viser'
    example = DualExample(viewer, args, plans, connections, run, replay=True,
                          starts=manifest.get('starts'), step_limit=manifest.get('step_limit'), feeders=manifest.get('feeders'))
    example.replay_events = json.loads((run / 'events.json').read_text())
    example.replay_event_index = 0
    return example
