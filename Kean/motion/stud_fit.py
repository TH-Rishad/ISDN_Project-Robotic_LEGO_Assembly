"""Opt-in stud resistance for the dual builder (scripts/dual_build.py --stud-fit).

The cell's stud proxies clear a brick's walls by 0.6 mm a side, so a brick
seats under its own weight and the clutch is only the snap gate. With this
on, a brick reaching the stud tops of its support is held there by a weld at
its current pose, standing in for the stud interference, until the measured
push through that weld reaches n x 8.9 N (planner F_INSERT_PER_STUD over its
mating studs). Then the weld's anchor moves straight down to the seat (held
x, y and yaw: the studs guide the slide) until the snap gate hands over to the
clutch welds. A brick that is never
pushed through stays on the stud tops and misses the gate (dz ~1.7 mm >
1.5 mm).

The builder's 1.5 N-per-finger side grip slips at ~3 N, so a stud-fit step
releases on the stud tops and presses the brick's top with the closed tips
(drop_press.top_press). Once the tips reach the held brick's studs, a
feed-forward force on the hand ramps over RAMP to 1.3 x the push needed, as a
Cartesian-impedance press would, and drops to zero when the brick goes through.
That arm's joint stiffness is raised STIFF x while it presses, and so is a
supporting arm's: at the cell's 400 N m/rad the hand skated 4 mm off a 2x2
under the push (dual_studfit_diag_009).

ponytail: a breakaway threshold followed by a guided slide, not a force spread
over the 1.7 mm stroke; make it a band of interference contacts if the stroke
itself matters. A free slide let the press twist a 1x1 5 deg on its single
stud (dual_studfit_diag_006). The hold reuses all of the brick's clutch welds,
each at the held pose against its own support, and restores their planned
relposes on release: one weld to the first support alone carried the whole
press and its moment into that brick and pried a seated 1x2 off
(dual_pyramid_s3_studfit_003). The recorder and joint breaker read only the
clutch flags (w[3]), so a hold never counts as a placement.
"""
from contextlib import contextmanager
import json
import math

import numpy as np

F_INSERT = 8.9      # N per mating stud (planner.F_INSERT_PER_STUD)
STUD_H = 0.0017     # m, stud height: the band in which the studs interfere
BAND = 0.0003       # m above the stud tops at which the hold engages
LATERAL = 0.002     # m, same lateral window as the snap gate
RAMP = 0.5          # s to full press force
MARGIN = 1.3        # press force over the push needed (newton_fit.press_force)
STIFF = 5.0         # arm joint stiffness gain while pressing (damping scales by its root)
SUPPORT_GRIP = 2.0  # x A_FINGER_KE for a supporting hand (~28 N a finger; a Franka Hand holds 70 N).
                    # At the cell's ~13 N, S3 step 13's planned support let a joint below
                    # it break at u 1.00 under less than the press it was sized for (_diag_015).
SETTLE = 0.25       # s after the push-through before the snap: the press bent S3's supported
                    # corbel 2.4 mm down and a next-frame gate read it (_diag_016)
HAND_KEEP_OUT = 0.020  # m, planner keep-out between the support grip and the pressed brick
TOUCH = 0.002       # m: tips this close above the held brick's stud tops count as on them (contact
                    # measured 0.05-0.15 mm above nominal in dual_studfit_diag_008)


def overhanging(P, brick, placed):
    """True when `brick`, or any brick under it down to the baseplate, has a
    stud cell with nothing beneath it: the press then loads a joint that ties
    an overhang to the rest. S3 step 15 sits on the bridge from the overhang to
    the column and broke that 2-stud joint at u 1.03-1.13 unsupported
    (dual_studfit_diag_018/019); the planner's rigid model passed it."""
    by_id = {b[0]: b for b in placed}
    todo, seen = [brick], set()
    while todo:
        b = todo.pop()
        if b[0] in seen:
            continue
        seen.add(b[0])
        own, k = P.cells(b)
        if k == 0:
            continue
        under = set()
        for p in placed:
            pc, pk = P.cells(p)
            if pk == k - 1:
                under |= own & pc
        if under != own:
            return True
        todo.extend(by_id[sid] for sid in P.supports(b, placed))
    return False


def near_press_assign(P, BR, original):
    """bracing.assign for --stud-fit plans. The same safety test, plus a support
    forced where `overhanging`. Among grips that make the step safe, the one
    nearest the press (the brick's bottom centre), not the lowest predicted
    utilisation: S3 step 13's support gripped the pier 34 mm from the press and
    the corbel under it sagged 1.6 mm (dual_pyramid_s3_studfit_005). With no
    safe grip, the lowest utilisation as before."""
    def assign(brick, placed, strategy, safety_factor=BR.ST.SAFETY_FACTOR, unbraced=None,
               clearance=None, grip_N=BR.GRIP_N, mu=1.0, selector=None):
        if strategy != 'weakest_joint' or not placed:
            return original(brick, placed, strategy, safety_factor, unbraced, clearance, grip_N, mu, selector)
        r0 = unbraced or BR.predicted(brick, placed)
        forced = overhanging(P, brick, placed)
        if r0.s * safety_factor < 1.0 and not forced:
            return None
        joint_bricks = {b for j, u in r0.util.items() if u * safety_factor >= 1.0 for b in j if b is not None}
        cands = BR.candidates(brick, placed, clearance)
        if not cands:
            return BR._record(brick, placed, None, r0, strategy, 'no_feasible_grasp')
        pool = [c for c in cands if set(BR.gripped(placed, *c)) & joint_bricks] or cands
        _, py, pz = P.brick_pose(brick)
        scored = []
        for y, z in pool:
            br = BR._brace_obj(placed, y, z, grip_N, mu)
            r = BR.predicted(brick, placed, br)
            scored.append((r.s, y, z, br, r))
        safe = [c for c in scored if c[0] * safety_factor < 1.0]
        pick = (min(safe, key=lambda c: (round(math.hypot(c[1] - py, c[2] - pz), 4), c[0])) if safe
                else min(scored, key=lambda c: c[0]))
        _, y, z, br, r = pick
        return BR._record(brick, placed, (y, z, br, r), r0, strategy,
                          'near_press_overhang' if forced else 'near_press')
    return assign


@contextmanager
def planning(D):
    """Plan with near_press_assign (the planner imports bracing at call time)."""
    import bracing as BR
    saved = BR.assign, BR.HAND_CLEARANCE_Y
    BR.assign = near_press_assign(D.P, BR, saved[0])
    # The top press is a closed fist over the brick, and the run checks the real
    # hands for every support pose (brace_search), so the planner's 28 mm
    # keep-out is relaxed: at 28 mm the nearest grip to S3 step 13's press was
    # the pier, 32 mm away.
    BR.HAND_CLEARANCE_Y = HAND_KEEP_OUT
    try:
        yield
    finally:
        BR.assign, BR.HAND_CLEARANCE_Y = saved


def press_offset(grasp, brick, bricks, footprint, pitch):
    """World xy from the brick's centre to where the top press goes in.

    The planner's grip offset is its press point (stability.best_press_point,
    which the generated supports are sized for). Pressed at its centre
    instead, the overhanging S3 2x6 corbel bent its supported tower and seated
    3.25 mm low, 10.7 deg off (dual_pyramid_s3_studfit_004); pressed at its
    mated-stud centroid, S3 step 13 broke a joint below its support at u 1.01
    (dual_studfit_diag_014). A fallback grip's offset is only where the pads
    fit, so those press at the centroid of the studs they mate (zero on the
    baseplate or fully supported).
    """
    if 'grip_fallback' not in grasp:
        return np.asarray(grasp.get('tcp_offset_m', [0., 0.]), float)
    def cells(b):
        nx, ny = footprint(b['type'], b['yaw_index'])
        i, j, _ = b['grid_pos']
        return {(i + a, j + c) for a in range(nx) for c in range(ny)}
    own, k = cells(brick), brick['grid_pos'][2]
    mated = set()
    for b in bricks:
        if b['grid_pos'][2] == k - 1:
            mated |= own & cells(b)
    if not mated:
        return np.zeros(2)
    return (np.mean(sorted(mated), axis=0) - np.mean(sorted(own), axis=0)) * pitch


def needed(step, brick, footprint):
    """n x 8.9 N over the mating studs; a first-course brick mates its whole
    footprint with the baseplate, which the plan's mating_studs leaves out."""
    n = sum(int(k) for _, k in step['mating_studs']) or int(np.prod(footprint(brick['type'], brick['yaw_index'])))
    return n * F_INSERT


class StudFit:
    def __init__(self, cell, D, path=None):
        self.cell, self.D, self.path = cell, D, path
        self.held = {}       # bid -> (weld, needed N) while held on the stud tops
        self.peak = {}       # bid -> largest push measured while held, N
        self.events = []
        self.seat = {}       # bid -> weld relpose at the seat below the held pose
        self.seating = {}    # bid -> (welds, time) after the push-through, until the snap
        self.offset = {}     # bid -> press point from the brick centre, world xy
        self.press_t = {}    # bid -> time the tips reached its studs
        self.applied = {}    # bid -> largest hand press force applied, N
        self.wrench = np.zeros((2, 6))
        self.stiff = [False, False]
        model = getattr(cell, 'model', None)
        self.gains = None if model is None or not hasattr(model, 'joint_target_ke') else (
            model.joint_target_ke.numpy().copy(), model.joint_target_kd.numpy().copy())
        for view in cell.views.values():
            view.snap = self._wrap_snap(view.snap)

    def _wrap_snap(self, snap):
        def wrapped(s):
            bid = s['brick_id']
            if bid in self.held:
                self.release(bid, 'not_pushed_through')
            if bid in self.seating:
                welds, _ = self.seating.pop(bid)
                self._set([(w, w[2]) for w in welds], False)   # the gate's clutch welds take over
            return snap(s)
        return wrapped

    def update(self):
        cell, q = self.cell, self.cell.state_0.body_q.numpy()
        wrench = np.zeros((2, 6))
        for view in cell.views.values():
            if not view.active or view.b_step is None:
                continue
            step = view.plan['sequence'][view.b_step]
            bid = step['brick_id']
            welds = cell.welds[bid]
            if welds[0][3] or bid in self.peak and bid not in self.held:
                continue                      # welded, or already pushed through
            p, tgt = q[cell.body[bid]], cell.target[bid][0]
            if bid in self.held:
                self.centre(view.B, p[:2] + self.offset[bid])
                self.check(bid)
                if bid in self.held:
                    k = 'AB'.index(view.owner)
                    wrench[k, 2] = -self.press_force(bid, view.B, q[self.D.EE + k * cell.n_arm_bodies][2] - tgt[2])
                continue
            dz = p[2] - tgt[2]
            if dz <= STUD_H + BAND and np.hypot(*(p[:2] - tgt[:2])) < LATERAL:
                brick = next(b for b in view.plan['bricks'] if b['id'] == bid)
                self.hold(bid, welds, needed(step, brick, self.D.P.footprint), q, dz)
                self.offset[bid] = press_offset(step['grasp'], brick, view.plan['bricks'], self.D.P.footprint, self.D.P.PITCH)
                for move in view.B.moves:
                    if move.get('dwell'):
                        move['wait'] = lambda arm=view.B, m=move, b=bid: self.through(b) or arm.t >= m['dwell']
        if hasattr(cell, 'press_w') and (wrench != self.wrench).any():
            self.wrench = wrench
            cell.press_w.assign(wrench.astype(np.float32))
            # A supporting arm stiffens with the press it braces: at its soft gains the S3
            # tower leaned 4 deg under a 53 N press and stayed there (_diag_017).
            roles = getattr(cell, 'roles', {})
            pressing = [bool(wrench[k, 2]) for k in range(2)]
            self.stiffen([pressing[k] or (pressing[1 - k] and roles.get('AB'[k]) == 'supporter') for k in range(2)])

    def stiffen(self, on):
        if on == self.stiff or self.gains is None:
            return
        self.stiff = on
        ke, kd = (g.copy() for g in self.gains)
        for k in range(2):
            if on[k]:
                ke[9*k:9*k+7] *= STIFF
                kd[9*k:9*k+7] *= np.sqrt(STIFF)
        # Keep the current finger gains (set_grip_role changes them for supports).
        model = self.cell.model
        cur_ke, cur_kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
        for k in range(2):
            ke[9*k+7:9*k+9], kd[9*k+7:9*k+9] = cur_ke[9*k+7:9*k+9], cur_kd[9*k+7:9*k+9]
        model.joint_target_ke.assign(ke)
        model.joint_target_kd.assign(kd)
        self.cell.solver.notify_model_changed(self.D.newton.solvers.SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    @staticmethod
    def centre(arm, xy):
        """Put the closed-tip press over the brick where it is held. The arm adds its
        in-hand bias to every command, so the move targets the brick minus that bias.
        On the nominal target, a 1x1 tipped 23 deg (dual_studfit_diag_003); without the
        bias, the tips came down 4-6 mm off a 2x2's centre and slid off it (_010)."""
        for move in [arm.cur, *arm.moves]:
            if move is not None and move['phase'] == 'press':
                move['pos'] = np.r_[np.asarray(xy, float) - np.asarray(arm.bias, float)[:2], move['pos'][2]]

    def press_force(self, bid, arm, tcp_dz):
        """Downward hand force, N: ramps once the closed tips are on the held brick's studs."""
        D = self.D
        tips_on = tcp_dz <= STUD_H + D.P.BRICK_H + D.ex.STUD_HEIGHT + D.TIP_BELOW_TCP + TOUCH
        if arm.phase != 'press':
            self.press_t.pop(bid, None)
            return 0.
        # Only the seat move and its dwell press: a force left on while hovering sank the
        # soft-held hand 6 mm, and it closed around a 2x2 instead of above it (_012).
        if not tips_on or not (arm.cur or {}).get('contact'):
            return 0.
        t0 = self.press_t.setdefault(bid, self.cell.sim_time)
        force = min(1., (self.cell.sim_time - t0) / RAMP) * MARGIN * self.held[bid][1]
        self.applied[bid] = max(self.applied.get(bid, 0.), force)
        return force

    def _set(self, poses, on):
        """[(weld, relpose)] -> set and switch each weld on or off."""
        model = self.cell.model
        rp, en = model.equality_constraint_relpose.numpy(), model.equality_constraint_enabled.numpy()
        for weld, relpose in poses:
            rp[weld[0]] = relpose
            en[weld[0]] = on
        model.equality_constraint_relpose.assign(rp)
        model.equality_constraint_enabled.assign(en)
        self.cell.solver.notify_model_changed(self.D.newton.solvers.SolverNotifyFlags.CONSTRAINT_PROPERTIES)

    def hold(self, bid, welds, force, q, dz):
        wp, cell = self.D.wp, self.cell
        p = q[cell.body[bid]].astype(float)
        p[2] += max(0., STUD_H - dz)          # arrived inside the band within a frame: back to the stud tops

        def rel(weld, z, quat):
            sid = weld[1]
            T1 = (wp.transform_identity() if sid is None else
                  wp.transform(wp.vec3(*q[cell.body[sid], :3]), wp.quat(*q[cell.body[sid], 3:])))
            return wp.transform_multiply(wp.transform_inverse(T1), wp.transform(wp.vec3(p[0], p[1], z), wp.quat(*quat)))
        # Pushed down over the studs it comes out level: keep only the held yaw. A dropped
        # brick held at its landing tilt seated 6.7 deg off (dual_pyramid_s3_studfit_002).
        x, y, z, w = p[3:]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        level = (0., 0., np.sin(yaw / 2), np.cos(yaw / 2))
        self.seat[bid] = [(wd, rel(wd, cell.target[bid][0][2], level)) for wd in welds]
        self._set([(wd, rel(wd, p[2], p[3:])) for wd in welds], True)
        self.held[bid] = (welds, force)
        self.peak[bid] = 0.
        self.log(bid, 'held', dz_mm=round(float(dz) * 1000, 2), needed_N=round(force, 1))

    def check(self, bid):
        welds, force_needed = self.held[bid]
        rows = self.D.weld_rows(self.cell.solver)
        if any(w[0] not in rows for w in welds):
            return
        # The welds hold the brick up as hard as it is pressed down.
        push = sum(float(self.D.brick_wrench(self.cell.solver, self.cell.model, w[0], rows[w[0]],
                                             self.cell.body[bid])[0][2]) for w in welds)
        self.peak[bid] = max(self.peak[bid], push)
        if push >= force_needed:
            self.release(bid, 'pushed_through')

    def through(self, bid):
        """Pushed through, slid to the seat, and the structure has sprung back."""
        return bid in self.seating and self.cell.sim_time >= self.seating[bid][1] + SETTLE

    def release(self, bid, outcome):
        welds, force_needed = self.held.pop(bid)
        if outcome == 'pushed_through':
            self._set(self.seat[bid], True)
            self.seating[bid] = (welds, self.cell.sim_time)
        else:
            self._set([(w, w[2]) for w in welds], False)
        self.log(bid, outcome, peak_N=round(self.peak[bid], 1), needed_N=round(force_needed, 1),
                 hand_N=round(self.applied.get(bid, 0.), 1))

    def log(self, bid, event, **data):
        row = dict(brick=bid, event=event, time=round(self.cell.sim_time, 3), **data)
        self.events.append(row)
        print('stud fit: %s' % row, flush=True)
        if self.path is not None:
            self.path.write_text(json.dumps(self.events, indent=2) + '\n')
