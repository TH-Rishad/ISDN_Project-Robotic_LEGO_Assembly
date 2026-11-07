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
That arm's joint stiffness is raised STIFF x while it presses: at the cell's
400 N m/rad the hand skated 4 mm off a 2x2 under the push (dual_studfit_diag_009).

ponytail: a breakaway threshold followed by a guided slide, not a force spread
over the 1.7 mm stroke; make it a band of interference contacts if the stroke
itself matters. A free slide let the press twist a 1x1 5 deg on its single
stud (dual_studfit_diag_006). The hold reuses the brick's first clutch weld and restores its
planned relpose on release. The recorder and joint breaker read only the
clutch flags (w[3]), so a hold never counts as a placement.
"""
import json

import numpy as np

F_INSERT = 8.9      # N per mating stud (planner.F_INSERT_PER_STUD)
STUD_H = 0.0017     # m, stud height: the band in which the studs interfere
BAND = 0.0003       # m above the stud tops at which the hold engages
LATERAL = 0.002     # m, same lateral window as the snap gate
RAMP = 0.5          # s to full press force
MARGIN = 1.3        # press force over the push needed (newton_fit.press_force)
STIFF = 5.0         # arm joint stiffness gain while pressing (damping scales by its root)
TOUCH = 0.002       # m: tips this close above the held brick's stud tops count as on them (contact
                    # measured 0.05-0.15 mm above nominal in dual_studfit_diag_008)


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
        self.seating = {}    # bid -> (weld, frame) after the push-through, until the snap
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
                weld, _ = self.seating.pop(bid)
                self._set(weld, weld[2], False)   # the gate's clutch welds take over
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
            weld = cell.welds[bid][0]
            if weld[3] or bid in self.peak and bid not in self.held:
                continue                      # welded, or already pushed through
            p, tgt = q[cell.body[bid]], cell.target[bid][0]
            if bid in self.held:
                self.centre(view.B, p[:2])
                self.check(bid)
                if bid in self.held:
                    k = 'AB'.index(view.owner)
                    wrench[k, 2] = -self.press_force(bid, view.B, q[self.D.EE + k * cell.n_arm_bodies][2] - tgt[2])
                continue
            dz = p[2] - tgt[2]
            if dz <= STUD_H + BAND and np.hypot(*(p[:2] - tgt[:2])) < LATERAL:
                brick = next(b for b in view.plan['bricks'] if b['id'] == bid)
                self.hold(bid, weld, needed(step, brick, self.D.P.footprint), q, dz)
                for move in view.B.moves:
                    if move.get('dwell'):
                        move['wait'] = lambda arm=view.B, m=move, b=bid: self.through(b) or arm.t >= m['dwell']
        if hasattr(cell, 'press_w') and (wrench != self.wrench).any():
            self.wrench = wrench
            cell.press_w.assign(wrench.astype(np.float32))
            self.stiffen([bool(wrench[k, 2]) for k in range(2)])

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

    def _set(self, weld, relpose, on):
        model = self.cell.model
        rp, en = model.equality_constraint_relpose.numpy(), model.equality_constraint_enabled.numpy()
        rp[weld[0]] = relpose
        en[weld[0]] = on
        model.equality_constraint_relpose.assign(rp)
        model.equality_constraint_enabled.assign(en)
        self.cell.solver.notify_model_changed(self.D.newton.solvers.SolverNotifyFlags.CONSTRAINT_PROPERTIES)

    def hold(self, bid, weld, force, q, dz):
        wp, cell = self.D.wp, self.cell
        p = q[cell.body[bid]].astype(float)
        p[2] += max(0., STUD_H - dz)          # arrived inside the band within a frame: back to the stud tops
        sid = weld[1]
        T1 = (wp.transform_identity() if sid is None else
              wp.transform(wp.vec3(*q[cell.body[sid], :3]), wp.quat(*q[cell.body[sid], 3:])))
        rel = lambda z: wp.transform_multiply(wp.transform_inverse(T1), wp.transform(wp.vec3(p[0], p[1], z), wp.quat(*p[3:])))
        self.seat[bid] = rel(cell.target[bid][0][2])
        self._set(weld, rel(p[2]), True)
        self.held[bid] = (weld, force)
        self.peak[bid] = 0.
        self.log(bid, 'held', dz_mm=round(float(dz) * 1000, 2), needed_N=round(force, 1))

    def check(self, bid):
        weld, force_needed = self.held[bid]
        rows = self.D.weld_rows(self.cell.solver)
        if weld[0] not in rows:
            return
        force, _ = self.D.brick_wrench(self.cell.solver, self.cell.model, weld[0], rows[weld[0]], self.cell.body[bid])
        push = float(force[2])               # the weld holds the brick up as hard as it is pressed down
        self.peak[bid] = max(self.peak[bid], push)
        if push >= force_needed:
            self.release(bid, 'pushed_through')

    def through(self, bid):
        """Pushed through and slid to the seat for at least one physics frame."""
        return bid in self.seating and self.cell.frame > self.seating[bid][1]

    def release(self, bid, outcome):
        weld, force_needed = self.held.pop(bid)
        if outcome == 'pushed_through':
            self._set(weld, self.seat[bid], True)
            self.seating[bid] = (weld, self.cell.frame)
        else:
            self._set(weld, weld[2], False)
        self.log(bid, outcome, peak_N=round(self.peak[bid], 1), needed_N=round(force_needed, 1),
                 hand_N=round(self.applied.get(bid, 0.), 1))

    def log(self, bid, event, **data):
        row = dict(brick=bid, event=event, time=round(self.cell.sim_time, 3), **data)
        self.events.append(row)
        print('stud fit: %s' % row, flush=True)
        if self.path is not None:
            self.path.write_text(json.dumps(self.events, indent=2) + '\n')
