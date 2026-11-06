"""Opt-in stud resistance for the dual builder (scripts/dual_build.py --stud-fit).

The cell's stud proxies clear a brick's walls by 0.6 mm a side, so a brick
seats under its own weight and the clutch is only the snap gate. With this
on, a brick reaching the stud tops of its support is held there by a weld at
its current pose, standing in for the stud interference, until the measured
push through that weld reaches n x 8.9 N (planner F_INSERT_PER_STUD over its
mating studs). Then it lets go and the brick may seat. A brick that is never
pushed through stays on the stud tops and misses the gate (dz ~1.7 mm >
1.5 mm).

ponytail: a breakaway threshold followed by a free slide, not a force spread
over the 1.7 mm stroke; make it a band of interference contacts if the stroke
itself matters. The hold reuses the brick's first clutch weld and restores its
planned relpose on release. The recorder and joint breaker read only the
clutch flags (w[3]), so a hold never counts as a placement.
"""
import json

import numpy as np

F_INSERT = 8.9      # N per mating stud (planner.F_INSERT_PER_STUD)
STUD_H = 0.0017     # m, stud height: the band in which the studs interfere
BAND = 0.0003       # m above the stud tops at which the hold engages
LATERAL = 0.002     # m, same lateral window as the snap gate


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
        for view in cell.views.values():
            view.snap = self._wrap_snap(view.snap)

    def _wrap_snap(self, snap):
        def wrapped(s):
            if s['brick_id'] in self.held:
                self.release(s['brick_id'], 'not_pushed_through')
            return snap(s)
        return wrapped

    def update(self):
        cell, q = self.cell, None
        for view in cell.views.values():
            if not view.active or view.b_step is None:
                continue
            step = view.plan['sequence'][view.b_step]
            bid = step['brick_id']
            weld = cell.welds[bid][0]
            if weld[3] or bid in self.peak and bid not in self.held:
                continue                      # welded, or already pushed through
            if bid in self.held:
                self.check(bid)
                continue
            q = cell.state_0.body_q.numpy() if q is None else q
            p, tgt = q[cell.body[bid]], cell.target[bid][0]
            dz = p[2] - tgt[2]
            if dz <= STUD_H + BAND and np.hypot(*(p[:2] - tgt[:2])) < LATERAL:
                brick = next(b for b in view.plan['bricks'] if b['id'] == bid)
                self.hold(bid, weld, needed(step, brick, self.D.P.footprint), q, dz)

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
        T2 = wp.transform(wp.vec3(*p[:3]), wp.quat(*p[3:]))
        sid = weld[1]
        T1 = (wp.transform_identity() if sid is None else
              wp.transform(wp.vec3(*q[cell.body[sid], :3]), wp.quat(*q[cell.body[sid], 3:])))
        self._set(weld, wp.transform_multiply(wp.transform_inverse(T1), T2), True)
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

    def release(self, bid, outcome):
        weld, force_needed = self.held.pop(bid)
        self._set(weld, weld[2], False)
        self.log(bid, outcome, peak_N=round(self.peak[bid], 1), needed_N=round(force_needed, 1))

    def log(self, bid, event, **data):
        row = dict(brick=bid, event=event, time=round(self.cell.sim_time, 3), **data)
        self.events.append(row)
        print('stud fit: %s' % row, flush=True)
        if self.path is not None:
            self.path.write_text(json.dumps(self.events, indent=2) + '\n')
