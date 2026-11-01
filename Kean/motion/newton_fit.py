"""Opt-in closed loop for a tight insert on the Newton cell. Default off.

KEAN_NEWTON_FIT=1 replaces the waypoint band on a critical socket with a
per-tick impedance command, an axial clutch wrench, and a 0.3 mm seat gate.
Kean/motion/wriggle.py stays the graded path and is not used for that step
while this flag is set. No policy is loaded. consider_policy names a
twin-trained file only after a Newton slot probe is within 10 points of the
twin; it does not open the file.

The graded acceptance path never enters this module's simulate wrapper.
That wrapper mirrors dual_arm_sim.simulate and adds the clutch wrench after
clear_forces, and it runs only while a fit insert is active.
"""

import math
import os
from types import MethodType

import numpy as np

from Kean.motion import wriggle

F_INSERT = 8.9
F_BREAK = 11.3
SEAT_M = 0.0003
STUD_H = 0.0018
PRESS_V = 0.008
SEARCH_V = 0.003
SPIRAL_R = 0.002
SEARCH_N = 8.0
HAND_MASS = 1.5
HAND_I = 0.004
ZETA = 1.0
KP = np.array([1200.0, 1200.0, 600.0])
KP_ROT = np.array([150.0, 150.0, 10.0])
ROCK = math.radians(1.5)
ROCK_HZ = 2.0
HOLD_S = 10.0
APPROACH = 0.012
TRANSFER_POINTS = 10.0
FLAG = "KEAN_NEWTON_FIT"


class TransferRefused(RuntimeError):
    pass


def enabled():
    return os.environ.get(FLAG, "0") == "1"


def seated(lateral_m, dz_m):
    return float(lateral_m) <= SEAT_M and abs(float(dz_m)) <= SEAT_M


def cap_descent(dz, dt, vmax=PRESS_V):
    return min(max(float(dz), 0.0), float(vmax) * float(dt))


def damping(k, inertia):
    return 2.0 * ZETA * np.sqrt(np.asarray(k, float) * float(inertia))


def impedance(err, vel, rot_err, ang_vel, press=0.0, mass=HAND_MASS):
    err = np.asarray(err, float)
    vel = np.asarray(vel, float)
    rot_err = np.asarray(rot_err, float)
    ang_vel = np.asarray(ang_vel, float)
    d = damping(KP, mass)
    dr = damping(KP_ROT, HAND_I)
    force = KP * err - d * vel
    force[2] -= float(press)
    torque = KP_ROT * rot_err - dr * ang_vel
    return np.concatenate([force, torque])


def joint_torque(jacobian, wrench):
    return np.asarray(jacobian, float).T @ np.asarray(wrench, float)


def clutch_axial(n_studs, dz_m, extract_n=0.0, released=False):
    """Upward force on the brick, and whether the connection has broken.

    In the stud band the clutch holds against insertion at n * 8.9 N.
    Once seated, an upward load above n * 11.3 N lets go. Extraction is
    the higher limit.
    """
    n = int(n_studs)
    if released or n <= 0:
        return 0.0, True
    dz = float(dz_m)
    if SEAT_M < dz <= STUD_H + 1e-12:
        return n * F_INSERT, False
    if abs(dz) <= SEAT_M and float(extract_n) > n * F_BREAK:
        return 0.0, True
    return 0.0, False


def spiral(t, radius=SPIRAL_R, speed=SEARCH_V):
    s = float(speed) * float(t)
    a = 0.001 / (2.0 * math.pi)
    th = math.sqrt(2.0 * s / a) if s > 0.0 else 0.0
    r = min(a * th, float(radius))
    return r * math.cos(th), r * math.sin(th)


def rock(t, amp=ROCK, hz=ROCK_HZ):
    w = float(amp) * math.sin(2.0 * math.pi * float(hz) * float(t))
    return w, 0.7 * w


def press_force(n_studs, phase, t):
    if phase == "search":
        return SEARCH_N
    return min(1.0, float(t) / 0.5) * 1.3 * int(n_studs) * F_INSERT


def closed_target(measured, target, t, dt, centre, phase):
    measured = np.asarray(measured, float).copy()
    target = np.asarray(target, float).copy()
    z_rel = measured[2] - target[2]
    if centre is None:
        centre = measured.copy()
    if phase == "descend" and z_rel <= STUD_H + 0.003:
        phase, centre, t = "search", measured.copy(), 0.0
    if phase == "press" or z_rel <= SEAT_M:
        phase = "press"
        goal = target.copy()
        goal[2] = measured[2] - cap_descent(z_rel, dt)
        return goal, np.zeros(3), phase, centre, t + dt
    if phase == "descend":
        goal = target.copy()
        vmax = 0.020 if z_rel > STUD_H + 0.003 else PRESS_V
        goal[2] = measured[2] - cap_descent(z_rel, dt, vmax)
        return goal, np.zeros(3), phase, centre, t + dt
    off = spiral(t)
    goal = np.array(centre, float).copy()
    goal[0] += off[0]
    goal[1] += off[1]
    goal[2] = measured[2] - cap_descent(max(z_rel, 0.0), dt)
    return goal, np.array([*rock(t), 0.0]), phase, centre, t + dt


def points(seated_n, n):
    if int(n) <= 0:
        return None
    return 100.0 * float(seated_n) / float(n)


def transfer_ok(twin_seated, twin_n, newton_seated, newton_n, tol=TRANSFER_POINTS):
    a, b = points(twin_seated, twin_n), points(newton_seated, newton_n)
    if a is None or b is None:
        return False
    return abs(a - b) <= float(tol)


def consider_policy(path, twin_seated, twin_n, newton_seated, newton_n):
    if not transfer_ok(twin_seated, twin_n, newton_seated, newton_n):
        raise TransferRefused(
            "Newton slot probe is not within %g points of the twin; policy not loaded" % TRANSFER_POINTS)
    return path


def studs_of(step, D, brick):
    mating = step.get("mating_studs") if isinstance(step, dict) else None
    if mating:
        return len(mating)
    P = getattr(D, "P", None)
    if P is not None and brick is not None:
        nx, ny = P.footprint(brick["type"], brick["yaw_index"])
        return int(nx * ny)
    return 1


def _collapse(ex, step):
    moves = ex.B.moves
    inserts = [i for i, m in enumerate(moves) if m["phase"] == "insert"]
    if not inserts:
        return None
    first, last = inserts[0], inserts[-1]
    final = np.array(moves[last]["pos"], float)
    snap = moves[last].get("then")
    yaw, grip = moves[first]["yaw"], moves[first]["grip"]
    approach = wriggle._move("insert", final + [0.0, 0.0, APPROACH], yaw, grip, 0.5)
    hold = wriggle._move("insert", final, yaw, grip, HOLD_S, then=snap)
    moves[first:last + 1] = [approach, hold]
    return final


class Fit:
    def __init__(self, example, D):
        self.example = example
        self.D = D
        self.armed = False
        self.saw_insert = False
        self.n_studs = 1
        self.released = False
        self.t = 0.0
        self.centre = None
        self.phase = "descend"
        self.target = None
        self.clutch_up = 0.0
        self.tau = None
        self.jacobian = None
        self.wrench = np.zeros(6)
        self.brick_body = None
        self.support_body = None

    def arm(self, step, target):
        bricks = {b["id"]: b for b in self.example.plan["bricks"]}
        bid = step["brick_id"]
        self.n_studs = studs_of(step, self.D, bricks.get(bid))
        self.target = np.array(target, float)
        self.armed = True
        self.saw_insert = False
        self.released = False
        self.t = 0.0
        self.centre = None
        self.phase = "descend"
        self.clutch_up = 0.0
        self.tau = None
        bodies = getattr(self.example, "body", None) or {}
        self.brick_body = bodies.get(bid)
        support = step.get("support_id")
        self.support_body = bodies.get(support) if support else None

    def override(self, dt, ee, cmd):
        pos, yaw, grip, tilt = cmd
        goal, rock_tilt, self.phase, self.centre, self.t = closed_target(
            ee, self.target, self.t, dt, self.centre, self.phase)
        self.saw_insert = True
        self._sense(ee, goal, rock_tilt)
        self.write_joint_f()
        return goal, yaw, grip, rock_tilt if self.phase == "search" else tilt

    def _sense(self, ee, goal, rot):
        dz = float(ee[2] - self.target[2])
        lateral = float(np.linalg.norm(np.asarray(ee[:2], float) - self.target[:2]))
        gate = getattr(self.example, "gate", None)
        if callable(gate) and getattr(self.example, "b_step", None) is not None:
            _, lateral_mm, dz_mm, _, _, _, _ = gate(self._bid())
            lateral, dz = lateral_mm / 1000.0, dz_mm / 1000.0
        self.clutch_up, self.released = clutch_axial(self.n_studs, dz, 0.0, self.released)
        press = press_force(self.n_studs, self.phase, self.t)
        self.wrench = impedance(goal - np.asarray(ee, float), np.zeros(3), rot, np.zeros(3), press)
        if self.jacobian is not None:
            self.tau = joint_torque(self.jacobian, self.wrench)
        if not seated(lateral, dz):
            self.wrench[2] -= self.clutch_up

    def _bid(self):
        step = self.example.plan["sequence"][self.example.b_step]
        return step["brick_id"]

    def write_joint_f(self):
        ctrl = getattr(self.example, "control", None)
        jf = getattr(ctrl, "joint_f", None)
        if jf is None or self.tau is None:
            return
        arr = np.array(jf.numpy(), float)
        arr[9:16] = self.tau
        jf.assign(arr)

    def body_wrenches(self):
        f = float(self.clutch_up)
        if abs(f) < 1e-9:
            return []
        rows = []
        if self.brick_body is not None:
            rows.append((int(self.brick_body), (0.0, 0.0, f, 0.0, 0.0, 0.0)))
        if self.support_body is not None:
            rows.append((int(self.support_body), (0.0, 0.0, -f, 0.0, 0.0, 0.0)))
        return rows


def _launch_clutch(example, D, fit, wp):
    rows = fit.body_wrenches()
    if not rows:
        return
    bodies = wp.array([b for b, _ in rows], dtype=wp.int32)
    wrench = wp.zeros(len(rows), dtype=wp.spatial_vector)
    for i, (_, w) in enumerate(rows):
        wrench[i] = wp.spatial_vector(*w)
    wp.launch(D.add_fixture_wrench, dim=len(rows), inputs=[bodies, wrench, example.state_0.body_f])


def _injecting_simulate(example, D, fit, original):
    needed = ("pipeline", "state_0", "state_1", "solver", "contacts", "control", "substeps", "viewer", "sim_dt")
    if not all(hasattr(example, name) for name in needed):
        fit.write_joint_f()
        return original()
    try:
        import warp as wp
    except ImportError:
        fit.write_joint_f()
        return original()
    example.pipeline.collide(example.state_0, example.contacts)
    fit.write_joint_f()
    for _ in range(example.substeps):
        example.state_0.clear_forces()
        example.viewer.apply_forces(example.state_0)
        if getattr(example, "fx", None):
            wp.launch(D.add_fixture_wrench, dim=len(example.fx_body),
                      inputs=[example.fx_body, example.fx_w, example.state_0.body_f])
        _launch_clutch(example, D, fit, wp)
        example.solver.step(example.state_0, example.state_1, example.control, example.contacts, example.sim_dt)
        example.state_0, example.state_1 = example.state_1, example.state_0


def install(example, D):
    if not enabled():
        return None
    fit = Fit(example, D)
    example.newton_fit = fit
    P = D.P
    original = example.queue_place

    def queue_place(ex, step):
        original(step)
        if not wriggle.needs_wriggle(ex, step, P):
            fit.armed = False
            return
        target = _collapse(ex, step)
        if target is not None:
            fit.arm(step, target)

    example.queue_place = MethodType(queue_place, example)
    arm = example.B
    orig_update = arm.update

    def update(dt, ee, _orig=orig_update):
        cmd = _orig(dt, ee)
        if fit.armed and arm.phase == "insert":
            return fit.override(dt, ee, cmd)
        return cmd

    arm.update = update
    if callable(getattr(example, "gate", None)):
        orig_gate = example.gate

        def gate(bid, _orig=orig_gate):
            q, lateral, dz, dyaw, tilt, ok, screen = _orig(bid)
            if fit.armed and not seated(lateral / 1000.0, dz / 1000.0):
                ok = False
            return q, lateral, dz, dyaw, tilt, ok, screen

        example.gate = gate
    if callable(getattr(example, "snap", None)):
        orig_snap = example.snap

        def snap(s, _orig=orig_snap):
            try:
                return _orig(s)
            finally:
                fit.armed = False

        example.snap = snap
    if callable(getattr(example, "simulate", None)):
        orig_sim = example.simulate

        def simulate(_orig=orig_sim):
            if not (fit.armed and arm.phase == "insert"):
                return _orig()
            return _injecting_simulate(example, D, fit, _orig)

        example.simulate = simulate
    if callable(getattr(example, "step", None)):
        orig_step = example.step

        def step(_orig=orig_step):
            graph = getattr(example, "graph", None)
            bypass = fit.armed and arm.phase == "insert" and graph is not None
            if bypass:
                example.graph = None
            try:
                return _orig()
            finally:
                if bypass:
                    example.graph = graph

        example.step = step
    return fit
