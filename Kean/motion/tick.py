"""Play a planned free-space path on the cell tick. The last 30 mm stays Cartesian."""

import numpy as np

from Kean.motion.contact_segment import CONTACT_M
from Kean.motion.plan_curobo import GraphCaptured, Request

def angle_wrap(a, period=2 * np.pi):
    period = float(period)
    return -((-float(a) + period / 2.0) % period - period / 2.0)


def resolve_wrap(module):
    fn = getattr(module, "wrap", None)
    return fn if callable(fn) else angle_wrap


FREE = frozenset({"to feeder", "transport", "park", "abort", "pre-insert", "clearance"})
# Retract starts at the insertion contact pose; keep this escape Cartesian.
STATIC = frozenset({"park", "hold", "stage", "settle"})
TAIL_SPEED = 0.020


def planner_yaw(example, arm, yaw, wrap_angle):
    """Match Example.step's choice of symmetric wrist orientation exactly."""
    if getattr(example, "direct", False):
        return float(yaw)
    shift = getattr(example, "yaw_shift", None)
    if arm is example.B and shift is not None:
        return float(yaw + shift)
    base_yaw = arm.base[1]
    return float(base_yaw + wrap_angle(yaw - base_yaw, np.pi))


def needs_plan(move, ee, contact_m=CONTACT_M):
    if move.get("phase") not in FREE or move.get("contact") or move.get("rel") is not None:
        return False
    return float(np.linalg.norm(np.asarray(move["pos"], float) - np.asarray(ee, float))) > contact_m + 1e-9


class Slot:
    def __init__(self, name):
        self.name = name
        self.mode = "idle"
        self.future = None
        self.q = None
        self.v = None
        self.play_q = None
        self.play_v = None
        self.i = 0
        self.tail = None
        self.goal = None
        self.move = None
        self.request = None
        self.other_q = None


class Tick:
    def __init__(self, planner, zone=None, fps=60, world=None, clearance=None, q_of=None, *, wrap_angle, resample_trajectory, strict=False, carried=None):
        self.strict = strict
        self.carried = carried
        self.resample_trajectory = resample_trajectory
        self.wrap_angle = wrap_angle
        self.planner = planner
        self.zone = zone
        self.fps = fps
        self.world = world
        self.clearance = clearance
        self.q_of = q_of or _q_of
        self.slots = {}
        self.example = None
        self.plans = 0
        self.plan_failures = []
        self.workarounds = []
        self.captures = 0
        self.segments = []
        self.min_gap = None
        self._vel_assign = None
        self._velocity_owned = set()

    def attach(self, example):
        self.example = example
        for arm in (example.A, example.B):
            self.slots[arm] = Slot(arm.name[0])
            orig = arm.update
            arm.update = lambda dt, ee, arm=arm, orig=orig: self._update(arm, orig, dt, ee)
        pos = example.control.joint_target_pos
        orig_assign = pos.assign

        def assign(tgt, _orig=orig_assign):
            self._write(tgt)
            return _orig(tgt)

        pos.assign = assign
        vel = getattr(example.control, "joint_target_vel", None)
        if vel is not None:
            self._vel_assign = vel.assign
        return self

    def other(self, arm):
        for candidate in self.slots:
            if candidate is not arm:
                return candidate
        return None

    def _moving(self, arm):
        slot = self.slots.get(arm)
        if slot is not None and slot.mode in ("waiting", "playing"):
            return True
        return arm.cur is not None and arm.phase not in STATIC

    def _in_zone(self, arm):
        if self.zone is None or not self._moving(arm):
            return False
        pts = [arm.ee]
        if arm.cur is not None:
            pts.append(arm.cur["pos"])
        slot = self.slots.get(arm)
        if slot is not None and slot.goal is not None:
            pts.append(slot.goal)
        return any(self.zone.contains(p) for p in pts)

    def _blocked(self, arm, target, ee):
        if self.zone is None:
            return False
        other = self.other(arm)
        if other is None or not self._in_zone(other):
            return False
        return self.zone.contains(target) or self.zone.contains(ee)

    def _hold(self, arm):
        p, yaw, g, tilt = arm.cmd
        return (np.asarray(p, float) + arm.bias, yaw, g, tilt)

    def _begin(self, arm, move, ee):
        slot = self.slots[arm]
        goal = np.asarray(move["pos"], float) + arm.bias
        boxes, other_q = (None, None) if self.world is None else self.world(self.example, slot.name)
        request = Request(
            slot.name, self.q_of(self.example, arm), ee, goal, np.array([0.0, 0.0, 0.0, 1.0]),
            zone_clear=True, yaw=planner_yaw(self.example, arm, move["yaw"], self.wrap_angle), boxes=boxes, other_q=other_q, full_path=self.strict,
            alternate_goals=[np.asarray(p, float) + arm.bias for p in move.get("alternate_goals", [])],
            grip=float(move["grip"]), carried=None if self.carried is None else self.carried(self.example, slot.name))
        slot.other_q = None if other_q is None else np.asarray(other_q[:7], float).copy()
        slot.future = self.planner.submit(request)
        slot.mode = "waiting"
        slot.goal = goal
        slot.move = move
        slot.request = request
        slot.q = request.q[:7].copy()
        slot.v = np.zeros(7)
        self.plans += 1
        self.segments.append({"arm": slot.name, "phase": move["phase"], "planned": True})
        p, _, _, _ = arm.cmd
        arm.cmd = (np.asarray(p, float), float(move["yaw"]), float(move["grip"]), np.asarray(move["tilt"], float))

    def _seed_ik(self, arm):
        # Assign in place so the captured IK graph keeps the same buffer.
        seed = self.example.ik_q.numpy()
        k = 0 if self.slots[arm].name == "A" else 1
        seed[k, :7] = self.q_of(self.example, arm)[:7]
        self.example.ik_q.assign(seed)

    def _release(self, arm, slot, ee):
        here = np.asarray(ee, float)
        # Arm commands exclude bias; planner poses and measured EE include it.
        start = here - arm.bias
        goal = np.asarray(arm.cur["pos"], float).copy()
        yaw, grip, tilt = arm.cur["yaw"], arm.cur["grip"], arm.cur["tilt"]
        arm.cmd = (start.copy(), yaw, grip, np.asarray(tilt, float).copy())
        arm.start = arm.cmd
        arm.t = 0.0
        arm.cur["pos"] = goal.copy()
        dist = float(np.linalg.norm(goal - start))
        arm.cur["dur"] = max(1.0 / self.fps, dist / TAIL_SPEED)
        slot.mode = "idle"
        slot.q = None
        slot.v = None
        slot.goal = None
        slot.move = arm.cur

    def _service(self, arm, orig, dt, ee):
        slot = self.slots[arm]
        if self.strict and slot.other_q is not None:
            other = self.other(arm)
            if np.max(np.abs(self.q_of(self.example, other)[:7] - slot.other_q)) > 0.02:
                return self._stop(arm, "motion_obstacle_moved")
        if slot.mode == "waiting":
            if not slot.future.done():
                return self._hold(arm)
            try:
                result = slot.future.result()
            except GraphCaptured:
                self.captures += 1
                raise
            except Exception as exc:
                self.plan_failures.append({"arm": slot.name, "phase": getattr(arm, "phase", None), "error": str(exc),
                                           "step": getattr(self.example, "b_step", None),
                                           "request": slot.request.snapshot()})
                if self.strict:
                    print("planner error %s %s: %s" % (slot.name, arm.phase, exc), flush=True)
                    return self._stop(arm, "motion_plan_failed")
                print("plan fallback %s %s: %s" % (slot.name, getattr(arm, "phase", ""), exc), flush=True)
                slot.mode = "idle"
                slot.q = None
                slot.v = None
                slot.move = arm.cur
                return orig(dt, ee)
            if result.get("joint_pos") is None:
                if self.strict:
                    return self._stop(arm, "motion_plan_empty")
                slot.mode = "idle"
                slot.q = None
                slot.v = None
                slot.move = arm.cur
                return orig(dt, ee)
            if self.strict and result.get("goal") is not None:
                slot.goal = np.asarray(result["goal"], float)
                arm.cur["pos"] = slot.goal - arm.bias
            if result.get("proposal_mode") == "full-sphere-filter":
                self.workarounds.append({"step": getattr(self.example, "b_step", None),
                                         "arm": slot.name, "phase": arm.phase,
                                         "method": "full-sphere-filter"})
            pos = np.asarray(result["joint_pos"], float)
            vel = result.get("joint_vel")
            if vel is None:
                vel = np.zeros_like(pos)
            if len(pos) < 2 or not result.get("dt"):
                slot.play_q, slot.play_v = pos, np.asarray(vel)
            else:
                slot.play_q, slot.play_v = self.resample_trajectory(
                    (pos, np.asarray(vel), result["dt"]), 1.0, fps=self.fps)
            slot.i = 0
            slot.tail = result["cartesian"]
            slot.mode = "playing"
        if slot.i < len(slot.play_q):
            slot.q = slot.play_q[slot.i]
            slot.v = None if slot.play_v is None else slot.play_v[slot.i]
            slot.i += 1
            if self.clearance is not None:
                gap = float(self.clearance(self.example, slot.name, slot.q))
                self.min_gap = gap if self.min_gap is None else min(self.min_gap, gap)
            return self._hold(arm)
        if self.strict:
            # Stay at the planned endpoint until measured joints have arrived.
            # Do not hand the endpoint back to unconstrained Cartesian IK.
            slot.i += 1
            if np.max(np.abs(self.q_of(self.example, arm)[:7] - slot.q[:7])) > 0.02:
                if slot.i > len(slot.play_q) + 2 * self.fps:
                    return self._stop(arm, "motion_tracking_failed")
                return self._hold(arm)
            move = arm.cur
            arm.cmd = (np.asarray(slot.goal, float) - arm.bias, move["yaw"], move["grip"], move["tilt"])
            arm.cur = None
            slot.mode = "idle"
            slot.q = slot.v = None
            if move["then"]:
                move["then"]()
            return self._hold(arm)
        self._release(arm, slot, ee)
        return orig(dt, ee)

    def _stop(self, arm, reason):
        slot = self.slots[arm]
        slot.mode = "stopped"
        slot.q = self.q_of(self.example, arm)[:7].copy()
        slot.v = np.zeros(7)
        self.example.fail(reason)
        print("motion stop %s %s: %s" % (slot.name, arm.phase, reason), flush=True)
        return self._hold(arm)

    def _update(self, arm, orig, dt, ee):
        arm.ee = np.asarray(ee, float).copy()
        slot = self.slots[arm]
        if slot.mode == "stopped":
            return self._hold(arm)
        if slot.mode in ("waiting", "playing"):
            self._seed_ik(arm)
            return self._service(arm, orig, dt, ee)
        pending = arm.cur if arm.cur is not None else (arm.moves[0] if arm.moves else None)
        if pending is not None and pending is not slot.move and (needs_plan(pending, ee) or (self.strict and pending.get("phase") in FREE and not pending.get("contact") and pending.get("rel") is None)):
            if (self.strict and self.other(arm) is not None and self._moving(self.other(arm))) or self._blocked(arm, pending["pos"], ee):
                return self._hold(arm)
            if arm.cur is None:
                orig(0.0, ee)
            self._begin(arm, arm.cur, ee)
            return self._hold(arm)
        return orig(dt, ee)

    def _write(self, tgt):
        vel = None
        for arm, slot in self.slots.items():
            k = 0 if slot.name == "A" else 1
            joints = slice(9 * k, 9 * k + 7)
            if slot.q is not None:
                tgt[joints] = slot.q
            # Feed-forward belongs only to playback. Clear it once on handoff,
            # including failures, without touching the fingers or the other arm.
            active = slot.q is not None
            if self._vel_assign is not None and (active or arm in self._velocity_owned):
                if vel is None:
                    vel = self.example.control.joint_target_vel.numpy()
                vel[joints] = slot.v if active and slot.v is not None else 0.0
                if active:
                    self._velocity_owned.add(arm)
                else:
                    self._velocity_owned.discard(arm)
        if self.strict and self.clearance is not None:
            # Guard even the shared Cartesian contact/escape phases against
            # arm-arm collisions. Intended finger-brick contacts are unaffected.
            measured = self.example.state_0.joint_q.numpy()[:18]
            n = max(1, int(np.ceil(np.max(np.abs(tgt[:18] - measured)) / 0.02)))
            for t in np.linspace(0., 1., n + 1):
                q = measured + t * (tgt[:18] - measured)
                a = self.clearance.spheres("A", q[:9])
                b = self.clearance.spheres("B", q[9:18])
                gap = self.clearance.probe.clearance_spheres(a, b)
                if not np.isfinite(gap) or gap < 0.001:
                    self.example.fail("arm_clearance")
                    detail = {"error": "arm clearance %.2f mm" % (gap * 1000),
                              "step": getattr(self.example, "b_step", None),
                              "phases": {a.name: a.phase for a in self.slots},
                              "modes": {a.name: s.mode for a, s in self.slots.items()},
                              "measured_q": measured.tolist(), "target_q": tgt[:18].tolist(),
                              "fraction": float(t)}
                    self.plan_failures.append(detail)
                    print("motion guard: %s" % detail, flush=True)
                    tgt[:18] = measured
                    if self._vel_assign is not None:
                        vel = self.example.control.joint_target_vel.numpy()
                        vel[:18] = 0
                    break
        if vel is not None:
            self._vel_assign(vel)

    def report(self):
        return {
            "strict": self.strict,
            "workarounds": self.workarounds,
            "failures": self.plan_failures,
            "plans": self.plans,
            "plan_failures": len(self.plan_failures),
            "captures": self.captures,
            "segments": len(self.segments),
            "min_clearance_m": self.min_gap,
            "clearance_scope": "executed joints during planned playback; full spheres, measured fingers",
        }


def _q_of(example, arm):
    k = 0 if arm.name[0] == "A" else 1
    return np.asarray(example.state_0.joint_q.numpy()[9 * k:9 * k + 7], float)
