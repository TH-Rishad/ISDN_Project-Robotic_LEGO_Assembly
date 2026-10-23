"""Free-space plan for either arm. Warm up before the loop. Do not plan the last 30 mm."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import importlib

from Kean.motion.contact_segment import sample, split

_planner = importlib.import_module("Sakib.brickassembly.planner")
PRE_Z = float(getattr(_planner, "PRE_INSERTION_DZ", 0.025))
TIME_SCALE = 0.5


class NotWarmed(RuntimeError):
    pass


class GraphCaptured(RuntimeError):
    pass


class PlannerUnavailable(RuntimeError):
    pass


class ZoneBusy(RuntimeError):
    pass


def same_start(planned, current, tol=0.5):
    """A flipped IK seed is not a plan from here."""
    return float(np.max(np.abs(np.asarray(planned, float).reshape(-1)[:7] - np.asarray(current, float).reshape(-1)[:7]))) <= tol


def pre_insert(target, dz=PRE_Z):
    pose = np.asarray(target, float).copy()
    pose[2] += dz
    return pose


def _scale(joint_pos, dt, time_scale):
    stretch = 1.0 / time_scale
    dt_out = dt * stretch
    if len(joint_pos) < 2:
        return np.zeros_like(joint_pos), dt_out
    vel = np.diff(joint_pos, axis=0) / dt_out
    return np.vstack([vel, vel[-1]]), dt_out


class GraphGuard:
    def __init__(self):
        self.warm = False
        self.after = 0

    def note(self, n):
        n = int(n)
        if self.warm and n:
            self.after += n
            raise GraphCaptured("cuda graph captured after warmup")
        return n

    def close_warmup(self):
        self.warm = True


class Request:
    def __init__(self, arm, q, ee, goal, quat, zone_clear=True, dt=0.02, yaw=0.0, boxes=None, other_q=None):
        self.arm = arm
        self.q = np.asarray(q, float)
        self.ee = np.asarray(ee, float)
        self.goal = np.asarray(goal, float)
        self.quat = np.asarray(quat, float)
        self.zone_clear = zone_clear
        self.dt = dt
        self.yaw = yaw
        self.boxes = boxes
        self.other_q = other_q


class Planner:
    def __init__(self, backend, time_scale=TIME_SCALE):
        self.backend = backend
        self.time_scale = time_scale
        self.guard = GraphGuard()
        self._pool = ThreadPoolExecutor(1)

    def warmup(self):
        self.backend.warmup(self.guard)
        self.guard.close_warmup()

    def plan(self, request):
        if not self.guard.warm:
            raise NotWarmed("warmup before the loop")
        if not request.zone_clear:
            raise ZoneBusy(request.arm)
        handoff, goal = split(request.ee, request.goal)
        tail = sample(handoff, goal)
        if np.allclose(handoff, request.ee):
            return {"arm": request.arm, "joint_pos": None, "joint_vel": None, "dt": None, "cartesian": tail}
        raw = self.backend.plan(request, handoff)
        self.guard.note(raw.get("captures", 0))
        pos = np.asarray(raw["joint_pos"], float)
        vel, dt = _scale(pos, raw.get("dt", request.dt), self.time_scale)
        return {"arm": request.arm, "joint_pos": pos, "joint_vel": vel, "dt": dt, "cartesian": tail}

    def submit(self, request):
        if not self.guard.warm:
            raise NotWarmed("warmup before the loop")
        return self._pool.submit(self.plan, request)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)


def _probe():
    from Kean.shared import load_probe
    try:
        return load_probe()
    except Exception as exc:
        raise PlannerUnavailable(str(exc)) from exc


class CuroboBackend:
    """In-process cuRobo via Sakib's probe. The socket server is not built."""

    def __init__(self):
        self.probe = None
        self.rig = None

    def warmup(self, guard):
        self.probe = _probe()
        # table + plate + S3's 16 bricks + 10 other-arm links, plus spare. The probe's 24 is too small.
        self.probe.N_CUBOID = max(int(self.probe.N_CUBOID), 40)
        self.rig = self.probe.Rig(("A", "B"))
        from .iter_budget import apply, scale_from_env
        scale = scale_from_env()
        for planner in self.rig.pl.values():
            changed = apply(planner, scale)
            if changed:
                print("iter budget x%.2f: %s" % (scale, ", ".join("%s %d->%d" % row for row in changed)), flush=True)
        self.probe.warm(self.rig)
        guard.note(0)

    def plan(self, request, handoff):
        if self.rig is None:
            raise PlannerUnavailable("warmup before the loop")
        probe, pl = self.probe, self.rig.pl[request.arm]
        if request.boxes is not None:
            gap, who = self._start_gap(request)
            if gap <= 0.0:
                raise PlannerUnavailable("start in %s by %.1f mm" % (who, -gap * 1000))
            self.rig.set_world(request.arm, request.boxes, request.other_q)
        pos, quat = probe.tool_pose_w(request.arm, handoff, request.yaw)
        result = pl.plan_pose(probe.goal_from(pl, pos, quat), probe.js7(request.q[:7]), max_attempts=5)
        if not probe.plan_ok(result):
            raise PlannerUnavailable("cuRobo plan failed")
        joint_pos, _, dt = probe.plan_traj(result, pl)
        if not same_start(joint_pos[0], request.q):
            raise PlannerUnavailable("plan leaves the current joints")
        since = probe.guard().since_mark()
        return {"joint_pos": joint_pos, "dt": dt, "captures": since.get("torch_captures", 0)}

    def _start_gap(self, request):
        q = np.asarray(request.q, float)[:7][None]
        spheres = self.probe.arm_world_spheres(request.arm, q, "plan")[0]
        return self.probe.clearance_boxes(spheres, request.boxes)
