"""Free-space plan for either arm. Warm up before the loop. Do not plan the last 30 mm."""

from concurrent.futures import Future

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
    def __init__(self, arm, q, ee, goal, quat, zone_clear=True, dt=0.02, yaw=0.0, boxes=None, other_q=None, full_path=False, grip=None, carried=None, alternate_goals=()):
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
        self.full_path = full_path
        self.grip = grip
        self.carried = carried
        self.alternate_goals = [np.asarray(p, float) for p in alternate_goals]

    def snapshot(self):
        """JSON-safe inputs for reproducing a failed query without rebuilding."""
        import json
        return json.loads(json.dumps(vars(self), default=lambda value: np.asarray(value).tolist()))


class Planner:
    def __init__(self, backend, time_scale=TIME_SCALE):
        self.backend = backend
        self.time_scale = time_scale
        self.guard = GraphGuard()
        # The shared probe caches mutable GPU kinematics buffers. Keep planning,
        # path checks and the simulation's command guard on the calling thread.
        self._closed = False

    def warmup(self):
        self.backend.warmup(self.guard)
        self.guard.close_warmup()

    def plan(self, request):
        if not self.guard.warm:
            raise NotWarmed("warmup before the loop")
        if not request.zone_clear:
            raise ZoneBusy(request.arm)
        handoff, goal = (request.goal, request.goal) if request.full_path else split(request.ee, request.goal)
        tail = sample(handoff, goal)
        if not request.full_path and np.allclose(handoff, request.ee):
            return {"arm": request.arm, "joint_pos": None, "joint_vel": None, "dt": None, "cartesian": tail}
        raw = self.backend.plan(request, handoff)
        self.guard.note(raw.get("captures", 0))
        pos = np.asarray(raw["joint_pos"], float)
        vel, dt = _scale(pos, raw.get("dt", request.dt), self.time_scale)
        return {"arm": request.arm, "joint_pos": pos, "joint_vel": vel, "dt": dt, "cartesian": tail,
                "goal": raw.get("goal"), "proposal_mode": raw.get("proposal_mode", "boxed")}

    def submit(self, request):
        if not self.guard.warm:
            raise NotWarmed("warmup before the loop")
        if self._closed:
            raise RuntimeError("planner is closed")
        future = Future()
        try:
            future.set_result(self.plan(request))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self._closed = True


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
        # Reserve four measured boxes per full-model link plus world/brick slots.
        # Set the capacity before constructing or warming either planner.
        self.probe.N_CUBOID = max(int(self.probe.N_CUBOID),
                                  32 + 4 * len(self.probe.link_slices("full")))
        self.rig = self.probe.Rig(("A", "B"))
        from .iter_budget import apply, scale_from_env
        scale = scale_from_env()
        for planner in self.rig.pl.values():
            changed = apply(planner, scale)
            if changed:
                print("iter budget x%.2f: %s" % (scale, ", ".join("%s %d->%d" % row for row in changed)), flush=True)
        self._seed_probe_brick()
        self.probe.warm(self.rig)
        guard.note(0)

    def plan(self, request, handoff):
        if self.rig is None:
            raise PlannerUnavailable("warmup before the loop")
        probe, pl = self.probe, self.rig.pl[request.arm]
        strict_scene = None
        relaxed_proposal = False
        if request.boxes is not None:
            gap, who = self._start_gap(request)
            if gap <= 0.0:
                raise PlannerUnavailable("start in %s by %.1f mm" % (who, -gap * 1000))
            if request.full_path:
                from .obstacles import other_arm_boxes
                if request.other_q is None or len(request.other_q) != 9:
                    raise PlannerUnavailable("strict paths require measured other-arm fingers")
                other = "A" if request.arm == "B" else "B"
                obstacles = other_arm_boxes(probe, other, request.other_q)
                strict_scene = probe.scene_for(request.arm, list(request.boxes) + obstacles)
                pl.update_world(strict_scene)
            else:
                self.rig.set_world(request.arm, request.boxes, None if request.other_q is None else request.other_q[:7])
        # The shared Rig already owns attachment slots and warmed query graphs.
        # A thin brick makes MorphIt return too few spheres and retry; fit the cuboid directly.
        attached = request.carried is not None
        if attached:
            from .brick_spheres import as_link_spheres
            pl.attachment_manager.update(as_link_spheres(pl.attachment_manager, request.carried), probe.js7(request.q[:7]))
        try:
            errors = []
            if request.full_path:
                from .path_check import validate
                try:
                    validate(probe, request, request.q[:7][None])
                except ValueError as exc:
                    raise PlannerUnavailable("measured start is invalid: %s" % exc) from exc
            # Only explicitly supplied free-space staging goals may change.
            # Each candidate still uses the same collision-aware planner and validator.
            goals = [handoff] + (request.alternate_goals if request.full_path else [])
            if len(goals) == 1 and request.full_path:
                goals *= 3
            # Cuboids enclose curved links and can close a passage that the
            # actual sphere model leaves open. A second pass proposes paths
            # without those cuboids; validate() still checks BOTH full arms,
            # fingers, carried brick and world along every candidate path.
            modes = ['boxed'] + (['full-sphere-filter'] if strict_scene is not None else [])
            for mode in modes:
                feedback_boxes = []
                candidates = list(goals)
                if mode == 'full-sphere-filter':
                    relaxed_proposal = True
                    pl.update_world(probe.scene_for(request.arm, list(request.boxes)))
                    print('planner workaround: trying proposals checked against full other-arm spheres', flush=True)
                for candidate in candidates:
                    pos, quat = probe.tool_pose_w(request.arm, candidate, request.yaw)
                    result = pl.plan_pose(probe.goal_from(pl, pos, quat), probe.js7(request.q[:7]), max_attempts=5)
                    if not probe.plan_ok(result):
                        errors.append("%s goal %s: cuRobo plan failed: %s" % (mode, np.round(candidate, 4).tolist(), "no result (IK/start feasibility)" if result is None else getattr(result, "status", "unknown status")))
                        continue
                    joint_pos, _, dt = probe.plan_traj(result, pl)
                    if not same_start(joint_pos[0], request.q, tol=0.02 if request.full_path else 0.5):
                        errors.append("plan leaves the current joints")
                        continue
                    if request.full_path:
                        from .path_check import validate
                        try:
                            validate(probe, request, np.vstack([request.q[:7], joint_pos]))
                        except ValueError as exc:
                            errors.append("%s goal %s: %s" % (mode, np.round(candidate, 4).tolist(), exc))
                            sphere = getattr(exc, 'other_sphere', None)
                            if relaxed_proposal and sphere is not None and len(feedback_boxes) < 8:
                                # Feed the actual failed clearance sample back into
                                # cuRobo, instead of asking for the same blind path.
                                repeated = sum(np.allclose(b['c'], sphere[:3], atol=1e-6, rtol=0)
                                               for b in feedback_boxes)
                                radius = float(sphere[3]) + .0015 * (1 + repeated)
                                feedback_boxes.append(dict(name='clearance_cut_%d' % len(feedback_boxes),
                                                           c=np.asarray(sphere[:3]).copy(),
                                                           dims=[2 * radius] * 3, yaw=0.0))
                                pl.update_world(probe.scene_for(request.arm, list(request.boxes) + feedback_boxes))
                                candidates.append(candidate)
                                print('planner recovery: avoid other-arm sphere %d, radius %.1f mm' %
                                      (len(feedback_boxes), radius * 1000), flush=True)
                            continue
                    since = probe.guard().since_mark()
                    if relaxed_proposal:
                        print("planner workaround: full-sphere validation accepted path without coarse other-arm boxes", flush=True)
                    if not np.allclose(candidate, handoff):
                        print("transport workaround: validated hover %s" % np.round(candidate, 4).tolist(), flush=True)
                    return {"joint_pos": joint_pos, "dt": dt, "goal": candidate,
                            "captures": since.get("torch_captures", 0), "proposal_mode": mode}
            raise PlannerUnavailable("; ".join(errors))
        finally:
            if relaxed_proposal:
                pl.update_world(strict_scene)
            if attached:
                self.rig.detach_brick(request.arm)

    def _seed_probe_brick(self):
        """Warm-up attaches a fixed 2x4. Seed it so that path does not call MorphIt either."""
        probe = self.probe
        if not hasattr(probe, "BRICK_DIMS") or not hasattr(probe, "BRICK_IN_TCP"):
            return
        from .brick_spheres import as_link_spheres
        brick = {"dims": list(probe.BRICK_DIMS), "center": list(probe.BRICK_IN_TCP),
                 "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}
        manager = next(iter(self.rig.pl.values())).attachment_manager
        self.rig.brick_spheres = as_link_spheres(manager, brick)

    def _start_gap(self, request):
        q = np.asarray(request.q, float)[:7][None]
        spheres = self.probe.arm_world_spheres(request.arm, q, "plan")[0]
        return self.probe.clearance_boxes(spheres, request.boxes)
