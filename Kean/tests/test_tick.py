import math
import sys
from concurrent.futures import Future
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import unittest

from Kean.motion.contact_segment import split
from Kean.motion.plan_curobo import GraphCaptured
from Kean.motion.tick import Tick, angle_wrap, needs_plan, planner_yaw
from Kean.support.sequence import Zone


# Exercise the exact shared pure function without importing Newton on CPU.
# Production receives D.wrap directly; this AST load is only a test fixture.
import ast
def _top(path, kind, name):
    if not path.is_file():
        return None
    return next((node for node in ast.parse(path.read_text()).body
                 if isinstance(node, kind) and getattr(node, "name", None) == name), None)


def _assign(path, name):
    if not path.is_file():
        return None
    return next((node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id == name for target in node.targets)), None)


source = Path(__file__).resolve().parents[2] / "Sakib/brickassembly/dual_arm_sim.py"
namespace = {"math": math, "np": np}
wrap_node = _top(source, ast.FunctionDef, "wrap")
if wrap_node is None:
    wrap = angle_wrap
else:
    exec(compile(ast.Module(body=[wrap_node], type_ignores=[]), str(source), "exec"), namespace)
    wrap = namespace["wrap"]
namespace["wrap"] = wrap
probe_source = source.parent / "scripts/12_curobo_probe.py"
resample_node = _top(probe_source, ast.FunctionDef, "resample")
if resample_node is None:
    def resample_trajectory(traj, scale, fps=60):
        q, v, dt = traj
        t = np.arange(len(q)) * dt
        if len(q) < 2 or t[-1] <= 0:
            return np.asarray(q), np.asarray(v)
        tau = np.arange(0, t[-1], scale / fps)
        qr = np.stack([np.interp(tau, t, q[:, j]) for j in range(q.shape[1])], 1)
        vr = np.stack([np.interp(tau, t, v[:, j]) for j in range(v.shape[1])], 1) * scale
        return np.vstack([qr, q[-1:]]), np.vstack([vr, np.zeros((1, v.shape[1]))])
else:
    probe_namespace = {"np": np}
    exec(compile(ast.Module(body=[resample_node], type_ignores=[]), str(probe_source), "exec"), probe_namespace)
    resample_trajectory = probe_namespace["resample"]


# Load the actual shared queue/controller class too: keep its arrival and wait
# semantics in adapter tests rather than maintaining a simplified second Arm.
arm_node = _top(source, ast.ClassDef, "Arm")
timeout_node = _assign(source, "WAIT_TIMEOUT")
body = [node for node in (timeout_node, arm_node) if node is not None]
if arm_node is None:
    raise ImportError("shared Arm class is not in dual_arm_sim.py")
if timeout_node is None:
    namespace["WAIT_TIMEOUT"] = 30.0
exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
Arm = namespace["Arm"]


class Buf:
    def __init__(self):
        self.data = np.zeros(18)
    def numpy(self):
        return self.data.copy()
    def assign(self, tgt):
        self.data = np.asarray(tgt, float).copy()


class Control:
    def __init__(self):
        self.joint_target_pos = Buf()
        self.joint_target_vel = Buf()


class Q:
    def numpy(self):
        return np.zeros(18)


class State:
    joint_q = Q()


class Ex:
    def __init__(self):
        self.A = Arm("A stabilizer", ((0, 0, 0), 0.0), [0.4, 0.0, 0.3])
        self.B = Arm("B placer", ((0, 0, 0), math.pi), [-0.4, 0.0, 0.3])
        self.control = Control()
        self.state_0 = State()
        self.ik_q = Buf()
        self.ik_q.assign(np.zeros((2, 9)))


class Later:
    def __init__(self):
        self.calls = 0
        self.request = None
        self.fut = Future()
    def submit(self, request):
        self.calls += 1
        self.request = request
        return self.fut


class TickTest(unittest.TestCase):
    def cell(self, planner):
        ex = Ex()
        tick = Tick(planner, zone=Zone(np.zeros(3), 0.05), wrap_angle=wrap, resample_trajectory=resample_trajectory).attach(ex)
        return ex, tick

    def test_short_and_contact_are_not_planned(self):
        ex, tick = self.cell(Later())
        ex.B.push("insert", ex.B.ee + [0, 0, -0.05], 0.0, 0.01, 1.0, contact=True)
        ex.B.update(0.02, ex.B.ee)
        self.assertEqual(tick.plans, 0)
        self.assertFalse(needs_plan({"phase": "transport", "pos": ex.B.ee + [0, 0, 0.02], "contact": False, "rel": None}, ex.B.ee))

    def test_plan_does_not_block_and_then_plays(self):
        planner = Later()
        ex, tick = self.cell(planner)
        start = np.array([0.30, 0.30, 0.30])
        ex.B.ee = start.copy()
        ex.B.cmd = (start.copy(), 0.0, 0.01, np.zeros(3))
        goal = np.array([0.0, 0.0, 0.05])
        ex.B.push("transport", goal, 0.4, 0.02, 1.5)
        ex.B.update(0.02, ex.B.ee)
        self.assertEqual(planner.calls, 1)
        self.assertEqual(tick.slots[ex.B].mode, "waiting")
        self.assertEqual(ex.B.t, 0.0)
        np.testing.assert_allclose(planner.request.goal, goal)
        handoff, _ = split(start, goal)
        q = np.zeros(7)
        planner.fut.set_result({
            "joint_pos": np.vstack([q, q + 0.2]), "joint_vel": np.zeros((2, 7)), "dt": 1.0 / 60,
            "cartesian": np.vstack([handoff, goal]),
        })
        ex.B.update(0.02, ex.B.ee)
        ex.control.joint_target_pos.assign(np.zeros(18))
        np.testing.assert_allclose(ex.control.joint_target_pos.data[9:16], q)
        for _ in range(4):
            if tick.slots[ex.B].q is None and ex.B.start is not None:
                break
            ex.B.update(0.02, ex.B.ee)
        np.testing.assert_allclose(ex.B.start[0], start)
        self.assertIsNone(tick.slots[ex.B].q)

    def test_contact_retract_stays_cartesian(self):
        planner = Later()
        ex, tick = self.cell(planner)
        ex.B.push("retract", ex.B.ee + [0, 0, 0.10], 0.0, 0.04, 1.0)
        ex.B.update(0.02, ex.B.ee)
        self.assertEqual(planner.calls, 0)
        self.assertGreater(ex.B.t, 0.0)

    def test_velocity_cleared_on_cartesian_handoff(self):
        ex, tick = self.cell(Later())
        slot = tick.slots[ex.B]
        slot.q = np.ones(7)
        slot.v = np.full(7, 0.5)
        ex.control.joint_target_vel.assign(np.full(18, 0.3))
        ex.control.joint_target_pos.assign(np.zeros(18))
        np.testing.assert_allclose(ex.control.joint_target_vel.data[9:16], 0.5)
        slot.q = slot.v = None
        ex.control.joint_target_pos.assign(np.zeros(18))
        np.testing.assert_allclose(ex.control.joint_target_vel.data[9:16], 0.0)
        np.testing.assert_allclose(ex.control.joint_target_vel.data[:9], 0.3)
        np.testing.assert_allclose(ex.control.joint_target_vel.data[16:], 0.3)

    def test_waiting_holds_measured_joints_and_updates_ee(self):
        planner = Later()
        ex, tick = self.cell(planner)
        tick.q_of = lambda example, arm: np.full(7, 0.4)
        ex.B.push("transport", [0.3, 0.3, 0.3], 0.0, 0.01, 1.0)
        ex.B.update(0.02, ex.B.ee)
        ex.control.joint_target_pos.assign(np.zeros(18))
        np.testing.assert_allclose(ex.control.joint_target_pos.data[9:16], 0.4)
        measured = np.array([-0.39, 0.0, 0.3])
        ex.B.update(0.02, measured)
        np.testing.assert_allclose(ex.B.ee, measured)
        planner.fut.set_exception(RuntimeError("no path"))
        ex.B.update(0.02, measured)
        self.assertIsNone(tick.slots[ex.B].q)
        ex.control.joint_target_pos.assign(np.zeros(18))
        np.testing.assert_allclose(ex.control.joint_target_pos.data[9:16], 0.0)
        np.testing.assert_allclose(ex.control.joint_target_vel.data[9:16], 0.0)

    def test_planner_uses_cell_wrist_convention(self):
        ex, _ = self.cell(Later())
        self.assertAlmostEqual(planner_yaw(ex, ex.B, 0.0, wrap), math.pi)
        self.assertAlmostEqual(planner_yaw(ex, ex.B, math.pi / 2, wrap), 3 * math.pi / 2)
        ex.yaw_shift = -math.pi
        self.assertAlmostEqual(planner_yaw(ex, ex.B, 0.2, wrap), 0.2 - math.pi)
        ex.direct = True
        self.assertAlmostEqual(planner_yaw(ex, ex.B, 0.2, wrap), 0.2)

    def test_request_yaw_matches_cell_at_and_around_boundaries(self):
        for yaw in (0.0, math.pi / 2 - 1e-8, math.pi / 2,
                    math.pi / 2 + 1e-8, -math.pi / 2, math.pi):
            planner = Later()
            ex, tick = self.cell(planner)
            ex.B.push("transport", [0.3, 0.3, 0.3], yaw, 0.01, 1.0)
            ex.B.update(0.02, ex.B.ee)
            expected = ex.B.base[1] + wrap(yaw - ex.B.base[1], math.pi)
            self.assertEqual(planner.request.yaw, expected)

    def test_bias_and_ik_seed_at_handoff(self):
        planner = Later()
        ex, tick = self.cell(planner)
        ex.B.bias = np.array([0.01, -0.02, 0.005])
        goal = np.array([0.1, 0.1, 0.2])
        ex.B.push("transport", goal, 0.0, 0.01, 1.0)
        ex.B.update(0.02, ex.B.ee)
        np.testing.assert_allclose(planner.request.goal, goal + ex.B.bias)
        tick.q_of = lambda example, arm: np.full(7, 0.25)
        planner.fut.set_result({"joint_pos": np.zeros((1, 7)),
                               "joint_vel": None, "dt": None,
                               "cartesian": np.array([goal + ex.B.bias])})
        measured = goal + ex.B.bias + [0, 0, 0.03]
        ex.B.update(0.02, measured)
        np.testing.assert_allclose(ex.ik_q.data[1, :7], 0.25)
        np.testing.assert_allclose(ex.ik_q.data[0], 0.0)
        ex.B.update(0.02, measured)
        np.testing.assert_allclose(ex.B.start[0] + ex.B.bias, measured)
        np.testing.assert_allclose(ex.B.cur["pos"], goal)

    def test_cartesian_tail_keeps_shared_arrival_and_wait_semantics(self):
        planner = Later()
        ex, tick = self.cell(planner)
        start = ex.B.ee.copy()
        goal = np.array([0.3, 0.3, 0.3])
        ready, completed = [False], []
        ex.B.push("transport", goal, 0.0, 0.01, 1.0,
                  wait=lambda: ready[0], then=lambda: completed.append(True))
        ex.B.update(0.02, start)
        planner.fut.set_result({"joint_pos": np.zeros((1, 7)), "joint_vel": None,
                               "dt": None, "cartesian": goal[None]})
        ex.B.update(0.02, start)
        ex.B.update(0.02, start)
        ex.B.t = ex.B.cur["dur"] - 0.01
        ex.B.update(0.02, start)
        self.assertIsNotNone(ex.B.cur)  # timer alone must not complete the move
        ex.B.update(0.02, goal)
        self.assertIsNotNone(ex.B.cur)  # arrival alone must not bypass the wait
        ready[0] = True
        ex.B.update(0.02, goal)
        self.assertIsNone(ex.B.cur)
        self.assertEqual(completed, [True])

    def test_other_arm_holds_the_zone(self):
        ex, tick = self.cell(Later())
        tick.slots[ex.A].mode = "playing"
        ex.A.ee = np.zeros(3)
        ex.A.cur = {"phase": "to brace", "pos": np.zeros(3)}
        ex.B.push("transport", np.zeros(3), 0.0, 0.01, 1.5)
        queued = len(ex.B.moves)
        ex.B.update(0.02, ex.B.ee)
        self.assertEqual(len(ex.B.moves), queued)
        self.assertIsNone(ex.B.cur)
        self.assertEqual(tick.plans, 0)

    def test_capture_after_warmup_fails_the_run(self):
        planner = Later()
        ex, tick = self.cell(planner)
        start = np.array([0.3, 0.0, 0.3])
        ex.B.ee = start.copy()
        ex.B.cmd = (start.copy(), 0.0, 0.01, np.zeros(3))
        ex.B.push("transport", [0.0, 0.0, 0.05], 0.0, 0.01, 1.5)
        ex.B.update(0.02, ex.B.ee)
        planner.fut.set_exception(GraphCaptured("cuda graph captured after warmup"))
        with self.assertRaises(GraphCaptured):
            ex.B.update(0.02, ex.B.ee)
        self.assertEqual(tick.captures, 1)


if __name__ == "__main__":
    unittest.main()
