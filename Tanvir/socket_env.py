"""SocketInsertionEnv: the fitting arm's (arm B's) wriggle-fit task in the v3.1 MuJoCo twin.

WRIGGLE_FIT_RL.md §3. Built on Sakib/brickassembly/tasks/insertion_env.py without editing it:
the floating 1.5 kg hand under the task-space impedance law, the same bricks, baseplate and
breakable clutch joint model (sim/joint_model/clutch.py), and the scripted insertion
controller (ScriptedBase) as the behaviour-cloning teacher and residual base. What changes:

  * any socket: a socket_family context (target on any layer, real neighbours and supports,
    all clutched, so a wriggle can break them), not InsertionEnv's fixed layer-0 layout;
  * cheap sensing (`sensing="cheap"`): the contact wrench as a motor-torque estimate would
    give it (30 ms lag, noise, drifting bias, an inertia-model error), instead of a wrist
    F/T sensor; "ft" (ideal wrist F/T, the upper bound) and "none" (no wrench channel; the
    impedance tracking error still carries contact) are the ablations;
  * actions (9-D, [-1, 1]): EE-frame dx dy dz, droll dpitch dyaw (tilt capped at 6 deg),
    press force, lateral stiffness scale (0.15-1 x K_xy), release;
  * fingers: two 17.5 mm pads on the pinch faces, their tips 4.1 mm above the brick bottom
    (the v4 cell's grasp), so a neighbour under a fingertip blocks the descent; the brick
    hangs from the hand on a weld that `release` switches off; once the hand has risen
    clear, the fingertips close into one pad that can press the brick's studs;
  * arm A's brace, stood in for by making the braced neighbour's stud joints stronger
    (x10 "ideal", x2.5 "weak") plus a keep-out box where arm A's fingers would be: touching
    it ends the episode (arm-arm contact);
  * reward (§3.5): +1 seated; -1 joint break, force over budget, arm contact or a released
    brick lost; -0.05 * max(0, u_nb - 0.6) per step on the structure's peak joint
    utilisation; -0.002 per step.

    env = SocketInsertionEnv(mode="e2e", sensing="cheap", seed=0)
    obs = env.reset(context=ctx, level=2, brace="ideal")   # or reset() with env.sampler set
    obs, r, done, info = env.step(action)                    # obs = {"actor", "critic"}

    python socket_env.py          # scripted episodes on a few contexts, with timings
"""

import math
import sys
import time
from collections import deque
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mujoco                                              # noqa: E402
import numpy as np                                         # noqa: E402

import socket_family as SF                                 # noqa: E402
import wriggle_brace as WB                                 # noqa: E402
import planner as P                                        # noqa: E402
import tasks.insertion_env as IE                           # noqa: E402
from tasks.insertion_env import InsertionEnv, ScriptedBase, _rot6, _yaw_rot   # noqa: E402
from sim.joint_model import clutch as CL                   # noqa: E402
from sim.mj import bricks as Bk                            # noqa: E402
from sim.mj import control as C                            # noqa: E402
from sim.mj import scene as S                              # noqa: E402

ACT_DIM = 9
HIST = 16
MAX_STEPS = IE.MAX_STEPS                  # 200 policy steps = 10 s
ERR_LEVELS = (0.5, 1.0, 2.0, 3.0, 4.0)    # mm: level L draws the aim error from U(previous level, L)
TILT_MAX = math.radians(6.0)
FACES = SF.FACES
TYPES = IE.TYPES

# fingers, as the v4 cell grips (dual_arm_sim GRASP_DZ / TIP_BELOW_TCP): tips 4.1 mm above the brick bottom
FINGER_W, PAD_T, PAD_GAP, PAD_H = 0.0175, 0.002, 0.0003, 0.020
TIP_ABOVE_BOTTOM = 0.0041
SHAPING_DEPTH = 0.012       # depth shaping: progress counts over the last 12 mm (brick 9.6 mm + studs)
SHAPING_GAMMA = 0.99        # = the PPO discount in train_wriggle.py
LOST_ERR = 0.003            # a released brick more than 3 mm off its seat laterally is lost
TIP_Z = -(IE.TCP_ABOVE_BOTTOM - TIP_ABOVE_BOTTOM)          # fingertip bottom, hand frame

# cheap wrench: what a joint-torque / motor-current estimate gives (Franka tau_ext class), [A]
# Noise is added per control tick BEFORE the 30 ms filter (sigma_f 3 N raw is ~0.55 N after it); the
# estimate is tared in free space at the start (as a real cell re-zeroes before contact), so what is
# left is a residual offset plus a drift; dm: inertia-model error, a fraction of the hand mass.
CHEAP = dict(tau=0.030, sigma_f=3.0, sigma_t=0.15, bias_f=0.5, bias_t=0.03, walk_f=0.02, walk_t=0.001,
             dm=0.3)

BUDGET_FLOOR = 50.0                       # N
BRACE_FACTOR = {"none": None, "ideal": 10.0, "weak": 2.5}
BRACE_P = {"none": 0.4, "ideal": 0.4, "weak": 0.2}

STEP_DIM = 36
CTX_DIM = 32
BRACE_DIM = 9
OBS_DIM = HIST * STEP_DIM + CTX_DIM + BRACE_DIM
PRIV_DIM = 28


def tilt_of(R):
    return math.acos(float(np.clip(R[2, 2], -1.0, 1.0)))


def context_features(ctx, axis, budget):
    """The socket as the plan describes it (no sensor): 32 numbers."""
    f = np.zeros(CTX_DIM, np.float32)
    f[0:4] = [x in ctx["faces"] for x in FACES]
    f[4:8] = [ctx["contact_fraction"][x] for x in FACES]
    f[8 + SF.CLASSES.index(ctx["class"])] = 1
    slot = [set(p) <= set(ctx["faces"]) for p in (("-x", "+x"), ("-y", "+y"))]
    f[14:16] = slot
    f[16:18] = ["x" in ctx["free_pinch_axes"], "y" in ctx["free_pinch_axes"]]
    f[18 + TYPES.index(ctx["type"])] = 1
    nx, ny = P.footprint(ctx["type"], ctx["yaw_index"])
    f[26:29] = [ctx["layer"] / 3.0, ctx["mating_studs"] / 8.0, nx * ny / 8.0]
    f[29] = budget / 100.0
    f[30:32] = [axis == "x", axis == "y"]
    return f


class ScriptedWriggle(ScriptedBase):
    """ScriptedBase, extended to the 9-D action, with insert-release-press for finger-blocked
    sockets: when the fingers stall on the neighbours' studs (brick bottom 4-9 mm above the
    believed seat, i.e. inside the slot mouth but not seated), release; rise until the
    fingertips clear the brick; press its studs with the closed tips, rocking +-1 deg."""

    def __init__(self, press, blocked):
        super().__init__(press)
        self.blocked = blocked
        self.post = None

    def __call__(self, env):
        if not env.released:
            a = ScriptedBase.__call__(self, env)
            rel = -1.0
            if self.blocked and self.phase in ("search", "press"):
                x, _ = env.hand_pose()
                if 0.004 < x[2] - env.believed[0][2] < 0.009:
                    rel = 1.0
            return np.concatenate([a, [1.0, rel]])
        dt = 1.0 / IE.POLICY_HZ
        x, R = env.hand_pose()
        dp, drot, press = np.zeros(3), np.zeros(3), 0.0
        if self.post is None:
            self.post, self.t, self.hist = "rise", 0.0, []
        self.t += dt
        if self.post == "rise":
            dp[2] = IE.DPOS
            drot = C.rot_error(env.believed[1], env.R_d)
            if env.tips_closed:
                self.post, self.t = "lower", 0.0
        elif self.post == "lower":
            dp[2] = -0.005 * dt
            if env.ft_world()[2] > 3.0:
                self.post, self.t = "press", 0.0
        else:
            press = min(1.0, self.t / 0.5) * self.press
            dp[2] = min(0.0, x[2] - 0.001 - env.x_d[2])
            w = math.radians(1.0) * math.sin(2 * math.pi * 2.0 * self.t)
            drot = C.rot_error(C.rotvec_to_mat([w, 0.7 * w, 0.0]) @ env.believed[1], env.R_d)
        a = np.zeros(ACT_DIM)
        a[:3] = np.clip(R.T @ dp / IE.DPOS, -1, 1)
        a[3:6] = np.clip(R.T @ drot / IE.DROT, -1, 1)
        a[6] = np.clip(press / env.budget, 0, 1) * 2 - 1
        a[7], a[8] = 1.0, -1.0
        return a


class SocketInsertionEnv(InsertionEnv):
    def __init__(self, mode="e2e", sensing="cheap", reward="sparse", dr=True, seed=0, fingers=True,
                 contexts=None, sampler=None, u_soft=0.6, res_scale=1.0, res_release=True, shaping=0.0,
                 pad_shift=False):
        """mode: "e2e" (policy only), "residual" (policy added to ScriptedWriggle) or "scripted"
        (ScriptedWriggle only; the action is ignored). sampler(rng) -> (context, level, brace)
        is used by reset() when no context is given; contexts is the default pool for it.
        Residual mode: the policy's action is scaled by res_scale before it is added to the base;
        res_release=False leaves the release channel to the base alone. shaping > 0 adds the
        potential-based depth term gamma*Phi(s') - Phi(s) of §3.5, Phi = shaping * depth_progress()."""
        super().__init__(stage=0, mode="residual" if mode != "e2e" else "e2e", reward=reward,
                         vision=False, dr=dr, seed=seed)
        self.wmode, self.sensing, self.fingers, self.u_soft = mode, sensing, fingers, u_soft
        self.contexts = contexts
        self.sampler = sampler or self._default_sampler
        self.res_scale, self.res_release = res_scale, res_release
        self.shaping = shaping
        self.pad_shift = pad_shift    # WRIGGLE_FIT_RL.md §6.10: grasp the brick off-centre, away from a flush neighbour

    def _default_sampler(self, rng):
        pool = self.contexts or [SF.family(int(rng.integers(1 << 30)), SF.CLASSES[rng.integers(6)])]
        ctx = pool[rng.integers(len(pool))]
        brace = list(BRACE_P)[rng.choice(3, p=list(BRACE_P.values()))]
        return ctx, int(rng.integers(len(ERR_LEVELS))), brace

    # --- episode construction ---------------------------------------------------------
    def reset(self, context=None, level=None, brace=None):
        if context is None:
            context, level, brace = self.sampler(self.rng)
        level = 2 if level is None else level
        brace = brace or "none"
        ctx = self.ctx = context
        self.level, self.brace_mode = level, brace
        by = {b["id"]: b for b in ctx["bricks"]}
        tb = by[ctx["target"]]
        btype = self.btype = tb["type"]
        nx, ny = P.footprint(btype, tb["yaw_index"])
        self.n_studs = nx * ny
        cells = [c for b in ctx["bricks"] for c in P.cells((b["id"], b["type"], *b["grid_pos"], b["yaw_index"]))[0]]
        ni, nj = max(i for i, _ in cells) + 3, max(j for _, j in cells) + 3
        origin = (-ni * P.PITCH / 2, -nj * P.PITCH / 2, S.Z0)
        plan = {"bricks": ctx["bricks"], "sequence": ctx["sequence"], "voxel_origin": list(origin)}

        def seat(b):
            bx, by_ = P.footprint(b["type"], b["yaw_index"])
            i, j, k = b["grid_pos"]
            return np.array([origin[0] + (i + bx / 2) * P.PITCH, origin[1] + (j + by_ / 2) * P.PITCH,
                             S.Z0 + k * P.BRICK_H])

        tgt = seat(tb)
        # brace (arm A stand-in): which neighbour, from the trigger or the plan
        self.braced = None
        if brace == "planner":
            self.braced = ctx.get("planner_brace")
        elif brace in ("ideal", "weak"):
            self.braced = WB.choose(ctx)[0]
        # pinch axis: a free one (across the short side first), else the less blocked one, and
        # never with a finger over the braced neighbour (arm A's fingers are there)
        free = ctx["free_pinch_axes"]
        pref = ["x", "y"] if nx <= ny else ["y", "x"]
        if free:
            axis = next(a for a in pref if a in free)
        else:
            cf = ctx["contact_fraction"]
            sides = SF.TA.side_cells(SF.tuples(ctx)[-1])
            nb_cells = (P.cells(next(t for t in SF.tuples(ctx) if t[0] == self.braced))[0]
                        if self.braced else set())
            axis = min(pref, key=lambda a: cf["-" + a] + cf["+" + a]
                       + 10 * any(set(sides[f]) & nb_cells for f in ("-" + a, "+" + a)))
        self.axis = axis
        self.blocked = not free or axis not in free
        # aim error (curriculum level) and start pose
        lo, hi = (0.0 if level == 0 else ERR_LEVELS[level - 1]) / 1000, ERR_LEVELS[level] / 1000
        r, th = self.rng.uniform(lo, hi), self.rng.uniform(0, 2 * math.pi)
        yaw_b = math.radians(self.rng.uniform(-1, 1) * (0.5 + 0.375 * ERR_LEVELS[level]))
        tgt_tcp = tgt + [0, 0, IE.TCP_ABOVE_BOTTOM]
        self.true_tgt = (tgt_tcp, np.eye(3))
        self.believed = (tgt_tcp + [r * math.cos(th), r * math.sin(th), 0.0], _yaw_rot(yaw_b))
        self.fine = self.believed
        self.aim_err_mm = r * 1000
        R0 = self.believed[1]
        hand0 = self.believed[0] + [*self.rng.uniform(-0.0005, 0.0005, 2), IE.PRE_Z]
        brick0 = hand0 - R0 @ np.array([0, 0, IE.TCP_ABOVE_BOTTOM])
        q0 = C.mat_to_quat(R0)
        Lx, Ly = Bk.outer(nx, ny)
        factor = BRACE_FACTOR.get(brace if brace != "planner" else "ideal")
        self.brace_factor = factor if self.braced else None
        keepout = None
        if self.braced:
            nb = by[self.braced]
            c = seat(nb)
            bx, by_ = P.footprint(nb["type"], nb["yaw_index"])
            # where arm A's fingertips are: over the braced brick's own footprint, 1 mm inside its edges,
            # starting above whatever is already stacked on it (a brick under others is gripped from the side)
            half = [bx * P.PITCH / 2 - 0.001, by_ * P.PITCH / 2 - 0.001]
            nb_cells = P.cells((nb["id"], nb["type"], *nb["grid_pos"], nb["yaw_index"]))[0]
            k_top = max(b["grid_pos"][2] for b in ctx["bricks"] if b["id"] != ctx["target"] and
                        P.cells((b["id"], b["type"], *b["grid_pos"], b["yaw_index"]))[0] & nb_cells)
            keepout = (c[:2], S.Z0 + (k_top + 1) * P.BRICK_H + Bk.STUD_H + 0.0005, half)

        def pre(spec):
            h = spec.worldbody.add_body(name="hand", pos=hand0.tolist(), quat=q0.tolist())
            h.add_freejoint(name="hand/free")
            h.mass, h.inertia = IE.HAND_MASS, [IE.HAND_I] * 3
            h.explicitinertial = True               # the pads below must not set the hand's mass
            if self.fingers:
                half = (Lx if axis == "x" else Ly) / 2 + PAD_GAP + PAD_T / 2
                sh = self._pad_offset(ctx, axis, Ly if axis == "x" else Lx)
                for s, name in ((1, "padp"), (-1, "padn")):
                    pos = [s * half, sh, TIP_Z + PAD_H / 2] if axis == "x" else [sh, s * half, TIP_Z + PAD_H / 2]
                    size = [PAD_T / 2, FINGER_W / 2, PAD_H / 2] if axis == "x" else [FINGER_W / 2, PAD_T / 2, PAD_H / 2]
                    g = h.add_geom(name="hand/" + name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos)
                    g.contype, g.conaffinity = Bk.HAND_TYPE, Bk.HAND_AFFINITY
                    g.rgba = [0.3, 0.3, 0.3, 1]
                    g.friction = list(S.PAD_FRICTION)
                    g.density = 0.0
                tip = [0.004, FINGER_W / 2, 0.002] if axis == "x" else [FINGER_W / 2, 0.004, 0.002]
                g = h.add_geom(name="hand/tips", type=mujoco.mjtGeom.mjGEOM_BOX, size=tip,
                               pos=[0, 0, TIP_Z + 0.002])
                # compiled collidable so the body's collision tree includes it; switched off after compile
                g.contype, g.conaffinity = Bk.HAND_TYPE, Bk.HAND_AFFINITY
                g.rgba = [0.3, 0.3, 0.3, 0.3]
                g.friction = list(S.PAD_FRICTION)
                g.density = 0.0

        def post(spec):
            spec.body("target").quat = q0.tolist()
            e = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                  name1="hand", name2="target", name="grasp")
            # explicit: anchor 0; relpose = the brick in the hand frame (TCP_ABOVE_BOTTOM below it); torquescale 1.
            # The compiler's default data carries an anchor of (0, 1, 0): that grasp gave ~170 mm under a 100 N press.
            e.data = [0, 0, 0, 0, 0, -IE.TCP_ABOVE_BOTTOM, 1, 0, 0, 0, 1] + [0] * (len(e.data) - 11)
            e.solref = list(S.WELD_SOLREF)
            e.solimp = list(S.WELD_SOLIMP)
            if keepout:
                (kx, ky), kz, (hx, hy) = keepout
                g = spec.worldbody.add_geom(name="armA_keepout", type=mujoco.mjtGeom.mjGEOM_BOX,
                                            size=[hx, hy, 0.02], pos=[kx, ky, kz + 0.02])
                g.contype, g.conaffinity = Bk.WORLD_TYPE, Bk.WORLD_AFFINITY
                g.rgba = [0.2, 0.6, 1.0, 0.4]

        pre_ids = [b["id"] for b in ctx["bricks"] if b["id"] != ctx["target"]]
        self.cell = S.build_cell(plan, preplaced=set(pre_ids), with_arms=False, pre=pre,
                                 spawn={ctx["target"]: brick0}, post=post)
        m, d = self.cell.model, self.cell.data
        self.m, self.d = m, d
        self.hand = m.body("hand").id
        self.bid = m.body(ctx["target"]).id
        self.tname = ctx["target"]
        self.eq = m.equality("grasp").id
        self.pads = [m.geom("hand/" + n).id for n in ("padp", "padn")] if self.fingers else []
        self.tips = m.geom("hand/tips").id if self.fingers else None
        if self.tips is not None:
            m.geom_contype[self.tips] = m.geom_conaffinity[self.tips] = 0
        self.keepout = m.geom("armA_keepout").id if keepout else None
        self.cm = CL.ClutchModel(self.cell)
        for s in ctx["sequence"]:
            if s["brick_id"] != ctx["target"]:
                self.cm.mate_now(s["brick_id"])
        self.hdof = m.jnt_dofadr[m.joint("hand/free").id]
        # domain randomisation: InsertionEnv's (friction, break force, gate, gains, F/T bias)
        f_ins, f_brk, gate0 = 8.9, 11.3, 0.0012
        self.cm.p.f_insert_per_stud = f_ins
        self.cm.p.gate_lateral = gate0
        mu, fb, gate = 0.6, f_brk, gate0
        if self.dr:
            mu = self.rng.uniform(0.3, 0.9)
            gate = gate0 * self.rng.uniform(0.75, 1.25)
            fb = f_brk * self.rng.uniform(0.8, 1.2)
            self.gain = self.rng.uniform(0.8, 1.25)
            self.ft_bias = self.rng.normal(0, [0.15, 0.15, 0.15, 0.01, 0.01, 0.01])
        else:
            self.gain, self.ft_bias = 1.0, np.zeros(6)
        brick_bodies = {m.body(b["id"]).id for b in ctx["bricks"]}
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in brick_bodies or "stud" in m.geom(g).name:
                m.geom_friction[g, 0] = mu
        self.cm.p.gate_lateral = gate
        for st in self.cm.conn_states:
            st.patch.coef *= st.patch.f_break / fb
            st.patch.f_break = fb
        self.dr_params = np.array([mu, fb, gate * 1000])
        self.f_ins = f_ins
        if self.brace_factor:
            for st in self.cm.bricks[self.braced].conns:
                st.patch.coef /= self.brace_factor
                st.patch.f_break *= self.brace_factor
        # InsertionEnv's budget, with a floor: scaled by studs alone a 1x2 gets 28-36 N, below the touchdown
        # transient of the impedance hand (InsertionEnv only ever used 2-wide bricks)
        self.budget = max(BUDGET_FLOOR, (self.rng.uniform(1.55, 2.0) if self.dr else 1.7) * self.n_studs * f_ins)
        self.press_nom = 1.3 * self.n_studs * f_ins
        self.target_conns = {id(st) for st in self.cm.bricks[self.tname].conns}
        mujoco.mj_forward(m, d)
        x, R = self.hand_pose()
        self.x_d, self.R_d = x.copy(), R.copy()
        self.k_scale = 1.0
        self.f_press = 0.0
        self.steps = 0
        self.released = self.tips_closed = False
        self.prev_action = np.zeros(ACT_DIM)
        self.peak = self.impulse = self.u_struct_peak = 0.0
        self.base = ScriptedWriggle(min(self.press_nom, 0.85 * self.budget), self.blocked)
        self.last_base = np.zeros(ACT_DIM)
        self.done_reason = None
        self.init_err = self._true_err()
        # sensing state
        self.ft_true = np.zeros(6)
        self.cw_lp = np.zeros(6)
        self.cw_bias = self.rng.uniform(-1, 1, 6) * np.array([CHEAP["bias_f"]] * 3 + [CHEAP["bias_t"]] * 3)
        self.v_prev = np.zeros(3)
        self.sensed = np.zeros(6)
        self.ctx_feat = context_features(ctx, axis, self.budget)
        self.hist = deque([np.zeros(STEP_DIM, np.float32)] * HIST, maxlen=HIST)
        self.hist.append(self._step_features())
        self.phi = self.shaping * self.depth_progress()
        return self._obs()

    def _pad_offset(self, ctx, axis, width):
        """Along-the-face offset of the two finger pads (hand frame, perpendicular to the pinch axis). The pads
        are FINGER_W = 17.5 mm wide, wider than a 1-wide brick (7.8 mm) or a 2-wide one (15.8 mm), so a pad
        centred on the brick overhangs it, and over a flush neighbour that overhang lands on the neighbour's
        studs and stops the descent (§6.10). With pad_shift, when exactly one of the two faces perpendicular
        to the pinch axis touches a neighbour, the pads sit off-centre, their edge 0.2 mm inside that face."""
        if not self.pad_shift:
            return 0.0
        side = [f for f in ctx["faces"] if f[1] != axis]
        if len(side) != 1 or FINGER_W <= width:
            return 0.0
        return (1.0 if side[0][0] == "-" else -1.0) * (FINGER_W / 2 - width / 2 + 0.0002)

    # --- sensing ------------------------------------------------------------------------
    def ft_world(self):
        """The wrench the controller and the policy see (world frame, N and N*m)."""
        return self.sensed

    def _sense_ctrl(self, dt):
        """Per control tick: the true low-passed F/T (safety and privileged) and the cheap estimate."""
        raw = self._ft_raw()
        a_ft = dt / (IE.FT_TAU + dt)
        self.ft_true += a_ft * (raw - self.ft_true)
        v = self.d.cvel[self.hand][3:].copy()
        acc = (v - self.v_prev) / dt
        self.v_prev = v
        est = raw.copy()
        est[:3] += CHEAP["dm"] * IE.HAND_MASS * acc        # inertia-model error of a torque-based estimate
        est += self.rng.normal(0, [CHEAP["sigma_f"]] * 3 + [CHEAP["sigma_t"]] * 3)
        a_cw = dt / (CHEAP["tau"] + dt)
        self.cw_lp += a_cw * (est - self.cw_lp)

    def _sense_policy(self):
        if self.sensing == "ft":
            self.sensed = self.ft_true + self.ft_bias
        elif self.sensing == "cheap":
            walk = np.array([CHEAP["walk_f"]] * 3 + [CHEAP["walk_t"]] * 3)
            lim = np.array([CHEAP["bias_f"]] * 3 + [CHEAP["bias_t"]] * 3)
            self.cw_bias = np.clip(self.cw_bias + self.rng.normal(0, walk), -4 * lim, 4 * lim)
            self.sensed = self.cw_lp + self.cw_bias
        else:
            self.sensed = np.zeros(6)

    # --- control ---------------------------------------------------------------------------
    def _impedance(self):
        m, d = self.m, self.d
        x, R = self.hand_pose()
        v = d.cvel[self.hand]
        w, vl = v[:3], v[3:]
        mass = IE.HAND_MASS + (0.0 if self.released else m.body_mass[self.bid])
        K = IE.KP_POS * self.gain
        if self.wmode != "e2e" and self.base.phase == "search" and not self.released:
            K = K.copy()
            K[:2] = 4000.0 * self.gain
        K = K * np.array([self.k_scale, self.k_scale, 1.0])
        D = 2 * IE.ZETA * np.sqrt(K * mass)
        F = K * (self.x_d - x) - D * vl
        F[2] += -self.f_press
        F -= mass * m.opt.gravity
        Kr = IE.KP_ROT_INSERT * self.gain
        Dr = 2 * IE.ZETA * np.sqrt(Kr * IE.HAND_I)
        e = R.T @ C.rot_error(self.R_d, R)
        d.xfrc_applied[self.hand, :3] = F
        d.xfrc_applied[self.hand, 3:] = R @ (Kr * e - Dr * (R.T @ w))

    def depth_progress(self):
        """Privileged: 0 with the brick bottom SHAPING_DEPTH or more above its seat, 1 when seated."""
        dz = self.d.xpos[self.bid][2] - (self.true_tgt[0][2] - IE.TCP_ABOVE_BOTTOM)
        return float(1.0 - np.clip(dz / SHAPING_DEPTH, 0.0, 1.0))

    def _brick_err(self):
        """Lateral distance of the brick itself from its true seat. _true_err() measures the
        hand, which after a release no longer says where the brick is."""
        return float(np.linalg.norm((self.d.xpos[self.bid] - self.true_tgt[0])[:2]))

    def _release(self):
        self.d.eq_active[self.eq] = 0
        for g in self.pads:
            self.m.geom_contype[g] = 0
            self.m.geom_conaffinity[g] = 0
        self.released = True

    def _maybe_close_tips(self):
        """Fingertips close once they are above the released brick's studs."""
        if not self.released or self.tips_closed or self.tips is None:
            return
        x, _ = self.hand_pose()
        top = self.d.xpos[self.bid][2] + P.BRICK_H + Bk.STUD_H
        if x[2] + TIP_Z > top + 0.0005:
            self.m.geom_contype[self.tips] = Bk.HAND_TYPE
            self.m.geom_conaffinity[self.tips] = Bk.HAND_AFFINITY
            self.tips_closed = True

    def _arm_contact(self):
        if self.keepout is None:
            return False
        for c in self.d.contact[:self.d.ncon]:
            if self.keepout in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.keepout else c.geom1
                if self.m.geom_bodyid[other] in (self.hand, self.bid):
                    return True
        return False

    def step(self, action):
        act = np.clip(np.asarray(action, float), -1, 1)
        base = self.base(self) if self.wmode != "e2e" else np.zeros(ACT_DIM)
        self.last_base = base
        if self.wmode == "scripted":
            a = base
        elif self.wmode == "residual":
            a, res = base.copy(), act * self.res_scale
            a[:6] = np.clip(base[:6] + res[:6], -1, 1)
            a[6] = np.clip(base[6] + 0.5 * res[6], -1, 1)
            a[7] = np.clip(base[7] + res[7], -1, 1)
            if self.res_release:
                a[8] = max(base[8], res[8])
        else:
            a = act
        x, R = self.hand_pose()
        self.x_d = self.x_d + R @ (a[:3] * IE.DPOS)
        off = self.x_d - x
        if np.linalg.norm(off) > IE.LEASH:
            self.x_d = x + off / np.linalg.norm(off) * IE.LEASH
        R_new = C.rotvec_to_mat(R @ (a[3:6] * IE.DROT)) @ self.R_d
        if tilt_of(R_new) > TILT_MAX:                          # keep the yaw part only
            R_new = C.rotvec_to_mat(R @ (np.array([0, 0, a[5]]) * IE.DROT)) @ self.R_d
        self.R_d = R_new
        self.f_press = (a[6] + 1) / 2 * self.budget
        self.k_scale = 0.15 + 0.85 * (a[7] + 1) / 2
        if a[8] > 0.5 and not self.released:
            self._release()
        m, d = self.m, self.d
        dt = IE.CTRL_EVERY * m.opt.timestep
        fz_max = 0.0
        for k in range(IE.PHYS_PER_POLICY):
            if k % IE.CTRL_EVERY == 0:
                self._impedance()
            mujoco.mj_step(m, d)
            self.cm.update(d)
            if k % IE.CTRL_EVERY == IE.CTRL_EVERY - 1:
                self._sense_ctrl(dt)
                fz_max = max(fz_max, float(np.linalg.norm(self.ft_true[:3])))
        if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            self.done_reason = "diverged"
            return self._obs(), -1.0, True, self._info()
        self._maybe_close_tips()
        self._sense_policy()
        self.steps += 1
        self.peak = max(self.peak, fz_max)
        self.impulse += max(0.0, self.ft_true[2]) / IE.POLICY_HZ
        u_struct = self._u_struct()
        self.u_struct_peak = max(self.u_struct_peak, u_struct)
        self.prev_action = a.copy()
        self.hist.append(self._step_features())
        state = self.cm.bricks[self.tname].state
        done, r = False, -0.002 - 0.05 * max(0.0, u_struct - self.u_soft)
        if state == CL.MATED:
            done, r, self.done_reason = True, 1.0, "success"
        elif fz_max > self.budget:
            done, r, self.done_reason = True, -1.0, "force_violation"
        elif self.cm.breaks():
            done, r, self.done_reason = True, -1.0, "joint_break"
        elif self._arm_contact():
            done, r, self.done_reason = True, -1.0, "arm_contact"
        elif self.released and self._brick_err() > LOST_ERR:
            done, r, self.done_reason = True, -1.0, "lost_brick"
        elif self.steps >= MAX_STEPS:
            done, self.done_reason = True, "timeout"
        if self.shaping:                                       # Phi = 0 at episode end: policy-invariant
            phi = 0.0 if done else self.shaping * self.depth_progress()
            r += SHAPING_GAMMA * phi - self.phi
            self.phi = phi
        return self._obs(), r, done, self._info()

    # --- observation ----------------------------------------------------------------------
    def _u_struct(self):
        return max((st.util for st in self.cm.conn_states
                    if st.state == CL.MATED and id(st) not in self.target_conns), default=0.0)

    def _step_features(self):
        x, R = self.hand_pose()
        v = self.d.cvel[self.hand]
        bx, bR = self.believed
        f = np.concatenate([(x - bx) / 0.01, _rot6(bR.T @ R), v[3:] / 0.05, v[:3],
                            (self.x_d - x) / 0.005,
                            np.concatenate([R.T @ self.sensed[:3] / 30.0, R.T @ self.sensed[3:] / 2.0]),
                            [(x[2] - bx[2]) / 0.01], self.prev_action,
                            [float(self.released), float(self.tips_closed)]])
        return f.astype(np.float32)

    def _brace_obs(self):
        b = np.zeros(BRACE_DIM, np.float32)
        if not self.braced:
            return b
        b[0] = 1.0
        nb_cells = P.cells(next((t for t in SF.tuples(self.ctx) if t[0] == self.braced)))[0]
        sides = SF.TA.side_cells(SF.tuples(self.ctx)[-1])
        for n, f in enumerate(FACES):
            b[1 + n] = bool(set(sides[f]) & nb_cells)
        b[5] = self.brace_factor / 10.0
        # arm A's reaction, as its own joint torques would estimate it: the braced brick's joint force, noised
        F = sum((st.wrench[:3] for st in self.cm.bricks[self.braced].conns), np.zeros(3))
        b[6:9] = (F + self.rng.normal(0, 1.0, 3)) / 30.0
        return b

    def _obs(self):
        actor = np.concatenate([np.concatenate(list(self.hist)), self.ctx_feat, self._brace_obs()]).astype(np.float32)
        x, R = self.hand_pose()
        st = self.cm.bricks[self.tname].state
        conns = self.cm.bricks[self.tname].conns
        nbc = 0
        for c in self.d.contact[:self.d.ncon]:
            pair = (self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2])
            nbc += self.bid in pair and self.hand not in pair
        ub = (max((s.util for s in self.cm.bricks[self.braced].conns), default=0.0) if self.braced else 0.0)
        priv = np.concatenate([
            self._delta(self.true_tgt),
            [st == CL.DISENGAGED, st == CL.ENGAGING, st == CL.MATED],
            [np.mean([c.state == CL.MATED for c in conns]) if conns else 0.0],
            [min(nbc, 50) / 50.0, self._u_struct(), ub],
            self.dr_params / np.array([1.0, 10.0, 1.0]),
            [self.steps / MAX_STEPS, self.peak / self.budget],
            self.ft_true[:3] / 30.0, self.ft_true[3:] / 2.0,
            [(x[2] - self.true_tgt[0][2]) / 0.01]]).astype(np.float32)
        return {"actor": actor, "critic": np.concatenate([actor, priv])}

    def _info(self):
        exy, edeg = self._true_err()
        return {"success": self.done_reason == "success", "reason": self.done_reason,
                "steps": self.steps, "peak_force_N": self.peak, "impulse_Ns": self.impulse,
                "budget_N": self.budget, "aim_err_mm": self.aim_err_mm, "final_err_mm": exy * 1000,
                "final_err_deg": edeg, "u_struct_peak": self.u_struct_peak, "context": self.ctx["id"],
                "class": self.ctx["class"], "level": self.level, "brace": self.brace_mode,
                "braced": self.braced, "released": self.released, "blocked": self.blocked,
                "type": self.btype, "layer": self.ctx["layer"],
                "broken": sorted({(e[2], (e[3] or {}).get("lower")) for e in self.cm.breaks()}, key=str)}


def run_episode(env, policy=None, **reset_kw):
    """One episode; policy(obs) -> action (None: zero action, i.e. the scripted base in scripted
    or residual mode). Returns the final info."""
    obs = env.reset(**reset_kw)
    done = False
    while not done:
        a = np.zeros(ACT_DIM) if policy is None else policy(obs["actor"])
        obs, _, done, info = env.step(a)
    return info


def main():
    env = SocketInsertionEnv(mode="scripted", sensing="cheap", seed=0)
    pocket = WB.probe_pocket()
    rep = {c["id"]: c for c in SF.repo_contexts()}
    cases = [(pocket, 1, "none"), (pocket, 1, "ideal"), (rep["S4_b_006_exec"], 2, "none"),
             (rep["pyramid_b_002_exec"], 0, "none"), (SF.family(7, "ENCLOSED", layer=0), 0, "none")]
    for ctx, level, brace in cases:
        t = time.time()
        info = run_episode(env, context=ctx, level=level, brace=brace)
        print("%-22s %-8s level %d brace %-5s -> %-15s steps %3d  peak %5.1f N  u_struct %.2f  released %s  "
              "braced %s  %.1fs" % (ctx["id"], ctx["class"], level, brace, info["reason"], info["steps"],
                                    info["peak_force_N"], info["u_struct_peak"], info["released"],
                                    info["braced"], time.time() - t))
    print("obs dim", OBS_DIM, "priv dim", PRIV_DIM, "act dim", ACT_DIM)


if __name__ == "__main__":
    main()
