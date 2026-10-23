"""WriggleSkillEnv: skill-level (option) RL for the wriggle fit. WRIGGLE_FIT_RL.md §6.9.

An episode is a SocketInsertionEnv episode (same contexts, DR, braces, cheap sensing, terminations
and §3.5 reward). The approach (ScriptedBase align + descend, up to first contact) stays scripted
and is not a decision. From first contact on, every decision is one *macro step* = K = 10 env ticks
(0.5 s): the policy outputs theta in [-1, 1]^10 that parameterises a wriggle primitive
(SkillController) run for those ticks. theta = 0 IS the scripted controller (ScriptedWriggle):
same spiral, same +-1.5 deg rock, same 8 N search press / seating press ramp, same release rule,
same post-release press; a zero-mean policy therefore starts exactly at the scripted base.

theta channels (mapped piecewise-linearly so that 0 -> the scripted value):

  0,1  centre nudge x, y        +-0.6 mm per macro step, accumulated on the spiral centre, kept within 4 mm of the believed target
  2,3  static tilt roll, pitch   +-3 deg about the believed orientation (search phase only; commanded tilt capped at 5.5 deg)
  4    yaw offset                +-1.5 deg about the believed yaw (search and press phases)
  5    rock amplitude            0 ... 1.5 (theta=0) ... 3 deg at 2 Hz (roll; pitch 0.7x; one full period per macro step)
  6    rock yaw mix              yaw rock = 0.5 * theta6 * roll rock (0 at theta=0: the scripted rock only tilts)
  7    press force               search press 2 ... 8 (0) ... 24 N; seating press 0.4 ... 1 (0) ... 1.2 x scripted, <= 0.9 budget
  8    lateral stiffness scale   0.15 ... 1 (theta <= 0: full); theta > 0 is the same as 0 (stiffer than K_xy is not available)
  9    spiral radius             0 ... 2 mm (0) ... 3 mm

Release for finger-blocked sockets is the scripted rule (not a decision). All of it is carried out
through the unchanged SocketInsertionEnv.step in its "scripted" mode: SkillController replaces
env.base, so the env calls it each tick for the 9-D action and then does its own impedance
control, sensing, reward and terminations.

    env = WriggleSkillEnv(sensing="cheap", seed=0)
    obs = env.reset(context=ctx, level=2, brace="none")      # obs = {"actor", "critic"} (macro)
    obs, r, done, info = env.step(theta)                       # r = sum of tick rewards + macro shaping
"""

import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                         # noqa: E402

import socket_env as SE                                    # noqa: E402
import tasks.insertion_env as IE                           # noqa: E402
from tasks.insertion_env import ScriptedBase               # noqa: E402
from sim.mj import control as C                            # noqa: E402

K_TICKS = 10                       # env ticks per macro step (0.5 s)
THETA_DIM = 10
GAMMA_MACRO = 0.97
DT = 1.0 / IE.POLICY_HZ
NUDGE = 0.0006                     # m per macro step at |theta| = 1
CENTRE_MAX = 0.004                 # m, accumulated centre offset from the believed target
TILT_STATIC = math.radians(3.0)
TILT_CAP = math.radians(5.5)       # commanded tilt (static + rock) stays below socket_env's 6 deg cap
YAW_OFF = math.radians(1.5)
ROCK_MAX = math.radians(3.0)
ROCK_MID = math.radians(1.5)
YAW_MIX = 0.5
SPIRAL_MAX, SPIRAL_MID = 0.003, 0.002
PHASES = ("search", "press")
MACRO_DIM = 2 + 18 + 6 + 6 + 2 + 3 + 3 + 2 + 2 + 2 + THETA_DIM + SE.CTX_DIM + SE.BRACE_DIM   # see _macro_obs
PRIV_DIM = SE.PRIV_DIM


def pw(th, lo, mid, hi):
    """Piecewise-linear map of th in [-1, 1]: lo at -1, mid at 0, hi at +1."""
    return mid + (hi - mid) * th if th >= 0 else mid + (mid - lo) * th


class SkillController(SE.ScriptedWriggle):
    """ScriptedWriggle whose wriggle (search / press) and post-release press follow `theta`."""

    def __init__(self, press, blocked):
        super().__init__(press, blocked)
        self.theta = np.zeros(THETA_DIM)
        self.nudge = np.zeros(2)          # m per tick, from theta[0:2]
        self.contact = False

    def set_theta(self, th):
        self.theta = np.clip(np.asarray(th, float), -1.0, 1.0)
        self.nudge = NUDGE * self.theta[:2] / K_TICKS

    def rock_amp(self):
        return pw(self.theta[5], 0.0, ROCK_MID, ROCK_MAX)

    def press_mult(self):
        return pw(self.theta[7], 0.25, 1.0, 3.0), pw(self.theta[7], 0.4, 1.0, 1.2)   # (search, seating)

    def __call__(self, env):
        if env.released:
            return self._post(env)
        if self.phase in ("align", "descend"):
            a = SE.ScriptedWriggle.__call__(self, env)
            if self.phase in PHASES:
                self.contact = True
            return a
        x, R = env.hand_pose()
        th, dt = self.theta, DT
        self.t += dt
        xd = env.x_d
        tb, Rb = env.believed
        z_rel = x[2] - tb[2]
        m_search, m_seat = self.press_mult()
        dp, press = np.zeros(3), 0.0
        yaw = YAW_OFF * th[4]
        if self.phase == "search":
            self.centre[:2] = self._clip_centre(self.centre[:2] + self.nudge, tb)
            s = 0.003 * self.t
            a_ = 0.001 / (2 * math.pi)
            ang = math.sqrt(2 * s / a_) if s > 0 else 0.0
            r = min(a_ * ang, pw(th[9], 0.0, SPIRAL_MID, SPIRAL_MAX))
            goal = self.centre[:2] + r * np.array([math.cos(ang), math.sin(ang)])
            dp[:2] = np.clip(goal - xd[:2], -IE.DPOS, IE.DPOS)
            dp[2] = min(0.0, x[2] - 0.001 - xd[2])
            press = 8.0 * m_search
            w = self.rock_amp() * math.sin(2 * math.pi * 2.0 * self.t)
            rv = np.array([TILT_STATIC * th[2] + w, TILT_STATIC * th[3] + 0.7 * w, yaw + YAW_MIX * th[6] * w])
            n = math.hypot(rv[0], rv[1])
            if n > TILT_CAP:
                rv[:2] *= TILT_CAP / n
            drot = C.rot_error(C.rotvec_to_mat(rv) @ Rb, env.R_d)
            if z_rel < 0.00148 or self.t > 5.0:
                self.phase, self.t, self.hist = "press", 0.0, []
        else:  # press: laterally free, level (plus the yaw offset); seating force ramp
            press = min(1.0, self.t / 0.5) * min(self.press * m_seat, 0.9 * env.budget)
            drot = C.rot_error(C.rotvec_to_mat([0.0, 0.0, yaw]) @ Rb, env.R_d)
            dp[:2] = x[:2] - xd[:2]
            dp[2] = min(0.0, x[2] - 0.001 - xd[2])
            self.hist.append(x[2])
            if len(self.hist) > 8 and self.hist[-9] - self.hist[-1] < 0.00003 and self.t > 0.9:
                self.phase, self.t = "search", 0.0
                self.centre = x.copy()
                self.centre[:2] = self._clip_centre(self.centre[:2], tb)
        k_scale = pw(th[8], 0.15, 1.0, 1.0)
        return self._action(R, dp, drot, press, env, (k_scale - 0.15) / 0.85 * 2 - 1, self._release(env))

    @staticmethod
    def _clip_centre(c, tb):
        off = c - tb[:2]
        n = np.linalg.norm(off)
        return tb[:2] + off * (CENTRE_MAX / n) if n > CENTRE_MAX else c

    def _release(self, env):
        if self.blocked and self.phase in PHASES:
            x, _ = env.hand_pose()
            if 0.004 < x[2] - env.believed[0][2] < 0.009:
                return 1.0
        return -1.0

    @staticmethod
    def _action(R, dp, drot, press, env, k, rel):
        a = np.zeros(SE.ACT_DIM)
        a[:3] = np.clip(R.T @ dp / IE.DPOS, -1, 1)
        a[3:6] = np.clip(R.T @ drot / IE.DROT, -1, 1)
        a[6] = np.clip(press / env.budget, 0, 1) * 2 - 1
        a[7], a[8] = k, rel
        return a

    def _post(self, env):
        """After release: rise, lower the closed tips onto the studs, press with a rock (ScriptedWriggle's
        sequence; the rock amplitude is the macro one scaled so that theta = 0 is the scripted +-1 deg)."""
        x, R = env.hand_pose()
        dp, drot, press = np.zeros(3), np.zeros(3), 0.0
        if self.post is None:
            self.post, self.t, self.hist = "rise", 0.0, []
        self.t += DT
        _, m_seat = self.press_mult()
        if self.post == "rise":
            dp[2] = IE.DPOS
            drot = C.rot_error(env.believed[1], env.R_d)
            if env.tips_closed:
                self.post, self.t = "lower", 0.0
        elif self.post == "lower":
            dp[2] = -0.005 * DT
            if env.ft_world()[2] > 3.0:
                self.post, self.t = "press", 0.0
        else:
            press = min(1.0, self.t / 0.5) * min(self.press * m_seat, 0.9 * env.budget)
            dp[2] = min(0.0, x[2] - 0.001 - env.x_d[2])
            w = (self.rock_amp() / ROCK_MID) * math.radians(1.0) * math.sin(2 * math.pi * 2.0 * self.t)
            drot = C.rot_error(C.rotvec_to_mat([w, 0.7 * w, YAW_MIX * self.theta[6] * w]) @ env.believed[1], env.R_d)
        return self._action(R, dp, drot, press, env, 1.0, -1.0)


class WriggleSkillEnv:
    """Macro-step wrapper around SocketInsertionEnv(mode="scripted") driven by SkillController."""

    def __init__(self, sensing="cheap", dr=True, seed=0, fingers=True, contexts=None, sampler=None,
                 shaping=0.5, gamma=GAMMA_MACRO, k_ticks=K_TICKS, pad_shift=False):
        self.env = SE.SocketInsertionEnv(mode="scripted", sensing=sensing, dr=dr, seed=seed, fingers=fingers,
                                         contexts=contexts, sampler=sampler, pad_shift=pad_shift)
        self.shaping, self.gamma, self.K = shaping, gamma, k_ticks
        self.zero = np.zeros(SE.ACT_DIM)

    @property
    def sampler(self):
        return self.env.sampler

    @sampler.setter
    def sampler(self, f):
        self.env.sampler = f

    # --- episode -----------------------------------------------------------------------
    def reset(self, context=None, level=None, brace=None):
        env = self.env
        env.reset(context=context, level=level, brace=brace)
        env.base = self.ctrl = SkillController(env.base.press, env.blocked)
        self.theta = np.zeros(THETA_DIM)
        self.macro_steps = 0
        self.dead = None
        r_pre, obs = 0.0, None
        while not self.ctrl.contact:                       # scripted approach: align + descend to first contact
            obs, r, done, info = env.step(self.zero)
            r_pre += r
            if done:                                       # failed (or even seated) before any decision
                self.dead = (r_pre, info)
                break
        if obs is None:
            obs = env._obs()
        self.obs_env = obs
        self.depth_prev = float(env.hist[-1][24])
        self.phi = self._phi(self.dead is not None)
        self.last_obs = self._macro_obs(list(env.hist)[-self.K:])
        return self.last_obs

    def _phi(self, done):
        return 0.0 if done else self.shaping * self.env.depth_progress()

    def step(self, theta):
        env, ctrl = self.env, self.ctrl
        if self.dead is not None:
            r, info = self.dead
            info = dict(info, macro_steps=0)
            return self.last_obs, r, True, info
        theta = np.clip(np.asarray(theta, float), -1.0, 1.0)
        ctrl.set_theta(theta)
        r_sum, done, info = 0.0, False, None
        for _ in range(self.K):
            self.obs_env, r, done, info = env.step(self.zero)
            r_sum += r
            if done:
                break
        self.theta = theta
        self.macro_steps += 1
        phi = self._phi(done)
        r_sum += self.gamma * phi - self.phi
        self.phi = phi
        self.last_obs = self._macro_obs(list(env.hist)[-self.K:])
        info = dict(info, macro_steps=self.macro_steps)
        return self.last_obs, r_sum, done, info

    # --- observation -------------------------------------------------------------------
    def _macro_obs(self, win):
        env, ctrl = self.env, self.ctrl
        W = np.array(win)                                  # K x 36 per-tick cheap features (socket_env._step_features)
        x, R = env.hand_pose()
        tb, Rb = env.believed
        depth = W[:, 24]
        rel = C.rot_error(Rb, R) / math.radians(5.0)       # tilt / yaw of the hand vs the believed pose
        centre = (ctrl.centre[:2] - tb[:2]) / CENTRE_MAX if ctrl.centre is not None else np.zeros(2)
        phase = [ctrl.phase == "search", ctrl.phase == "press", float(env.released)]
        a_tail = self.obs_env["actor"][-(SE.CTX_DIM + SE.BRACE_DIM):]
        actor = np.concatenate([
            [depth[-1], depth[-1] - self.depth_prev],      # depth progress (10 mm units) and its change over the window
            W[:, 18:24].mean(0), W[:, 18:24].min(0), W[:, 18:24].max(0),   # cheap wrench estimate, hand frame
            W[:, 15:18].mean(0), W[-1, 15:18],             # impedance tracking error
            W[:, 9:12].mean(0), W[:, 12:15].mean(0),       # hand linear / angular velocity
            W[-1, 0:2],                                    # hand xy vs the believed target
            rel,                                           # hand roll, pitch, yaw vs the believed pose
            phase,                                         # controller phase: search / press / released
            centre,                                        # accumulated centre offset
            [W[-1, 35], env.steps / SE.MAX_STEPS],         # tips closed, elapsed time
            [ctrl.t / 5.0, float(ctrl.contact)],           # time in the current phase
            self.theta, a_tail]).astype(np.float32)
        self.depth_prev = float(depth[-1])
        assert actor.shape == (MACRO_DIM,), actor.shape
        priv = self.obs_env["critic"][-PRIV_DIM:]
        return {"actor": actor, "critic": np.concatenate([actor, priv])}


def run_episode(env, policy=None, **reset_kw):
    """One macro episode; policy(actor_obs) -> theta (None: theta = 0). Returns the final info."""
    obs = env.reset(**reset_kw)
    done, info = env.dead is not None, None
    while not done:
        th = np.zeros(THETA_DIM) if policy is None else policy(obs["actor"])
        obs, _, done, info = env.step(th)
    if info is None:
        _, info = env.dead
        info = dict(info, macro_steps=0)
    return info
