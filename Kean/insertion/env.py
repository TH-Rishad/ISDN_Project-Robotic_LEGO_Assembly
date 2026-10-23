"""Insertion episode. Starts at the pre-insert pose. Residual on a straight press. Sparse +1 / -1."""

import numpy as np

from Kean.insertion.obs import from_shared, observe

from Sakib.brickassembly.sim.joint_model.capacity import (
    F_INSERT_PER_STUD as F_INSERT,
    F_BREAK_PER_STUD as F_BREAK,
)
SUCCESS = 1.0
FAIL = -1.0
MAX_STEPS = 200
ACT_DIM = 7
PRIV_DIM = 10


def press_n(a6, n_studs):
    return (float(np.clip(a6, -1.0, 1.0)) + 1.0) / 2.0 * int(n_studs) * F_INSERT


def scripted_insert(step, max_steps=MAX_STEPS):
    action = np.zeros(ACT_DIM)
    action[2] = -1.0
    ramp = min(1.0, step / max(1, int(0.3 * max_steps)))
    action[6] = -1.0 + 2.0 * ramp
    return action


def residual(action, base):
    action = np.clip(np.asarray(action, float), -1.0, 1.0)
    out = np.asarray(base, float).copy()
    out[:6] = np.clip(out[:6] + action[:6], -1.0, 1.0)
    out[6] = np.clip(out[6] + 0.5 * action[6], -1.0, 1.0)
    return out


class InsertionEnv:
    def __init__(self, cell):
        self.cell = cell
        self.steps = 0

    def observe(self):
        if hasattr(self.cell, "observe"):
            return from_shared(self.cell.observe())
        # Lightweight fixtures supply raw sense data; live adapters use the
        # shared Sim.observe / ObsSpec.encode contract above.
        sense = self.cell.sense()
        return observe(sense["frames"], sense["q"], sense["qd"], sense["hands"], sense.get("stride", 4))

    def critic_vector(self):
        priv = self.cell.privileged()
        n = max(int(self.cell.n_studs), 1)
        clutch = np.zeros(3, np.float32)
        clutch[int(priv.get("clutch", 0)) % 3] = 1.0
        delta = np.asarray(priv.get("true_delta", np.zeros(3)), np.float32).reshape(3)
        return np.concatenate([
            delta,
            [priv.get("yaw_err", 0.0)],
            clutch,
            [priv.get("util", 0.0), priv.get("force_n", 0.0) / (n * F_BREAK), float(bool(priv.get("dropped", False)))],
        ]).astype(np.float32)

    def reset(self):
        self.steps = 0
        self.cell.reset()
        return self.observe()

    def step(self, action):
        base = self.cell.scripted_action(self.steps)
        command = residual(action, base)
        n = int(self.cell.n_studs)
        press = press_n(command[6], n)
        self.cell.apply(command, press)
        self.steps += 1
        priv = self.cell.privileged()
        done, reward = False, 0.0
        if priv.get("snap"):
            done, reward = True, SUCCESS
        elif priv.get("broken") or priv.get("dropped") or priv.get("force_n", 0.0) > n * F_BREAK:
            done, reward = True, FAIL
        elif self.steps >= MAX_STEPS:
            done = True
        return self.observe(), reward, done, {"press_n": press}
