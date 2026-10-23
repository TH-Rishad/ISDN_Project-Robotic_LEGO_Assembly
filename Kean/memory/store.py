"""Brick poses and frozen grasp frames. A held brick is inferred, not tracked."""

import numpy as np

REASONS = ("init", "step_done", "grip_adjust")


def _qnorm(q):
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def qmul(a, b):
    ax, ay, az, aw = np.asarray(a, float)
    bx, by, bz, bw = np.asarray(b, float)
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def qinv(q):
    x, y, z, w = _qnorm(q)
    return np.array([-x, -y, -z, w])


def qrot(q, v):
    p = np.array([v[0], v[1], v[2], 0.0])
    return qmul(qmul(q, p), qinv(q))[:3]


def yaw_quat(yaw):
    return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)])


def compose(hand_xyz, hand_q, off_xyz, off_q):
    return hand_xyz + qrot(hand_q, off_xyz), qmul(hand_q, off_q)


def relative(hand_xyz, hand_q, brick_xyz, brick_q):
    iq = qinv(hand_q)
    return qrot(iq, brick_xyz - hand_xyz), qmul(iq, brick_q)


class Grasp:
    def __init__(self, arm, brick_id, off_xyz, off_q):
        self.arm = arm
        self.brick_id = brick_id
        self.off_xyz = np.asarray(off_xyz, float).copy()
        self.off_q = _qnorm(off_q)


class Store:
    def __init__(self):
        self.bricks = {}
        self.grasps = {}

    def remember(self, brick_id, xyz, quat):
        self.bricks[brick_id] = (np.asarray(xyz, float).copy(), _qnorm(quat))

    def remember_yaw(self, brick_id, xyz, yaw):
        self.remember(brick_id, xyz, yaw_quat(yaw))

    def freeze(self, arm, brick_id, hand_xyz, hand_q, brick_xyz, brick_q):
        off_xyz, off_q = relative(hand_xyz, hand_q, brick_xyz, brick_q)
        self.grasps[arm] = Grasp(arm, brick_id, off_xyz, off_q)
        self.remember(brick_id, brick_xyz, brick_q)

    def release(self, arm):
        self.grasps.pop(arm, None)

    def held_ids(self):
        return {g.brick_id for g in self.grasps.values()}

    def infer(self, arm, hand_xyz, hand_q):
        g = self.grasps[arm]
        return compose(np.asarray(hand_xyz, float), hand_q, g.off_xyz, g.off_q)

    def pose(self, brick_id):
        return self.bricks[brick_id]
