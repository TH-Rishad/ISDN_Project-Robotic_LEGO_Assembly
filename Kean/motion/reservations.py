"""Conservative full-arm swept reservations for Kean's segment playback.

Planning stays synchronous. Independent free-space segments may overlap.
Cartesian contact/escape moves reserve the arm's measured volume padded by
the move's displacement and must stay inside it while another arm moves.
"""
from types import SimpleNamespace

import numpy as np

from Kean.motion.path_check import dense_joints


def separated(a, b, margin=.025):
    alo, ahi = a
    blo, bhi = b
    overlap = np.all((alo[:, None] <= bhi[None] + margin) &
                     (blo[None] <= ahi[:, None] + margin), axis=2)
    return not bool(overlap.any())


class Reservations:
    def __init__(self, clearance):
        self.clearance = clearance
        self.active = {}
        self.expected = {}
        self.parallel_frames = 0
        self.waits = 0
        self.last_error = None

    def can_begin(self, tick, arm, planned):
        name = tick.slots[arm].name
        other = tick.other(arm)
        if other is not None and tick._moving(other) and tick.slots[other].name not in self.active:
            return False
        if planned or name in self.active:
            return True
        return self.admit(name, *self.cartesian(tick, arm))

    def cartesian(self, tick, arm):
        """Measured arm volume padded by this Cartesian move's reach."""
        name = tick.slots[arm].name
        move = arm.cur if arm.cur is not None else arm.moves[0]
        q = np.asarray(tick.q_of(tick.example, arm)[:7], float)
        here, start = np.asarray(arm.ee, float), np.asarray(arm.cmd[0], float) + arm.bias
        reach = (np.linalg.norm(np.asarray(move['pos'], float) + arm.bias - here)
                 + np.linalg.norm(start - here))
        turn = abs((float(move['yaw']) - float(arm.cmd[1]) + np.pi) % (2 * np.pi) - np.pi)
        # ponytail: 1.5x TCP reach bounds elbow drift for short IK moves; the
        # runtime inside() check stops the arm if this ever under-predicts.
        pad = 1.5 * reach + .15 * turn + .01
        carried = None if tick.carried is None else tick.carried(tick.example, name)
        parts = [self.bounds(name, SimpleNamespace(q=q, grip=float(g), carried=carried), q[None])
                 for g in (arm.cmd[2], move['grip'])]
        lo = np.minimum(parts[0][0], parts[1][0]) - pad
        hi = np.maximum(parts[0][1], parts[1][1]) + pad
        return SimpleNamespace(q=q), None, (lo, hi)

    def bounds(self, name, request, path):
        probe = self.clearance.probe
        q = dense_joints(np.vstack([request.q[:7], path]))
        spheres = probe.arm_world_spheres(name, q, 'full')
        p, rotations = probe.tcp_poses(name, q)
        for link, sign in (('fr3_leftfinger', 1), ('fr3_rightfinger', -1)):
            ids = self.clearance.slices[link]
            spheres[:, ids, :3] += sign * (request.grip - probe.FINGER_LOCK) * rotations[:, None, :, 1]
        lows, highs = [], []
        for ids in self.clearance.slices.values():
            if len(ids):
                s = spheres[:, ids]
                lows.append((s[..., :3] - s[..., 3, None]).min(axis=(0, 1)))
                highs.append((s[..., :3] + s[..., 3, None]).max(axis=(0, 1)))
        if request.carried is not None:
            c = p + np.einsum('tij,j->ti', rotations, request.carried['center'])
            radius = np.linalg.norm(request.carried['dims']) / 2
            lows.append(c.min(axis=0) - radius)
            highs.append(c.max(axis=0) + radius)
        return np.asarray(lows), np.asarray(highs)

    def admit(self, name, request, path, bound=None):
        bound = self.bounds(name, request, path) if bound is None else bound
        for other, other_bound in self.active.items():
            if other != name and not separated(bound, other_bound):
                self.waits += 1
                return False
        for other, other_bound in self.active.items():
            if other != name:
                self.expected[name] = other_bound
                self.expected[other] = bound
        self.active[name] = bound
        return True

    def allows_other(self, name, other, joints):
        bound = self.expected.get(name)
        if bound is None:
            self.last_error = 'no admitted concurrent path'
            return False
        return self.inside(bound, other, joints)

    def inside(self, bound, name, joints):
        spheres = self.clearance.spheres(name, joints)
        lo, hi = bound
        i = 0
        for link, ids in self.clearance.slices.items():
            if not len(ids):
                continue
            s = spheres[ids]
            if np.any(s[:, :3] - s[:, 3, None] < lo[i] - .01) or np.any(s[:, :3] + s[:, 3, None] > hi[i] + .01):
                self.last_error = dict(link=link, lower_excess=float((lo[i] - (s[:, :3] - s[:, 3, None])).max()),
                                       upper_excess=float(((s[:, :3] + s[:, 3, None]) - hi[i]).max()))
                return False
            i += 1
        return True

    def keeps_own(self, name, joints):
        """A Cartesian move overlapping other motion must stay in its reservation."""
        if len(self.active) < 2 or name not in self.active:
            return True
        return self.inside(self.active[name], name, joints)

    def release(self, name):
        self.active.pop(name, None)
        self.expected.pop(name, None)

    def step(self):
        self.parallel_frames += int(len(self.active) == 2)
