"""Conservative full-arm swept reservations for Kean's segment playback.

Planning stays synchronous. Independent free-space segments may overlap;
Cartesian contact/escape phases take an exclusive motion reservation.
"""
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
        self.exclusive = None
        self.parallel_frames = 0
        self.waits = 0
        self.last_error = None

    def can_begin(self, tick, arm, planned):
        name = tick.slots[arm].name
        other = tick.other(arm)
        if self.exclusive not in (None, name):
            return False
        if other is not None and tick._moving(other):
            if not planned or tick.slots[other].mode != 'playing' or tick.slots[other].name not in self.active:
                return False
        if not planned:
            if any(a != name for a in self.active):
                return False
            self.exclusive = name
        return True

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

    def admit(self, name, request, path):
        bound = self.bounds(name, request, path)
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
        spheres = self.clearance.spheres(other, joints)
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

    def release(self, name):
        self.active.pop(name, None)
        self.expected.pop(name, None)
        if self.exclusive == name:
            self.exclusive = None

    def step(self):
        self.parallel_frames += int(len(self.active) == 2)
