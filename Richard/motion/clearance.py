"""Executed free-space clearance using the probe's full collision spheres.

Finger spheres are translated to measured openings, as in P3 step 8. This
measures playback frames; contact segments remain covered by ContactMonitor.
"""
import numpy as np


def grouped_sphere_gap(a, b, groups, measure):
    """Exact sphere minimum, pruning link pairs by enclosing-box lower bounds."""
    aa, bb = [a[ids] for ids in groups], [b[ids] for ids in groups]
    def bounds(spheres):
        return ((spheres[:, :3] - spheres[:, 3, None]).min(axis=0),
                (spheres[:, :3] + spheres[:, 3, None]).max(axis=0))
    pairs = []
    other_bounds = list(map(bounds, bb))
    for i, (lo, hi) in enumerate(map(bounds, aa)):
        for j, (other_lo, other_hi) in enumerate(other_bounds):
            separation = np.maximum(0., np.maximum(lo - other_hi, other_lo - hi))
            lower = float(np.linalg.norm(separation))
            # Overlapping boxes also contain potentially penetrating spheres.
            pairs.append((lower - 1e-6 if lower > 0 else -np.inf, i, j))
    best = np.inf
    for lower, i, j in sorted(pairs):
        if lower >= best:
            break
        best = min(best, measure(aa[i], bb[j]))
    return best


class Clearance:
    def __init__(self, probe, world, cache=False):
        self.probe, self.world = probe, world
        # Cache before playback instead of rebuilding the link map each tick.
        self.slices = probe.link_slices('full')
        self.cache = {} if cache else None
        self.groups = [np.asarray(ids) for ids in self.slices.values() if len(ids)]

    def pair_gap(self, own, other):
        if (self.cache is not None and len(own) == len(other) == sum(map(len, self.groups))
                and np.isfinite(own).all() and np.isfinite(other).all()):
            return grouped_sphere_gap(own, other, self.groups, self.probe.clearance_spheres)
        return self.probe.clearance_spheres(own, other)

    def spheres(self, name, joints):
        key = (name, tuple(np.asarray(joints[:9], float)))
        if self.cache is not None and key in self.cache:
            return self.cache[key].copy()
        probe = self.probe
        q = np.asarray(joints[:7])[None]
        spheres = probe.arm_world_spheres(name, q, 'full')[0].copy()
        _, rotations = probe.tcp_poses(name, q)
        y = rotations[0, :, 1]
        for link, opening, sign in (('fr3_leftfinger', joints[7], 1),
                                    ('fr3_rightfinger', joints[8], -1)):
            spheres[self.slices[link], :3] += sign * (opening - probe.FINGER_LOCK) * y
        if self.cache is not None:
            if len(self.cache) >= 16:
                self.cache.clear()
            self.cache[key] = spheres.copy()
        return spheres

    def __call__(self, example, name, commanded_q):
        joints = example.state_0.joint_q.numpy()
        k = 0 if name == 'A' else 1
        own = self.spheres(name, joints[9*k:9*k+9])
        other_name = 'B' if name == 'A' else 'A'
        other = self.spheres(other_name, joints[9*(1-k):9*(1-k)+9])
        boxes, _ = self.world(example, name)
        gap, _ = self.probe.clearance_boxes(own, boxes)
        return min(gap, self.pair_gap(own, other))
