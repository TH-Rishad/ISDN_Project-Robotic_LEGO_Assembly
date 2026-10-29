"""Check sampled joint paths against the full arm model and carried brick."""
import numpy as np

from Kean.motion.clearance import Clearance


class PathCollision(ValueError):
    """A rejected sample, optionally identifying an other-arm sphere to avoid."""
    def __init__(self, message, other_sphere=None):
        super().__init__(message)
        self.other_sphere = other_sphere


def dense_joints(path, max_step=0.02):
    path = np.asarray(path, float)
    if path.ndim != 2 or path.shape[1] != 7 or not len(path) or not np.isfinite(path).all():
        raise ValueError('invalid joint trajectory')
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / max_step)))
        out.extend(a + (b - a) * t / n for t in range(1, n + 1))
    return np.asarray(out)


def validate(probe, request, path):
    """Conservative sampled check; this is not a continuous-collision proof."""
    checker = Clearance(probe, None)
    other_name = 'A' if request.arm == 'B' else 'B'
    other_q = np.asarray(request.other_q, float)
    if len(other_q) != 9:
        raise ValueError('strict paths require measured other-arm fingers')
    other = checker.spheres(other_name, other_q)
    samples = dense_joints(path)
    for index, q in enumerate(samples):
        avoid = None
        own = checker.spheres(request.arm, np.r_[q, request.grip, request.grip])
        gap, obstacle = probe.clearance_boxes(own, request.boxes)
        pair_gap = probe.clearance_spheres(own, other)
        if pair_gap < gap:
            gap, obstacle = pair_gap, 'other arm'
            if gap < .001:
                distances = np.linalg.norm(own[:, None, :3] - other[None, :, :3], axis=-1)
                distances -= own[:, None, 3] + other[None, :, 3]
                avoid = other[np.unravel_index(np.argmin(distances), distances.shape)[1]].copy()
        if request.carried is not None:
            p, R = probe.tcp_poses(request.arm, q[None])
            brick = request.carried
            center = p[0] + R[0] @ np.asarray(brick['center'])
            # A circumsphere encloses all corners regardless of brick rotation.
            sphere = np.r_[center, np.linalg.norm(brick['dims']) / 2][None]
            bgap, who = probe.clearance_boxes(sphere, request.boxes)
            agap = probe.clearance_spheres(sphere, other)
            if min(bgap, agap) < gap:
                gap, obstacle = min(bgap, agap), 'carried brick / ' + (who if bgap <= agap else 'other arm')
                avoid = (other[np.argmin(np.linalg.norm(other[:, :3] - center, axis=1)
                                         - other[:, 3])].copy() if agap < bgap else None)
        if not np.isfinite(gap) or gap < 0.001:
            raise PathCollision('path clearance %.2f mm at %s (sample %d/%d)' %
                                (gap * 1000, obstacle, index, len(samples) - 1), avoid)
