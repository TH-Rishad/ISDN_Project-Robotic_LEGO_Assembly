"""Measured other-arm obstacles without filling the empty space between links."""
import numpy as np

from Kean.motion.clearance import Clearance


def other_arm_boxes(probe, name, joints, groups_per_link=4):
    """Bound small groups of full-model spheres, including measured fingers.

    A single PCA box around a curved link fills its concavity. This is especially
    restrictive when B must pass above A's bracing hand. Splitting retains every
    collision sphere but avoids much of that artificial occupied volume.
    """
    checker = Clearance(probe, None)
    spheres = checker.spheres(name, joints)
    boxes = []
    for link, indices in checker.slices.items():
        if not len(indices):
            continue
        groups = [np.asarray(indices, dtype=int)]
        while len(groups) < groups_per_link:
            candidates = [i for i, ids in enumerate(groups) if len(ids) > 1]
            if not candidates:
                break
            i = max(candidates, key=lambda i: np.ptp(spheres[groups[i], :3], axis=0).max())
            ids = groups.pop(i)
            centers = spheres[ids, :3]
            axis = np.ptp(centers, axis=0).argmax()
            ordered = ids[np.argsort(centers[:, axis], kind='stable')]
            groups.extend(np.array_split(ordered, 2))
        for i, ids in enumerate(groups):
            c, r = spheres[ids, :3], spheres[ids, 3, None]
            mu = c.mean(axis=0)
            axes = np.linalg.svd(c - mu, full_matrices=False)[2] if len(c) > 2 else np.eye(3)
            if np.linalg.det(axes) < 0:
                axes[2] *= -1
            local = (c - mu) @ axes.T
            lo, hi = (local - r).min(axis=0), (local + r).max(axis=0)
            boxes.append(dict(name=f'other_{link}_{i}', dims=hi - lo,
                              c=mu + ((lo + hi) / 2) @ axes, R=axes.T))
    return boxes
