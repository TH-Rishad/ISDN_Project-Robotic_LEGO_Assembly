"""Inscribed spheres for a carried brick. MorphIt is not used.

A thin cuboid makes MorphIt return one sphere or none and retry the full
optimization. The attachment slot only needs spheres that stay inside the box.
"""

import numpy as np

_CACHE = {}


def fit_cuboid(dims, n=12):
    """Sphere centres in the brick frame and one shared radius. Nothing protrudes."""
    dims = np.maximum(np.asarray(dims, float).reshape(3), 1e-4)
    n = max(1, int(n))
    key = (n, tuple(np.round(dims, 6)))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    best = None
    for nx in range(1, n + 1):
        for ny in range(1, n // nx + 1):
            counts = np.array([nx, ny, n // (nx * ny)])
            cell = dims / counts
            radius = 0.5 * float(cell.min())
            gap = 0.5 * float(np.linalg.norm(cell)) - radius
            score = (gap, -int(counts.prod()))
            if best is None or score < best[0]:
                best = (score, counts, radius, cell)
    counts, radius, cell = best[1:]
    radius = max(1e-4, radius - 1e-6)
    axes = [np.linspace(-dims[i] / 2 + cell[i] / 2, dims[i] / 2 - cell[i] / 2, counts[i]) for i in range(3)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    hit = (centers, np.full(len(centers), radius))
    _CACHE[key] = hit
    print("brick spheres: %d inscribed (morphit skipped)" % len(centers), flush=True)
    return hit


def _R(quat_wxyz):
    q = np.asarray(quat_wxyz, float).reshape(4)
    nrm = np.linalg.norm(q)
    w, x, y, z = q / nrm if nrm > 1e-8 else (1.0, 0.0, 0.0, 0.0)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def posed(dims, center, quat_wxyz, n=12):
    """Spheres in the parent frame, columns x, y, z, radius."""
    centers, radii = fit_cuboid(dims, n)
    world = centers @ _R(quat_wxyz).T + np.asarray(center, float).reshape(3)
    return np.column_stack([world, radii])


def as_link_spheres(manager, brick, n=12):
    """Attachment tensor. Stays at or under the 16-slot `attached_object` link."""
    import torch

    spheres = posed(brick["dims"], brick["center"], brick["quat_wxyz"], n)
    ref = manager.kinematics_params.link_spheres
    return torch.as_tensor(spheres, dtype=ref.dtype, device=ref.device).contiguous()
