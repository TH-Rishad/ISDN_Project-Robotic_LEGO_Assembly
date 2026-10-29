"""Ground-truth cuboids for a WP2 plan. The held brick is not one of them."""

import math

import numpy as np

PICK = frozenset({"park", "to feeder", "descend", "settle", "grasp", "release", "retract", "clearance"})


def yaw_of(q):
    x, y, z, w = np.asarray(q, float)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def held_brick(arm, phase, brick_id):
    if arm != "B" or brick_id is None or phase in PICK:
        return None
    return brick_id


def brick_boxes(body_q, bodies, dims, skip=None, *, box_of):
    boxes = []
    for bid, body in bodies.items():
        if bid == skip:
            continue
        length, width = dims[bid]
        pose = np.asarray(body_q[body], float)
        info = {bid: {"L": float(length), "W": float(width),
                      "bottom": pose[:3], "yaw": yaw_of(pose[3:7])}}
        boxes.append(box_of(bid, info, name=bid))
    return boxes


def plate_box(origin, ni, nj, pitch, z0):
    ox, oy = origin[0], origin[1]
    return {
        "name": "plate",
        "dims": ((ni + 4) * pitch, (nj + 4) * pitch, z0),
        "c": (ox + ni * pitch / 2.0, oy + nj * pitch / 2.0, z0 / 2.0),
        "yaw": 0.0,
    }


def zone_of(origin, ni, nj, pitch):
    from Kean.support.sequence import Zone
    ox, oy = float(origin[0]), float(origin[1])
    center = np.array([ox + ni * pitch / 2.0, oy + nj * pitch / 2.0, 0.0])
    half = 0.5 * float(np.hypot(ni * pitch, nj * pitch))
    return Zone(center, half)
