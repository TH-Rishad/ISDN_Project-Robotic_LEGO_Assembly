"""Support pose from bracing.assign. The selector is not rebuilt here."""

import numpy as np

_FIELDS = ("brace_pose", "brace_tilt_rotvec", "brace_axis", "brace_force_N")


class NoSupport(ValueError):
    pass


def support_target(brace):
    if not brace or not brace.get("feasible", False):
        raise NoSupport("brace not feasible")
    missing = [k for k in _FIELDS if k not in brace]
    if missing:
        raise NoSupport("missing %s" % ",".join(missing))
    axis = np.asarray(brace["brace_axis"], float)
    return {
        "pose": np.asarray(brace["brace_pose"], float).copy(),
        "yaw": float(np.arctan2(axis[0], axis[1])),
        "tilt": np.asarray(brace["brace_tilt_rotvec"], float).copy(),
        "grip_n": float(brace["brace_force_N"]),
        "brick_id": brace.get("target_brick_id"),
    }
