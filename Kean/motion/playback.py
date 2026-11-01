"""Apply a saved cell recording onto arm joints and brick poses. No Newton import."""

import numpy as np


def recording_file(run_dir, shape):
    matches = sorted(run_dir.glob("%s_*.npz" % shape))
    if len(matches) != 1:
        raise FileNotFoundError("%s has %d recordings for %s" % (run_dir, len(matches), shape))
    return matches[0]


def plan_file(run_dir, shape, strategy="weakest_joint"):
    path = run_dir / "plans" / ("%s_%s.json" % (shape, strategy))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def apply_recorded(joint_q, body_q, brick_ids, body_index, q_exec, brick_q):
    """Write one recorded frame. joint_q and body_q are returned as copies."""
    joints = np.array(joint_q, float, copy=True)
    bodies = np.array(body_q, float, copy=True)
    command = np.asarray(q_exec, float).reshape(-1)
    poses = np.asarray(brick_q, float)
    if joints.shape[0] < 18 or command.shape[0] < 18:
        raise ValueError("recording and model both need 18 arm joints")
    if len(brick_ids) != len(poses):
        raise ValueError("brick pose rows do not match brick ids")
    missing = [bid for bid in brick_ids if bid not in body_index]
    if missing:
        raise ValueError("recording bricks are not in this model: %s" % ", ".join(missing))
    joints[:18] = command[:18]
    for i, bid in enumerate(brick_ids):
        bodies[body_index[bid]] = poses[i]
    return joints, bodies
