"""Actor observation: camera frames and both arms' proprioception. No brick pose."""

import numpy as np

CAMERAS = ("top", "wrist_A", "wrist_B")
PROPRIO_F = 54


def proprio_vector(q, qd, hands):
    q = np.asarray(q, np.float32).reshape(18)
    qd = np.asarray(qd, np.float32).reshape(18)
    hands = [np.asarray(h, np.float32).reshape(7) for h in hands]
    return np.concatenate([q, qd, hands[0], hands[1], q[7:9], q[16:18]])


def observe(frames, q, qd, hands, stride=4):
    packed = {}
    shapes = []
    for cam in CAMERAS:
        rgb = np.asarray(frames[cam]["rgb"])[::stride, ::stride]
        depth = np.asarray(frames[cam]["depth"])[::stride, ::stride]
        packed[cam] = {"rgb": np.ascontiguousarray(rgb), "depth": np.ascontiguousarray(depth)}
        shapes.append(rgb.shape[:2])
    if len(set(shapes)) != 1:
        raise ValueError("camera frames differ in size")
    return {"frames": packed, "proprio": proprio_vector(q, qd, hands)}


def from_shared(encoded):
    """Project rl_env.ObsSpec.encode output onto Kean's actor contract.

    Sakib owns capture, FK, quaternion-sign handling and image subsampling.
    Only cameras and proprioception are exposed to the insertion actor.
    """
    proprio = np.asarray(encoded["proprio"], np.float32).reshape(PROPRIO_F)
    images = encoded["images"]
    frames = {cam: {"rgb": images[cam + "_rgb"], "depth": images[cam + "_depth"]}
              for cam in CAMERAS}
    return {"frames": frames, "proprio": proprio}


def stack_frames(frames):
    chans = []
    for cam in CAMERAS:
        rgb = np.asarray(frames[cam]["rgb"], np.float32)
        if rgb.ndim == 2:
            rgb = rgb[..., None]
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        rgb = rgb[..., :3] / 255.0
        depth = np.clip(np.asarray(frames[cam]["depth"], np.float32), 0.0, 2.0) / 2.0
        chans.append(np.transpose(rgb, (2, 0, 1)))
        chans.append(depth[None])
    return np.concatenate(chans, 0).astype(np.float32)
