"""When a scan is allowed. Held bricks are not overwritten, except the one whose grip is being adjusted."""

import numpy as np

from Kean.memory.store import REASONS, yaw_quat


class ScanRefused(RuntimeError):
    pass


def _quat_of(det):
    if "quat" in det and det["quat"] is not None:
        return np.asarray(det["quat"], float)
    return yaw_quat(float(det["yaw"]))


class ScanSchedule:
    def __init__(self):
        self.reason = None
        self.adjusting = None

    def allow_init(self):
        self.reason = "init"
        self.adjusting = None

    def allow_step_done(self):
        self.reason = "step_done"
        self.adjusting = None

    def begin_adjust(self, brick_id):
        self.reason = "grip_adjust"
        self.adjusting = brick_id

    def end_adjust(self):
        self.reason = None
        self.adjusting = None

    def apply(self, store, detections):
        if self.reason not in REASONS:
            raise ScanRefused(self.reason)
        held = store.held_ids()
        for det in detections:
            brick_id = det["id"]
            if brick_id in held and not (self.reason == "grip_adjust" and brick_id == self.adjusting):
                continue
            store.remember(brick_id, det["xyz"], _quat_of(det))
        if self.reason != "grip_adjust":
            self.reason = None


def lifted(store, prev, phase, arm, brick_id, hand_xyz, hand_q, brick_xyz, brick_q):
    if prev != "grasp" or phase in ("grasp", "descend", "settle") or brick_id in store.held_ids():
        return False
    store.freeze(arm, brick_id, hand_xyz, hand_q, brick_xyz, brick_q)
    return True
