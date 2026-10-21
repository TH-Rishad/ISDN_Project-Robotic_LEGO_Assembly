"""Insertion starts at the pre-insert pose. Press stays at n * 8.9 N."""

import numpy as np

from Kean.insertion.env import F_INSERT, scripted_insert
from Kean.insertion.obs import CAMERAS


class NotAtPreInsert(RuntimeError):
    pass


class OverPress(RuntimeError):
    pass


def arrived(arm):
    nxt = arm.moves[0]["phase"] if arm.moves else None
    return arm.phase == "pre-insert" and arm.cur is None and nxt == "insert"


class KinematicInsert:
    """CPU stand-in. A negative z command lowers the brick. Snap is the only success."""

    def __init__(self, n_studs=8):
        self.n_studs = int(n_studs)
        self.z = 0.025
        self.snapped = False
        self.broken = False
        self.dropped = False
        self.force_n = 0.0

    def reset(self):
        self.z = 0.025
        self.snapped = False
        self.broken = False
        self.dropped = False
        self.force_n = 0.0

    def sense(self):
        frames = {cam: {"rgb": np.zeros((8, 8, 3), np.uint8), "depth": np.ones((8, 8), np.float32)} for cam in CAMERAS}
        return {"frames": frames, "q": np.zeros(18), "qd": np.zeros(18), "hands": [np.zeros(7), np.zeros(7)], "stride": 1}

    def scripted_action(self, step):
        return scripted_insert(step)

    def apply(self, command, press):
        self.force_n = float(press)
        self.z += float(command[2]) * 0.001
        if self.z <= 0.0 and press > 0.0:
            self.snapped = True

    def privileged(self):
        return {
            "snap": self.snapped, "broken": self.broken, "dropped": self.dropped, "force_n": self.force_n,
            "true_delta": np.array([0.0, 0.0, self.z]), "clutch": 1 if self.snapped else 0, "util": 0.0,
        }


class PreInsertCell:
    def __init__(self, inner, at_pre_insert=True):
        if not at_pre_insert:
            raise NotAtPreInsert("insertion starts at the pre-insert pose")
        self.inner = inner
        self.n_studs = inner.n_studs

    def reset(self):
        return self.inner.reset()

    def sense(self):
        return self.inner.sense()

    def scripted_action(self, step):
        return self.inner.scripted_action(step)

    def privileged(self):
        return self.inner.privileged()

    def apply(self, command, press):
        cap = self.n_studs * F_INSERT
        if float(press) > cap + 1e-6:
            raise OverPress("press %.3f N above %.3f N" % (press, cap))
        self.inner.apply(command, press)
