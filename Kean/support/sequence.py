"""Arm A settles before Arm B moves. At most one arm moves inside the build cylinder."""

import numpy as np

MARGIN = 0.12


class ZoneBusy(RuntimeError):
    pass


class NotSettled(RuntimeError):
    pass


class Zone:
    def __init__(self, center, half_diagonal, margin=MARGIN):
        self.center = np.asarray(center, float)
        self.radius = float(half_diagonal) + float(margin)

    def contains(self, pos):
        d = np.asarray(pos, float)[:2] - self.center[:2]
        return float(np.dot(d, d)) <= self.radius * self.radius


class Handoff:
    def __init__(self, zone):
        self.zone = zone
        self.a = "parked"
        self.b = "parked"

    def _state(self, arm):
        return self.a if arm == "A" else self.b

    def _set(self, arm, state):
        if arm == "A":
            self.a = state
        else:
            self.b = state

    def begin(self, arm, target, other_pos):
        other = "B" if arm == "A" else "A"
        if arm == "B" and self.a != "holding":
            raise NotSettled("A has not settled")
        if self._state(other) == "moving" and (self.zone.contains(target) or self.zone.contains(other_pos)):
            raise ZoneBusy(arm)
        self._set(arm, "moving")

    def settled(self, arm):
        self._set(arm, "holding" if arm == "A" else "parked")
