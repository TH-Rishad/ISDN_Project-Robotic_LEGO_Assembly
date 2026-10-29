"""Build-zone geometry for the motion adapter (Arm A/B exclusion cylinder)."""

import numpy as np

MARGIN = 0.12


class Zone:
    def __init__(self, center, half_diagonal, margin=MARGIN):
        self.center = np.asarray(center, float)
        self.radius = float(half_diagonal) + float(margin)

    def contains(self, pos):
        d = np.asarray(pos, float)[:2] - self.center[:2]
        return float(np.dot(d, d)) <= self.radius * self.radius
