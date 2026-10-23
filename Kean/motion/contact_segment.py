"""The last 30 mm is a straight line, checked every 2 mm. MotionGen does not plan it."""

import numpy as np

CONTACT_M = 0.030
STEP_M = 0.002


def split(start, goal, contact_m=CONTACT_M):
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    delta = goal - start
    dist = float(np.linalg.norm(delta))
    if dist <= contact_m or dist == 0.0:
        return start.copy(), goal.copy()
    return goal - delta / dist * contact_m, goal.copy()


def sample(start, goal, step_m=STEP_M):
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    delta = goal - start
    dist = float(np.linalg.norm(delta))
    if dist == 0.0:
        return start.reshape(1, -1)
    n = int(np.ceil(dist / step_m))
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return start + delta * t
