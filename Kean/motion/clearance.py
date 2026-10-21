"""Executed free-space clearance using the probe's full collision spheres.

Finger spheres are translated to measured openings, as in P3 step 8. This
measures playback frames; contact segments remain covered by ContactMonitor.
"""
import numpy as np


class Clearance:
    def __init__(self, probe, world):
        self.probe, self.world = probe, world
        # Cache before playback instead of rebuilding the link map each tick.
        self.slices = probe.link_slices('full')

    def spheres(self, name, joints):
        probe = self.probe
        q = np.asarray(joints[:7])[None]
        spheres = probe.arm_world_spheres(name, q, 'full')[0].copy()
        _, rotations = probe.tcp_poses(name, q)
        y = rotations[0, :, 1]
        for link, opening, sign in (('fr3_leftfinger', joints[7], 1),
                                    ('fr3_rightfinger', joints[8], -1)):
            spheres[self.slices[link], :3] += sign * (opening - probe.FINGER_LOCK) * y
        return spheres

    def __call__(self, example, name, commanded_q):
        joints = example.state_0.joint_q.numpy()
        k = 0 if name == 'A' else 1
        own = self.spheres(name, joints[9*k:9*k+9])
        other_name = 'B' if name == 'A' else 'A'
        other = self.spheres(other_name, joints[9*(1-k):9*(1-k)+9])
        boxes, _ = self.world(example, name)
        gap, _ = self.probe.clearance_boxes(own, boxes)
        return min(gap, self.probe.clearance_spheres(own, other))
