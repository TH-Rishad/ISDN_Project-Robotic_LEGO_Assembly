import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import unittest

from Kean.motion.brick_spheres import fit_cuboid, posed


class BrickSphereTest(unittest.TestCase):
    def test_arch_bricks_stay_inside_and_are_not_a_single_sphere(self):
        for dims in ([0.0158, 0.0158, 0.0114], [0.0318, 0.0158, 0.0096], [0.0478, 0.0158, 0.0114]):
            centers, radii = fit_cuboid(dims)
            self.assertGreater(len(centers), 1)
            self.assertLessEqual(len(centers), 12)
            half = np.asarray(dims) / 2
            self.assertTrue(np.all(np.abs(centers) + radii[:, None] <= half + 1e-9))

    def test_pose_is_applied_in_the_parent_frame(self):
        dims = [0.0318, 0.0158, 0.0096]
        centers, radii = fit_cuboid(dims)
        spheres = posed(dims, [0.1, -0.2, 0.03], [1, 0, 0, 0])
        np.testing.assert_allclose(spheres[:, :3], centers + [0.1, -0.2, 0.03])
        np.testing.assert_allclose(spheres[:, 3], radii)
        turned = posed(dims, [0, 0, 0], [0, 0, 0, 1])
        np.testing.assert_allclose(turned[:, 0], -centers[:, 0], atol=1e-9)
        np.testing.assert_allclose(turned[:, 1], -centers[:, 1], atol=1e-9)

    def test_carried_plan_does_not_call_morphit(self):
        from Kean.motion.plan_curobo import CuroboBackend, Request

        probe = SimpleNamespace(
            tool_pose_w=lambda *args: (np.zeros(3), [1, 0, 0, 0]),
            js7=lambda q: q,
            goal_from=lambda *args: None,
            plan_ok=lambda result: True,
            plan_traj=lambda *args: (np.zeros((2, 7)), None, 0.02),
            guard=lambda: SimpleNamespace(since_mark=lambda: {}),
        )
        attachment = SimpleNamespace(update=Mock(), fit_spheres=Mock())
        planner = SimpleNamespace(plan_pose=Mock(return_value=object()), attachment_manager=attachment)
        backend = CuroboBackend()
        backend.probe = probe
        backend.rig = SimpleNamespace(pl={"B": planner}, detach_brick=Mock())
        request = Request("B", np.zeros(9), np.zeros(3), np.ones(3), [0, 0, 0, 1],
                          carried=dict(dims=[0.0318, 0.0158, 0.0096], center=[0, 0, 0.02], quat_wxyz=[1, 0, 0, 0]))
        with patch("Kean.motion.brick_spheres.as_link_spheres", return_value="sph") as fit:
            backend.plan(request, request.goal)
        attachment.fit_spheres.assert_not_called()
        fit.assert_called_once()
        self.assertIs(attachment.update.call_args.args[0], "sph")
        backend.rig.detach_brick.assert_called_once_with("B")


if __name__ == "__main__":
    unittest.main()
