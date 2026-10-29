import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest

from Kean.motion.iter_budget import _SCALED, apply


class Opt:
    def __init__(self):
        self.n_iters = 100
        self.inner_iters = 20
        self.interpolation_steps = 5000


class Solver:
    def __init__(self):
        self.optimizer = Opt()
        self.warmup_iters = 5


class Planner:
    def __init__(self):
        self.trajopt_solver = Solver()


class TrajCfg:
    def __init__(self):
        self.num_iters = 100
        self.inner_iters = 25


class IkCfg:
    def __init__(self):
        self.num_iters = 100
        self.inner_iters = 20


class GraphCfg:
    def __init__(self):
        self.max_path_finding_iterations = 10


class SplitPlanner:
    def __init__(self):
        self.trajopt_solver_config = TrajCfg()
        self.ik_solver_config = IkCfg()
        self.graph_planner_config = GraphCfg()


class IterBudgetTest(unittest.TestCase):
    def tearDown(self):
        _SCALED.clear()

    def test_scales_iter_fields_only(self):
        planner = Planner()
        changed = apply(planner, 0.25)
        self.assertEqual(planner.trajopt_solver.optimizer.n_iters, 25)
        self.assertEqual(planner.trajopt_solver.optimizer.inner_iters, 5)
        self.assertEqual(planner.trajopt_solver.optimizer.interpolation_steps, 5000)
        self.assertEqual(planner.trajopt_solver.warmup_iters, 5)
        self.assertEqual(len(changed), 2)

    def test_floor_and_idempotent(self):
        planner = Planner()
        planner.trajopt_solver.optimizer.inner_iters = 5
        apply(planner, 0.25)
        self.assertEqual(planner.trajopt_solver.optimizer.inner_iters, 4)
        again = apply(planner, 0.25)
        self.assertEqual(again, [])
        self.assertEqual(planner.trajopt_solver.optimizer.n_iters, 25)

    def test_num_iters_stays_a_multiple_of_inner_iters(self):
        planner = SplitPlanner()
        apply(planner, 0.25)
        cfg = planner.trajopt_solver_config
        self.assertEqual(cfg.inner_iters, 6)
        self.assertEqual(cfg.num_iters, 24)
        self.assertEqual(cfg.num_iters % cfg.inner_iters, 0)

    def test_leaves_ik_and_graph_search(self):
        planner = SplitPlanner()
        apply(planner, 0.25)
        self.assertEqual(planner.ik_solver_config.num_iters, 100)
        self.assertEqual(planner.ik_solver_config.inner_iters, 20)
        self.assertEqual(planner.graph_planner_config.max_path_finding_iterations, 10)

    def test_scale_one_is_a_noop(self):
        planner = Planner()
        self.assertEqual(apply(planner, 1.0), [])
        self.assertEqual(planner.trajopt_solver.optimizer.n_iters, 100)


if __name__ == "__main__":
    unittest.main()


class DefaultBudgetTest(unittest.TestCase):
    def test_shared_optimizer_settings_are_default(self):
        import os
        from unittest.mock import patch
        from Kean.motion.iter_budget import scale_from_env
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(scale_from_env(), 1.0)
