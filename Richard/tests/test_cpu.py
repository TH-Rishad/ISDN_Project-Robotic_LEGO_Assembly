"""CPU self-checks for the Richard control flow. No Newton required.

    python -m unittest Richard.tests.test_cpu -v        (from the repo root)

Covers what is provable without the GPU cell: criteria 1 and 2 of the
100-point rubric (direct structure input; generated sequence + supports),
plus import safety of the GPU-lazy modules.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Richard.plans import ACCEPTANCE, generate, preview, preview_ok  # noqa: E402
from Richard.verify import assess, package_from_plans, plan_problems, support_steps  # noqa: E402


class ImportSafety(unittest.TestCase):
    def test_runner_imports_without_newton(self):
        import Richard.runner  # noqa: F401

    def test_verify_imports_without_newton(self):
        import Richard.verify  # noqa: F401


class PlanGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plans = {name: generate(name, kind) for name, kind in ACCEPTANCE}

    def test_direct_structure_ids(self):
        for name, plan in self.plans.items():
            self.assertEqual(plan.get("structure_id"), name)
            source = plan.get("source") or {}
            self.assertFalse(source.get("image"))
            self.assertNotEqual(source.get("dataset"), "image")

    def test_at_least_eight_bricks_each(self):
        for name, plan in self.plans.items():
            self.assertGreaterEqual(len(plan["sequence"]), 8, name)

    def test_generated_supports(self):
        for name, plan in self.plans.items():
            supports = support_steps(plan)
            self.assertTrue(supports, "%s has no generated support" % name)
            validation = plan.get("validation") or {}
            self.assertEqual(validation.get("bracing_strategy"), "weakest_joint")
            self.assertEqual(validation.get("brace_required_count"), len(supports))

    def test_no_plan_problems(self):
        for name, plan in self.plans.items():
            self.assertEqual(plan_problems(plan), [], name)

    def test_plan_check_assessment(self):
        report = assess(package_from_plans(self.plans), "plan-check")
        self.assertEqual(report["criteria"]["1"]["status"], "PASS")
        self.assertEqual(report["criteria"]["2"]["status"], "PASS")
        self.assertEqual(report["criteria"]["3"]["status"], "UNVERIFIED")
        self.assertEqual(report["exit_code"], 2)

    def test_preview_is_clean(self):
        self.assertEqual(preview_ok(preview()), [])


if __name__ == "__main__":
    unittest.main()
