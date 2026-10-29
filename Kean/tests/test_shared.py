"""Integration boundaries that can be checked without importing the GPU stack."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Kean import shared
from Kean.motion.plan_curobo import Planner


class SharedTest(unittest.TestCase):
    def test_probe_keeps_shared_models_and_redirects_only_reports(self):
        profiles = {name: {"yml": Path("/shared") / (name + ".yml")}
                    for name in ("full", "plan", "mixed", "lite")}
        probe = SimpleNamespace(PROFILES=profiles, HERE=Path("/shared"),
                                FR3_YML=Path("/shared/full.yml"))
        before = {name: row["yml"] for name, row in profiles.items()}
        shared.configure_probe(probe)
        self.assertEqual(probe.OUT, shared.KEAN / "results/p3")
        self.assertEqual({name: row["yml"] for name, row in profiles.items()}, before)
        self.assertEqual(probe.HERE, Path("/shared"))
        self.assertEqual(probe.FR3_YML, Path("/shared/full.yml"))

    def test_planner_closes_when_work_raises(self):
        planner = Planner(SimpleNamespace(warmup=lambda guard: None))
        planner.warmup()
        with self.assertRaisesRegex(ValueError, "failure"):
            with planner:
                raise ValueError("failure")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            planner.submit(None)

    def test_probe_import_uses_shared_overlay_and_cleans_failed_cache(self):
        import importlib
        shared.add_cell_path()
        vendor = importlib.import_module("motion.vendor")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            source = Path(tmp) / "probe.py"
            source.write_text("from motion.vendor import add_overlay\nadd_overlay()\n"
                              "raise ModuleNotFoundError('missing', name='missing_dep')\n")
            hook = Mock()
            had = hasattr(vendor, "add_overlay")
            original = getattr(vendor, "add_overlay", None)
            if not had:
                vendor.add_overlay = lambda: None
            try:
                with patch.object(vendor, "add_overlay", hook), patch.object(shared, "PROBE", source):
                    with self.assertRaises(ModuleNotFoundError):
                        shared.load_probe()
                    hook.assert_called_once_with()
                    self.assertIs(vendor.add_overlay, hook)
                    self.assertNotIn("Kean._curobo_probe", sys.modules)
                if had:
                    self.assertIs(vendor.add_overlay, original)
            finally:
                if not had:
                    del vendor.add_overlay

    def test_p3_model_generation_stays_with_sakib(self):
        from Kean.scripts.p3_env import parse_args
        import contextlib
        import io
        for args in (["--all"], ["--step", "2"], ["--step=2"], ["--step=02"], ["--al"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)
        options, remaining = parse_args(["--summary"])
        self.assertFalse(options.kean_extensions)
        self.assertEqual(remaining, ["--summary"])

    def test_p3_default_does_not_modify_the_shared_probe(self):
        from Kean.scripts import p3_env
        from Kean.motion import contact_check, iter_budget, sphere_fit
        probe = SimpleNamespace(main=Mock())
        with patch.object(p3_env, "load_probe", return_value=probe), patch.object(sys, "argv", ["p3", "--summary"]):
            with patch.object(contact_check, "install") as contacts, patch.object(iter_budget, "install") as budget, patch.object(sphere_fit, "install") as fit:
                p3_env.main()
                probe.main.assert_called_once_with()
                contacts.assert_not_called()
                budget.assert_not_called()
                fit.assert_not_called()
