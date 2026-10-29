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


if __name__ == "__main__":
    unittest.main()
