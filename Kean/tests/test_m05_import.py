"""Runner bootstrap must select Kean's motion even in a cell environment."""
import subprocess
import sys
import unittest
from pathlib import Path


class RunnerImportTest(unittest.TestCase):
    def test_conflicting_cached_package_and_pythonpath(self):
        kean = Path(__file__).resolve().parents[1]
        code = """
import runpy
import sys
import types
from pathlib import Path
kean = Path(sys.argv[1])
# Reproduce Kean already being on PYTHONPATH, behind another package.
sys.path.append(str(kean))
wrong = types.ModuleType('motion')
wrong.__path__ = []
sys.modules['motion'] = wrong
sys.modules['motion.gate'] = types.ModuleType('motion.gate')
ns = runpy.run_path(str(kean / 'scripts' / 'm05.py'), run_name='m05_import_test')
assert sys.modules['motion'] is wrong
assert Path(sys.modules['Kean.motion.gate'].__file__).resolve() == kean / 'motion' / 'gate.py'
assert len(ns['jobs'](5, 3)) == 23
"""
        result = subprocess.run([sys.executable, "-c", code, str(kean)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
