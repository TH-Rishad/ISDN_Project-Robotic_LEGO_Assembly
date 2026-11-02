import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Kean.scripts.m05 import resolve_out
from Kean.scripts.test_m05 import run

ROOT = Path(__file__).resolve().parents[2]


class TestRunnerTest(unittest.TestCase):
    def test_relative_out_ignores_launcher_cwd(self):
        kean = ROOT / "Kean"
        self.assertEqual(resolve_out("Kean/results/m05_collision"), kean / "results" / "m05_collision")
        self.assertEqual(resolve_out("results/m05_collision"), kean / "results" / "m05_collision")
    def test_failed_command_keeps_output_and_exit_status(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            log = Path(tmp) / "failure.log"
            status = run([sys.executable, "-c", "print('diagnostic'); raise SystemExit(7)"], log)
            self.assertEqual(status, 7)
            self.assertIn("diagnostic", log.read_text())

    def test_dry_run_isolated_output_and_invalid_job(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            out = Path(tmp).resolve()
            base = [sys.executable, str(ROOT / "Kean/scripts/m05.py"), "--dry-run", "--out", str(out)]
            result = subprocess.run(base, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((out / "jobs.json").read_text())["job_count"], 23)
            result = subprocess.run(base + ["--job", "-1"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("job index", result.stderr)

    def test_score_reads_saved_jobs_without_newton(self):
        from Kean.motion.gate import jobs
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            out = Path(tmp).resolve()
            (out / "runs").mkdir()
            for job in jobs():
                row = {
                    "shape": job["shape"], "seed": job["seed"], "stress": job["stress"],
                    "strategy": job["strategy"], "placed": job["bricks"], "events": [],
                    "cycle_s": [1.0], "min_clearance_m": 0.01,
                    "motion_report": {"plan_failures": 0, "captures": 0},
                }
                name = "%s_%s_s%d.json" % (job["shape"], job["strategy"], job["seed"])
                (out / "runs" / name).write_text(json.dumps(row))
            result = subprocess.run(
                [sys.executable, str(ROOT / "Kean/scripts/m05.py"), "--score", "--out", str(out)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads((out / "summary.json").read_text())["ok"])
            occupied = subprocess.run(
                [sys.executable, str(ROOT / "Kean/scripts/m05.py"), "--job", "0", "--out", str(out)],
                capture_output=True, text=True)
            self.assertEqual(occupied.returncode, 2)
            self.assertIn("refusing", occupied.stderr)


class InterpreterTest(unittest.TestCase):
    def test_all_launchers_use_only_shared_venv(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp).resolve()
            scripts = root / "Kean/scripts"
            scripts.mkdir(parents=True)
            for name in ("run.sh", "m05.sh", "test_m05.sh", "accept.sh"):
                shutil.copy2(ROOT / "Kean/scripts" / name, scripts / name)
            shared_launcher = root / "Sakib/brickassembly/scripts/run.sh"
            shared_launcher.parent.mkdir(parents=True)
            shared_launcher.write_text('#!/bin/sh\nexec "$ROBOFAB_PY" -u "$@"\n')
            venv = root / "Sakib/.venv/bin/python"
            venv.parent.mkdir(parents=True)
            venv.write_text('#!/bin/sh\necho shared-venv\nprintf "%s\\n" "$@"\n')
            venv.chmod(0o755)
            env = dict(os.environ, P3_PY="/wrong/python", ROBOFAB_PY="/wrong/python")
            for name in ("m05.sh", "test_m05.sh", "accept.sh"):
                result = subprocess.run(["sh", str(scripts / name), "--help"], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("shared-venv", result.stdout)
                self.assertIn("-u", result.stdout)
                self.assertIn("--help", result.stdout)
            venv.unlink()
            result = subprocess.run(["sh", str(scripts / "test_m05.sh")], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Sakib/.venv is unavailable", result.stderr)
