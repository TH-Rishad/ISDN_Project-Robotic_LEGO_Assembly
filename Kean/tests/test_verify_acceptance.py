import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Kean.verify_acceptance import (
    EXIT_FAIL,
    EXIT_PASS,
    EXIT_UNVERIFIED,
    assess,
    directory_occupied,
    hold_timing,
    package_from_plans,
    plan_problems,
    support_steps,
    verify_run,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "Kean" / "scripts" / "verify_acceptance.py"


def plan(name, n=8, support=7, count=1):
    bricks = [{"id": "b_%03d" % i, "type": "2x2"} for i in range(n)]
    sequence = []
    for i in range(n):
        step = {"step": i, "brick_id": bricks[i]["id"], "requires_brace": i == support, "placer_arm": "B"}
        if i == support:
            step["brace"] = {
                "arm": "A", "feasible": True, "target_brick_id": "b_000",
                "brace_pose": [0.0, 0.0, 0.01], "strategy": "weakest_joint",
            }
        sequence.append(step)
    return {
        "structure_id": name,
        "source": {"dataset": "hand_authored"},
        "bricks": bricks,
        "sequence": sequence,
        "validation": {
            "bracing_strategy": "weakest_joint",
            "brace_required_count": count,
            "stability_method": "stablelego_force_balance",
            "brace_model": "grasp_lp",
        },
    }


def frames(step, during="hold", before=True, after=True):
    rows = []
    if before:
        rows.append({"frame": 1, "b_step": step, "a_phase": "hold", "b_phase": "pre-insert"})
    rows.append({"frame": 2, "b_step": step, "a_phase": during, "b_phase": "insert"})
    if after:
        rows.append({"frame": 3, "b_step": step, "a_phase": "open", "b_phase": "retract"})
    return rows


def episodes(saved):
    return [
        {
            "structure": saved["structure_id"], "step": step["step"], "brick_id": step["brick_id"],
            "success": True, "failure_mode": None, "backend": "newton_mujoco", "joint_model": "weld_pool",
        }
        for step in saved["sequence"]
    ]


def recording(saved):
    step = support_steps(saved)[0]
    return {"frames": frames(step), "fired": [step["brick_id"] for step in saved["sequence"]], "failure": ""}


def cell(saved, done=True):
    n = len(saved["sequence"])
    row = {"structure": saved["structure_id"], "placed": n, "total": n, "error": None, "event_counts": {"penetration": 1}}
    if done is not None:
        row["done"] = done
    return row


def package(arch, s3, provenance=True, done=True, arch_episodes=None, during="hold"):
    arch_rec = recording(arch)
    s3_rec = recording(s3)
    if during != "hold":
        arch_rec["frames"] = frames(support_steps(arch)[0], during=during)
    structures = []
    for saved, rec in ((arch, arch_rec), (s3, s3_rec)):
        structures.append({
            "name": saved["structure_id"],
            "plan": saved,
            "plan_path": "plans/%s.json" % saved["structure_id"],
            "plan_sha256": saved["structure_id"] + "-hash",
            "episodes": episodes(saved) if arch_episodes is None or saved["structure_id"] != "arch" else arch_episodes,
            "cells": [cell(saved, done=done)],
            "summary_runs": [{"shape": saved["structure_id"], "done": True, "failure": None, "placed": len(saved["sequence"]), "total": len(saved["sequence"])}] if done else [],
            "recording": rec,
        })
    payload = {
        "structures": structures,
        "duplicates": [],
        "contacts": [{"structure": "arch", "counts": {"penetration": 1}}],
        "ignored_flags": ["ok", "full_acceptance"],
        "provenance": {
            "git_revision": "abc",
            "dirty_hashes": {},
            "config": {"image_input": False, "strategy": "weakest_joint", "seeds": {"arch": 0, "S3": 0}},
            "simulator": {"backend": "newton_mujoco"},
            "plan_hashes": {"arch": "arch-hash", "S3": "S3-hash"},
        } if provenance else None,
    }
    return payload


class VerifierTest(unittest.TestCase):
    def test_import_does_not_load_newton(self):
        self.assertNotIn("newton", sys.modules)
        self.assertNotIn("dual_arm_sim", sys.modules)

    def test_support_count_is_derived(self):
        saved = plan("arch")
        self.assertEqual(support_steps(saved), [7])
        saved["validation"]["brace_required_count"] = 2
        self.assertTrue(any("brace_required_count" in item for item in plan_problems(saved)))

    def test_complete_package_passes_and_keeps_contacts(self):
        report = assess(package(plan("arch"), plan("S3")), "evidence")
        self.assertEqual(report["exit_code"], EXIT_PASS)
        self.assertEqual(report["contacts"][0]["counts"]["penetration"], 1)
        self.assertEqual(report["hardware"], "UNVERIFIED")

    def test_plan_check_leaves_execution_unverified(self):
        report = assess(package_from_plans({"arch": plan("arch"), "S3": plan("S3")}), "plan-check")
        self.assertEqual(report["criteria"]["3"]["status"], "UNVERIFIED")
        self.assertEqual(report["exit_code"], EXIT_UNVERIFIED)

    def test_one_structure_does_not_pass(self):
        report = assess(package_from_plans({"arch": plan("arch")}), "plan-check")
        self.assertNotEqual(report["exit_code"], EXIT_PASS)
        evidence = package(plan("arch"), plan("S3"))
        evidence["structures"] = evidence["structures"][:1]
        report = assess(evidence, "evidence")
        self.assertEqual(report["exit_code"], EXIT_FAIL)

    def test_duplicate_structure_fails(self):
        payload = package(plan("arch"), plan("S3"))
        payload["duplicates"] = ["arch"]
        self.assertEqual(assess(payload, "evidence")["exit_code"], EXIT_FAIL)

    def test_fewer_than_eight_bricks_fails(self):
        report = assess(package(plan("arch", n=7, support=6), plan("S3")), "evidence")
        self.assertEqual(report["exit_code"], EXIT_FAIL)
        self.assertEqual(report["criteria"]["2"]["status"], "FAIL")

    def test_missing_plan_is_not_a_pass(self):
        payload = package(plan("arch"), plan("S3"))
        payload["structures"][1]["plan"] = None
        payload["structures"][1]["plan_sha256"] = None
        report = assess(payload, "evidence")
        self.assertNotEqual(report["exit_code"], EXIT_PASS)
        self.assertNotEqual(report["criteria"]["2"]["status"], "PASS")

    def test_altered_plan_hash_fails(self):
        payload = package(plan("arch"), plan("S3"))
        payload["provenance"]["plan_hashes"]["arch"] = "changed"
        report = assess(payload, "evidence")
        self.assertEqual(report["exit_code"], EXIT_FAIL)
        self.assertTrue(any("hash" in check["detail"] for check in report["criteria"]["3"]["checks"]))

    def test_missing_provenance_is_unverified(self):
        report = assess(package(plan("arch"), plan("S3"), provenance=False), "evidence")
        self.assertEqual(report["criteria"]["3"]["status"], "UNVERIFIED")
        self.assertEqual(report["exit_code"], EXIT_UNVERIFIED)

    def test_missing_completion_is_unverified(self):
        payload = package(plan("arch"), plan("S3"), done=None)
        report = assess(payload, "evidence")
        self.assertEqual(report["criteria"]["3"]["status"], "UNVERIFIED")
        self.assertNotEqual(report["exit_code"], EXIT_PASS)

    def test_incomplete_placement_ignores_saved_success(self):
        arch = plan("arch")
        payload = package(arch, plan("S3"), arch_episodes=episodes(arch)[:-1])
        report = assess(payload, "evidence")
        self.assertEqual(report["exit_code"], EXIT_FAIL)
        self.assertIn("ok", report["ignored_flags"])
        self.assertIn("full_acceptance", report["ignored_flags"])

    def test_support_held_at_the_wrong_time_fails(self):
        self.assertEqual(hold_timing(frames(7, during="park"), 7)["status"], "FAIL")
        self.assertEqual(hold_timing(frames(7, before=False), 7)["status"], "FAIL")
        self.assertEqual(hold_timing(frames(7, after=False), 7)["status"], "FAIL")
        report = assess(package(plan("arch"), plan("S3"), during="park"), "evidence")
        self.assertEqual(report["exit_code"], EXIT_FAIL)

    def test_run_directory_and_script_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "partial"
            (run / "plans").mkdir(parents=True)
            saved = plan("arch")
            (run / "plans" / "arch_weakest_joint.json").write_text(json.dumps(saved))
            (run / "summary.json").write_text(json.dumps({
                "ok": True, "full_acceptance": True, "scope": "arch",
                "runs": [{"shape": "arch", "done": True, "failure": None, "placed": 8, "total": 8}],
            }))
            report = verify_run(run)
            self.assertEqual(report["exit_code"], EXIT_FAIL)
            out = root / "verification"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--run", str(run), "--out", str(out)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, EXIT_FAIL, result.stderr)
            self.assertTrue((out / "verification.json").exists())
            self.assertTrue((out / "verification.md").exists())

    def test_occupied_directory_is_refused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            out = Path(tmp).resolve()
            (out / "summary.json").write_text("{}")
            self.assertTrue(directory_occupied(out))
            result = subprocess.run(
                [sys.executable, str(ROOT / "Kean/scripts/accept.py"), "--dry-run", "--out", str(out)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stderr)


if __name__ == "__main__":
    unittest.main()
