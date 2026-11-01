"""Recompute instructor acceptance from plans and records.

Summary ``ok`` and ``full_acceptance`` are ignored. Missing provenance is
UNVERIFIED, not a pass. Importing this module does not import Newton.
"""

import hashlib
import json
import platform
import socket
import subprocess
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_UNVERIFIED = 2
EXPECTED = ("arch", "S3")
MIN_BRICKS = 8
HOLD = "hold"
INSERT = "insert"
RELEASED = frozenset({"open", "retract", "park"})
OCCUPIED = ("summary.json", "manifest.json", "cell_rows.jsonl", "episodes.jsonl", "provenance.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_occupied(path):
    path = Path(path)
    if not path.exists():
        return False
    if any((path / name).exists() for name in OCCUPIED):
        return True
    if any(path.glob("*.npz")):
        return True
    plans = path / "plans"
    return plans.is_dir() and any(plans.glob("*.json"))


def support_steps(plan):
    return [step["step"] for step in plan.get("sequence") or [] if step.get("requires_brace")]


def _git(repo, args):
    try:
        result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def capture_provenance(repo, config):
    repo = Path(repo)
    revision = _git(repo, ["rev-parse", "HEAD"])
    porcelain = _git(repo, ["status", "--porcelain", "--", "Kean"]) or ""
    dirty = {}
    for line in porcelain.splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        file = repo / path
        if file.is_file():
            dirty[path] = sha256_file(file)
    return {
        "git_revision": revision,
        "dirty_hashes": dirty,
        "config": config,
        "simulator": None,
        "plan_hashes": {},
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
    }


def _check(status, detail, evidence=None):
    return {"status": status, "detail": detail, "evidence": evidence or []}


def _fail(detail, evidence=None):
    return _check("FAIL", detail, evidence)


def _unverified(detail, evidence=None):
    return _check("UNVERIFIED", detail, evidence)


def _pass(detail, evidence=None):
    return _check("PASS", detail, evidence)


def rollup(checks):
    states = [row["status"] for row in checks]
    if any(state == "FAIL" for state in states):
        return "FAIL"
    if any(state != "PASS" for state in states):
        return "UNVERIFIED"
    return "PASS"


def _pose_ok(pose):
    return isinstance(pose, (list, tuple)) and len(pose) == 3 and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in pose)


def plan_problems(plan):
    problems = []
    if not plan:
        return ["missing plan"]
    source = plan.get("source") or {}
    if source.get("dataset") == "image" or source.get("image") or plan.get("image"):
        problems.append("image input")
    if not plan.get("structure_id"):
        problems.append("no structure id")
    sequence = plan.get("sequence") or []
    ids = [step.get("brick_id") for step in sequence]
    declared = [brick.get("id") for brick in plan.get("bricks") or []]
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(declared):
        problems.append("sequence does not cover each target brick once")
    if len(set(ids)) < MIN_BRICKS:
        problems.append("fewer than eight unique bricks")
    validation = plan.get("validation") or {}
    if validation.get("bracing_strategy") in (None, "none"):
        problems.append("strategy missing or none")
    if "stability_method" not in validation:
        problems.append("plan has no generator validation")
    seen = []
    supports = []
    for step in sequence:
        seen.append(step.get("brick_id"))
        if not step.get("requires_brace"):
            continue
        brace = step.get("brace") or {}
        supports.append(step.get("step"))
        if brace.get("feasible") is False:
            problems.append("step %s support infeasible" % step.get("step"))
        if brace.get("arm") not in (None, "A"):
            problems.append("step %s support is not Arm A" % step.get("step"))
        if not _pose_ok(brace.get("brace_pose")):
            problems.append("step %s support pose invalid" % step.get("step"))
        if brace.get("target_brick_id") not in seen[:-1]:
            problems.append("step %s support target was not already placed" % step.get("step"))
    count = validation.get("brace_required_count")
    if count is not None and count != len(supports):
        problems.append("brace_required_count %s disagrees with %s requires_brace steps" % (count, len(supports)))
    if not supports:
        problems.append("no generated support")
    return problems


def hold_timing(frames, step):
    rows = [row for row in frames if row.get("b_step") == step]
    inserts = [row for row in rows if row.get("b_phase") == INSERT]
    if not inserts:
        return _fail("no insert frames for step %s" % step)
    if any(row.get("a_phase") != HOLD for row in inserts):
        return _fail("Arm A was not holding during insertion of step %s" % step)
    first = min(row["frame"] for row in inserts)
    last = max(row["frame"] for row in inserts)
    if not any(row["frame"] < first and row.get("a_phase") == HOLD for row in rows):
        return _fail("Arm A was not holding before insertion of step %s" % step)
    if not any(row["frame"] > last and row.get("a_phase") in RELEASED for row in rows):
        return _fail("Arm A did not release after insertion of step %s" % step)
    return _pass("step %s held before and during insertion, then released" % step)


def _load_json(path):
    return json.loads(Path(path).read_text())


def _load_jsonl(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _recording(path):
    try:
        import numpy as np
    except ImportError:
        return None, "numpy unavailable"
    with np.load(path, allow_pickle=True) as record:
        names = [str(item) for item in record["phase_names"]]
        frames = [
            {
                "frame": int(record["frame"][index]),
                "a_phase": names[int(record["a_phase"][index])],
                "b_phase": names[int(record["b_phase"][index])],
                "b_step": int(record["b_step"][index]),
            }
            for index in range(len(record["frame"]))
        ]
        fired = [str(bid) for bid, frame in zip(record["brick_ids"], record["fire_frame_by_brick"]) if int(frame) >= 0]
        failure = str(record["failure"])
    return {"frames": frames, "fired": fired, "failure": failure}, None


def load_run(path):
    path = Path(path)
    plans = {}
    duplicates = []
    plan_dir = path / "plans"
    if plan_dir.is_dir():
        for file in sorted(plan_dir.glob("*.json")):
            plan = _load_json(file)
            name = plan.get("structure_id") or file.stem.split("_", 1)[0]
            if name in plans:
                duplicates.append(name)
            plans[name] = {"plan": plan, "path": file, "sha256": sha256_file(file)}
    summary = _load_json(path / "summary.json") if (path / "summary.json").exists() else {}
    summary_runs = {}
    for run in summary.get("runs") or []:
        summary_runs.setdefault(run.get("shape"), []).append(run)
    episodes = _load_jsonl(path / "episodes.jsonl") if (path / "episodes.jsonl").exists() else []
    cells = _load_jsonl(path / "cell_rows.jsonl") if (path / "cell_rows.jsonl").exists() else []
    provenance = _load_json(path / "provenance.json") if (path / "provenance.json").exists() else None
    names = set(plans) | set(summary_runs)
    structures = []
    for name in sorted(names):
        saved = plans.get(name) or {}
        recordings = sorted(path.glob(name + "_*.npz"))
        recording, recording_error = (None, None)
        if len(recordings) == 1:
            recording, recording_error = _recording(recordings[0])
        elif len(recordings) > 1:
            recording_error = "multiple recordings"
        structures.append({
            "name": name,
            "plan": saved.get("plan"),
            "plan_path": saved.get("path"),
            "plan_sha256": saved.get("sha256"),
            "episodes": [row for row in episodes if row.get("structure") == name],
            "cells": [row for row in cells if row.get("structure") == name],
            "summary_runs": summary_runs.get(name) or [],
            "recording": recording,
            "recording_path": recordings[0] if len(recordings) == 1 else None,
            "recording_error": recording_error,
        })
    contacts = []
    for row in cells:
        counts = row.get("event_counts") or {}
        if counts:
            contacts.append({"structure": row.get("structure"), "counts": counts})
    return {
        "path": path,
        "structures": structures,
        "duplicates": duplicates,
        "provenance": provenance,
        "contacts": contacts,
        "ignored_flags": [key for key in ("ok", "full_acceptance") if key in summary],
        "summary_scope": summary.get("scope"),
    }


def _placement(plan, episodes, fired):
    expected = [(step["step"], step["brick_id"]) for step in plan["sequence"]]
    if not episodes and fired is None:
        return _unverified("no placement records")
    if episodes:
        got = [(row.get("step"), row.get("brick_id")) for row in episodes]
        if len(got) != len(set(got)) or set(got) != set(expected):
            return _fail("placements missing, extra, or duplicated")
        if any(not row.get("success") or row.get("failure_mode") for row in episodes):
            return _fail("a placement failed")
    if fired is not None and set(fired) != {brick for _, brick in expected}:
        return _fail("recording weld set does not match the plan")
    return _pass("every planned brick placed once")


def _completion(cells, runs, recording):
    dones = []
    for row in cells:
        if "done" in row:
            dones.append(bool(row["done"]))
    for row in runs:
        if "done" in row:
            dones.append(bool(row["done"]))
    if not dones:
        return _unverified("no explicit completion record")
    if not all(dones):
        return _fail("execution did not complete")
    if any(row.get("error") for row in cells) or any(row.get("failure") for row in runs):
        return _fail("execution failure recorded")
    if recording and recording.get("failure"):
        return _fail("recording failure %s" % recording["failure"])
    if not cells and not recording:
        return _unverified("completion is only in the summary run")
    return _pass("explicit completion and no execution failure")


def _provenance(provenance, hashes):
    if not provenance:
        return _unverified("provenance.json missing")
    missing = []
    if not provenance.get("git_revision"):
        missing.append("git_revision")
    if not isinstance(provenance.get("dirty_hashes"), dict):
        missing.append("dirty_hashes")
    config = provenance.get("config") or {}
    if config.get("image_input") is not False:
        missing.append("config.image_input")
    if not config.get("strategy") or config.get("strategy") == "none":
        missing.append("config.strategy")
    seeds = config.get("seeds") or {}
    if not seeds:
        missing.append("config.seeds")
    if not (provenance.get("simulator") or {}).get("backend"):
        missing.append("simulator")
    stored = provenance.get("plan_hashes") or {}
    for name, digest in hashes.items():
        if name not in stored:
            missing.append("plan_hashes.%s" % name)
        elif stored[name] != digest:
            return _fail("plan hash does not match provenance for %s" % name)
    if missing:
        return _unverified("missing " + ", ".join(missing))
    return _pass("bound to revision, dirty hashes, config, seed and simulator")


def assess(package, mode, expected=EXPECTED):
    structures = package.get("structures") or []
    names = [row["name"] for row in structures]
    criterion1 = []
    criterion2 = []
    criterion3 = []
    image = any("image input" in " ".join(plan_problems(row.get("plan"))) for row in structures if row.get("plan"))
    config = (package.get("provenance") or {}).get("config") or {}
    if config.get("image_input") or image:
        criterion1.append(_fail("image conversion was used"))
    elif not structures:
        criterion1.append(_unverified("no structure input in this package"))
    elif any(not (row.get("plan") or {}).get("structure_id") for row in structures):
        criterion1.append(_unverified("a structure id is missing"))
    else:
        criterion1.append(_pass("direct structure ids, no image input"))
    if package.get("duplicates") or len(names) != len(set(names)):
        criterion2.append(_fail("duplicate structure"))
    if set(names) != set(expected):
        detail = "structures are %s, expected %s" % (sorted(set(names)), list(expected))
        criterion2.append(_unverified(detail) if not names else _fail(detail))
    for row in structures:
        problems = plan_problems(row.get("plan"))
        evidence = [str(row["plan_path"])] if row.get("plan_path") else []
        if row.get("plan") is None:
            criterion2.append(_unverified("missing plan for %s" % row["name"], evidence))
            continue
        structural = [item for item in problems if item != "image input"]
        if structural:
            criterion2.append(_fail("%s: %s" % (row["name"], "; ".join(structural)), evidence))
        else:
            criterion2.append(_pass(
                "%s: %s bricks, supports %s" % (row["name"], len(row["plan"]["sequence"]), support_steps(row["plan"])),
                evidence,
            ))
    if mode == "plan-check":
        criterion3.append(_unverified("execution was not run"))
    else:
        if set(names) != set(expected) or package.get("duplicates"):
            criterion3.append(_fail("a one-structure or duplicate package cannot pass"))
        hashes = {row["name"]: row["plan_sha256"] for row in structures if row.get("plan_sha256")}
        criterion3.append(_provenance(package.get("provenance"), hashes))
        for row in structures:
            if not row.get("plan"):
                criterion3.append(_unverified("cannot cross-check %s without its plan" % row["name"]))
                continue
            recording = row.get("recording")
            fired = None if recording is None else recording.get("fired")
            criterion3.append(_placement(row["plan"], row["episodes"], fired))
            criterion3.append(_completion(row["cells"], row["summary_runs"], recording))
            if row.get("recording_error"):
                criterion3.append(_unverified("%s recording: %s" % (row["name"], row["recording_error"])))
            elif recording is None:
                criterion3.append(_unverified("no per-frame recording for %s" % row["name"]))
            else:
                for step in support_steps(row["plan"]):
                    criterion3.append(hold_timing(recording["frames"], step))
    criteria = {
        "1": {"title": "Direct structure input", "status": rollup(criterion1), "checks": criterion1},
        "2": {"title": "Generated sequence and supports", "status": rollup(criterion2) if criterion2 else "UNVERIFIED", "checks": criterion2},
        "3": {"title": "Two braced dual-arm assemblies", "status": rollup(criterion3), "checks": criterion3},
    }
    report = {
        "mode": mode,
        "criteria": criteria,
        "contacts": package.get("contacts") or [],
        "ignored_flags": package.get("ignored_flags") or [],
        "hardware": "UNVERIFIED",
        "hardware_note": "Newton simulation only. A physical robot demonstration is not in this package.",
        "structures": [
            {
                "name": row["name"],
                "bricks": None if not row.get("plan") else len(row["plan"].get("sequence") or []),
                "supports": None if not row.get("plan") else support_steps(row["plan"]),
                "plan": None if not row.get("plan_path") else str(row["plan_path"]),
            }
            for row in structures
        ],
    }
    report["exit_code"] = exit_code(report)
    return report


def exit_code(report):
    states = [report["criteria"][key]["status"] for key in ("1", "2", "3")]
    if any(state == "FAIL" for state in states):
        return EXIT_FAIL
    if any(state != "PASS" for state in states):
        return EXIT_UNVERIFIED
    return EXIT_PASS


def render_md(report):
    lines = [
        "# Acceptance verification",
        "",
        "Mode: %s" % report["mode"],
        "",
        "Exit 0 is a complete pass. Exit 1 means a criterion FAIL. Exit 2 means invalid or incomplete evidence (UNVERIFIED) with no FAIL.",
        "Saved `ok` and `full_acceptance` flags are ignored. Contact events do not change the three criteria.",
        "",
    ]
    for key in ("1", "2", "3"):
        row = report["criteria"][key]
        lines.append("## Criterion %s: %s — %s" % (key, row["title"], row["status"]))
        lines.append("")
        for check in row["checks"]:
            lines.append("- %s: %s" % (check["status"], check["detail"]))
        lines.append("")
    lines.append("## Contacts")
    lines.append("")
    if report["contacts"]:
        for row in report["contacts"]:
            lines.append("- %s: %s" % (row["structure"], row["counts"]))
    else:
        lines.append("- none recorded")
    lines.append("")
    lines.append("## Hardware")
    lines.append("")
    lines.append(report["hardware_note"])
    lines.append("")
    if report["ignored_flags"]:
        lines.append("Ignored summary flags: %s." % ", ".join(report["ignored_flags"]))
        lines.append("")
    lines.append("Exit code: %d" % report["exit_code"])
    lines.append("")
    return "\n".join(lines)


def write_report(report, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "verification.md").write_text(render_md(report))
    return report["exit_code"]


def package_from_plans(plans):
    return {
        "structures": [
            {
                "name": name,
                "plan": plan,
                "plan_path": None,
                "plan_sha256": None,
                "episodes": [],
                "cells": [],
                "summary_runs": [],
                "recording": None,
            }
            for name, plan in plans.items()
        ],
        "duplicates": [],
        "provenance": None,
        "contacts": [],
        "ignored_flags": [],
    }


def verify_run(path, expected=EXPECTED):
    return assess(load_run(path), "evidence", expected)


def note_simulator(provenance, episodes):
    for row in episodes:
        if row.get("backend"):
            provenance["simulator"] = {"backend": row["backend"], "joint_model": row.get("joint_model")}
            break
    return provenance


def _brief(path):
    path = Path(path)
    if not path.is_dir():
        return {"present": False}
    summary_path = path / "summary.json"
    summary = _load_json(summary_path) if summary_path.exists() else {}
    runs = []
    for run in summary.get("runs") or []:
        runs.append({
            "shape": run.get("shape"),
            "placed": run.get("placed"),
            "total": run.get("total"),
            "frames": run.get("frames"),
            "done": run.get("done"),
            "failure": run.get("failure"),
            "required_supports": run.get("required_supports"),
            "sha256": sha256_file(summary_path) if summary_path.exists() else None,
        })
    plans = {}
    for file in sorted((path / "plans").glob("*.json")) if (path / "plans").is_dir() else []:
        plan = _load_json(file)
        plans[plan.get("structure_id") or file.stem] = {
            "sha256": sha256_file(file),
            "bricks": len(plan.get("sequence") or []),
            "supports": support_steps(plan),
            "brace_required_count": (plan.get("validation") or {}).get("brace_required_count"),
        }
    return {
        "present": True,
        "files": sorted(item.name for item in path.iterdir() if item.is_file()) + sorted(
            "plans/" + item.name for item in (path / "plans").glob("*.json")
        ) if (path / "plans").is_dir() else sorted(item.name for item in path.iterdir() if item.is_file()),
        "scope": summary.get("scope"),
        "runs": runs,
        "plans": plans,
        "provenance": (path / "provenance.json").exists(),
        "cell_rows": sum(1 for line in (path / "cell_rows.jsonl").read_text().splitlines() if line.strip()) if (path / "cell_rows.jsonl").exists() else 0,
        "episodes": sum(1 for line in (path / "episodes.jsonl").read_text().splitlines() if line.strip()) if (path / "episodes.jsonl").exists() else 0,
    }


def inventory(results):
    results = Path(results)
    names = ("acceptance", "acceptance_arch_wriggle2", "acceptance_s3_wriggle")
    rows = {name: _brief(results / name) for name in names}
    combined = rows["acceptance"]
    separate = rows["acceptance_arch_wriggle2"].get("runs", [{}])[:1] + rows["acceptance_s3_wriggle"].get("runs", [{}])[:1]
    combined_runs = {run.get("shape"): run for run in combined.get("runs") or []}
    same = {}
    for other in separate:
        shape = other.get("shape")
        left = combined_runs.get(shape) or {}
        same[shape] = {
            "same_frames": left.get("frames") == other.get("frames"),
            "combined_frames": left.get("frames"),
            "separate_frames": other.get("frames"),
            "same_supports": left.get("required_supports") == other.get("required_supports"),
        }
    plan_same = {}
    for shape, saved in (combined.get("plans") or {}).items():
        other = None
        for name in ("acceptance_arch_wriggle2", "acceptance_s3_wriggle"):
            other = (rows[name].get("plans") or {}).get(shape) or other
        plan_same[shape] = None if other is None else saved.get("sha256") == other.get("sha256")
    return {
        "runs": rows,
        "combined_matches_separate_runs": same,
        "plan_hash_matches_separate": plan_same,
        "explicit_aggregation_manifest": False,
        "conclusion": (
            "results/acceptance holds both structures in one directory and has no aggregation manifest. "
            "Plan hashes match the separate runs. Arch frames match (%s) but S3 frames do not (%s vs %s). "
            "No provenance.json or source revision is stored, so the combined summary is unverified."
        ) % (
            (same.get("arch") or {}).get("combined_frames"),
            (same.get("S3") or {}).get("combined_frames"),
            (same.get("S3") or {}).get("separate_frames"),
        ),
    }
