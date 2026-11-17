"""Richard control flow: two braced dual-arm builds, minimal 100-point path.

Criteria (course feedback, 100 points):
  1. Target structure as direct input (no image conversion).
  2. Automatically generated assembly sequence and support locations.
  3. Both arms autonomously assemble two different structures, each with at
     least eight bricks and at least one section requiring temporary support.

Modes:
  --plan-check   Generate both plans and verify criteria 1-2 on CPU. No Newton.
  --live         Execute arch and S3 in the Newton cell (HPC), then judge.
  --verify DIR   Recompute all three criteria from a saved run package (CPU).

Exit codes follow verify.py: 0 pass, 1 fail, 2 unverified (plan-check: 0/1).
"""

import argparse
import json
import sys
from pathlib import Path

RICHARD = Path(__file__).resolve().parent
ROOT = RICHARD.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Richard.plans import ACCEPTANCE, judge, preview, preview_ok  # noqa: E402
from Richard.verify import (  # noqa: E402
    capture_provenance, directory_occupied, note_simulator, sha256_file,
)

OUT = RICHARD / "results" / "acceptance"
JOBS = tuple(
    {"shape": name, "kind": kind, "seed": 0, "strategy": "weakest_joint"}
    for name, kind in ACCEPTANCE
)


def resolve_out(raw):
    path = Path(raw).expanduser()
    out = (path if path.is_absolute() else ROOT / path).resolve()
    if not out.is_relative_to(RICHARD):
        raise SystemExit("--out must be inside Richard/ (resolved %s)" % out)
    return out


def plan_check(out):
    payload = preview()
    reasons = preview_ok(payload)
    payload["ok"] = not reasons
    payload["reasons"] = reasons
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(payload, indent=1) + "\n")
    for row in payload["structures"]:
        braced = [s["step"] for s in row["steps"] if s["requires_brace"]]
        print("plan-check: %s — %d bricks, supports at steps %s" % (row["structure"], row["bricks"], braced))
    print("plan-check: ok=%s %s" % (payload["ok"], ", ".join(reasons) or "clean"))
    if reasons:
        raise SystemExit(1)


def live(out, args):
    from Richard.runner import run_job
    runs = []
    jobs = [j for j in JOBS if args.shape is None or j["shape"] == args.shape]
    selected = [job["shape"] for job in jobs]
    provenance = capture_provenance(ROOT, {
        "image_input": False,
        "strategy": "weakest_joint",
        "max_frames": args.max_frames,
        "seeds": {job["shape"]: job["seed"] for job in jobs},
    })
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    (out / "runs").mkdir(exist_ok=True)
    summary = {}
    reasons = []
    for job in jobs:
        print("live: %s seed %d" % (job["shape"], job["seed"]), flush=True)
        row, _ = run_job(job, out, args.max_frames, record=True)
        plan = out / "plans" / ("%s_%s.json" % (job["shape"], job["strategy"]))
        if plan.exists():
            provenance["plan_hashes"][job["shape"]] = sha256_file(plan)
        if (out / "episodes.jsonl").exists():
            note_simulator(provenance, [
                json.loads(line)
                for line in (out / "episodes.jsonl").read_text().splitlines() if line.strip()
            ])
        (out / "runs" / ("%s.json" % job["shape"])).write_text(json.dumps(row, indent=1, default=str) + "\n")
        runs.append(row)
        reasons = judge(runs, expected_shapes=selected)
        finished = len(runs) == len(jobs) and not reasons
        summary = {
            "scope": args.shape or "all",
            "full_acceptance": args.shape is None and finished,
            "ok": finished,
            "reasons": reasons,
            "runs": runs,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
        (out / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print("live: ok=%s %s" % (summary.get("ok"), ", ".join(reasons) or "clean"))
    if reasons:
        raise SystemExit(1)


def verify_saved(out, report_out):
    from Richard.verify import verify_run, write_report
    report = verify_run(out)
    code = write_report(report, report_out)
    for key in ("1", "2", "3"):
        row = report["criteria"][key]
        print("verify: criterion %s %s — %s" % (key, row["title"], row["status"]))
    print("verify: exit %d (0 pass, 1 fail, 2 unverified)" % code)
    raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-check", action="store_true", help="generate both plans; CPU only")
    mode.add_argument("--live", action="store_true", help="run both structures in the Newton cell (HPC)")
    mode.add_argument("--verify", type=Path, default=None, metavar="RUN_DIR", help="verify a saved run package; CPU only")
    parser.add_argument("--shape", choices=tuple(name for name, _ in ACCEPTANCE),
                        help="diagnose one structure; does not certify both")
    parser.add_argument("--max-frames", type=int, default=60000)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replace", action="store_true", help="overwrite an existing run directory")
    args = parser.parse_args()

    if args.verify is not None:
        report_out = resolve_out(args.out) if args.out != OUT else RICHARD / "results" / ("verification_" + args.verify.name)
        verify_saved(args.verify.expanduser().resolve(), report_out)
        return

    out = resolve_out(args.out)
    if directory_occupied(out) and not args.replace:
        parser.error("refusing to overwrite %s; pass a new --out" % out)
    out.mkdir(parents=True, exist_ok=True)
    if args.plan_check:
        plan_check(out)
    else:
        live(out, args)


if __name__ == "__main__":
    main()
