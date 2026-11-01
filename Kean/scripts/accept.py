"""Arch and S3 from a structure id. Dry-run generates the plan. Live needs HPC."""

import argparse
import json
import sys
from pathlib import Path

KEAN = Path(__file__).resolve().parents[1]
ROOT = KEAN.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Kean.acceptance import judge, preview, preview_ok  # noqa: E402
from Kean.verify_acceptance import capture_provenance, directory_occupied, note_simulator, sha256_file  # noqa: E402

OUT = KEAN / "results" / "acceptance"
JOBS = (
    {"shape": "arch", "kind": "shape", "seed": 0, "strategy": "weakest_joint", "stress": False, "bricks": 11},
    {"shape": "S3", "kind": "structure", "seed": 0, "strategy": "weakest_joint", "stress": False, "bricks": 16},
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="generate the plan only")
    parser.add_argument("--live", action="store_true", help="run arch and S3 in the Newton cell")
    parser.add_argument("--shape", choices=("arch", "S3"),
                        help="diagnose one structure; does not certify both structures")
    parser.add_argument("--max-frames", type=int, default=60000)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replace", action="store_true", help="overwrite an existing run directory")
    args = parser.parse_args()
    if args.live and args.dry_run:
        parser.error("choose one of --dry-run or --live")
    output_path = args.out.expanduser()
    out = (output_path if output_path.is_absolute() else ROOT / output_path).resolve()
    if not out.is_relative_to(KEAN):
        parser.error("--out must be inside Kean/")
    if directory_occupied(out) and not args.replace:
        parser.error("refusing to overwrite %s; pass a new --out" % out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.live:
        payload = preview()
        reasons = preview_ok(payload)
        payload["ok"] = not reasons
        payload["reasons"] = reasons
        (out / "manifest.json").write_text(json.dumps(payload, indent=1) + "\n")
        print("accept: preview ok=%s %s" % (payload["ok"], ", ".join(reasons) or "clean"))
        if reasons:
            raise SystemExit(1)
        return
    from Kean.scripts.m05 import run_job
    runs = []
    jobs = [j for j in JOBS if args.shape is None or j["shape"] == args.shape]
    selected = [job["shape"] for job in jobs]
    provenance = capture_provenance(ROOT, {
        "image_input": False,
        "strategy": "weakest_joint",
        "max_frames": args.max_frames,
        "strict_motion": True,
        "planned_motion": True,
        "fresh_plan": True,
        "seeds": {job["shape"]: job["seed"] for job in jobs},
    })
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    (out / "runs").mkdir(exist_ok=True)
    for job in jobs:
        print("accept: %s seed %d" % (job["shape"], job["seed"]), flush=True)
        row, _ = run_job(job, out, args.max_frames, record=True, fresh_plan=True, planned_motion=True, strict_motion=True)
        plan = out / "plans" / ("%s_%s.json" % (job["shape"], job["strategy"]))
        if plan.exists():
            provenance["plan_hashes"][job["shape"]] = sha256_file(plan)
        if (out / "episodes.jsonl").exists():
            note_simulator(provenance, [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines() if line.strip()])
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
    print("accept: ok=%s %s" % (summary["ok"], ", ".join(reasons) or "clean"))
    if reasons:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
