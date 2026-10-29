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
    args = parser.parse_args()
    if args.live and args.dry_run:
        parser.error("choose one of --dry-run or --live")
    output_path = args.out.expanduser()
    out = (output_path if output_path.is_absolute() else ROOT / output_path).resolve()
    if not out.is_relative_to(KEAN):
        parser.error("--out must be inside Kean/")
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
    for job in jobs:
        print("accept: %s seed %d" % (job["shape"], job["seed"]), flush=True)
        row, _ = run_job(job, out, args.max_frames, record=True, fresh_plan=True, planned_motion=True, strict_motion=True)
        runs.append(row)
    reasons = judge(runs, expected_shapes=[j["shape"] for j in jobs])
    summary = {"scope": args.shape or "all", "full_acceptance": args.shape is None and not reasons, "ok": not reasons, "reasons": reasons, "runs": runs}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print("accept: ok=%s %s" % (summary["ok"], ", ".join(reasons) or "clean"))
    if reasons:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
