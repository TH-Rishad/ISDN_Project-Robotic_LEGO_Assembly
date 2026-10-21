"""Run CPU preflight and one recorded GPU job into a fresh evidence directory."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
KEAN = ROOT / "Kean"


def run(command, log):
    print("test:", " ".join(map(str, command)), flush=True)
    # Propagate isolation to every child interpreter, including unit-test children.
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    with log.open("w") as stream:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, env=env) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            return process.wait()


def metadata():
    sources = list(KEAN.rglob("*.py")) + list((KEAN / "scripts").glob("*.sh"))
    sources += [ROOT / "Sakib/brickassembly" / name for name in
                ("dual_arm_sim.py", "scripts/12_curobo_probe.py", "cell/record.py", "cell/contacts.py")]
    return {"python": sys.executable, "version": sys.version, "platform": platform.platform(),
            "settings": {key: os.environ.get(key) for key in
                         ("P3_ITER_SCALE", "P3_LATENCY_MS")},
            "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(sources)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--max-frames", type=int, default=60000)
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S-%f")
    out = KEAN / "results" / "checks" / stamp
    out.mkdir(parents=True)
    print("Evidence:", out, flush=True)
    meta = metadata()
    meta.update(cpu_only=args.cpu_only, max_frames=args.max_frames, job=0)
    (out / "environment.json").write_text(json.dumps(meta, indent=2) + "\n")
    stages = [("unit", [sys.executable, "-m", "unittest", "discover", "-s", "Kean/tests", "-v"]),
              ("manifest", [sys.executable, "Kean/scripts/m05.py", "--dry-run", "--out", str(out)])]
    if not args.cpu_only:
        stages += [("gpu", [sys.executable, "Kean/scripts/check_env.py", "--gpu"]),
            ("job", [sys.executable, "Kean/scripts/m05.py", "--job", "0", "--record",
                      "--experimental-grasp-offset", "--out", str(out),
                      "--max-frames", str(args.max_frames)])]
    results = {}
    for name, command in stages:
        results[name] = run(command, out / (name + ".log"))
        (out / "status.json").write_text(json.dumps(results, indent=2) + "\n")
        if results[name]:
            print("Stopped at %s (exit %s). Evidence: %s" % (name, results[name], out))
            return results[name] if results[name] > 0 else 1
    print("CPU preflight passed." if args.cpu_only else "Recorded job passed.")
    print("Evidence:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
