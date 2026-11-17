"""30-second edited videos of the two verified dual-arm builds (GPU, offscreen).

    FFMPEG=/path/to/ffmpeg Sakib/.venv/bin/python final_report/render_videos.py [arch|pyramid ...]

Each video is a list of shots: (video start s, video end s, sim start s, sim end s,
camera at start, camera at end, caption). A camera is (position, look-at target);
it moves smoothly from the first to the second over the shot. Frames are piped straight into ffmpeg's libx264.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Kean" / "scripts"))

import cv2  # noqa: E402
import newton  # noqa: E402
import warp as wp  # noqa: E402
from play_run import load_record, show_frame  # noqa: E402
from Kean.dual_build import load_example  # noqa: E402

W, H, FPS = 1920, 1080, 30
OUT = Path(__file__).resolve().parent / "videos"

# Cameras: (position, target). Pad A (arch / pyramid) is at y = +0.16 m, pad B (S3) at y = -0.16 m.
# Wide shots look steeply down from above the cell: from lower angles the arms, which reach out
# to feeders on both sides of the pads, keep filling the frame.
OV0 = ((0.0, -0.82, 1.62), (0.0, 0.02, 0.04))
OV1 = ((0.0, -0.74, 1.46), (0.0, 0.02, 0.04))
OV2 = ((0.0, -0.68, 1.34), (0.0, 0.02, 0.04))
A_SIDE_W = ((-0.36, 0.16, 0.15), (0.0, 0.16, 0.07))
A_SIDE = ((-0.30, 0.16, 0.13), (0.0, 0.16, 0.066))
PYR_SIDE = ((-0.27, 0.16, 0.11), (0.0, 0.16, 0.038))
PYR_34 = ((-0.20, 0.38, 0.20), (0.0, 0.16, 0.04))
S3_SIDE = ((-0.30, -0.16, 0.15), (0.0, -0.16, 0.086))
S3_SIDE_W = ((-0.38, -0.16, 0.17), (0.0, -0.16, 0.085))
A_FIN = ((-0.34, 0.16, 0.10), (0.0, 0.16, 0.05))
A_FIN_IN = ((-0.24, 0.16, 0.085), (0.0, 0.16, 0.05))
S3_FIN = ((-0.40, -0.16, 0.14), (0.0, -0.16, 0.075))
S3_FIN_IN = ((-0.30, -0.16, 0.12), (0.0, -0.16, 0.075))

# Stud-fit run 003: a low view along the arch from the builder's side shows each same-column hold;
# S3 from inside the cell (its supports reach the piers from the outer side).
ARCH_LOW = ((-0.24, 0.16, 0.09), (0.0, 0.16, 0.03))
S3_IN = ((-0.16, 0.0, 0.15), (0.0, -0.17, 0.04))

VIDEOS = {
    "arch": dict(
        run="Kean/results/dual_arch_s3_012", out="dual_arch_s3_30s.mp4",
        title="Dual-arm LEGO assembly  |  arm A: arch (11 bricks)   arm B: S3 bridge (16 bricks)",
        shots=[
            (0, 3, 0, 45, OV0, OV1, "Two arms, two structures: A builds the arch, B builds the S3 bridge"),
            (3, 11, 45, 240, OV1, OV2, "Both arms build at the same time; reserved motions keep them apart"),
            (11, 15.5, 257, 281, A_SIDE_W, A_SIDE, "Arch crown: arm B holds the arch while arm A presses the crown"),
            (15.5, 19, 290, 308, S3_SIDE, S3_SIDE, "S3 step 11: arm A holds the pier while B places the corbel"),
            (19, 22.5, 340, 357, S3_SIDE, S3_SIDE, "S3 step 13: arm A grips the pier top under the overhang"),
            (22.5, 26, 372, 411, S3_SIDE_W, S3_SIDE_W, "Steps 14 and 15: the S3 bridge is complete"),
            (26, 28, 411, 411, A_FIN, A_FIN_IN, "Arch 11/11  |  S3 16/16  |  4 supports held  |  verifier: pass"),
            (28, 30, 411, 411, S3_FIN, S3_FIN_IN, "Arch 11/11  |  S3 16/16  |  4 supports held  |  verifier: pass"),
        ]),
    "pyramid": dict(
        run="Kean/results/dual_pyramid_s3_006", out="dual_pyramid_s3_30s.mp4",
        title="Dual-arm LEGO assembly  |  arm A: user's pyramid drawing (11 bricks)   arm B: S3 bridge (16 bricks)",
        shots=[
            (0, 3, 0, 45, OV0, OV1, "Arm A builds the pyramid from a drawing, arm B builds the S3 bridge"),
            (3, 10, 45, 235, OV1, OV2, "Both arms build at the same time; reserved motions keep them apart"),
            (10, 13.5, 244, 255, PYR_SIDE, PYR_SIDE, "Boxed-in 2x2: released above the studs, then pressed with closed fingertips"),
            (13.5, 15.5, 260, 273, PYR_34, PYR_34, "Top brick placed: pyramid complete"),
            (15.5, 19, 288, 302, S3_SIDE, S3_SIDE, "Arm A, finished, now holds S3 while B places the corbel"),
            (19, 22.5, 333, 346, S3_SIDE, S3_SIDE, "S3 step 13: arm A grips the pier top under the overhang"),
            (22.5, 26, 365, 400, S3_SIDE_W, S3_SIDE_W, "Steps 14 and 15: the S3 bridge is complete"),
            (26, 28, 400, 400, A_FIN, A_FIN_IN, "Pyramid 11/11  |  S3 16/16  |  3 supports held  |  verifier: pass"),
            (28, 30, 400, 400, S3_FIN, S3_FIN_IN, "Pyramid 11/11  |  S3 16/16  |  3 supports held  |  verifier: pass"),
        ]),
    # Stud-fit runs. pyramid_studfit is a local GPU rerun of Kean's dual_pyramid_s3_studfit_030 (same sources,
    # same 74 events, timings within 1 s); the HPC recording.npz is not in git.
    "arch_studfit": dict(
        run="Kean/results/dual_arch_s3_studfit_007", out="dual_arch_s3_studfit_30s.mp4",
        title="Dual-arm LEGO assembly with stud fit  |  arm A: arch (11 bricks)   arm B: S3 bridge (16 bricks)",
        shots=[
            (0, 3, 0, 45, OV0, OV1, "Stud fit on: bricks are pressed onto studs that resist insertion"),
            (3, 8, 45, 170, OV1, OV2, "Both arms build at the same time; reserved motions keep them apart"),
            (8, 12, 170, 268, ARCH_LOW, ARCH_LOW, "Arch steps 6-9: the stacks carry each press, so no support is needed"),
            (12, 15.5, 278, 305, ARCH_LOW, ARCH_LOW, "Arch crown: arm B holds a stack while arm A presses the crown"),
            (15.5, 18, 305, 362, S3_IN, S3_IN, "S3 steps 10-12: corbels and pier bricks go in without support"),
            (18, 21, 362, 388, S3_IN, S3_IN, "S3 step 13: arm A holds the corbel while arm B presses the bridge brick"),
            (21, 24, 444, 468, S3_IN, S3_IN, "S3 step 14: arm A holds the bridge while arm B presses"),
            (24, 27, 492, 516, S3_IN, S3_IN, "S3 step 15: arm A holds the bridge while arm B presses the top brick"),
            (27, 28.5, 516, 516, A_FIN, A_FIN_IN, "Arch 11/11  |  S3 16/16  |  4 supports held  |  stud fit  |  verifier: pass"),
            (28.5, 30, 516, 516, S3_FIN, S3_FIN_IN, "Arch 11/11  |  S3 16/16  |  4 supports held  |  stud fit  |  verifier: pass"),
        ]),
    "pyramid_studfit": dict(
        run="Kean/results/dual_pyramid_s3_studfit_030_local", out="dual_pyramid_s3_studfit_030_30s.mp4",
        title="Dual-arm LEGO assembly with stud fit  |  arm A: pyramid (11 bricks)   arm B: S3 bridge (16 bricks)",
        shots=[
            (0, 3, 0, 45, OV0, OV1, "Stud fit on: bricks are pressed onto studs that resist insertion"),
            (3, 10, 45, 247, OV1, OV2, "Both arms build at the same time; reserved motions keep them apart"),
            (10, 13, 266, 286, S3_SIDE, S3_SIDE, "S3 step 9: arm A holds the structure while B presses"),
            (13, 16, 324, 345, S3_SIDE, S3_SIDE, "S3 step 11: arm A holds the pier while B places the corbel"),
            (16, 18.5, 345, 366, PYR_34, PYR_34, "Top brick placed: pyramid complete"),
            (18.5, 22, 385, 407, S3_SIDE, S3_SIDE, "S3 step 13: arm A grips the pier top under the overhang"),
            (22, 26, 455, 512, S3_SIDE_W, S3_SIDE_W, "Steps 14 and 15, both supported: the S3 bridge is complete"),
            (26, 28, 512, 512, A_FIN, A_FIN_IN, "Pyramid 11/11  |  S3 16/16  |  5 supports held  |  stud-fit  |  verifier: pass"),
            (28, 30, 512, 512, S3_FIN, S3_FIN_IN, "Pyramid 11/11  |  S3 16/16  |  5 supports held  |  stud-fit  |  verifier: pass"),
        ]),
}


def ease(x):
    return x * x * (3 - 2 * x)


def camera(spec):
    """(pos, pitch deg, yaw deg) for a (pos, target) pair."""
    pos, target = map(np.asarray, spec)
    d = np.asarray(target, float) - pos
    return pos, math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))), math.degrees(math.atan2(d[1], d[0]))


def blend(a, b, s):
    """Interpolate two cameras."""
    return tuple(tuple(np.asarray(p) + s * (np.asarray(q) - np.asarray(p))) for p, q in zip(a, b))


def banner(img, text, y, scale, top):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    pad = 16
    y0, y1 = (y, y + th + base + 2 * pad) if top else (y - th - base - 2 * pad, y)
    over = img.copy()
    cv2.rectangle(over, (0, y0), (img.shape[1], y1), (20, 22, 28), -1)
    cv2.addWeighted(over, .62, img, .38, 0, img)
    cv2.putText(img, text, ((img.shape[1] - tw) // 2, y1 - pad - base), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (245, 245, 245), 2, cv2.LINE_AA)


def render(name, ffmpeg):
    cfg = VIDEOS[name]
    run = ROOT / cfg["run"]
    manifest = json.loads((run / "manifest.json").read_text())
    record = load_record(run / manifest["recording"])
    viewer = newton.viewer.ViewerGL(width=W, height=H, headless=True)
    example = load_example(run, viewer)
    OUT.mkdir(exist_ok=True)
    path = OUT / cfg["out"]
    enc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                            "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(path)], stdin=subprocess.PIPE)
    times = record["sim_time"]
    last = None
    for v0, v1, t0, t1, c0, c1, caption in cfg["shots"]:
        n = round((v1 - v0) * FPS)
        for k in range(n):
            x = k / max(n - 1, 1)
            t = min(t0 + x * (t1 - t0), float(times[-1]))
            pos, pitch, yaw = camera(blend(c0, c1, ease(x)))
            viewer.set_camera(pos=wp.vec3(*map(float, pos)), pitch=pitch, yaw=yaw)
            index = min(int(np.searchsorted(times, t)), len(times) - 1)
            for _ in range(2 if k == 0 else 1):        # the first render after a cut can lag one frame
                show_frame(example, record, index)
                example.render()
            img = np.ascontiguousarray(viewer.get_frame().numpy()[..., ::-1])
            banner(img, cfg["title"], 0, .85, True)
            banner(img, caption, H, 1.05, False)
            speed = "final" if t1 == t0 else "%.0fx speed" % ((t1 - t0) / (v1 - v0))
            label = "sim t = %3.0f s   %s" % (t + cfg.get("time_offset", 0.), speed)
            cv2.putText(img, label, (W - 380, 100), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
            enc.stdin.write(img.tobytes())
        if caption != last:
            print("%s: %4.1f-%4.1f s  sim %3.0f-%3.0f  %s" % (name, v0, v1, t0, t1, caption), flush=True)
            last = caption
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg failed")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or list(VIDEOS):
        render(name, os.environ.get("FFMPEG", "ffmpeg"))
