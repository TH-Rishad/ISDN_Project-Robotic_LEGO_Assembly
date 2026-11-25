"""Render PNG stills from a saved dual-build recording: stills.py RUN OUT_DIR TAG:T:X,Y,Z,PITCH,YAW ..."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Kean" / "scripts"))

import cv2  # noqa: E402
import warp as wp  # noqa: E402
import newton  # noqa: E402
from play_run import load_record, show_frame  # noqa: E402
from Kean.dual_build import load_example  # noqa: E402

run, out = ROOT / sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((run / "manifest.json").read_text())
record = load_record(run / manifest["recording"])
viewer = newton.viewer.ViewerGL(width=1600, height=1000, headless=True)
example = load_example(run, viewer)
for job in sys.argv[3:]:
    tag, t, cam = job.split(":")
    x, y, z, pitch, yaw = map(float, cam.split(","))
    viewer.set_camera(pos=wp.vec3(x, y, z), pitch=pitch, yaw=yaw)
    index = min(int(np.searchsorted(record["sim_time"], float(t))), len(record["frame"]) - 1)
    for _ in range(2):  # first render after a camera move can lag one frame
        show_frame(example, record, index)
        example.render()
    image = viewer.get_frame().numpy()[..., ::-1]
    cv2.imwrite(str(out / (tag + ".png")), np.ascontiguousarray(image))
    print("wrote", tag, "t=%.1f" % record["sim_time"][index], flush=True)
