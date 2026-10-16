"""Camera page for dual_arm_sim, same process as the Viser 3D viewer.

Started only when Sakib's hook finds Kean/kean.id. The image is the same tile
stack as ViewerGL --show-cameras: RGB over depth, columns top, wrist_A,
wrist_B, and the last look when V5 is running. Built by Example.refresh_feeds.
"""

import io
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

PORT = int(os.environ.get("KEAN_CAM_PORT", "8081"))
EVERY = 10

_png = b""
_lock = threading.Lock()
_errors = 0
_server = None


def _mosaic(images):
    stack = np.asarray(images)
    if stack.ndim == 3:
        return stack
    cols = stack.shape[0] // 2
    top = np.concatenate([stack[i] for i in range(cols)], axis=1)
    bot = np.concatenate([stack[i] for i in range(cols, stack.shape[0])], axis=1)
    return np.concatenate([top, bot], axis=0)


def _encode(rgb):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG")
    return buf.getvalue()


def _publish(images):
    global _png
    data = _encode(_mosaic(images))
    with _lock:
        _png = data


def _grab(example):
    if example.cams is None:
        from cell.cameras import Cameras
        example.cams = Cameras(example.model, seed=example.seed)
    example.refresh_feeds()


def _page():
    return b"""<!doctype html>
<meta charset="utf-8">
<title>cameras</title>
<style>
body { margin: 12px; background: #111; color: #eee; font-family: sans-serif; }
img { max-width: 100%; background: #000; }
p { color: #aaa; }
</style>
<h3>cameras: top | wrist_A | wrist_B (RGB over depth)</h3>
<img id="feeds" alt="camera feeds">
<p>top row RGB, bottom row depth. A last-look column is added when --look and V5 run.</p>
<script>
function tick() { document.getElementById("feeds").src = "/cam/feeds.png?" + Date.now(); }
setInterval(tick, 400);
tick();
</script>
"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            body = _page()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path != "/cam/feeds.png":
            self.send_error(404)
            return
        with _lock:
            data = _png
        if not data:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        return


def _serve():
    global _server
    _server = ThreadingHTTPServer(("0.0.0.0", PORT), _Handler)
    _server.serve_forever()


def attach(example):
    """Serve Sakib's camera tiles. No-op if already attached."""
    global _errors
    if getattr(example, "_kean_cams", False):
        return
    example._kean_cams = True
    example.show_cams = False
    example.viewer.log_image = lambda name, images, *a, **k: _publish(images)
    every = max(1, int(getattr(example, "cam_every", None) or EVERY))
    threading.Thread(target=_serve, name="kean-cams", daemon=True).start()
    print("[kean] camera server http://0.0.0.0:%d  (RGB over depth)" % PORT, flush=True)

    orig = example.step

    def step():
        global _errors
        orig()
        if example.frame % every:
            return
        try:
            _grab(example)
        except Exception as e:
            if _errors < 3:
                print("[kean] camera frame failed: %s" % e, flush=True)
            _errors += 1

    example.step = step
