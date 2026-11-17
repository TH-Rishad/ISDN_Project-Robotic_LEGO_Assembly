#!/usr/bin/env bash
# Provision the single-interpreter environment Richard's control flow needs.
#
#   bash Richard/scripts/setup_pod.sh [REPO_ROOT]
#
# Written for a fresh RunPod GPU pod (Ubuntu 24.04 image, Python 3.12, CUDA 12.8,
# RTX 5090). It is the exact sequence that produced a working cell on
# 2026-11-17: Newton 1.2.1 + Warp 1.13.0 + MuJoCo-Warp 3.8.0.3 + cuRobo
# 0.8.0.post1.dev43 in ONE venv at Sakib/.venv (the path Richard/scripts/run.sh
# selects). Torch is reused from the image via --system-site-packages, so the
# cu128 build the image ships is not downloaded again.
#
# Two things the upstream env.lock splits across two interpreters are merged
# here on purpose: cuRobo 0.8 is imported in-process by the motion planner, so
# it must live in the same interpreter as Newton. It works: cuRobo 0.8 JIT-
# compiles through cuda-core and needs no CUDA extension build.
#
# Everything is pinned to Sakib/brickassembly/env.lock. Do not bump a pin
# without re-running scripts/12_curobo_probe.py --step 1 and the acceptance
# path; the FR3 collision-sphere fit and the plan latencies were measured there.
set -eu

REPO="${1:-$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)}"
VENV="$REPO/Sakib/.venv"
CUROBO_SHA=78fd485fa82d9b9a063fb4985e371814587e666a   # v0.8.0-43-g78fd485

echo "repo: $REPO"
[ -f "$REPO/Sakib/brickassembly/dual_arm_sim.py" ] || {
    echo "setup_pod: run from the repository root (no Sakib/brickassembly here)" >&2
    exit 1
}

# 1. venv sharing the image's torch (CUDA 12.8 build) --------------------------
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -q --upgrade pip

# 2. setuptools < 82: Python 3.12 dropped distutils, and setuptools >= 82
#    breaks the torch version pin the image ships.
"$VENV/bin/pip" install -q "setuptools<82" wheel

# 3. Newton stack, exactly as pinned ------------------------------------------
"$VENV/bin/pip" install -q \
    "warp-lang==1.13.0" \
    "newton==1.2.1" \
    "mujoco-warp==3.8.0.3" \
    "mujoco==3.8.0"

# 4. cell-side imports: images, LP stability, vision gate, viewers ------------
"$VENV/bin/pip" install -q \
    numpy scipy pillow opencv-python-headless \
    viser trimesh yourdfpy numpy-quaternion

# 5. cuRobo in-process (the planner imports it, it is not a separate process) --
if [ ! -d "$REPO/../curobo/.git" ] && [ ! -d "$REPO/curobo/.git" ]; then
    git clone -q https://github.com/NVlabs/curobo.git "$(dirname "$REPO")/curobo"
fi
CUROBO_DIR="$(dirname "$REPO")/curobo"
[ -d "$CUROBO_DIR/.git" ] || CUROBO_DIR="$REPO/curobo"
git -C "$CUROBO_DIR" checkout -q "$CUROBO_SHA"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
PATH="$CUDA_HOME/bin:$PATH" \
    "$VENV/bin/pip" install -q -e "$CUROBO_DIR[cu12]" --no-build-isolation

# 6. Newton downloads the Franka URDF through GitPython -----------------------
"$VENV/bin/pip" install -q GitPython

# 7. Verify: the probe's step 1 imports cuRobo next to Newton and plans on FR3
ROBOFAB_PY="$VENV/bin/python" \
    bash "$REPO/Sakib/brickassembly/scripts/run.sh" scripts/12_curobo_probe.py --step 1

echo
echo "environment ready: $VENV"
echo "next: bash Richard/scripts/main.sh --live --out Richard/results/acceptance_pod"
