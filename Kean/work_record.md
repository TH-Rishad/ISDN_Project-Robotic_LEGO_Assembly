# Work Record — Kean

**Date:** 2026-10-16
**Purpose:** Track testing and verification work on Sakib's folder (brickassembly + related) to confirm code works.
**Location of tests:** gpu04 (HPC, Linux, NVIDIA A30, RoboFabEnv)
**Scope:** Minimal smoke / verification only — not full gates or exhaustive experiments.

## Environment Verified
- Python 3.12.3
- Warp 1.13.0 (CUDA 12.9, Driver 12.8)
- Newton 1.2.1 + mujoco-warp
- scripts/run.sh launcher functional
- GPU kernels compile / cache correctly

## Tests Executed & Results

| Command | Result | Notes |
|---------|--------|-------|
| `bash scripts/run.sh tests/test_aim.py` | 18/18 passed (3.3–3.9s) | Fast, clean |
| `bash scripts/run.sh tests/test_brace_bandit.py` | 11/11 passed (11.35s) | Minor fork deprecation warnings (harmless) |
| `bash scripts/run.sh tests/test_vision.py` | 18/18 passed (~113s) | Deprecation warnings from collada (not our code) |
| `bash scripts/run.sh tests/test_v4_vision.py` | 17/17 passed (4.56s) | Clean |
| `bash scripts/run.sh tests/test_arm_protocol.py` | 7/7 passed (3.65s) | Clean |
| `bash scripts/run.sh tests/test_manual.py` | 3/3 passed (0.28s) | Clean |
| `bash scripts/run.sh dual_arm_sim.py --shape cube --viewer null --test --num-frames 40000` | 8/8 placed | Full build success; wrote proto_episodes.jsonl |
| `bash scripts/run.sh dual_arm_sim.py --shape cube --viewer null --test --num-frames 40000 --look --aim fk_vision` | 8/8 placed | Vision path also succeeds |
| `bash scripts/run.sh dual_arm_sim.py --shape cube --viewer null --num-frames 500` | Ran without crash | Smoke test (short) |
| `bash scripts/run.sh experiments/v4_vision.py e1 --dry-run` | 36 entries, 0 to-run | Harness works |
| `bash scripts/run.sh scripts/status.py` | Full status printed | 260 ledger entries; v4_phase2_gate + v4_G_V1 ACC visible |
| `bash scripts/run.sh scripts/status.py --open` | Historical ledger shown | No new blockers; old v3.1 + early v4 items |
| `bash scripts/run.sh scripts/00_env_check.py` | Newton ok; report written | G0: some PASS, cuRobo/impedance BLOCKED (expected) |

## Cleanup Performed
- `git checkout -- results/proto_episodes.jsonl tests/*.json` after runs that overwrite tracked files (per project rules in SETUP.md / CLAUDE.md).
- Post-cleanup `git status`: clean for tracked files.

## Summary
Core components in Sakib's folder work on the target GPU environment:
- Launcher (`run.sh`)
- Simulator (`dual_arm_sim.py`) — both plain and vision (`--look --aim fk_vision`)
- Fast unit tests (aim, vision, arm protocol, etc.)
- Status and env-check scripts
- Newton/Warp stack loads and runs physics

No crashes on initialization or basic execution. Full 8/8 builds succeed on cube. Vision path functional at smoke level. Environment checks pass where hardware matches (Newton core OK).

## Interactive / Remote Viewer Attempts (2026-10-16)

**Setup:** mlieq@hpc4.ust.hk → gpu04 (headless). Access via SSH tunnel + Viser.

**Working:**
- `bash scripts/run.sh dual_arm_sim.py --shape arch --viewer viser`
  - Viser server starts on :8080
  - Tunnel works (local browser http://localhost:8080)
  - Full arch build visible in main 3D scene (11/11 placed)
  - User sees two arms working from side view

**Issues with camera inputs:**
- `--show-cameras` (and `--look --show-cameras`) does not expose separate camera panels.
- User reports still seeing only the single 3D scene (no top/wrist camera feeds visible as separate windows or panels).
- Cameras appear to be logged but not surfacing in the Viser browser UI.

**Manual control:**
- `bash scripts/run.sh dual_arm_sim.py --shape cube --manual --show-cameras --viewer viser`
  - Fails immediately:
    ```
    pyglet.display.xlib.NoSuchDisplayException: Cannot connect to "None"
    ```
  - pyglet (used for keyboard in manual_step) requires a real display.

**Sakib's local setup (from Sakib):**
- Uses manual control code.
- Cameras open in separate floating windows.
- This works because Sakib runs locally on a machine with display (no `--viewer viser`, no tunnel, native GL/pyglet windows).

**X11 forwarding attempt:**
- Suggested: `ssh -X ...` then run without `--viewer viser`
- Result: "Cluster doesn't work this way." (X11 forwarding not functional/allowed on this cluster.)

**Current status:**
- Main 3D scene via Viser tunnel works reliably.
- Remote camera feeds and manual control not yet usable in current remote workflow.
- Difference: Sakib = local desktop with floating windows. Us = HPC headless + tunnel → limited to Viser 3D scene.

**Cause (from VIEWING.md / argparse):** `--show-cameras` and `--manual` are ViewerGL-only. `log_image` does not appear in Viser. Null viewer also shows nothing. X11 forwarding is unavailable on this cluster.

**Chosen method:** live 3D via existing `--viewer viser`. No edits outside `Kean/`. Camera image panels stay unavailable in Viser (`log_image` is ViewerGL-only).

**Camera server (built, same process):**
- Identifier: `Kean/kean.id` (delete it to disable; Sakib's sim then runs unchanged).
- Hook: `Sakib/brickassembly/dual_arm_sim.py` `_kean_viz()` — if `kean.id` exists, imports `Kean/viz_cameras.py` and calls `attach(example)`.
- `viz_cameras.attach` serves top / wrist_A / wrist_B JPEGs on `0.0.0.0:8081` (env `KEAN_CAM_PORT`) in a thread of the sim process. Viser 3D stays on :8080. Physics unchanged. Do not pass `--show-cameras` (that path is ViewerGL-only; the hook turns it off).
- Node: gpu02 (was gpu04). Tunnel: `ssh -L 8080:gpu02:8080 -L 8081:gpu02:8081 mlieq@hpc4.ust.hk`
- Pages: `http://localhost:8080` (3D), `http://localhost:8081` (cameras).
- These three files must be on gpu02 before the run.

**8081 vs Sakib's camera window:** the page now shows one mosaic from `Example.refresh_feeds` / `push_feeds`: top row RGB, bottom row depth, columns top / wrist_A / wrist_B, plus last look when V5 runs. Overlays are Sakib's (`overlay_top`, `overlay_wrist`). Still missing: the separate Workspace text window. PNG, not the GL window's raw upload.

**Other visualization methods (usable from HPC):**
1. Keep `--viewer viser` for the live 3D scene only (orbit with mouse for a top view; that is not the camera feed).
2. Save camera frames to disk, then `scp` and open locally: `--viewer null --save-captures DIR --look --aim fk_vision`.
3. Record a USD and open it locally (Blender / usdview): `--viewer usd --output-path run.usd`.
4. `--record-all PATH.npz` for offline pose/contact inspection (`cell/record.py`), not a picture.

**Next tracking:** All future work records for testing/verification will be appended here in `Kean/`.

---
*Created by opencode on behalf of testing effort.*