# Work Record — Kean

**Date:** 2026-10-16
**Purpose:** Track testing and verification work on Sakib's folder (brickassembly + related) to confirm code works.
**Location of tests:** HPC compute node (Linux, NVIDIA A30, RoboFabEnv). Node name changes per allocation.
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

**Setup:** mlieq@hpc4.ust.hk → allocated compute node (headless). Access via SSH tunnel + Viser.

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
- Tunnel (replace NODE with the allocated node): `ssh -L 8080:NODE:8080 -L 8081:NODE:8081 mlieq@hpc4.ust.hk`
- Pages: `http://localhost:8080` (3D), `http://localhost:8081` (cameras).
- These three files must be on the compute node before the run.

**8081 vs Sakib's camera window:** the page now shows one mosaic from `Example.refresh_feeds` / `push_feeds`: top row RGB, bottom row depth, columns top / wrist_A / wrist_B, plus last look when V5 runs. Overlays are Sakib's (`overlay_top`, `overlay_wrist`). Still missing: the separate Workspace text window. PNG, not the GL window's raw upload.

**Other visualization methods (usable from HPC):**
1. Keep `--viewer viser` for the live 3D scene only (orbit with mouse for a top view; that is not the camera feed).
2. Save camera frames to disk, then `scp` and open locally: `--viewer null --save-captures DIR --look --aim fk_vision`.
3. Record a USD and open it locally (Blender / usdview): `--viewer usd --output-path run.usd`.
4. `--record-all PATH.npz` for offline pose/contact inspection (`cell/record.py`), not a picture.

**Next tracking:** All future work records for testing/verification will be appended here in `Kean/`.

---

## M0.5 wiring (2026-10-21)

Local tests: 49 passed, 1 skipped (torch not installed). HPC login from this machine was refused, so the gate has not been run.

- `motion/tick.py` plans free-space moves on the cell tick, holds while the future is open, plays joint targets, and leaves the last 30 mm Cartesian. A capture after warmup fails the run. One arm in the zone.
- `motion/gate.py` is the M0.5 list: cube, arch, hollow_box, S3 × 5 seeds = 235 placements, plus S3 brace-all × 3. Arm–arm is split by braced placement. `neighbour_rub` is not counted.
- `scripts/m05.sh` is the HPC command. `--dry-run` wrote `results/m05/jobs.json` (23 jobs, 235 placements, 3 stress).
- Insertion still starts only at pre-insert. Press above n·8.9 N is refused. `scripts/insert.py` trains the kinematic cell; it does not add press.
- HPC run 2026-10-21: cell built, warmup died on `No module named 'motion.vendor'`. Kean's `motion` package was already imported, so the probe never saw Sakib's. `CellMotion` hides it for that import only. Re-run `bash Kean/scripts/m05.sh`.

---
*Created by opencode on behalf of testing effort.*
## M0.5 contact exit and controller handoff (2026-10-21)

The pasted HPC run snapped b_000, rejected B retract because its start overlapped the plate by 0.4 mm, then missed b_001. Keep retract on the existing Cartesian controller because it starts at the insertion contact pose. Tick now holds measured joint positions with zero velocity while planning, clears its feed-forward velocity on return to Cartesian control (including planner fallback), and refreshes measured end-effector positions while servicing plans.

Validation: `python3 -m unittest discover -s Kean/tests -v`: 55 passed, 1 skipped (torch unavailable). Added regression checks for contact retract, velocity cleanup, and waiting/failure handoff. Full Newton/cuRobo GPU behavior and the subsequent placement remain unverified locally; rerun `bash Kean/scripts/m05.sh --job 0` on HPC after syncing changes.

## M0.5 wrist and Cartesian handoff (2026-10-21)

After the earlier fixes, HPC still snapped b_000 then missed b_001 by 151.6 mm. With permission to read the shared controller, compared Example.step/Arm.update with Kean's Tick adapter. Found and corrected three integration mismatches in Kean only:

- Planner yaw now uses the cell's base-relative, pi-periodic wrist convention (and honors direct/yaw_shift modes). For B, nominal yaw 0 becomes world yaw pi in the cell; the previous request sent world yaw 0 to cuRobo.
- During waiting/playback and before Cartesian handoff, seed the existing batched IK buffer from measured arm joints, preserving the other arm and the captured buffer allocation.
- Plan to the bias-adjusted world goal, then start Cartesian interpolation at measured EE minus bias, preserving the nominal waypoint. Previously the bias was omitted in planning and added twice at handoff.

Validation: 58 tests passed, 1 skipped (torch unavailable), including wrist convention, biased world goal/handoff, and IK seed regression checks. No shared controller files edited. HPC Newton/cuRobo execution remains unverified; these are verified adapter defects, not yet a confirmed successful eight-brick build.

## Reuse and folder cleanup (2026-10-21)

Reviewed all Kean source areas and relevant shared implementations before continuing placement debugging. Migrated imports to Kean.*, removed package-cache swapping, consolidated the shared probe loader, confined P3 generated outputs/models to Kean, reused shared constants/pre-insert offset/PPO log-probability helper, and scoped planner/log lifetimes. Added README.md and review.md with a reuse map, execution instructions and unresolved integration gaps. Historical results preserved; no Sakib edits or GPU experiment reruns. Validation: 61 passed, 1 skipped (torch unavailable); compile and shell syntax checks passed. The second-brick issue remains open by design for the next debugging pass.

## Recorded test preparation (2026-10-21)

Added scripts/test_m05.sh and test_m05.py: one activated interpreter runs unit tests, isolated manifest generation, GPU/probe preflight and recorded cube job 0. Timestamped evidence directories preserve logs, source hashes and exit codes. m05 now accepts --out and --record, reuses the shared Recorder, validates CLI ranges and calls Example.finish in a finally block. Existing gate logic and historical reports are unchanged. CPU end-to-end preflight: 63 passed, 1 skipped; manifest passed. GPU execution remains pending on HPC.

## Test launcher interpreter correction (2026-10-21)

HPC passed all 64 tests, but the new launcher selected RoboFabEnv, whose torch emitted a driver compatibility warning and whose environment lacked Newton. Corrected test_m05.sh to prefer the same Sakib/.venv/bin/python as the working M0.5/P3 launchers; explicit P3_PY remains authoritative and invalid overrides fail clearly. GPU preflight now prints torch version/path and CUDA build before checking availability/importing Newton. Detached the training loss before converting it for reporting. Local validation: 64 passed, 1 skipped; interpreter-selection regression covers default venv, explicit override and missing interpreter. No dependency installation or driver changes made; GPU rerun remains pending.

## Recorded fourth-brick failure (2026-10-21)

Inspected synced run 20261021-170812-641375 with Sakib's cell.record.check: 4,729 valid frames, three fired bricks, no weld breaks or arm-arm flags. B joint-command jumps >2 rad occur on 90-degree grasp steps 0 and 3; steps 1 and 2 (zero nominal yaw) are smooth. Step 3 transport jumps 2.186 rad at frame 4244 before the pre-insert failure, and ends with the brick at approximately (13.2,40.7,104.8) mm rather than over the (0,8) mm target.

Found a precise bug in the earlier yaw fix: D.wrap uses (-pi/2,pi/2], but Kean used [-pi/2,pi/2). At nominal yaw pi/2 and B base yaw pi, the cell chooses 3pi/2 while the adapter chose pi/2. Removed that duplicate math: Tick now receives D.wrap directly. CPU tests load the shared pure wrap function via AST without importing Newton and check actual Request yaw at/around the boundaries. Validation: 65 passed, 1 skipped. Added record_analysis.json alongside the synced evidence. The corrected physical run and independent contact/clearance failures remain unverified.

## Contact geometry and measured clearance (2026-10-21)

Run 171655 completed 8/8 with no planning failures. Its eight unintended events were both B fingers against the preceding same-course brick on steps 1/3/5/7; all are shared-plan accessible=False grasps. Added a bounded pickup-offset search using the existing probe finger_boxes and obb_obb_gap, testing closed and open fingers against actual placed bricks at the pressed height. Search chooses the smallest 0.25 mm increment within 2 mm giving 0.5 mm nominal box clearance; no candidate fails explicitly. The shared queue builder receives the temporary pickup location so its hover/settle closures agree; feeder metadata is restored, insertion targets stay unchanged, and the existing aim controller estimates the held offset. Physical grasp stability is not yet validated; the shared planner notes off-centre grip swing as a risk.

Wired a Clearance adapter into Tick: shared full-model spheres at measured joints, translated to each measured finger opening, checked against world boxes and the other arm. Link indexing is cached before playback. Summary explicitly limits this metric to planned playback; contact events remain independently scored. No gate exemptions or thresholds changed. CPU validation: 70 passed, 1 skipped; physical contact elimination, positive GPU clearance and runtime cost require HPC rerun.

## Single shared environment (2026-10-21)

At the user's request, all Kean shell entry points now delegate to scripts/run.sh and use only Sakib/.venv/bin/python. Removed interpreter fallback/legacy overrides and Conda site-packages injection; inherited PYTHONPATH/PYTHONHOME and user site-packages are excluded, including test subprocesses. cuRobo must be installed in that environment (editable source installation is supported). The probe's legacy motion.vendor overlay hook is disabled for module import and restored afterward; missing dependencies give the shared setup command.

Reused Sakib/setup_env.sh --cuda cu128 --with-curobo as the provisioning path. Added check_env.py, used by the GPU test preflight, to report import locations and verify CUDA/probe/models in the shared venv. README now documents setup and plain launch commands. No Sakib files/environment packages were changed locally; Sakib/.venv is absent on this Mac, so actual dependency provisioning and GPU validation require HPC. CPU suite: 72 passed, 1 skipped (torch unavailable); shell syntax and changed Python compilation passed. Regression checks cover no added Conda paths, disabled/restored overlay hook on failure, strict interpreter selection for all launchers, and no fallback when the venv is absent.

## Reuse/ownership audit before more cluster tests (2026-10-21)

User clarified that Sakib owns environment building and Kean should focus on motion and RL. Replaced local interpreter routing with delegation to Sakib's launcher, removed environment-loader patching/package repair advice, kept shared collision models authoritative, and made historical P3 modifications opt-in. Default iteration scale is 1.0. Tick now consumes the shared trajectory resampler; world boxes consume shared brick_box/TABLE. Removed no-op GT memory bookkeeping. The unvalidated pickup offset is opt-in, preserving the default shared grasp queue. Added a projection of shared encoded observations for the RL actor. Retained camera policy/task code because the shared twin training loop is task-specific; documented shared-interface gaps, inactive memory/support prototypes and remaining integration limits in the rewritten README/review. No environment installation, shared-file edits or cluster runs.

Audit validation: 75 passed, 1 skipped (torch absent locally). Motion tests now execute Sakib's actual pure Arm class, wrap and resample bodies, including handoff arrival/wait/callback contracts, rather than a simplified second controller. Python compilation, shell syntax and M0.5/insertion CLI help passed. No HPC job launched; shared grasp/contact integration remains unresolved and explicitly documented.

## Synced cube job 0 (2026-10-21)

Evidence is `results/checks/20261021-191332-980676`. NVIDIA L20, torch CUDA 12.8, Sakib `.venv`. Status: unit, manifest, gpu, and job all exited 0. Unit log: 76 passed in 5.190 s.

Cube, weakest_joint, seed 0, `--experimental-grasp-offset`: 8/8 snapped, 8540 frames, 142.33 s, 0 braced, no cell failure. Offsets from `summary.json`: b_001 +1.5 mm in x, b_003 +1.25 mm in y, b_005 +1.25 mm in x, b_007 +1.25 mm in y. Job log lateral errors were 0.19, 0.13, 0.05, 0.07, 0.07, 0.08, 0.08, and 0.09 mm.

Gate summary: ok, no reasons. 0 plan failures, 0 captures, 0 arm-arm, 0 unintended, 0 displaced, 0 penetration, 18 neighbour_rub. Those rubs are held-brick to plate or neighbour during insert, which the gate does not count. `min_clearance_m` is 0.0461, planned playback only.

Compared with synced run `20261021-171655-549530`: also 8/8, but 8 unintended B-finger contacts on steps 1/3/5/7, `min_clearance_m` null, gate failed. The offset is still opt-in. Arch, hollow_box, and S3 are not in this sync.

`20261021-185956-521081` is only a unit exit 1 from an incomplete tree sync. It did not reach a job.
