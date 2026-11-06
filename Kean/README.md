# Kean: two braced dual-arm builds

Sakib owns the environment, the Newton cell, and the sequence and support generators. Tanvir owns the tight-fit skill. Kean runs the generators on a structure id and finishes the arch and S3. Do not copy Tanvir's modules into `Kean/`.

## Concurrent independent builders

The new `dual_build` mode gives A and B separate structures, studded pads and
feeders in one Newton simulation. It reuses Kean's cuRobo planner, joint
playback, insertion controller and generated brace search. Support requests
are collected before pickup; the helper finishes its current placement and
parks before helping. Simultaneous pending requests give A priority, then
serve B's retained request. A completed builder remains available to help.

From the repository root:

```bash
bash Kean/scripts/dual_build.sh --build-a arch --build-b S3 \
  --viewer viser --port 8081 --out Kean/results/dual_arch_s3_live_001
```

Open `http://localhost:8081`. Omit `--viewer viser` for headless execution.
Choose a fresh output directory each time. The existing `Sakib/.venv` supplies
the interpreter and GPU dependencies.

Swept full-arm/finger and carried-brick reservations admit compatible
motions together. Free-space segments reserve their planned sweep; Cartesian
contact/escape moves reserve the measured arm volume padded by the move's
reach and stop if they leave it. During a supported insertion the helper's
measured joints hold its brace. The validated full build is
`results/dual_arch_s3_012`. Both structures, both pads and
all feeder bricks remain in the clearance world. Support acknowledgment
requires measured contact force on both fingers; failed clearance or lost
support stops execution. These are sampled simulation checks.

Before moving a helper, the runner checks its approach, finger-closing sweep
and the carried brick's insertion corridor. Additional generated grip heights
and lateral positions must retain the original predicted support capacity.
The executor uses the generator's grasp offsets. Pickup heights may rise by
up to 2 mm to clear lower-course studs, and the hand lifts while releasing.
Precision sockets use Kean's existing wriggle phases with 0.1 mm lateral
motion, 0.1 degree rocking and a longer final press. A loaded builder waits
at its checked park pose while the helper approaches.

Detailed contact evidence distinguishes intended engagement of a lower stud
with its planned upper mating brick from unintended collisions. Stud contacts
retain their measured depth and shape IDs in `contacts.json`; the verifier
requires the expected geometry and caps this contact at the 1.7 mm stud
height. Mesh studs additionally require a contact point above the brick shell
and within a stud footprint; the verifier recomputes that point from the
recorded body pose. Non-stud wall collisions, other brick pairs and deeper
penetrations fail.

Runs save both plans and layout, per-arm progress/roles, scheduler events,
contact results, motion failures and a complete recording. Verify or replay:

```bash
Sakib/.venv/bin/python -m Kean.verify_dual --run Kean/results/dual_arch_s3_live_001
bash Kean/scripts/play_run.sh --run Kean/results/dual_arch_s3_live_001 --port 8081 --loop
# Offscreen MP4 (OpenCV mp4v, 30 fps, fixed overview camera) at 8x:
bash Kean/scripts/play_run.sh --run Kean/results/dual_arch_s3_live_001 --speed 8 \
  --mp4 Kean/results/dual_arch_s3_live_001/replay_8x.mp4
```

`results/dual_arch_s3_012/dual_arch_s3_012_8x.mp4` is the 8x video of the
validated build, transcoded to H.264 for browser playback.

`--build-a pyramid` (or `--build-b`) builds the user's drawing
`Sakib/brickassembly/blueprints/user/pyramid_*.png`. When no side grip clears
flush neighbours at the seat, the builder tries the other axis and edge grips,
then drop-and-press (release above the neighbours' studs, press with closed
fingertips). It seats the pyramid's multi-stud tight bricks; a dropped 1x1
still tips on its single stud (`results/dual_pyramid_s3_004`). See the plan.

`--step-limit N` executes a short diagnostic. `--start-step-a N` and
`--start-step-b N` pre-place earlier bricks for support diagnostics. Such runs
are explicitly marked diagnostic and cannot pass as complete builds. See
[the implementation plan](DUAL_BUILDER_PLAN.md) for the handshake and validation criteria.

Meeting the three requirements below is a full mark. The M0.5 gate, insertion RL, perception, and the other master-plan items are omitted.

## Instructor acceptance (full mark)

1. Take the target structure as direct input. Converting an image into a target structure is not required.
2. Automatically generate the assembly sequence and determine where temporary support is needed.
3. Use both robotic arms to autonomously assemble two different structures, each containing at least eight LEGO bricks and at least one section requiring temporary support.

Input is `--structure` or a named sample such as `--shape` that does not read an image. Do not require `--front`, `--side`, or `--top`. Generate the order and the support poses with `planner.build_plan` and `bracing.assign` (`--strategy weakest_joint` or `nearest`, never `none` on an acceptance run). Do not hand-author the sequence or the support steps. Call Tanvir only if the cell insert misses a step. Do not train.

The two structures are the arch and S3. Support counts are the plan's `requires_brace` steps; `validation.brace_required_count` must match that list. Do not hard-code two or three. The saved plans currently read arch step 10 (count 1) and S3 steps 11, 13, 14 (count 3). Arm A must hold each generated support before and during Arm B's insert, then release. Cube and hollow_box have no required support, so a passing cube job does not meet item 3.

## Workspaces and what is finished (2026-10-25)

| Workspace | Has | Finished | Not finished |
| --- | --- | --- | --- |
| `Sakib/` | `brickassembly/` (cell, planner, bracing, stability, vision, RL env, plans, ledger), `robofab_sim/`, `BrickSim/`, `Docs/`, `setup_env.sh` | Unbraced cube 8/8, arch 11/11, hollow_box 12/12. Sequence and vertical-press support generators. Perception V1 closed. v3.1 frozen. | G-V1. Live default is `--strategy none`, so braced acceptance builds are not the current path. |
| `Kean/` | `motion/` planned-path adapter, acceptance runner, CPU verifier | `results/instructor_run_001` on gpu12: arch 11/11, S3 16/16, verifier exit 0. | Physical robot. `KEAN_NEWTON_FIT` stays off. |
| `Tanvir/` | `tight_fit_audit.py`, `socket_family.py`, `socket_env.py`, `wriggle_brace.py`, `skill_env.py`, trainers, `report/` | `ScriptedWriggle` is the best policy. Pad shift is the measured gain. Five PPO runs and one ES run did not beat scripted. Proposal written. | Not in the Newton cell. Held-out, no brace, 1–2 mm: 26/68. |
| `Richard/` | `.gitkeep` | Nothing | Everything |

Do not rebuild the planner, cell, or scripted wriggle. Do not rerun into `results/instructor_run_001`. Perception, insertion RL, the 235-placement gate, and a learned wriggle are omitted.

## Run with Sakib's supplied environment

No Conda activation or Kean installer is required. The thin `scripts/run.sh` wrapper delegates to `Sakib/brickassembly/scripts/run.sh`, selecting Sakib's `.venv` through its existing `ROBOFAB_PY` interface. If that environment or a dependency is unavailable, send the diagnostic to Sakib.

```sh
python3 Kean/scripts/verify_acceptance.py --plan-check --out Kean/results/instructor_preview
python3 Kean/scripts/verify_acceptance.py --run Kean/results/instructor_run_001 --out Kean/results/verification_instructor_run_001 --replace
bash Kean/scripts/run.sh Kean/scripts/check_env.py --gpu
```

CPU verification does not need the GPU environment. `--plan-check` generates both plans and exits 2: criteria 1 and 2 can pass, criterion 3 stays UNVERIFIED. `--run` recomputes the verdict from the saved plan, episodes, cell rows and recording. It ignores `ok` and `full_acceptance`. Exit 0 only when all three criteria PASS. Exit 1 is a criterion FAIL. Exit 2 is UNVERIFIED evidence with no FAIL. A one-structure run cannot exit 0. Contact counts are reported and do not change the three criteria. If an example `--out` already exists, pass a new suffix.

`accept.sh` refuses to write into a directory that already has a run. Pass a new `--out`. `--replace` is the explicit overwrite. Each finished structure is saved under `runs/` before the next one starts, with `provenance.json` (git revision, dirty-file hashes, config, seed, simulator backend, plan hashes). Missing provenance is unverified, not a pass. Recorder NPZ files are gitignored, so a shared submission has to include them outside git.

The GPU check and live run belong on HPC. If Sakib's environment is missing, send that diagnostic to Sakib. The evidence is Newton simulation, not a physical robot. Replay a saved summary with `bash Kean/scripts/run.sh Kean/scripts/replay_motion.py <summary.json>`.

CPU checks can also run with a suitable local interpreter:

```sh
python3 -m unittest discover -s Kean/tests -v
python3 Kean/scripts/m05.py --dry-run
```

Routine `test_m05.sh` checks focus on brace search and motion recovery. Use
`bash Kean/scripts/test_m05.sh --cpu-only --full-tests` when a full historical
regression run is needed. After a passing run, repeat only checks affected by
subsequent changes.

## Motion recovery (2026-10-28)

Planning now runs synchronously on the simulation thread. The simulation pauses
while a query runs, avoiding concurrent access to the shared probe's cached GPU
kinematics buffers. This trades responsiveness for a simpler execution model.

If planning with the other arm's enclosing cuboids fails, strict motion can
propose a path without those cuboids. Every proposed path must still pass the
full-arm sphere checks against the measured other arm, fingers, carried brick
and world before playback. cuRobo retains its self-collision and world checks.
This is a bounded fallback for overly conservative cuboids, not an unchecked
Cartesian move. Sampled validation is not a continuous-collision guarantee.
Accepted fallbacks appear in `motion_report.workarounds`.
If a fallback trajectory fails the other-arm clearance check, the planner adds
the offending sphere as a local obstacle and retries, with up to eight added
obstacles. This supplies collision feedback while preserving the 1 mm threshold.

Replay the saved arch planning failure and S3 command-guard snapshot first on
HPC, without reconstructing the preceding placements:

```sh
bash Kean/scripts/run.sh Kean/scripts/replay_motion.py Kean/results/acceptance/summary.json
```

The guard replay prints three independently recomputed minimum clearances.
A passing replay only establishes feasibility for the recorded state. Check
the complete physical builds separately, preserving the previous summary:

```sh
bash Kean/scripts/accept.sh --live --out Kean/results/acceptance_serialized
```

These workarounds have CPU regression coverage; GPU feasibility and complete
physical acceptance still require the HPC run. Generated supports and final
acceptance requirements remain in force.

Strict acceptance also searches up to eight generated brace configurations per
required support. It tests the original pose, alternative lean angles and
generated support locations with no worse predicted utilisation or increased
grip force. At the end of A's open-finger guard approach, it checks B's complete
staged-to-hover and hover-to-pre-insert paths against A's measured joints, with
both open and closed fingers. Only then may A close. On rejection A retracts
and parks before trying the next configuration; exhaustion fails the build.
Actual B execution is planned again against the measured closed brace.
Candidate poses, rejection reasons and acceptance are saved in each run's
`brace_search` field in the summary. This search needs the live cell and cannot
be exercised by replaying the old fixed-brace request alone:

```sh
bash Kean/scripts/accept.sh --live --out Kean/results/acceptance_brace_search
```

## Shared interfaces and Kean's additions

| Concern | Reused implementation | Kean's remaining responsibility |
| --- | --- | --- |
| Environment and interpreter routing | `Sakib/.venv`, shared `scripts/run.sh` | Thin launcher; report missing prerequisites |
| Scene, controller, grasp/brace queue and clutch | `dual_arm_sim.Example`, `Arm.update`, `queue_place`, `queue_brace` | Joint-path playback and Cartesian handoff |
| Build/brace plans | `planner.sequence`, `planner.build_plan`, `bracing.assign`, `stability` | Run them on a structure id and execute every generated support |
| Motion planning | Cell waypoint IK; P3 probe only if that collides or misses | Do not close P3 or the 235-placement gate first |
| Collision models and brick geometry | P3 `PROFILES`, `TABLE`, `brick_box` | Dynamic scene inputs and geometry-query adapters |
| Clearance | P3 sphere/FK/gap routines | Measured-joint/finger-opening adapter and reporting |
| Contact events and recordings | `ContactMonitor`, `Recorder`, `Example.finish` | Job labels and M0.5 acceptance summary |
| Tight insert, only if the cell misses | `Tanvir/socket_env.py` `ScriptedWriggle` (replayed in `motion/wriggle.py`) | Call. Do not copy, retrain, or require for the mark unless a step fails |
| Insertion RL, perception, memory | Omitted (removed) | Not part of acceptance |

Import local modules as `Kean.*`; the top-level `motion` package belongs to Sakib. `shared.py` loads the existing P3 probe once. Model paths stay exactly as supplied by Sakib. The shared vendor loader runs as Sakib defines it; Kean does not inject Conda packages or override that loader.

## Experiments and prototypes

Default M0.5 uses shared grasp execution and optimizer iterations. `P3_ITER_SCALE` explicitly opts into Kean's iteration-budget experiment; its default is now 1.0. The pickup-offset search stays opt-in (`m05.py --experimental-grasp-offset`). Cube job 0 in `results/checks/20261021-191332-980676` placed 8/8, had no unintended contacts, and passed the motion gate; planned-playback clearance was 0.046 m. That cube has no temporary-support section, so it does not meet instructor acceptance. Hollow_box is extra regression only. Contact acceptance thresholds are unchanged.

The insertion-RL, memory-scan, grip-adjustment and camera-bridge prototypes and the opt-in P3 extension (`p3.sh`/`p3_env.py`, sphere repair, contact rescoring) were removed (2026-10-29) — they were not part of the acceptance path. Tight steps call `motion/wriggle.py`, which replays Tanvir's `ScriptedWriggle` search phases on the cell's waypoint queue.

`motion/newton_fit.py` is the opt-in Newton port (impedance, axial clutch, 0.3 mm seat gate). It is off unless `KEAN_NEWTON_FIT=1`, and it does not load or train a policy. Arch and S3 keep the waypoint replay until a Newton slot probe is within 10 points of the twin.

See [review.md](review.md) for the audit and validation limits.

## Current status (2026-11-01)

Historical notes above that say GPU acceptance is still pending, or that S3 has two supports, are records of that date. Support counts are read from the plans, not fixed at two or three.

`results/instructor_run_001` is the certified package (gpu12, seed 0, `weakest_joint`, revision `d9fc5d79`). The verifier exits 0: arch 11/11 with support step 10, S3 16/16 with supports 11, 13, and 14. Contacts were arch 6 unintended / 4 penetration and S3 11 unintended / 5 penetration. Those counts do not change the three criteria. Replay clearances with `bash Kean/scripts/run.sh Kean/scripts/replay_motion.py Kean/results/instructor_run_001/summary.json`. Show the saved assembly, without stepping the cell, with `bash Kean/scripts/play_run.sh --run Kean/results/instructor_run_001 --shape arch` and open the tunnel at http://localhost:8080. The `.npz` recordings are gitignored; a sync that follows gitignore must still copy them. `results/acceptance` remains the older package with no provenance. Do not pass `--replace` on `instructor_run_001`.
