# Kean: two braced dual-arm builds

Sakib owns the environment, the Newton cell, and the sequence and support generators. Tanvir owns the tight-fit skill. Kean runs the generators on a structure id and finishes the arch and S3. Do not copy Tanvir's modules into `Kean/`.

Meeting the three requirements below is a full mark. The M0.5 gate, insertion RL, perception, and the other master-plan items are omitted.

## Instructor acceptance (full mark)

1. Take the target structure as direct input. Converting an image into a target structure is not required.
2. Automatically generate the assembly sequence and determine where temporary support is needed.
3. Use both robotic arms to autonomously assemble two different structures, each containing at least eight LEGO bricks and at least one section requiring temporary support.

Input is `--structure` or a named sample such as `--shape` that does not read an image. Do not require `--front`, `--side`, or `--top`. Generate the order and the support poses with `planner.build_plan` and `bracing.assign` (`--strategy weakest_joint` or `nearest`, never `none` on an acceptance run). Do not hand-author the sequence or the support steps. Call Tanvir only if the cell insert misses a step. Do not train.

The two structures are the arch (11 bricks, `brace_required_count` 1 under `weakest_joint`) and S3 (16 bricks, `brace_required_count` 2, `single_arm_fails`). Arm A must hold each generated support pose before Arm B inserts that step. Both builds must finish. Cube and hollow_box have `brace_required_count` 0, so a passing cube job does not meet item 3.

## Workspaces and what is finished (2026-10-25)

| Workspace | Has | Finished | Not finished |
| --- | --- | --- | --- |
| `Sakib/` | `brickassembly/` (cell, planner, bracing, stability, vision, RL env, plans, ledger), `robofab_sim/`, `BrickSim/`, `Docs/`, `setup_env.sh` | Unbraced cube 8/8, arch 11/11, hollow_box 12/12. Sequence and vertical-press support generators. Perception V1 closed. v3.1 frozen. | G-V1. Live default is `--strategy none`, so braced acceptance builds are not the current path. |
| `Kean/` | `motion/` planned-path adapter and the acceptance runner (`acceptance.py`, `scripts/accept.sh`) | arch 11/11 and S3 16/16 in the cell, with Arm A holding each generated support and the scripted wriggle seating the arch crown. | — |
| `Tanvir/` | `tight_fit_audit.py`, `socket_family.py`, `socket_env.py`, `wriggle_brace.py`, `skill_env.py`, trainers, `report/` | `ScriptedWriggle` is the best policy. Pad shift is the measured gain. Five PPO runs and one ES run did not beat scripted. Proposal written. | Not in the Newton cell. Held-out, no brace, 1–2 mm: 26/68. |
| `Richard/` | `.gitkeep` | Nothing | Everything |

Do not rebuild those finished pieces. Next work is `bash Kean/scripts/accept.sh --live`: arch and S3, generated support held by Arm A. Perception, insertion RL, the 235-placement gate, and a learned wriggle are omitted.

## Run with Sakib's supplied environment

No Conda activation or Kean installer is required. The thin `scripts/run.sh` wrapper delegates to `Sakib/brickassembly/scripts/run.sh`, selecting Sakib's `.venv` through its existing `ROBOFAB_PY` interface. If that environment or a dependency is unavailable, send the diagnostic to Sakib.

```sh
bash Kean/scripts/run.sh Kean/scripts/check_env.py --gpu
bash Kean/scripts/test_m05.sh
python3 Kean/scripts/accept.py --dry-run
bash Kean/scripts/accept.sh --live
```

The smoke check imports Sakib's actual motion probe and reports its model paths. It does not install, repair or pin packages. The recorded test runs CPU checks, manifest generation, that smoke check and cube job 0. It saves logs, source hashes, per-stage exit codes, shared cell rows and the shared Recorder's NPZ in `Kean/results/checks/<Hong-Kong-timestamp>/`. `--max-frames N` bounds simulation frames, not planner wall time.

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

Import local modules as `Kean.*`; the top-level `motion` package belongs to Sakib. `shared.py` loads the existing P3 script once and redirects only report output to Kean. Model paths stay exactly as supplied by Sakib. Old `Kean/results/p3/fr3*.yml` copies are retained as historical artifacts and are no longer selected by the runtime. The shared vendor loader runs as Sakib defines it; Kean does not inject Conda packages or override that loader.

## Experiments and prototypes

Default M0.5 uses shared grasp execution and optimizer iterations. `P3_ITER_SCALE` explicitly opts into Kean's iteration-budget experiment; its default is now 1.0. The pickup-offset search stays opt-in (`m05.py --experimental-grasp-offset`). Cube job 0 in `results/checks/20261021-191332-980676` placed 8/8, had no unintended contacts, and passed the motion gate; planned-playback clearance was 0.046 m. That cube has no temporary-support section, so it does not meet instructor acceptance. Hollow_box is extra regression only. Contact acceptance thresholds are unchanged.

`p3.sh` runs the shared probe without automatic sphere repair, contact rescoring or summary-source rewriting. Historical extensions require explicit `--kean-extensions`. Collision-model generation (`--step 2` / `--all`) is delegated to Sakib and rejected by this wrapper. P3 report files continue to be written under Kean.

The insertion-RL, memory-scan, grip-adjustment and camera-bridge prototypes were removed (2026-10-29) — they were not part of the acceptance path. Tight steps call `motion/wriggle.py`, which replays Tanvir's `ScriptedWriggle` search phases on the cell's waypoint queue.

See [review.md](review.md) for the audit and validation limits.

## Final acceptance

Both structures assemble in the cell with every generated support held by Arm A:

- `--shape arch` places 11/11; the crown (`b_010`) seats with the scripted
  wriggle (lateral < 0.15 mm, dz < 0.8 mm, tilt < 2.1 deg).
- `--shape S3` places 16/16 with its three required supports.

The strict acceptance path searches up to eight brace configurations per
required support (including the opposite pier) before closing, and replays
Tanvir's `ScriptedWriggle` search phases for precision sockets. Run the
combined acceptance with:

```bash
bash Kean/scripts/accept.sh --live
```

A single-structure summary records `scope=<shape>` and `full_acceptance=false`;
it does not certify the combined goal. Omit `--shape` for the final combined
acceptance.
