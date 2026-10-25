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
| `Kean/` | `motion/` planned-path adapter, M0.5 scripts, `viz_cameras.py`, `memory/` and `support/` prototypes, `insertion/` smoke fixture, `acceptance.py` | Cube job 0 with opt-in grasp offset: 8/8, gate passed, 0 braces (`results/checks/20261021-191332-980676`). CPU preview generates arch and S3. | Cell runs of arch and S3, with Arm A holding each generated support. |
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
| Tight insert, only if the cell misses | `Tanvir/socket_env.py` `ScriptedWriggle`, `pad_shift`, `wriggle_brace.choose` | Call. Do not copy, retrain, or require for the mark unless a step fails |
| Insertion RL, perception, memory | Existing prototypes | Omitted. Do not extend |

Import local modules as `Kean.*`; the top-level `motion` package belongs to Sakib. `shared.py` loads the existing P3 script once and redirects only report output to Kean. Model paths stay exactly as supplied by Sakib. Old `Kean/results/p3/fr3*.yml` copies are retained as historical artifacts and are no longer selected by the runtime. The shared vendor loader runs as Sakib defines it; Kean does not inject Conda packages or override that loader.

## Experiments and prototypes

Default M0.5 uses shared grasp execution and optimizer iterations. `P3_ITER_SCALE` explicitly opts into Kean's iteration-budget experiment; its default is now 1.0. The pickup-offset search stays opt-in (`m05.py --experimental-grasp-offset`). The recorded launcher passes that flag. Cube job 0 in `results/checks/20261021-191332-980676` placed 8/8, had no unintended contacts, and passed the motion gate; planned-playback clearance was 0.046 m. That cube has no temporary-support section, so it does not meet instructor acceptance. The same cube without the offset, `results/checks/20261021-171655-549530`, had 8 unintended finger contacts and no measured clearance. Arch and S3, with generated support actually executed by both arms, are still not run. Hollow_box is extra regression only. Contact acceptance thresholds are unchanged.

`p3.sh` runs the shared probe without automatic sphere repair, contact rescoring or summary-source rewriting. Historical extensions require explicit `--kean-extensions`. Collision-model generation (`--step 2` / `--all`) is delegated to Sakib and rejected by this wrapper. P3 report files continue to be written under Kean.

`memory/`, `support/hold.py`, Handoff and GripAdjust are isolated prototypes, not a second live perception/support stack. M0.5's no-op Store bookkeeping has been removed. Future live memory should adapt Sakib's `top_detect.remember`, `track_held` and sensing interfaces while enforcing the task's scan schedule; the local full-pose prototypes are retained for review, not claimed as integrated.

`KinematicInsert` is a CPU fixture with synthetic observations. `scripts/insert.py` is only a network/update smoke experiment. It is not a Newton inserter, not evidence of a trained policy, and not a substitute for `Tanvir/socket_env.py`. A live cell adapter can supply `observe()` using the shared encoded observation; `from_shared` discards all fields except cameras and proprioception. Tight steps call `ScriptedWriggle` instead of growing this fixture into another twin. The frozen MuJoCo twin's 121-D environment is not substituted for this task.

See [review.md](review.md) for the audit, remaining shared-interface needs and validation limits. Sync the complete source tree after these interface changes.
