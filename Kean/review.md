# Reuse audit — 2026-10-21

> Updated 2026-10-29: instructor acceptance now passes on HPC — arch 11/11 and S3
> 16/16 with every generated support held by Arm A, the arch crown seated by the
> scripted wriggle (`motion/wriggle.py`). The omitted prototypes listed below
> (`insertion/`, `memory/`, `support/hold.py` + `grip_adjust.py`, `viz_cameras.py`)
> were removed, not kept. The body of this audit is the 2026-10-21 record.

This review supersedes the earlier cleanup notes. It covers all Kean source areas, scripts, tests and existing evidence, compared with Sakib's current launcher, cell controller, P3 probe, perception, observation encoder, camera/contact/record interfaces, motion skills and PPO code. No cluster run, environment installation or shared-source edit was made.

## Workspaces and finished work (2026-10-25)

- `Sakib/`: cell, planner, vertical-press brace, vision, RL env, standalone sim, BrickSim, docs, environment. Finished unbraced cube, arch, and hollow_box builds, and the sequence/support generators. Perception V1 closed. G-V1 failed. v3.1 frozen. Default strategy is `none`.
- `Kean/`: planned-motion adapter, M0.5 runner, camera bridge, unwired memory/support prototypes, kinematic insertion fixture, acceptance runner. Finished only cube job 0 with the grasp offset (8/8, 0 braces). Arch and S3 are generated in the CPU preview and not run in the cell. Tanvir is a fallback if the cell insert misses, not a second inserter.
- `Tanvir/`: tight-fit audit, socket family, scripted wriggle, pad shift, sideways brace, skill/ES search, proposal. Finished: scripted wriggle beats every learned run; pad shift is the gain. Not in the Newton cell.
- `Richard/`: empty.

Instructor acceptance is not finished. Do not redo the finished pieces above.

## Instructor acceptance (required, 2026-10-25)

The course mark is all three of the following. Master-plan extras are omitted: perception, insertion RL, the camera actor, the memory scan, grip adjustment, the 235-placement M0.5 gate, cube and hollow_box regression, P3 ledger close, G3, G4, and a learned wriggle. They do not add to the mark and must not block the two builds.

1. Take the target structure as direct input. Converting an image into a target structure is not required.
2. Automatically generate the assembly sequence and determine where temporary support is needed.
3. Use both robotic arms to autonomously assemble two different structures, each containing at least eight LEGO bricks and at least one section requiring temporary support.

The cube job below does not meet item 3: `brace_required_count` is 0. Hollow_box is the same. Acceptance evidence still required: arch (11 bricks, one generated support) and S3 (16 bricks, two generated supports), each finished by both arms, with Arm A holding every generated support before Arm B inserts. Call `planner.build_plan` and `bracing.assign`. Call Tanvir only if the cell insert misses a step. Do not hand-author the sequence, do not use strategy `none`, do not require an image blueprint, and do not copy Tanvir's tight-fit stack into `Kean/`.

## Unnecessary parallel machinery removed from active paths

- **Interpreter routing:** Kean's wrapper now calls Sakib's existing `scripts/run.sh` with its supported `ROBOFAB_PY` parameter. Kean does not build or repair the environment. The former package inventory/prefix checker is reduced to a real probe import/GPU/model smoke check. Setup commands and installation advice have been removed from current Kean instructions/errors; provisioning belongs to Sakib.
- **Probe environment hooks:** removed the custom cuRobo locator and temporary replacement of `motion.vendor.add_overlay`. The shared probe imports normally, including Sakib's own dependency hook. Kean adds no Conda package path.
- **Collision-model copies:** stopped copying/repointing PROFILES or changing probe.HERE. Runtime uses Sakib's model files directly. Kean only redirects experiment reports. Generating/refitting shared models is blocked in the Kean P3 wrapper and left with Sakib.
- **Default P3 alterations:** sphere repair, declared-graze rescoring and source-rewritten latency summaries no longer install automatically. Historical experiment modules remain available through explicit `--kean-extensions`. Optimizer iteration scaling defaults to 1.0, preserving Sakib's configuration.
- **Trajectory interpolation and controller tests:** replaced the simplified test-only Arm with the actual shared pure class (loaded without GPU imports). Deleted Tick's duplicate interpolation routine; actual motion and CPU adapter tests now use the shared probe's `resample`. The adapter only handles missing velocity/single-sample edge cases. It receives the actual shared D.wrap as well.
- **Brick/table geometry:** deleted Kean's table dictionary and brick-envelope construction/height constant. Runtime passes actual poses/dimensions to the existing probe.brick_box and uses probe.TABLE. Variable-size plate placement and world-state wiring remain adapter responsibilities.
- **Unused GT memory:** removed per-frame detector-shaped GT snapshots and Store/ScanSchedule/frozen-grasp bookkeeping from M0.5. They did not drive its planner or controller and overstated memory integration.
- **Grasp offset:** still opt-in (`--experimental-grasp-offset`), not the default shared queue. Cube job 0 with the flag passed; see the synced summary below. Sakib's planner still warns that an off-centre grip can swing, and the other shapes are unrun, so it is not promoted to every job.
- **RL observations:** added a projection of the existing `rl_env.ObsSpec.encode` output, using the shared camera sampling, FK and quaternion-sign convention. Actor inputs contain only images/proprioception; task, bricks and privileged fields are dropped. Existing raw-data packing remains solely for lightweight fixtures and the unintegrated smoke cell.

## Kean code retained for a concrete reason

- `motion/plan_curobo.py`, `tick.py`, `contact_segment.py`: the shared Newton cell has waypoint IK but no integrated asynchronous planned-path playback with a reserved contact tail. These are the motion deliverable, not a duplicate cell.
- `motion/world.py`, `clearance.py`: adapters from live example state to the shared planning/geometry APIs. The clearance adapter also handles unequal measured left/right openings; P3's finger_shifted_spheres takes a single opening for both fingers. No new collision-distance algorithm is implemented.
- `motion/gate.py`, `scripts/m05.py`, the recorded-test wrapper: task-specific experiment orchestration and acceptance. They reuse the cell plan, brace selector, monitor, recorder and finish path. The hand-built Example argument Namespace is an integration seam: the shared cell exposes no CPU-light reusable default-argument factory, and importing rl_env.example_args also imports gym/vision/Newton. A public factory would be a suitable shared-interface request, not a reason to copy more cell code.
- `insertion/obs.py`, `env.py`, `train.py`: a camera/proprioception adapter and a kinematic smoke fixture only. Tanvir already has the tight-fit inserter (`socket_env.ScriptedWriggle`, `SocketInsertionEnv` with `pad_shift`, `skill_env.SkillController`, `tight_fit_audit`, `socket_family`, `wriggle_brace`). Do not expand `Kean/insertion/` into a second twin, wriggle, pad shift, or PPO/ES trainer. No learned policy in `Tanvir/` beat the scripted wriggle. Reuse `squash_logp` if a residual on that base is resumed later. Do not copy the twin environment or Tanvir's modules, and do not claim the synthetic smoke cell is physical RL.
- `memory/`, `support/`: retained as isolated prototypes with tests. Runtime support remains Example's staged/braced/retracted handshake and queue_brace. Runtime perception should remain top_detect/vision. A future scan-budget/full-pose adapter must account for the shared track_held representation (top-face position/yaw and stable detection identity), rather than treating it as interchangeable with full-pose plan IDs.
- `viz_cameras.py`: no equivalent HTTP camera endpoint was found. It displays existing Example.refresh_feeds tiles; it does not implement camera rendering/perception. Its global server lifecycle is a known limitation, not a new motion/RL workstream.

## Remaining issues before claiming the full gate

- Synced run `results/checks/20261021-191332-980676` is cube, weakest_joint, seed 0, with the pickup offset. It placed 8/8 in 8540 frames. Summary: `ok` true, 0 plan failures, 0 captures, 0 arm-arm, 0 unintended, 0 displaced, 0 penetration, 18 neighbour_rub events during insert, and `min_clearance_m` 0.0461. Offsets were 1.5 mm or 1.25 mm on steps 1, 3, 5, and 7. Lateral snap error stayed under 0.2 mm. This is one unbraced job, not the 235-placement gate, and not instructor acceptance.
- The earlier synced cube, `results/checks/20261021-171655-549530`, also placed 8/8 but failed: 8 unintended B-finger contacts on steps 1/3/5/7 and `min_clearance_m` null. The offset run does not relabel those events. Neighbour rubs remain uncounted, as before.
- Grip adjustment is still the task-level path when a 2 mm shift cannot clear a grasp. It is not wired into the live cell.
- Full-sphere clearance measures executed joints during planned playback only. ContactMonitor still scores contact segments. Held-brick attachment and the scan-memory-driven world are not integrated; runtime world input is explicitly GT diagnostic state.
- Stress jobs are still excluded from gate.fold's contact aggregation. A separate stress acceptance result is needed; the current summary cannot prove the stress requirement.
- Planner errors still permit a Cartesian fallback (counted as a gate failure). The abort policy and counting graph initializations as well as torch captures remain open.
- CPU motion tests now load the actual shared pure Arm class, wrap and resample bodies without importing Newton. Arrival/wait/callback behavior is checked at Cartesian handoff. Unit passes still do not establish physical grasp stability or tracking.
- The camera RL trainer is a prototype. Horizon-end value bootstrap and exact sample-budget handling still need attention, and the live Newton action/reset/reward adapter is absent. Do not build that physics environment. Tight insertion is `Tanvir/socket_env.py`. Wiring `ScriptedWriggle` at pre-insert is the missing integration, not a new skill.

## Validation

Local audit result: **75 passed, 1 skipped** (PyTorch unavailable). That audit launched no cluster job.

Synced HPC result, NVIDIA L20, Sakib's `.venv`: run `20261021-191332-980676` had unit, manifest, GPU preflight, and cube job 0 all exit 0. The unit log is 76 passed. `20261021-185956-521081` stopped at unit exit 1 before the source tree was complete; it is not a motion result.
