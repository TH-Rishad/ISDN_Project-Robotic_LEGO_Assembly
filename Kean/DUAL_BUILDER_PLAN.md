# Dual-builder execution

Status: approved and implemented; GPU validation is in progress.
Branch: `feat/dual-builder-coordination`.

Execution refinements from GPU validation: support-waiting builders move with
their carried brick to a checked park pose before the helper approaches.
The executor honors generated grasp offsets and checks pickup height and
release clearance. Brace search also tests finer heights, lateral positions
and the parallel gripper's equivalent wrist orientation, preserving the
generator's feasibility and support-capacity checks. The full finger-closing
sweep and carried-brick insertion corridor must clear before a helper moves.
Dual-build precision sockets retain Kean's wriggle phases with 0.1 mm / 0.1
degree limits and a longer press; existing single-build defaults are retained.

Contact evidence identifies individual collision shapes. During a planned
insertion or release, contact between a lower mating stud cylinder and its
upper brick is logged as `stud_mating` up to the calibrated 1.7 mm stud height.
For studs embedded in the full mesh, the contact must lie above the shell and
within 0.3 mm of the calibrated stud radius. The verifier recomputes the local
contact coordinate from the saved world contact and recorded lower-brick pose.
The raw depth, shape types and local stud position remain in `contacts.json`.
The verifier checks those fields against the saved mating references. Other
deep contacts, wrong brick pairs and excessive penetration still fail; the
1 mm free-space/arm clearance checks are unchanged. Every deep shape contact
is classified separately so a valid stud contact cannot hide a wall collision
on the same body pair.

## Validation in progress

| Check | Evidence | Status |
| --- | --- | --- |
| Kean CPU regressions | 158 tests | PASS |
| Shared handoff protocol, aiming and manual controls | Existing shared tests | PASS |
| Concurrent placement by both arms | `results/dual_pick_probe_004` | PASS; diagnostic |
| Upper-course clearance | `results/dual_upper_course_001` | PASS; diagnostic |
| A-first tie and reciprocal support | `results/dual_support_tie_016` | PASS; diagnostic |
| Later S3 supports and completed-arm helping | `results/dual_late_support_004` | PASS; diagnostic |
| Latest later-support regression | `results/dual_late_support_005` | PASS; diagnostic |
| Contact-evidence pipeline | `results/dual_stud_policy_001` | PASS; diagnostic |
| Stud/wall distinction, mixed contacts and excessive depth | CPU policy/verifier regressions | PASS |
| Mesh-stud contact evidence | `results/dual_mesh_stud_001` | PASS; diagnostic |
| Complete build with detailed contact evidence | `results/dual_arch_s3_010` | Running |

Diagnostic scenes pre-place earlier bricks and do not establish full-build
completion. A complete from-empty dual build has not yet been accepted.

## Requested behavior

Each arm owns a separate structure, its assembly plan, and its progress.
Both arms build concurrently where their motions are compatible. When an
arm reaches a step requiring support, the other arm finishes its current
placement, then helps with that step. Both arms can alternate between
builder and supporter. Simultaneous pending support requests give arm A
priority. A completed builder remains available to support the other arm.

The initial demonstration will use arm A building arch (11 bricks) and arm B
building S3 (16 bricks), unless the user chooses another pairing. Both already
have generated support steps: arch 10; S3 11, 13, 14. These indices/counts
remain generated plan data, not scheduler constants.

## Existing controller to preserve

Use Kean's current `CuroboBackend`, `Planner`, planned joint playback,
velocity feed-forward, full collision spheres, measured finger openings,
carried-brick checks, clearance-validated fallback paths, and brace search.
Keep the existing snap gate, clutch/weld model, support-force model, and
scripted tight insertion behavior.

Generalize the functions currently bound to B as placer and A as stabilizer
to accept the active builder, helper, and build context. This includes
placement queues, brick aiming, brace queues/readout, fixed-joint brace hold,
support candidate checks, retraction handoff, and picking/bracing finger
settings. Both arms must receive the existing appropriate controller
settings for their current role; simply assigning A a placement queue would
leave it with the wrong grip configuration and state references.

Most new orchestration belongs under `Kean/`. Extend shared scene and
primitive code in `Sakib/brickassembly` where necessary, retaining its
existing single-build defaults. Existing Kean commands remain regression
paths. There is one physics world and one simulation clock.

## Scene, plans, and geometry

1. Add a second studded baseplate/build pad and a dedicated brick feeder for
   arm A, alongside B's feeder. Include visible workspace labels, separate
   staging/parking poses, and the platforms in the physics and planner worlds.
2. Start with two pads separated along world Y, both in the arms' mutual
   reachable region. Centers near `(0, -0.16)` and `(0, +0.16)` meters are
   layout candidates, not validated placements. Keep the current arm bases
   initially. Choose final pad/feeder positions using reachability and full-arm
   clearance preflight, including reciprocal support approaches and the
   envelopes of the completed structures. Store the validated layout with the
   run so it can be reproduced.
3. Generate each assembly sequence and support plan using the current
   planner/bracing code. Give each plan its own origin and world transform;
   do not let mutable global `VOXEL_ORIGIN` leak between builds. Translate all
   targets, support poses, connection centers, and related world geometry
   consistently.
4. Namespace brick IDs by build, including every mating/support reference.
   Each build has its own step counter, held brick, placement history, welds,
   support events, and travel height. Neither build uses a shared `b_step` as
   its authoritative state.
5. Give each structure its own welded collision group. The current common
   group suppresses collisions among all placed bricks, which would conceal
   collisions between separate structures. Maintain only the existing
   intended exclusions within each build.

## Communication and support handshake

Use one deterministic in-process coordinator with explicit requests and
acknowledgments. The simulator already contains both arms, so network
services and additional middleware are unnecessary.

Support requests carry requester arm, build ID, step ID, and the generated
support requirements. Readiness and completion come from the controllers'
measured states and callbacks, not a timer pretending the action finished.

For an A request (the reverse is identical):

1. A discovers that its next generated step requires support and requests
   assistance **before picking up that brick**. It waits empty-handed at its
   checked safe pose.
2. B reserves its next task for helping A. If B is doing an ordinary placement,
   B completes placement, release, and safe retraction. It does not start
   another own-build step. If already free, B can acknowledge immediately.
3. Once B is available, A picks up its brick and reaches the existing safe
   staging state. B approaches the generated support, runs the role-adjusted
   brace search and guards, closes, and confirms its hold.
4. A proceeds only after that confirmation. B holds measured joints during
   A's placement and release.
5. A clears the shared area before B releases support and retracts through a
   checked route. The cooperative step ends only after both are clear.
6. The coordinator services any already-pending reciprocal request, then
   allows ordinary independent building to resume.

| Situation | Scheduling rule |
| --- | --- |
| Neither arm requests support | Both can build when their motion reservations are compatible. |
| One requests; the helper is midway through an ordinary step | Helper finishes that step, safely retracts, then assists. |
| Both have pending requests before a support operation starts | A's step is supported first, then B's pending step. B remains empty-handed while helping A. |
| A request arrives during an already-active B support operation | Finish the active operation before changing roles. A's priority does not interrupt an engaged brace. |
| One structure finishes first | That arm stays available as helper until both builds finish. |

Checking both pending requests before dispatch prevents update-order races.
Keeping the losing request queued and servicing it next prevents starvation.
No scheduler state can require both arms to hold bricks while each waits for
the other to become a supporter. No in-flight placement is interrupted to
grant a support request.

## Concurrent motion and collisions

Kean's current strict tick permits only one moving arm and stops a path if
the other arm changes from its original measured pose. Removing these checks
alone would invalidate the collision model.

Keep GPU planning synchronous on the simulation thread, as in Kean now, and
allow accepted trajectories to execute concurrently:

- Reserve checked motion segments before playback. Use conservative swept
  full-arm/finger and carried-brick geometry to admit independent concurrent
  segments; if the reservations conflict, hold the new segment at a checked
  waiting pose until the active segment clears. Include contact and retreat
  phases in reservations, not just free-space transport.
- Validate both structures, both platforms, all feeder bricks, and the other
  arm's carried brick as obstacles. Exclude only the active arm's own carried
  object from its static world, while checking its attached geometry.
- Retain Kean's measured-state and next-command checks. Replace the blanket
  moving-other-arm rejection only when that motion is covered by an accepted
  reservation. Unexpected motion or tracking error invalidates the affected
  paths and holds the arms for revalidation.
- Retain the current 1 mm planning clearance floor and dense checking, with
  geometry/motion uncertainty included conservatively in reservations. Allow
  intended pick, mating, and brace contacts only for the active objects and
  phases. Inspect actual contact events as well as predicted clearance.
- During cooperative insertion, reserve the relevant build workspace for the
  builder/helper pair. The helper remains stationary in its measured brace
  hold while the builder follows its validated placement path.
- On an invalid path, support loss, collision, or bounded progress failure,
  hold measured commands and clear velocity feed-forward. Any recovery or
  retraction route must also pass clearance checks; do not blindly release an
  active brace on a timeout.

This first version uses conservative reservations with existing individual
arm planners. It does not require replacing Kean with a new joint 14-DOF
planner. Concurrency must be demonstrated in recorded execution; serially
alternating all placements is not sufficient.

## Proposed code changes and interface

| Area | Change |
| --- | --- |
| New Kean dual-build runner/coordinator | Own the two build contexts, support handshake, task grants, and completion. |
| Shared scene and action primitives | Add two build pads/feeders and make placement, aiming, support and readout accept explicit arm/build arguments. |
| `Kean/motion/tick.py`, `path_check.py`, `world.py`, `clearance.py` | Validate/reserve concurrent trajectories and account for both worlds and both carried objects. |
| `Kean/motion/coordinated.py`, `brace_hold.py`, `brace_search.py`, `wriggle.py` | Support either builder/helper assignment using the existing behavior. |
| Contact monitor, recorder, playback, verifier | Record per-arm build/step/role and support ownership; replay the complete two-build cell; retain existing recording support. |

Run from the repository root with Sakib's existing environment:

```bash
bash Kean/scripts/dual_build.sh --build-a arch --build-b S3 \
  --viewer viser --port 8081 --out Kean/results/dual_arch_s3_live_001
```

Choose a fresh output directory for each run. Omit `--viewer viser` for a
headless run. Replay the complete recorded cell and verify its evidence with:

```bash
bash Kean/scripts/play_run.sh --run Kean/results/dual_arch_s3_live_001 --port 8081 --loop
Sakib/.venv/bin/python -m Kean.verify_dual --run Kean/results/dual_arch_s3_live_001
```

`--step-limit`, `--start-step-a` and `--start-step-b` are diagnostic options.
Runs using these options are marked diagnostic and never count as a full build.
For example, `--start-step-a 10 --start-step-b 11 --step-limit 1` exercises
simultaneous support requests using nominal earlier placements.

The run manifest will store both plans, layout, configuration and provenance.
Events will identify request, helper-ready, support-held, placement-complete,
clearance-complete, support-released, and resumed-building, with arm/build/step
IDs and simulation timestamps. The viewer will show both structures, each
arm's role and progress, pending requests, and reasons for waiting. NPZ
recordings retain both joint trajectories and all bricks and add the
per-arm/build state required for playback and verification.

## Implementation order and acceptance

1. Implement and unit-test the coordinator: unilateral requests, helpers
   finishing a current step, simultaneous requests with A first, retained B
   requests, role reversal, completed-build helpers, and progress failures.
2. Add the shared scene and role-neutral primitives. Preflight both pads and
   feeders, then test each arm independently placing while the other supports.
   Check the different role-dependent finger settings and brace holds.
3. Add concurrent segment reservations and collision tests, including both
   moving arms, either/both carrying bricks, an occupied other workspace,
   moving-obstacle invalidation, and intended-versus-unintended contacts.
4. Extend logging/playback, then run small GPU concurrency cases and the full
   arch/S3 pair. Exercise the reciprocal roles and simultaneous-request case;
   use explicitly labeled nominal-start test scenes when a forced tie is
   needed, rather than counting pre-placed bricks as a full build.
5. Run the existing Kean CPU regressions and affected shared tests. Preserve
   the existing single-build acceptance behavior and the saved instructor
   evidence.

The final demonstration must show both arms placing their own bricks, actual
overlapping independent work, completed arch and S3 structures, every generated
support held by the other arm, A winning a simultaneous request, B subsequently
being served, and both arms finishing safely. Unintended arm-arm,
arm/other-structure, or carried-brick collisions, joint breaks, unserved
requests, and deadlock/timeouts fail the new mode's checks. Intended mating,
grasping, and bracing contacts are classified explicitly. Save replayable
recordings and the scheduler event history for review.
