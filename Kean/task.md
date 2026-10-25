# Kean — two braced dual-arm builds

**Execution setup.** Local development is CPU-only macOS. GPU work runs on HPC4. Keep new files and edits under `Kean/` by default and avoid editing shared folders. Reading outside `Kean/`, including `Sakib/`, is allowed without additional permission.

**Ownership.** Sakib owns environment provisioning, the cell, the sequence and support generators, grasp/brace execution, and collision models. Tanvir owns the tight-fit skill. Kean runs the generators and finishes the two builds. Kean scripts must consume the supplied environment, not install or repair it.

**Full mark.** The three instructor requirements below are sufficient. Master-plan items that they do not need are omitted.

**Reuse first.** Inspect existing code and documentation before implementing. Reuse shared controllers, utilities, and interfaces; add adapters or missing behavior under `Kean/` rather than duplicating existing implementations. If a shared-file edit is necessary, follow the authorization guidance in `Kean/AGENTS.md`.

**v3.1 is frozen.** Do not re-run G3 or G4, and do not port the twin PPO onto the Newton weld. **v4 is the cell.** Sakib owns the environment. This file is the job.

**Finished, do not redo.** Sakib: unbraced cube 8/8, arch 11/11, hollow_box 12/12, plus `planner.build_plan` and `bracing.assign`. Kean: cube job 0, grasp offset, 8/8, motion gate passed, 0 braces. Tanvir: `ScriptedWriggle` plus pad shift; learned search did not beat it. Richard: empty. Not finished: arch and S3 finished in the cell, with every generated support held by Arm A.

---

## Instructor acceptance (full mark)

These three are the mark. Do not drop or weaken any of them. Do not add master-plan work as a further requirement.

1. Take the target structure as direct input. Converting an image into a target structure is not required.
2. Automatically generate the assembly sequence and determine where temporary support is needed.
3. Use both robotic arms to autonomously assemble two different structures, each containing at least eight LEGO bricks and at least one section requiring temporary support.

Direct input is a structure id or an equivalent brick list already in memory (`--structure`, or a named sample such as `--shape` that does not read an image). Do not require `--front`, `--side`, `--top`, or any silhouette-to-structure step. Image conversion may exist in the shared repo; it is not part of this job and must not block acceptance.

Sequence and support are generated, not supplied by hand. Call `planner.sequence` and `planner.build_plan` with `bracing.assign` (`weakest_joint` or `nearest`). Do not use `--strategy none` for the acceptance runs. The plan must record, per step, the order, `requires_brace`, and the support pose when a brace is required. Do not hand-author the brick order or which steps are supported.

The two structures are the arch and S3. The arch has 11 bricks and, under `weakest_joint` / `grasp_lp`, `brace_required_count` 1 (the crown). S3 has 16 bricks and `brace_required_count` 2 under the same generators (`single_arm_fails` is true). Both must be assembled autonomously: Arm A reaches each generated support pose and holds before Arm B inserts on that step; Arm B picks, transports, and inserts every brick; the build completes. A run that skips a required support, or that only places an unbraced cube or hollow box, does not count.

Autonomous means no manual stepping and no hand-labeled brace list. Ground-truth structure input is allowed. A learned inserter is not required.

---

## Omitted from the master plan

Not required for the mark. Do not build them, do not train them, and do not block the two builds on them.

- Image-to-structure, perception V1/V2/V3, G-V1, and vision-only acting.
- Insertion RL, residual PPO, the camera actor, reward shaping, the place-bandit, and the brace-bandit.
- The memory scan, grip adjustment, and a live tracker for a grasped brick.
- The M0.5 235-placement gate, cube and hollow_box regression, brace-all stress, and a P3 ledger close.
- G3, G4, and any port of the v3.1 twin PPO onto the Newton weld.
- A learned policy that beats `ScriptedWriggle`. Five PPO runs and one evolution-strategy run already failed.

If the cell's scripted insert cannot seat a step on the arch or S3, call Tanvir's `ScriptedWriggle` and, for a sideways neighbour, `wriggle_brace.choose`. That is reuse, not a new work package. Do not copy those modules and do not train.

---

## Contract

Sakib supplies the cell and the generators (`planner.py`, `bracing.py`, `stability.py`, `queue_place`, `queue_brace`). Kean runs those generators on a structure id and executes the sequence and support poses. Tanvir is a fallback for a missed insert, not a required run. Do not treat a pre-written pose list as a substitute for generation. Do not copy Tanvir's modules into `Kean/`.

Arm A supports. It moves to the generated support pose and holds before Arm B inserts that step. Arm B does the pick, the transport, and the insert. Only one arm moves inside the build zone. Grip adjustment is not required.

Use the cell's existing insert. Call Tanvir only when that insert misses a step on the arch or S3. Do not start insertion RL.

---

## What to do

1. **Generate.** From a structure id, run `planner.build_plan` with `bracing.assign`. Record the order and every `requires_brace` pose.
2. **Support.** Move Arm A to each generated support pose and hold it before Arm B inserts. Do not rebuild `bracing.py` or `tasks/brace_bandit.py`.
3. **Assemble.** Both arms finish the arch and S3. Reuse the cell's motion and insert. A planned path is allowed only where waypoint motion collides or misses. Do not close P3 or the 235-placement gate first.

`memory/`, `insertion/`, and the camera actor already exist as prototypes. They are omitted. Do not extend them. `curobo_server.py` is written only if in-process cuRobo is required and fails. `fr3.yml` stays in `Sakib/brickassembly/motion/`.

---

## Do not build

- A second cell, clutch, sequence generator, or perception stack.
- An image-to-structure converter.
- A new brace selector. Execute `bracing.assign`. Call `wriggle_brace.choose` only if a tight step on the arch or S3 breaks a neighbour.
- A second tight-fit stack, pad shift, or PPO/ES trainer.
- The place-bandit, the memory scan, and a live tracker for a grasped brick.

---

## Constraints that still apply

- More press is not progress. Use 8.9 N insertion and 11.3 N break per stud. Do not invent force limits.
- On the arch and S3, Arm A must actually hold each generated support. Zero arm–arm events on those steps.
- If a plan is used, it must not block the sim tick. Warm up every CUDA graph before the loop. A capture during the loop fails the run.

## Read

1. `Sakib/brickassembly/dual_arm_sim.py` `queue_place` and `queue_brace`
2. `Sakib/brickassembly/planner.py` `build_plan` and `bracing.py` `assign`
3. `Tanvir/socket_env.py` `ScriptedWriggle` only if the cell insert misses a step
4. `Sakib/Docs/master_report.md` is the frozen record, not this job
