# Kean — motion, support, and insertion RL

**Execution setup.** Local development is CPU-only macOS. GPU work (cuRobo, Newton, RL training) runs on HPC4. Keep new files and edits under `Kean/` by default and avoid editing shared folders. Reading outside `Kean/`, including `Sakib/`, is allowed without additional permission.

**Ownership.** Sakib owns environment provisioning, shared simulation, sensing, grasp/brace execution, and collision models. Kean owns planned-motion integration and camera/proprioception insertion RL. Kean scripts must consume the supplied environment, not install or repair it.

**Reuse first.** Inspect existing code and documentation before implementing. Reuse shared controllers, utilities, and interfaces; add adapters or missing behavior under `Kean/` rather than duplicating existing implementations. If a shared-file edit is necessary, follow the authorization guidance in `Kean/AGENTS.md`.

**v3.1 is frozen.** Do not re-run G3 or G4, and do not port the twin PPO onto the Newton weld. **v4 is the cell.** Sakib owns the environment. This file is the job.

---

## Contract

Sakib supplies the cell, the sequence, and the support pose. Kean takes a brick pose and an insertion target.

Arm A supports. It moves to the support pose and holds before Arm B transports. It also holds the brick when Arm B must adjust its grip. Arm B does the pick, the transport, and the insert. Only one arm moves inside the build zone.

Motion ends at the pre-insert pose: above the target socket, +25 mm in z. Insertion RL starts there, brick already grasped. It does not learn pick, transport, or grip adjustment.

---

## What to develop

1. **Memory.** Scan brick poses at initialization and after each completed step. On grasp, freeze the brick in the gripper frame and infer it from forward kinematics. Do not track a held brick. Rescan only while a grip is being adjusted, then freeze the new frame.
2. **Support.** Move Arm A to the support pose on a collision-free path, then hold. When a grip adjustment is required, Arm A holds the brick, Arm B regrasps, the scan runs, and Arm A returns to its hold. Consume the support pose from `bracing.py`. Do not rebuild `tasks/brace_bandit.py`.
3. **Motion.** Collision-free path for either arm from the stored pose to the handoff pose. Arm B moves only after Arm A has settled. The last 30 mm of grasp, insert, and support are straight Cartesian segments, not planned paths.
4. **Insertion RL.** Residual PPO on the scripted insert. Sparse snap reward. The policy sees camera frames and both arms' proprioception only.

```
Kean/
├── memory/
│   ├── store.py
│   └── scan.py
├── support/
│   ├── hold.py
│   ├── sequence.py
│   └── grip_adjust.py
├── motion/
│   ├── plan_curobo.py
│   ├── contact_segment.py
│   └── curobo_server.py
├── insertion/
│   ├── obs.py
│   ├── env.py
│   └── train.py
├── tests/
├── scripts/
│   └── p3.sh
└── results/
    ├── p3/
    ├── m05/
    └── insertion/
```

`curobo_server.py` is written only if in-process cuRobo fails. `fr3.yml` stays in `Sakib/brickassembly/motion/`.

---

## Observation

The actor gets frames from `top`, `wrist_A`, and `wrist_B`, plus both arms' joint positions, joint velocities, and finger openings, and the hand poses from forward kinematics.

Brick position and orientation are a memory, not a live input. A held brick's frame is the frozen grasp composed with the hand's forward kinematics.

Not actor inputs: the true socket error, clutch state, weld utilization, a camera embedding, and a wrist force-torque history. The cell has no wrist force-torque sensor. Those signals are reward and critic only.

This is not the master-plan 183-D observation and not the frozen twin's 121-D observation.

---

## Reward and action

Action is a residual on the scripted insert: an end-effector increment at 20 Hz, plus a press channel. Cap the press at the stud budget. Do not invent force limits. Use 8.9 N insertion and 11.3 N break per stud.

Reward weights, chosen: +1 on snap, −1 on a joint break, a force above n·11.3 N, or a drop, and 0 otherwise. This is the frozen twin's scale. The spec's +50 / −10 is not used.

A grip adjustment is a support move. It is not an insertion action.

---

## Do not build

- Sakib's cell, clutch, sequence, or perception stack.
- The brace selector (`tasks/brace_bandit.py`). Execute the pose it emits.
- The v4 place-bandit. It is a draft and is not adopted.
- A live tracker for a brick that is already grasped.

---

## Inherited constraints

- Current cell motion is waypoint IK in `dual_arm_sim.py`. Planned motion is the new work. P3 (`scripts/12_curobo_probe.py`) has no ledger pass. Close it on HPC before trusting `plan_curobo.py`.
- M0.5 is the motion gate: cube, arch, hollow_box, S3, zero arm–arm events. Report braced and unbraced separately.
- G3 and G4 failed on joint breaks and on press, not on transport. More press is not progress.
- Plans must not block the sim tick. Warm up every CUDA graph before the loop. A capture during the loop fails the run.

## Read

1. `Sakib/Docs/plan_v4.md` §2.3, §2.4, P3, WP2
2. `Sakib/brickassembly/rl_env.py` for what a policy may read
3. `Sakib/brickassembly/dual_arm_sim.py` `queue_place` and `queue_brace`
4. `Sakib/Docs/master_report.md` §WP5 and §7.7 M8 only as the frozen record, not as this MDP
