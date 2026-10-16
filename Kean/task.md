# Kean — motion planning and insertion RL

**Execution setup**: Local development (OpenCode) is CPU-only. GPU-heavy work (RL training, full simulations) runs on the HPC4 cluster. Code should support CPU MuJoCo twin for local testing and GPU execution on the cluster.

Your stated scope: **motion planning that moves bricks**, and **reinforcement learning that inserts them**. The repo never names you, but that split is the project's placer-arm job. Stabilizer bracing, the joint model, and perception are other tracks. Do not rebuild them.

The proposal states the handoff in one sentence: motion planning does coarse pick and transport; learned control takes over near the target for contact-rich alignment and press-fit (`GROUP 10 Preliminary Proposal.docx`, Introduction, goal 2).

There are two generations. **v3.1 is frozen.** **v4 is the plan of record.** Read v4 before changing anything that still runs.

---

## What you own

| Job | Spec | Code that already exists | Gate |
|---|---|---|---|
| Coarse motion: grasp, collision-free transport, pre-insert handoff | v3.1 WP4; v4 WP2 / M0.5 | `Sakib/brickassembly/motion/`, `orchestration/` | G3 (failed, frozen); M0.5 (not closed) |
| Precision insertion RL, residual on the scripted skill | v3.1 WP5 | `Sakib/brickassembly/tasks/` | G4 (failed, frozen) |
| Tight-space placement primitives (draft only) | `Sakib/Docs/plan_v4_rl_place.md` | not built (`tasks/place_bandit.py` does not exist) | G-P0 (not adopted) |

## What you do not own

- **Brace RL** (`Sakib/Docs/plan_v4_rl_brace.md`, `tasks/brace_bandit.py`). That is the stabilizer arm, not insertion.
- **Joint / clutch model** (`sim/joint_model/`, v4 `JointBreaker`). You consume it. A one-sided wrench or a bad force budget will look like a policy failure.
- **Perception** (`Sakib/Docs/plan_v4_perception.md`). v3.1 skipped it (A3). v4 vision is a separate phase and is already past G-V1.
- **Sequence and brace assignment** (`planner.py`, `bracing.py`, `stability.py`). You execute the plan they emit.

---

## 1. Motion planning — moving bricks

### Contract

Coarse motion delivers the brick to a **pre-insertion pose**: above the target socket, **+25 mm in z**, with injected error taken from the measured perception error, or from the planner's own placement error if vision is unused (`Sakib/Docs/master_report.md` §2.5).

The stabilizer plans and settles **before** the placer transports, then is a static obstacle for the placer's plan. That is prioritized planning, not a shortcut. State it that way if you write it up.

### v3.1 (frozen, MuJoCo twin)

Spec: `Sakib/Docs/master_report.md` §WP4. Stack: cuRobo `MotionGen`, world model = table, plate, feeders, growing structure, other arm.

Method, in order:

1. Four antipodal grasps per brick. Score `clearance_margin_mm − 2·|offset_from_centroid_mm| + 5·aligned_with_place_orientation`. Fingertips 4.8 mm above the brick base. Grip width = footprint − 1 mm. Nominal grip 15 N, then a 1 s shake test; prune failures.
2. `curobo_planner.py`: MotionGen, incremental world updates. Warm up once. Target after warmup: plan < 50 ms (WP0).
3. `dual_arm.py`: stabilizer first. On failure: secondary brace pose, then serialize, then recovery.
4. Scripted insertion (also the WP5 residual base): descend at 5 mm/s; on contact, 2 mm Archimedean spiral at 3 mm/s holding 8 N down, ±1.5° wiggle about x/y at 2 Hz. Timeout 10 s.
5. `py_trees` behaviour tree. Do not hand-roll a state machine. Nominal path: brace → perceive → grasp → transport to pre-insert → insert → verify snap → release → unbrace. Up to 3 retries, then lift-and-regrasp.

Behaviour tree and skills live in:

- `Sakib/brickassembly/motion/skills.py` — `pick`, `transport`, `insert`, `release`, `brace`
- `Sakib/brickassembly/orchestration/twin_executor.py`
- `Sakib/brickassembly/orchestration/rl_inserter.py` — swaps the scripted inserter for the policy

v3.1 amendments that change the spec (`master_report.md` §7.7):

- **M5.** 15 N grip lets the brick slide under a 70 N press. Grip = clip(1.2 × press, 20, 70) N per finger. Close at ≤ 0.08 m/s. Grasp slides toward the mating-patch centroid.
- **M7.** Fast descent to 3 mm above stud tops, then 5 mm/s. Press ramped to 1.3·n·f_insert with a speed limit (above 8 mm/s the snap-through hits at ~40 mm/s). Stuck (no 0.03 mm in 0.4 s) → search at 4000 N/m laterally. "In the cavity" requires a level wrist (< 0.12 N·m). A joint break anywhere ends the assembly.
- Accessibility was **not** cuRobo in the twin. It was a finger-clearance rule + IK + a hand-collision check (`sim/mj/checks.py`).

**G3 result (do not re-run as if it were open):** scripted pipeline, ground truth, weakest-joint bracing, n = 20. S1 17/20 (target ≥ 90%), S2 5/20 (target ≥ 90%), S3 0/20 (target ≥ 60%). **G3 failed.** Failures are joint breaks during later placements, not failed transports. Log: `Sakib/brickassembly/results/g3/`. Report: `Sakib/Docs/results_report.md` §4.

The scripted search also regressed: from ≤ 2 mm lateral error it passed 9/10 at the first twin commit and **2/10** at the frozen code. Above ~1.1 mm the spiral times out (`v31_search_regression`). Do not "fix" it in the frozen twin; it would invalidate G3 and A6.

### v4 (plan of record — this is the open motion job)

Plan: `Sakib/Docs/plan_v4.md` §2.3, §2.4, P3, **WP2 → milestone M0.5**.

Current cell (`dual_arm_sim.py`) moves with **waypoint IK**, not cuRobo. Collision-free planned motion is M0.5 and is not the default path. r5 explicitly says the prototype is "waypoint IK + hand-off + dry-run clearance, not planned motion."

Your v4 deliverable:

- `motion/plan_curobo.py` — does not exist yet. Probe script does: `Sakib/brickassembly/scripts/12_curobo_probe.py`.
- `motion/fr3.yml` — FR3 spheres from `curobo.sphere_fit`. cuRobo ships only Panda `franka.yml`.
- `tests/test_motion.py` — 200 random queries per arm with the other arm static, plus an executed subset.

Planner: **cuRobo 0.8 MotionPlanner, one per arm.** Route order (U5):

1. In-process from a `.vendor` overlay. Do not install into the Isaac env.
2. Out of process: `motion/curobo_server.py` under `isdnenv`.
3. Newton-native only if both fail: IK with sphere avoidance + a short RRT-Connect.

Rules that will bite you:

- Free space: time-scaled trajectories, joint pos **and** vel feed-forward. Start at 50% of velocity/acceleration limits.
- Last 30 mm of grasp, insert, and brace are **straight Cartesian segments** through `newton.ik.IKSolver`, collision-checked per link at 2 mm steps. Not planned by MotionGen.
- World model in WP2 may be ground truth. Perception poses only when WP4 (v4 integration) switches it. Never mix them silently.
- Plans must not block the sim tick. `ThreadPoolExecutor(1)`; the tree leaf stays RUNNING until the future resolves.
- **CUDA-graph discipline.** Capture every cuRobo graph in a warm-up before the loop. Collision cache pre-sized (`MotionPlannerCfg(collision_cache=…)`). A graph capture during the loop invalidates Newton. Guard: 0 post-warm-up captures or the run fails. Fallbacks: `use_cuda_graph=False`, then synchronous plans (~0.1 s pause), then the socket route.
- Shared zone: cylinder around the build, radius = structure half-diagonal + 0.12 m. At most one arm moves inside it. Tray is B's alone.
- Padding = max(5 mm, executed-deviation p99 + 2 mm). P3 measured 12.08 mm at time scale 1.0; that number is cuRobo evidence, not the waypoint margin.

**P3 gate** (probe, before you build the planner): runs in-process with warp 1.13; median plan ≤ 100 ms warm; success ≥ 95% on feasible queries; at the chosen time scale, p99 deviation + 2 mm ≤ padding; 0 arm–arm events on 20 executed plans; soak ≥ 200 plans with 0 CUDA errors and 0 post-warm-up captures; 0 contact-segment false positives on the cube's inserts.

**M0.5 gate** (ground truth; cube, arch, hollow_box, S3 × 5 seeds = 235 placements, 20 braced; plus S3 `--brace-all` × 3):

- 0 arm–arm events (report braced and unbraced separately)
- ≤ 1 unintended event per 50 placements
- 0 displaced bricks, 0 penetration events, executed clearance > 0 on every free-space segment
- completion ≥ M0; mean cycle per brick ≤ 1.5× P0
- shown live

Experiment E3 compares prototype waypoint IK against this planned motion. That comparison is the motion result, not a side plot.

---

## 2. Reinforcement learning — inserting bricks

### Contract

The episode **starts at the pre-insertion pose** with the brick already grasped and the stabilizer already bracing. You do not learn pick or transport. Policy rate 20 Hz. Action is an EE-frame increment, integrated, then tracked by task-space impedance at 240 Hz (`master_report.md` §2.5).

Impedance (do not retune casually):

```
K_p trans : diag(1200, 1200, 600) N/m    # soft in z on purpose
K_p rot   : diag(40, 40, 40) N·m/rad
damping   : 1.0
```

Soft z is what lets the brick find the socket. Compliance in z is not a bug.

### v3.1 WP5 (frozen results — the insertion MDP you inherit)

Spec: `Sakib/Docs/master_report.md` §WP5, §7.1, §7.2. Code: `Sakib/brickassembly/tasks/insertion_env.py`, `tasks/ppo.py`.

Primary method is **residual RL on the WP4 scripted controller**, sparse reward only (Marougkas). The scripted base carries the shaping. The residual learns the correction.

Run order is normative:

| Run | What | Role |
|---|---|---|
| R1 | residual + sparse | **primary** |
| R2 | residual + dense | ablation A-dense |
| R3 | end-to-end + dense, privileged state | conventional baseline (A1, A4) |
| R4 | R1 with vision group zeroed | ablation A3, the vision decision |
| R5 | vision student | only if R4 loses |

Sparse reward:

```
+50  snap verified
-10  force violation, drop, or timeout
  0  otherwise
```

Dense terms exist only for R2/R3. `r_engage` is the one that matters. A pure distance reward plateaus at hovering above the socket.

Curriculum (advance if success over last 100 episodes > 80%; regress below 50%):

| Stage | Pos error | Rot error | Socket | Bricks |
|---|---|---|---|---|
| 0 | U(0, 2 mm) | U(0, 2°) | isolated | 2x4 |
| 1 | U(0, 5 mm) | U(0, 5°) | isolated | 2x4, 2x2 |
| 2 | U(0, 10 mm) | U(0, 8°) | 1 neighbour | all 2xN |
| 3 | U(0, 15 mm) | U(0, 12°) | 2–3 neighbours | all 8 |
| 4 | measured handoff error | — | ≤ 4 neighbours | all 8 |

**G4 target:** ≥ 85% success at stage 3, mean peak force < 30 N, success-head AUC > 0.85. **G4 failed.** Best stage-3 success was 24% (R4). No calibrated-joint run reached stage 3 in training. Source: `Sakib/Docs/results_report.md` §6, `results/wp5/`.

Held-out, 100 episodes, calibrated joint, stage 3: R1 19%, R2 8%, R3 19%, R4 24%. Stage 0 is much higher (R3 99%, R4 81%, R1 66%). The policy does not transfer to large initial error.

What the frozen code actually is (M8), not the Isaac Lab spec:

- Reduced cell: 1.5 kg floating hand (Panda operational-space inertia), brick rigidly grasped, same clutch. Not 2048 Isaac envs.
- Action is **7-D**: the 6-D pose increment **plus a press-force channel**. Without it the §2.5 impedance can push only ~6 N, and a 2x4 needs ~71 N (8 × 8.9 N/stud).
- Actor obs is **121-D**, not 183. The "vision" group is a 9-D pose estimate (σ 0.3 mm / 0.3°), not a CNN. R4 zeroes it.
- 300 k samples per run, not 120 M. PPO is `tasks/ppo.py`, not `rsl_rl`.
- Spec PPO (if you ever get a GPU stack): 2048 envs, rollout 32, minibatch 16384, 4 epochs, clip 0.2, lr 3e-4 cosine, entropy 0.002 → 0.0005, MLP [512, 256, 128] ELU, separate actor/critic.

Commands (from `Sakib/brickassembly/`):

```bash
bash scripts/run.sh -m tasks.ppo --run R1 --samples 300000
bash scripts/run.sh -m tasks.ppo --eval-all
bash scripts/run.sh -m experiments.runner g6 --policy R1 --seeds 1
```

Integration (WP7, also frozen): RL-first, scripted fallback on the third retry (`orchestration/rl_inserter.py`). Poses must be in a task frame turned by the seated hand's yaw. World-frame obs at the plan's 90° yaw put the policy out of distribution (success head read 0.002). Stop the press within 2 ms of seating or of crossing the force budget. A half-seated brick is finished by the scripted fallback, not by another policy attempt (M14).

**G6 result:** RL-first is worse than scripted-only. Full system completes S1 once in five seeds, against 17/20 for the scripted pipeline. The policy seats 54% of attempts; runs die on **joint breaks**, and the policy presses harder (mean peak 90 N vs 59 N). Do not treat "more press" as progress.

A10, the result worth keeping: a policy trained on a light hand-tuned joint (3 N insert / 5 N break per stud) seats 23% on the calibrated joint vs 66% for one trained on the calibrated joint, with force limits held equal. If the force budget is also taken from the light model, training collapses to a policy that never presses. **Do not invent force limits.** Use WP2's numbers: 8.9 N insertion / 11.3 N break per stud, 1.2 mm gate.

### v4 insertion is not the same job

v4's open RL plan for the placer is **not** contact-rich snap insertion. It is a draft, not adopted: `Sakib/Docs/plan_v4_rl_place.md` (ledger `v4_placer_rl_primitives`).

Why it exists: side-pinch cannot finish a filled layer. Fingertips sit ~4.1 mm above the brick bottom; a flush neighbour's studs stop the hand ~7.2 mm above seated. The user's pyramid died at step 2 with dz +7.84 mm (`v4_pipeline_pyramid_inaccessible_grasp`). That is gripper geometry, not a motion skill. Re-sequencing cannot remove every tight step.

The draft asks the policy to pick a primitive per brick:

- `insert` — current `queue_place`, unchanged
- `drop_press` — release above the neighbours' studs, then press down with closed fingertips

Algorithm, if P0 says learning is worth it: the brace track's horizon-1 contextual bandit, not PPO. One decision per brick. Reward = snap-gate success. Files to build: `tasks/place_bandit.py`, `tests/test_place_bandit.py`. Extend `dual_arm_sim.py` `queue_place` behind `--place-json`.

P0 is a 16-setting screen, not training. Outcomes are pre-registered: FIXED-RULE (hand the best setting to the pipeline, do not train), LEARN (only if per-context best beats the fixed rule by ≥ 0.10), or STOP-G (top-press cannot seat ≥ 50% of tight steps). "A fixed rule suffices" is an accepted outcome. Do not start P1 until G-P0 says LEARN, and do not treat this draft as adopted.

v4's contact-rich detent (n·f_insert, the v3.1 force regime) is **WP5b**, and it is still "decide after M1". Do not port the twin PPO onto the Newton weld clutch and call it the same task. The weld is axial/prying only, breaks at utilization ≥ 1, and the build ends at the first break.

---

## Status you should assume

| Item | State |
|---|---|
| v3.1 motion + scripted insert | implemented, frozen, **G3 fail** |
| v3.1 insertion PPO | implemented, frozen, **G4 fail** (stage 3 ≪ 85%) |
| v3.1 vision for insertion | skipped; A3 null. Do not rebuild WP6 for the twin |
| v4 cell motion | waypoint IK, working on cube / arch / hollow_box unbraced |
| v4 cuRobo planner (`plan_curobo.py`) | specified, **not in the tree** |
| v4 placer RL (`plan_v4_rl_place.md`) | **draft, not adopted** |
| v4 brace RL | separate track; not your insertion job |

---

## Read in this order

1. `Sakib/Docs/master_report.md` §2.5, §WP4, §WP5, §7.1, §7.7 M5–M8 and M14
2. `Sakib/Docs/results_report.md` §4 and §6
3. `Sakib/brickassembly/motion/skills.py` and `tasks/insertion_env.py` (the docstrings are the executed spec)
4. `Sakib/Docs/plan_v4.md` §2.3, §2.4, P3, WP2
5. `Sakib/Docs/plan_v4_rl_place.md` only after you know it is still a draft

Ledger of record: `Sakib/brickassembly/ledger.jsonl`. Later entries supersede earlier ones. Do not edit `WORKLOG.md` by hand.
