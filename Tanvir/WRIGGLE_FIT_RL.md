# Wriggle-Fit RL: the fitting arm's tight-socket skill in the dual-arm LEGO cell

Author: Tanvir · 2026-10-21 · Status (2026-10-23): twin, scripted baseline, PPO runs (residual and skill-level), pad-shift fix and ES search built and run. No learned policy beat the scripted wriggle; the pad shift (§6.10) is the one measured gain. Held-out repo, no brace, aim 1–2 mm: 26/68 (target ≥ 85 %). See §6.9–§6.11 and `report/Wriggle_Fit_Design_Proposal.md`

**Goal.** In the dual-arm cell, arm A braces the structure and arm B fits bricks. This design is for **arm B's fitting skill at crucial layouts**: *wriggling* a brick into a tight socket, such as a gap in the middle of a layer with bricks around it. It should succeed from a coarse aim (1–4 mm off) using only cheap, built-in sensing: no wrist force/torque (F/T) sensor and no fine vision. The aim is to get crucial, load-bearing layers seated without the neighbours being levered off, while arm A braces whatever is at risk.

**Scope.** Everything here lives in `Tanvir/`. The rest of the repo is used read-only: Sakib's planner, plans, generators and the MuJoCo insertion twin. The brace policy itself (where and when arm A braces) is Sakib's track (`Sakib/Docs/plan_v4_rl_brace.md`, `tasks/brace_bandit.py`) and is **not** rebuilt here. This design defines what the fitting arm needs from the brace and what it gives back.

---

## 0. Where this sits in the dual-arm system

```
plan step ──► is the socket tight? (tight_fit_audit features, §1)
              │ no  ──► normal insert (existing scripted / RL inserter)
              │ yes ──► 1. arm A braces: the planner's weakest-joint brace, if any,
              │            or a "wriggle brace" on the most fragile neighbour (§3.8)
              │         2. arm A settles, then is a static obstacle for arm B
              │         3. arm B runs the Wriggle-Fit policy (§3)
              │         4. arm B seats, releases and retreats; arm A unbraces
```

**Why the two arms need each other at tight steps** (measured, §2.1). A brace and a wriggle fix *different* failures:

| neighbours | 1 mm aim error | 2 mm | 3 mm |
|---|---|---|---|
| not braced, scripted wiggle | 4/6 (2 breaks) | 3/6 (3 breaks) | 2/6 (3 breaks) |
| braced (stand-in), scripted wiggle | **6/6** | 3/6 (3 timeouts) | 2/6 (4 timeouts) |

- **Bracing removes the breaks.** Every break was the same brick: the 2-stud side neighbour. So a single brace on the most fragile neighbour, which one arm can do, is enough.
- **Bracing does not find the hole.** With the neighbours safe, the scripted wiggle times out from 2 mm upward. That is what the learned fitting policy is for.

**The planner does not brace at tight steps today.** The planner's brace trigger (`bracing.py`, weakest-joint LP) models only the vertical insertion press levering an overhang. Of the 8 executed tight steps, only the arch crown (step 10) also gets a brace. S2 step 6, S4 steps 3/7/10/13, S5 step 4 and hollow_box step 7 get none, nor do any of the pyramid's finger-blocked steps. The sideways loads of a wriggle are not in the model. §3.8 adds that trigger.

---

## 1. Answer first: can the existing assemblies train this?

**They are a good *test set* and a poor *training set*. They contain real tight steps, but almost all are of one kind, and the "middle opening" case (a hole surrounded on all sides) does not occur in any of them.**

`tight_fit_audit.py` walks every step of every assembly in the repo, in the order the simulator actually ran it. For each step it classifies the socket by which faces of the brick touch an already-placed brick in the same layer:

| class | faces touching a neighbour | why it matters |
|---|---|---|
| FREE | 0 | nothing to fit against |
| FLUSH | 1 | one wall: usable as a guide |
| CORNER | 2 adjacent | an L-shaped guide; still open on two sides |
| **SLOT** | 2 opposite | the brick drops *between* two bricks |
| **POCKET** | 3 | |
| **ENCLOSED** | 4 | the "middle opening" case |

The grid is exact. The stud pitch is 8.0 mm and a brick's footprint is 8n − 0.2 mm, so in a SLOT, POCKET or ENCLOSED socket the brick has **0.1 mm of play per side**. The cell's aim error is about 0.8 mm (1σ; `insertion_env.BELIEF_SIGMA`). Those three classes are therefore *precision-critical*: the brick cannot drop in on aim alone.

### Results (full tables in `results/tight_fit_audit.md`; per-step data in `results/tight_fit_audit.json`)

| source | bricks | precision-critical steps (as built) | if the brick were placed last in its layer |
|---|---|---|---|
| S1 column, S3 bridge, S3C, S3L | 33 | 0 | 0 |
| S2 staggered wall | 12 | 1 SLOT (step 6, load-bearing) | 4 SLOT |
| S4 staircase | 19 | 4 SLOT (steps 3, 7, 10, 13, all load-bearing) | 6 SLOT |
| S5 tower + balcony | 26 | 1 SLOT (step 4, **21 bricks rest on it**) | 7 SLOT |
| S2A | 7 | 0 | 1 SLOT |
| cube | 8 | 0 | 0 |
| hollow_box | 12 | 1 SLOT (step 7) | 6 SLOT |
| arch | 11 | 1 SLOT (step 10, the crown) | 1 SLOT |
| pyramid (user drawing) | 11 | 0 SLOT, but **5 CORNER/FLUSH steps the fingers cannot reach** (below) | **2 POCKET** |
| corbel family, 30 generated structures (brace track) | 441 | 0 | 2 SLOT |

What this means:

1. **Crucial steps exist and they matter.** S5 step 4 is a 2x4 dropped between two bricks with 21 bricks built on it, and every S4 SLOT is load-bearing. Failing one of these, or breaking a neighbour while wriggling, ends the build. The results report says the same: G3 failures were "nearly every[one] a joint break during a later placement".
2. **They are almost all the same shape.** Every SLOT runs along y: the brick goes end-to-end between two in-line bricks, and the fingers can still pinch it on the free x faces (`free pinch axes = x`, `pinch_blocked = 0`). That is a 1-D precision problem.
3. **Nothing is POCKET or ENCLOSED as built.** Most benchmark structures are walls 2 studs deep, which cannot enclose a brick. The only POCKETs come from re-ordering the pyramid. Placing a brick last within its own layer is always buildable, because support comes only from the layer below, so this re-ordering is legitimate.
4. **The pyramid shows a second, separate difficulty: finger access.** Steps 2, 3, 4, 6, 7, 8, 9 have a placed neighbour under a fingertip (`planner.pinch_blocked` = 2, 4, 4, 1, 1, 1, 1). Steps 2, 3, 4, 7 and 9 have no free pinch axis at all. That is exactly where the v4 cell stopped with the brick 7.8 mm high (`Sakib/Docs/plan_v4_rl_place.md`, facts 1–2). The precision is fine there, but the gripper cannot go down.

**Conclusion.** The repo gives about 30 realistic tight contexts (8 executed SLOTs; 19 more by re-ordering, of which 17 are SLOTs and 2 POCKETs; plus 7 finger-blocked steps). That is enough to **evaluate** on, so hold them out. It is far too few and too narrow to **train** on, and has no ENCLOSED case. Training needs a procedural socket generator (§4.2).

---

## 2. What the simulators can and cannot teach

### 2.1 The v3.1 MuJoCo twin models the slot (checked)

`slot_probe.py` runs Sakib's `InsertionEnv` with its scripted controller only: align, descend, then on contact a 2 mm spiral search with a ±1.5° wiggle. The aim is deliberately off by a fixed amount. The test uses a 2x4 brick, 6 seeds per cell, CPU, about 1–2 s per episode (`results/slot_probe.md`):

| socket | aim error 0 mm | 1 mm | 2 mm | 3 mm | 4 mm |
|---|---|---|---|---|---|
| open (no neighbours) | 6/6 | 6/6 | 6/6 | 1/6 | 0/6 |
| pocket (3 flush neighbours) | 5/6 | 4/6 (2 joint breaks) | 3/6 (3 breaks) | 2/6 (3 breaks) | 0/6 |

- The neighbours genuinely constrain the brick. Brick-to-neighbour contacts rise from about 20 to about 1,000 per episode as the aim error grows. The contacts are between the target's walls and the neighbours' top plates and studs. Wall-to-wall pairs are filtered out by the collision bits in `sim/mj/bricks.py`, but the top plate still blocks.
- **The failure in a tight socket is not "cannot find the hole". It is breaking the neighbours' stud joints while searching.** An open socket at 1–2 mm is always solved, and the pocket loses half its episodes to `joint_break`. The scripted spiral levers against the neighbours. This is the behaviour the RL policy has to learn to avoid.
- **The neighbour that breaks is always the 1x2 side neighbour (`nb2`, 2 studs).** With `--brace all`, the neighbours' stud joints are made 10× stronger as a stand-in for arm A holding them. The breaks disappear (1 mm: 4/6 → 6/6), but at 2 mm and above the same episodes now time out (3/6 and 2/6 still seated). The brace protects the structure; it does not do the fitting (§0).
- With n = 6 these numbers are indicative only. Phase 0 (§6) re-measures them at n ≥ 100.

### 2.2 Earlier RL in the same twin (Sakib, `Docs/results_report.md` §6)

- The best policies seat **66–99 % at stage 0** (no neighbours) and only **19–24 % at stage 3**, which is a 3-neighbour POCKET on the baseplate. Stage-3 failures were 17–34 joint breaks and 41–59 timeouts per 100 episodes.
- No run on the calibrated joint *reached* stage 3 in training. The curriculum jumped from 1 neighbour with 10 mm error to 3 neighbours with 15 mm error, and each run was only 300 k steps. So stage 3 was never really trained; it was tested out of distribution.
- An end-to-end policy with a press-force channel beat the residual policy (99 % vs 38 % at stage 0). Dense reward shaping *hurt* (66 % → 38 %). A fine pose estimate (vision) did not help (A3).
- The policy's main input was a 6-axis **wrist F/T history**, which is the costly sensor this design removes.

### 2.3 The v4 Newton cell cannot teach precision as it stands

`dual_arm_sim.py` welds a brick in place once it is within **2 mm laterally, 1.5 mm vertically and 5°** (`GATE`, line 159), and its colliders are shrunk inward by `COLLIDER_INSET`. A 0.1 mm-per-side slot is therefore not a tight fit there. **Train in the MuJoCo twin.** Port to Newton only after its snap gate is tightened to about 0.3 mm, or replaced by the clutch model, and the inset is accounted for (§7).

---

## 3. The algorithm

### 3.1 Formulation

This is a partially observable episodic task with one decision every 50 ms (20 Hz). An episode starts with the brick held 25 mm above its believed seat, matching the existing pre-insertion handoff. It ends on:

- seated (the clutch reaches MATED),
- any joint of the structure breaking (in the twin, placed bricks stay welded until their joint breaks, so the planned "neighbour moved > 0.3 mm" rule reduces to this),
- the force budget being exceeded,
- arm B's hand or brick touching arm A's keep-out zone (§3.8),
- a released brick ending more than 3 mm from its seat, or
- a timeout of 10 s.

Implemented in `socket_env.py` (`SocketInsertionEnv`).

### 3.2 Sensing: cheap only

| signal | source on a real arm | how the twin models it |
|---|---|---|
| joint positions and velocities, end-effector pose from forward kinematics | joint encoders (free) | hand pose and velocity |
| **estimated contact wrench** | motor torque or current: e.g. Franka's built-in external-torque estimate `tau_ext` | the true contact wrench on hand and brick, plus: an inertia-model error of 30 % of the hand mass times its acceleration; white noise of 3 N / 0.15 N·m per 2 ms control tick, *before* a 30 ms low-pass (about 0.55 N after it); and an offset tared to ±0.5 N at the start that drifts as a random walk (0.02 N per step, capped at ±2 N). In free space this reads 0.4–1.2 N (`tests/test_wriggle.py`). It replaces `ft_hist`. |
| impedance tracking error `x_d − x` | the controller's own state | always present, so even the "no wrench" ablation still feels contact through it |
| **depth progress** `z − z_seat,believed` | forward kinematics | the hand z against the believed seat. This is the strongest cheap cue: the brick *drops* when it lines up with the slot. |
| socket description from the plan | the plan (no sensor at all) | 32 numbers: faces touching a neighbour plus contact fractions, class, slot axes, free pinch axes, brick type, layer, mating studs, size, force budget, pinch axis used (`socket_env.context_features`) |
| coarse target pose | plan plus a calibrated cell | aim-error level L draws the lateral error from U(previous level, L) with L ∈ {0.5, 1, 2, 3, 4} mm, and the yaw error from ±(0.5 + 0.375·L)° |
| **brace state** | the plan plus arm A's own joint torques | active flag, which face the braced neighbour touches, brace strength, and the braced brick's joint force plus 1 N noise as arm A's felt reaction (9 numbers) |

The actor sees the last 16 steps of a 36-number per-step vector (pose and rotation relative to the believed target, velocities, tracking error, wrench estimate, depth, previous action, released and tips-closed flags), plus the context and brace numbers: 617 in all.

These are **not** given to the policy: wrist F/T, fine vision, the true offset to the socket, and contact states. An ideal F/T sensor is kept only as an **upper-bound ablation**.

### 3.3 Actions (20 Hz, in the end-effector frame, each squashed to [−1, 1])

| channel | range | purpose |
|---|---|---|
| Δx, Δy, Δz | ±2 mm per step (`InsertionEnv.DPOS`, shared with the scripted teacher), through the existing action integrator with a 10 mm leash | move |
| Δroll, Δpitch | ±2° per step (`DROT`); total tilt capped at 6° | **tilt-in**: lead with one edge |
| Δyaw | ±2° per step | square the brick up to the slot. This matters: a 2x4 jams in its slot beyond about 0.36° of yaw (§6.2). |
| press force | 0 … budget (an action of 0 already means half the budget) | seat it (the extra channel that made the end-to-end policy work in A4) |
| lateral stiffness scale | 0.15 … 1 × K_xy | **go soft so the walls centre the brick**: variable impedance |
| release | > 0.5 opens the grasp (once) | insert, release, press for finger-blocked sockets (§3.7) |

These channels are what make a "wriggle" possible. The classic way to fit a tight peg (Whitney's jamming and wedging analysis) is: tilt, catch one edge on the far wall to use it as a reference, slide until the edge drops, then rotate level while compliant and press. A soft lateral stiffness lets the 0.1 mm walls do the centring instead of the sensor. The policy learns when and how much to do each of these; none of it is hard-coded.

### 3.4 Policy and critic

- **Actor:** the last 16 steps of cheap observations, stacked, into a 2 × 256 MLP (Sakib's `tasks/ppo.ActorCritic`, reused). It outputs a squashed Gaussian over the actions. A GRU over the same window is the planned upgrade if stacking proves too short-sighted. Memory is required because a single noisy force reading is ambiguous. *Sequences* tell the story: force rising while depth is not, followed by a drop, means "edge caught, now slipping in".
- **Critic (asymmetric, privileged):** the actor's inputs plus the true offset to the socket, the brick-to-neighbour contact count, each neighbour's **joint utilisation** (`ClutchModel.max_util()`), the clutch state and the randomised physics parameters. The critic's extra inputs are limited to ones that can be inferred from force and depth history. This follows Sakib's D12 and the informed asymmetric actor-critic idea, to avoid a teacher the student cannot imitate.
- **Algorithm:** PPO (rsl_rl, or the twin's own `tasks/ppo.py`).
  - **Warm start by behaviour cloning** of the scripted teacher (`ScriptedWriggle`: Sakib's scripted base plus insert, release, press), on the episodes it seats, so training starts near the scripted performance. This was the A4 lesson (end-to-end with a press channel, not residual), **but here the cloned policy failed closed-loop, so the main run is residual (§6.3).** A tilt-in teacher variant is not built yet.
  - The residual policy is kept as an ablation.

### 3.5 Reward

```
r = +1                       seated (episode ends)
    −1                       a neighbour joint breaks, or a neighbour moves > 0.3 mm (episode ends)
    −1                       force over budget (episode ends)
    −1                       arm B (hand, fingers or held brick) touches arm A (episode ends)
    −0.05·max(0, u_nb − 0.6) per step: u_nb = highest neighbour-joint utilisation (privileged, reward only)
    −0.002                   per step (finish quickly)
```

The neighbour-utilisation term is the core of the design. The probe in §2.1 shows the failure is *breaking neighbours*, and this term charges for loading them **before** they break. It is a gentle penalty, not dense shaping toward the goal (dense shaping hurt in Sakib's study).

An optional shaping term on depth, `γΦ(s') − Φ(s)`, where Φ is the true depth reached, does not change which policy is optimal. Keep it as an ablation only.

### 3.6 Curriculum: one difficulty at a time, chosen automatically

Each training context is a tuple:

```
(socket class, aim error, yaw error, sensor noise, layer and support stiffness, brick type)
```

The axes span:

- **socket class:** FREE → FLUSH → CORNER → SLOT → POCKET → ENCLOSED
- **aim error:** 0.5 → 4 mm
- **yaw error:** 0.5 → 4°
- **sensor noise:** 0 → full
- **layer:** 0 on the baseplate → layers 1–3 sitting on bricks with few supporting studs

Contexts are sampled by **success-band prioritised replay**: keep a running success rate per context bucket and sample mostly from buckets between 20 % and 80 % success, plus 10 % at uniform random. This replaces Sakib's fixed 4-stage ladder, which never reached stage 3.

**Budget:** 5–20 M environment steps, not 300 k. The CPU twin measured here runs about 100–150 policy steps/s per core, so 20 M steps is about 40–55 core-hours: run with 32–64 workers on the HPC4 cluster (Kean's setup). The local GTX 1660 Ti is fine for the network update and for development.

### 3.7 Phase 2: finger-blocked and ENCLOSED sockets (insert, release, press)

When a neighbour sits under a fingertip, a side pinch cannot seat the brick. The fingertips hit the neighbours' studs when the brick is still 7.2 mm above its seat (plan_v4_rl_place fact 2). By then the brick bottom is already 2.4 mm *inside* the slot, below the neighbours' top faces. That is enough for the slot walls to hold it in place laterally.

So the skill becomes:

1. Wriggle in while holding the brick, until depth progress is at least 2 mm.
2. **Release.**
3. Close the fingers.
4. Press-and-rock on the brick's stud tops.

The cell already presses stud tops with closed fingertips (the legacy brace). This is the drop-and-press primitive of `plan_v4_rl_place.md`, but with the release moment and the rocking *learned* and closed-loop, instead of chosen from a fixed grid.

**As implemented** (`socket_env.py`). `InsertionEnv`'s hand had no fingers and the brick was a rigid child of it, so neither finger blocking nor release was possible. The subclass changes three things:

- **Finger pads.** Two 17.5 mm pads sit 0.3 mm outside the pinch faces, their tips 4.1 mm above the brick bottom (the v4 cell's grasp). The pinch axis is a free one if any, else the less blocked one, and never one with a finger over the braced neighbour.
- **Releasable grasp.** The brick hangs from the hand on a weld constraint that `release` switches off. The weld's anchor and relative pose must be set explicitly: MuJoCo's default data gave a grasp that collapsed by about 170 mm under a 100 N press.
- **Closing fingertips.** After release, a fingertip pad switches on once the hand has risen clear of the brick's studs. It is compiled as collidable and switched off at build time, because MuJoCo leaves non-collidable geoms out of a body's collision tree for good.

The scripted teacher (`ScriptedWriggle`) releases when the fingers stall with the brick 4–9 mm above the believed seat, then rises, lowers the closed tips onto the studs and presses with a ±1° rock. In smoke runs it seats ENCLOSED and finger-blocked pyramid sockets that a pinch alone cannot.

### 3.8 Working with the bracing arm

**When to brace at a tight step (the "wriggle brace" trigger).** For a step whose class is SLOT, POCKET or ENCLOSED, or that is finger-blocked:

1. For each same-layer neighbour touching the socket, take its support: the number of studs holding it down and its joint capacity (`planner.supports`, `sim/joint_model/capacity`).
2. Estimate its utilisation under a sideways wriggle load: a lateral push F_LAT at its top edge, one brick height above its stud joint, split over its supports by stud count, with the twin's capacity model.
3. If the worst neighbour reaches U_TRIGGER = 0.6, request a brace on it. Arm A grips that neighbour, or presses its stud tops, and holds it.

**As implemented** (`wriggle_brace.py`). F_LAT = 8 N is calibrated on the slot probe. It flags the 1x2 side neighbour (u = 0.85), the brick that broke in every unbraced pocket episode, and neither 2x2 (u = 0.21). On the repo contexts it braces 1x1/1x2 neighbours in the pyramid (u up to 1.7) and none of the SLOTs, whose neighbours sit on 4–8 studs (u ≤ 0.28).

This is computed in `Tanvir/` from the plan and handed over as a plan annotation. It does not edit Sakib's `bracing.py`. If the planner already braces this step (arch step 10), the planner's brace wins and the fitting policy sees it through the brace observation.

**Ordering.** Arm A braces and settles first, then is a static obstacle for arm B: the prioritised planning the project already uses. Arm B's policy never commands arm A.

**Training with a brace present.** Each training episode draws a brace condition:

- none (40 %),
- an ideal brace on the neighbour the trigger picks (40 %), simulated as that brick's stud joints being 10× stronger, plus a keep-out box where arm A's fingers sit (over the braced brick, 1 mm inside its edges, 40 mm tall), or
- a weak brace (20 %): joints 2.5× stronger, same keep-out. A late-engaging brace is not built yet.

If the trigger flags no neighbour, the episode runs unbraced.

The policy therefore learns to fit both with and without help, and learns not to hit arm A. The brace observation tells it which case it is in.

**What the fitting arm gives back to the brace track.** Logged per episode: the neighbour utilisation trace, which neighbour came closest to breaking, and arm A's reaction wrench. These are training labels for Sakib's brace selector at tight steps, a step type his brace data does not yet cover.

---

## 4. Training and test data

### 4.1 Test set: held out, taken from the repo's real assemblies

From `results/tight_fit_audit.json`:

- **The 8 executed SLOT steps:** S2 step 6; S4 steps 3, 7, 10, 13; S5 step 4; hollow_box step 7; arch step 10.
- **17 re-ordered SLOTs** (`class_if_last_in_layer`) and the 2 pyramid POCKETs.
- **The 7 finger-blocked pyramid steps** (Phase 2).

Each context is rebuilt in the twin with its real neighbours and real supports. The neighbours are clutched, not welded rigidly, so wriggling can break them, exactly as it would in the full build. Each is evaluated three ways: no brace, the planner's own brace, and the wriggle brace of §3.8. These contexts are never trained on.

### 4.2 Training set: a procedural socket family (`socket_family.py`)

1. Draw the target brick (any of the 8 types, weighted toward 2-wide) and its layer: 0, 1 or 2 with probability 0.5 / 0.35 / 0.15.
2. Place one neighbour per required face, flush, with a random type (1x2, 1x4, 2x2, 2x3, 2x4) and a random overlap along the face. Reject draws whose class comes out different from the one requested.
3. For layers above 0, tile support layers under everything with Sakib's `blueprint.tile`. In half of the draws, one neighbour keeps only 1–2 supported studs: a fragile, crucial-layer neighbour.
4. Check every brick is supported in sequence order.

Train, validation and test splits use disjoint seed ranges (`SPLITS`), and contexts are deterministic in (seed, class). Generation takes milliseconds, so the pool can be as large as wanted. This guarantees the ENCLOSED case that the repo lacks. `repo_contexts()` rebuilds the 34 held-out repo contexts the same way.

### 4.3 Warm-up

The existing `InsertionEnv` stages 0–2, unchanged.

---

## 5. Experiments and success criteria

**Main result:** on the held-out repo contexts at 2 mm aim error with *cheap sensing only*:

- **at least 85 % seated with zero neighbour breaks**, against the scripted base (about 50 % in the pocket today, §2.1), and
- the same contexts' **open** version as the ceiling.

| ablation | question |
|---|---|
| cheap wrench vs ideal wrist F/T vs no force at all (pose and depth only) | how much does the costly sensor buy? (The thesis: little.) |
| with vs without the stiffness channel | does compliance do the centring? |
| with vs without tilt channels | is tilt-in what solves SLOT and POCKET? |
| with vs without the neighbour-utilisation penalty | does it remove the joint breaks? |
| success-band curriculum vs Sakib's fixed stages | does the policy reach POCKET and ENCLOSED in training? |
| BC-warm-started end-to-end vs residual | re-checks A4 in the tight setting |
| **{no brace, planner brace, wriggle brace} × {scripted wiggle, learned wiggle}** | the dual-arm result: how much of tight-step success comes from bracing, how much from the learned fit, and whether they add up. The §0 probe predicts bracing fixes the breaks and the policy fixes the timeouts. |
| trained with vs without brace conditions | does a policy trained only unbraced misuse a brace, or collide with arm A? |

**Report:** mean ± 95 % CI over 5 training seeds × 100 evaluation episodes per context, with peak force, neighbour peak utilisation and time to seat.

---

## 6. Build plan (all files in `Tanvir/`)

| phase | deliverable | status |
|---|---|---|
| 0 | `tight_fit_audit.py`: socket classes for every repo assembly | **done** |
| 0 | `slot_probe.py`: checks the twin models the slot; scripted baseline, unbraced and braced (`--brace none all`) | **done** (n = 6; re-run at n = 100) |
| 1 | `socket_env.py`: `SocketInsertionEnv(InsertionEnv)`. Its own reset (any socket_family or repo context, neighbours clutched, finger pads, releasable weld grasp, arm-A brace stand-in and keep-out) and step (9-D action, cheap sensing, §3.5 reward and terminations). It inherits `InsertionEnv`'s helpers and scripted base. Subclassing means no edits to Sakib's code. | **done** |
| 1 | `wriggle_brace.py`: the §3.8 trigger | **done** (calibrated on the probe) |
| 1 | `socket_family.py`: the procedural family and the 34 repo contexts | **done** |
| 1 | `tests/test_wriggle.py`: family validity and splits, repo contexts vs the audit, brace choice, env reset/step/release, cheap sensing in free space, brace stand-in, curriculum unlocking | **done**, 13/13 pass |
| 2 | `train_wriggle.py`: BC warm start, PPO (Sakib's `ActorCritic`), success-band curriculum, worker processes | **done** (see §6.3 for the run) |
| 2 | `eval_wriggle.py`: repo, family-test and probe contexts × aim-error levels × brace conditions; per-episode rows and tables | **done** |
| 3 | Phase 2 insert, release, press | **built into phase 1** (`release` action, `ScriptedWriggle`); learned release is part of the main run |
| 4 | Port to Newton (tightened gate) and run inside full dual-arm builds of S4, S5 and the pyramid, with arm A's real brace (`dual_arm_sim.py --strategy ...`) | optional |

**Gate after phase 1:**

- The scripted base in `SocketInsertionEnv` reproduces §2.1 within ±10 %.
- An ENCLOSED context is constructible and solvable from 0 mm aim error.
- With the cheap wrench in place of F/T, the scripted base loses no more than 10 points.

### 6.1 Bugs found while building (all fixed; the tests cover them)

- **Grasp weld.** MuJoCo's default weld data carries an anchor of (0, 1, 0). That grasp collapsed by about 170 mm under a 100 N press, so presses never reached the brick. The anchor and relative pose are now set explicitly (brick 12 mm below the TCP, 0.04 mm compliance at 100 N).
- **Hand mass.** Adding finger geoms let the compiler recompute the hand's mass from them, about 1 g instead of 1.5 kg, and the first step diverged. The hand's inertia is now explicit and the pads have zero density.
- **Fingertips.** A geom compiled as non-collidable is left out of its body's collision tree for good, so the closed tips never touched anything and the hand fell through the world. They are now compiled collidable and switched off after the build.
- **Force budget for small bricks.** Scaled by studs alone, a 1x2 got 28–36 N, below the hand's touchdown transient (Sakib's env only used 2-wide bricks). The budget now has a 50 N floor.
- **Keep-out over a stacked brick.** Arm A's keep-out box was placed on the braced brick even with bricks stacked on it, so it overlapped placed bricks and broke them at once. It now starts above the column.
- **Cheap-sensor noise.** Noise added after the filter crossed the scripted 3 N contact threshold in mid-air. Noise now goes in before the 30 ms filter, and the estimate is tared at the start.

### 6.2 Scripted baseline (`results/eval/scripted_cheap.md`)

`ScriptedWriggle` with cheap sensing and fingers: 34 repo contexts plus 30 family-test sockets (5 per class), aim-error levels 0 / 2 / 4 (≤ 0.5, 1–2, 3–4 mm), 2 seeds each.

| contexts | brace | ≤ 0.5 mm | 1–2 mm | 3–4 mm | main failures |
|---|---|---|---|---|---|
| repo (held out) | none | 33/68 | 22/68 | 9/68 | joint break 87, timeout 36, force 15 |
| repo (held out) | wriggle brace | 31/68 | 22/68 | 9/68 | joint break 70, timeout 36, **arm contact 20** |
| family test | none | 30/60 | 21/60 | 3/60 | timeout 57, joint break 49, force 14 |
| family test | wriggle brace | 28/60 | 17/60 | 3/60 | **arm contact 40**, timeout 50, joint break 21 |

What it shows:

1. **The scripted wriggle tops out near 50 % even at ≤ 0.5 mm and collapses by 3–4 mm.** That is the gap the policy must close.
2. **Two physical failure modes dominate:**
   - *Yaw jam.* A 2x4 jams in its slot beyond about 0.36° of yaw, and the scripted rock only tilts. One traced probe episode wedged 5.9 mm high, then slammed through at 166 N.
   - *Joint breaks.* These hit neighbours and also deeper supports. For example, a brick placed last on S5's layer 7 breaks the layer-4 corbel, which Sakib's planner braces anyway.
3. **The wriggle brace halves the joint breaks** (family 49 → 21, repo 87 → 70). But the scripted controller cannot steer around arm A, so it trades them for arm contacts and seats no more. Learning to use the brace without touching arm A is exactly the policy's job (§3.8).
4. **The pyramid's finger-blocked steps are the hardest:** 5/54 seated.
5. **The probe pocket in this env is finger-blocked.** The 1x2 on +x and the 2x2s on ±y leave no free pinch axis, which Sakib's fingerless env hid.

The "planner" brace column in that file predates the keep-out fix and is invalid. A re-run with the fixed keep-out seats the arch crown 2/2.

### 6.3 Learning: what happened

- **End-to-end with behaviour cloning failed closed-loop.** BC on 123 seated scripted episodes (10,179 samples) fit to a loss of 0.005. Yet the cloned policy seated **0/72** family-test episodes, mostly timeouts: small per-step errors in the integrated position target accumulate, and the teacher's internal phase is not fully visible in the observation. PPO from that start (317k samples) never left the first curriculum bucket. That run is kept as `results/runs/main_attempt1/`.
- **Deviation: the main run is residual** (`--mode residual`): PPO corrections added to `ScriptedWriggle`, zero-initialised, exploration std 0.15 (Sakib's R1–R4 setting). It starts exactly at the scripted baseline. This reverses the A4-based preference in §3.4 for this task; end-to-end stays an ablation, which needs DAgger-style data collection to make its warm start work.
- Results of the residual run: §6.4.

### 6.4 Residual run `main`: it collapsed (negative result)

`train_wriggle.py --name main --mode residual --samples 1500000 --workers 7`: 1.5 M samples, 2 h 51 min on 7 CPU workers (`results/runs/main/`).

**Training curve** (success over the last 200 training episodes, `train.jsonl`):

| samples | 0 | 55 k | 110 k | 215 k | 430 k | 860 k | 1.5 M |
|---|---|---|---|---|---|---|---|
| success | 0.85 | 0.34 | 0.38 | 0.23 | 0.13 | 0.04 | 0.05 |
| buckets unlocked | 1 | 10 | 10 | 10 | 10 | 10 | 10 |

- It started at the scripted level (85 % on FREE at ≤ 0.5 mm), as a zero-initialised residual should.
- The curriculum opened 10 of 30 buckets in the first 55 k samples: FREE and FLUSH at most levels, plus CORNER at level 1. It never reached SLOT, POCKET or ENCLOSED.
- After that, success fell steadily, even on FREE at ≤ 0.5 mm, which ended at 8 % (`buckets.json`). Exploration std drifted up from 0.14 to 0.16 late in the run, and policy entropy rose with it.

**Held-out evaluation** (`results/eval/main_repo_none.md`): 34 repo contexts, no brace, levels 0 / 2 / 4, 2 seeds each.

| controller | ≤ 0.5 mm | 1–2 mm | 3–4 mm | joint breaks | force over budget | timeouts | lost brick |
|---|---|---|---|---|---|---|---|
| scripted (§6.2) | 33/68 | 22/68 | 9/68 | 87 | 15 | 36 | – |
| residual `main` | 2/68 | 0/68 | 0/68 | 93 | 36 | 60 | 13 |

The trained residual **destroys the scripted base's skill**: 2/204 against 64/204. The family, probe and brace evaluations were not run, because a policy this far below its own base would not answer anything they ask.

**Likely causes.** These are hypotheses, not yet tested:

1. **Unbounded residual.** Nothing ties the policy to the base. With a sparse ±1 reward and few successes after the curriculum widened, the policy gradient is mostly noise, and the mean residual drifts until it breaks the scripted motion.
2. **Residual on the release channel.** A shifted release threshold lets go of the brick early (13 lost bricks, more timeouts). The scripted release logic was already tuned.
3. **Curriculum widened too fast.** Ten buckets opened from FREE/0's early success alone, so the success rate fell to about 35 % within 55 k samples, before the value function had learned anything.
4. **Final, not best, checkpoint.** `policy.pt` is the last update. No intermediate checkpoint was kept, so the early, near-scripted policy is lost.

**Fixes** (implemented; §6.5 for the test run):

- penalise residual size (an L2 term on the mean residual, or a KL term to the zero residual), and cap the residual scale at about 0.3 of the action range;
- leave `release` to the scripted base (no residual on that channel);
- train the critic for about 20 k samples with the actor frozen before any actor update;
- open a new bucket only when its neighbour holds ≥ 50 % *and* the overall rate stays above 40 %;
- save a checkpoint every 100 k samples, and select the best one on the family validation split.

At about 135 steps/s locally, the §3.6 budget of 5–20 M samples needs the cluster.

### 6.5 Residual safeguards, as implemented

`train_wriggle.py --safe` switches all of them on. Each also has its own flag.

| safeguard | flag (`--safe` value) | where |
|---|---|---|
| residual capped at a fraction of the action range | `--res-scale 0.3` | `SocketInsertionEnv(res_scale=...)` |
| release left to the scripted base | `--no-res-release` | `SocketInsertionEnv(res_release=False)` |
| L2 penalty on the actor's mean residual | `--res-l2 0.1` (0.01 in `safe_short`) | PPO loss |
| critic trained alone first | `--critic-warmup 20000` (samples) | PPO loss: value term only until then |
| curriculum gate: nothing unlocks while the last 200 episodes seat < 40 % | `--unlock-gate 0.4` | `Buckets.update(overall, gate)` |
| checkpoints, the zero-residual start included | `--ckpt-every 100000` | `ckpt_<samples>.pt` |
| best checkpoint on the family validation split | `--select` | `select_best()` → `best.pt`, `select.json` |

The env defaults (`res_scale=1`, release on) keep the old behaviour, so the `main` checkpoint still evaluates as it was trained. Because `ckpt_0000000.pt` is the scripted base itself, selection can never return a policy that is worse than the scripted base on validation. Selection runs 72 validation episodes per checkpoint: 3 contexts per class, levels 0 and 2, no brace and wriggle brace. Two new tests cover the gate and the residual cap and release lock (15/15 pass).

**Short local test run** (`results/runs/safe_short/`): `--safe`, 300 k samples, 40 min on 7 workers, plus 13 min of selection.

| samples | 0 | 23 k | 45 k | 90 k | 135 k | 200 k | 300 k |
|---|---|---|---|---|---|---|---|
| success, last 200 episodes (`safe_short`) | 0.92 | 0.56 | 0.42 | 0.36 | 0.47 | 0.40 | 0.43 |
| same, old `main` run | 0.85 | – | 0.34 | 0.38 | 0.27 | 0.23 | 0.12 |
| buckets unlocked (`safe_short`) | 1 | 8 | 10 | 10 | 11 | 11 | 11 |

Validation selection (`select.json`), seated / 72 episodes:

| checkpoint | 0 (the scripted base) | 100 k | 200 k | 300 k |
|---|---|---|---|---|
| seated | **37** | 26 | 22 | 24 |

What it shows:

1. **The collapse is fixed.** Training success holds at about 40 % after the curriculum widens, where `main` was at 12 % and still falling at the same point. The easy buckets keep their skill: FREE at levels 0, 1 and 2 seats 86 %, 94 % and 74 % (`main` ended at 8 %, 4 % and 12 %).
2. **But it learns nothing better than the base yet.** Every trained checkpoint scores below the scripted base on validation, so `best.pt` *is* the scripted base. Held-out repo numbers are therefore §6.2's scripted row, so the repo evaluation was not re-run. The curriculum reached CORNER (0–8 %) and never SLOT, POCKET or ENCLOSED.
3. **The L2 term is too weak at 0.01.** The penalised mean residual grew steadily, from 0.007 to about 1.3 (summed over 9 channels: about 0.37 per channel before tanh, about 0.11 of the action range after the 0.3 cap).
4. **Bricks are still lost** (1–2 per update) with the release channel locked. So drops come from the motion itself (pulling up or shoving sideways), not just from early release. Worth tracing.

**Suggested next steps:**

1. Raise `--res-l2` to about 0.1.
2. Trace the lost-brick episodes.
3. Run 2–5 M samples on the cluster.

Locally, a longer run would mostly cost hours before telling us whether any checkpoint beats checkpoint 0.

### 6.6 Dropped bricks: a false termination (fixed)

`trace_lost.py` replays the scripted controller on the contexts where it releases: the 7 pyramid repo steps plus 3 family contexts each of CORNER, POCKET and ENCLOSED, at levels 0–4, 1 seed. For each `lost_brick` episode it logs the hand's and the brick's lateral error, the brick's height above its seat, its tilt, and what it touches. 50 episodes released: 22 seated, 19 timed out and **9 ended "lost_brick"**.

**In none of the 9 was the brick lost.** At the end the brick was 0.1–2.5 mm off its seat laterally, while the *hand* was 3.0–3.9 mm off. The check called Sakib's `_true_err()`, which measures the hand. After a release, the hand stays compliant: it springs sideways by about 1 mm on release and drifts while pressing the studs. A hand error above 3 mm therefore ended episodes whose brick was fine:

- **4 were essentially seated:** pyramid steps 2 and 9 and a CORNER at level 0, each with the brick ≤ 0.1 mm off and 0–0.3 mm high, cut off mid-press.
- **3 were cut off right after release:** pyramid step 3 at levels 3 and 4, and the CORNER at level 3. Each ended 1 step after release (the hand's sideways spring), with the brick 1.6–2.0 mm off and in the slot mouth.
- **2 POCKETs were wedged, not lost:** the brick sat 6 mm high at about 11° tilt. That is a real failure, but a jam (timeout), not a drop.

**Fix:** `lost_brick` now uses `_brick_err()`, the brick's own lateral distance from its seat (> 3 mm, `LOST_ERR`), with a test. The scripted baseline and the `main` / `safe_short` numbers above were measured with the old check. `results/eval/scripted_cheap_v2.*` re-runs the baseline with the fixed check, and `results/runs/safe_short2/` re-runs the short training with the L2 penalty raised to 0.1 (§6.7).

### 6.7 After the fix: baseline re-run and `safe_short2`

**Scripted baseline with the fixed check** (`results/eval/scripted_cheap_v2.md`; same contexts as §6.2):

| contexts | brace | ≤ 0.5 mm | 1–2 mm | 3–4 mm | lost brick (old → new) |
|---|---|---|---|---|---|
| repo | none | 33/68 | 22/68 | 9/68 | 2 → 2 |
| repo | wriggle brace | 31/68 | 22/68 | 9/68 | 2 → 2 |
| family test | none | 32/60 (was 30) | 23/60 (was 21) | 3/60 | 6 → 5 |
| family test | wriggle brace | 30/60 (was 28) | 19/60 (was 17) | 3/60 | 8 → 5 |

The fix moves the baseline little. The repo numbers are unchanged, and the family test gains 2 seated at each of the two lower levels. Most falsely terminated episodes would not have seated anyway: they become timeouts. The lost bricks that remain are real, with the brick more than 3 mm off its seat.

**`safe_short2`** (`results/runs/safe_short2/`): `--safe` with `--res-l2 0.1` and the fixed check, 300 k samples.

| samples | 0 | 45 k | 90 k | 135 k | 180 k | 270 k | 300 k |
|---|---|---|---|---|---|---|---|
| success, last 200 episodes | 0.92 | 0.39 | 0.41 | 0.47 | 0.42 | 0.47 | 0.41 |
| mean residual penalty term (Σ μ²) | 0 | 0.08 | 0.19 | 0.15 | 0.19 | 0.20 | 0.17 |

| checkpoint (validation, seated / 72) | 0 (scripted) | 100 k | 200 k | 300 k |
|---|---|---|---|---|
| `safe_short` (L2 0.01) | **37** | 26 | 22 | 24 |
| `safe_short2` (L2 0.1) | **37** | 31 | 23 | 27 |

- **The stronger penalty does its job.** The residual stays small (Σ μ² ≈ 0.2, against 1.3 at 0.01), and training success is stable at 40–47 %.
- **The curriculum got one step further:** 12 buckets, including the first SLOT bucket (SLOT at ≤ 0.5 mm, 26 % in training).
- **Still no checkpoint beats the scripted base on validation.** The best trained checkpoint is 31/72 against 37/72, so `best.pt` is again the scripted base. Three local runs now agree: within 300 k samples, PPO on this residual does not find an improvement. It either drifts away from the base (`main`) or stays near it and is slightly worse.

**What this implies.** The safeguards remove the failure modes, but the learning signal at this budget is too weak:

- the reward is a sparse ±1;
- each update covers only about 10 finished episodes;
- the success rate is around 40 %.

The options are a much larger sample budget on the cluster (§3.6: 5–20 M), or a denser learning signal. The §3.5 depth-potential shaping ablation is the cheapest of the latter, and so is more parallel environments per update (larger batches). The shaping is tried in §6.8.

### 6.8 Depth shaping (`safe_shaped`)

The env now has an option `shaping=k`. It adds `γΦ(s') − Φ(s)` to the reward, where:

- Φ = k · depth progress;
- depth progress is 0 with the brick bottom 12 mm or more above its seat and 1 when seated (privileged, reward only);
- Φ = 0 at episode end, so the discounted sum telescopes to −Φ(s₀) and the optimal policy is unchanged (a test checks this).

`train_wriggle.py --shaping k` turns it on. The run `safe_shaped` is identical to `safe_short2` (same seed, `--safe`, 300 k samples) except for `--shaping 0.5`.

| checkpoint (validation, seated / 72) | 0 (scripted) | 100 k | 200 k | 300 k |
|---|---|---|---|---|
| `safe_short2` (no shaping) | **37** | 31 | 23 | 27 |
| `safe_shaped` (shaping 0.5) | **37** | 27 | 29 | 32 |

| final training bucket rates | FREE ≤ 0.5 mm | FLUSH ≤ 0.5 mm | CORNER ≤ 0.5 mm | CORNER 1 mm | SLOT ≤ 0.5 mm |
|---|---|---|---|---|---|
| `safe_short2` | 0.78 | 0.44 | 0.08 | 0.12 | 0.26 |
| `safe_shaped` | 0.84 | 0.32 | 0.24 | 0.18 | 0.28 |

- **The shaped run is the only one whose validation score rises with training:** 27 → 29 → 32. Its last checkpoint is the best trained one so far, and it is better on CORNER in training.
- **It still does not beat the scripted base** (32 against 37 / 72), so `best.pt` is again the base.
- **The differences between trained runs are within noise.** With 72 episodes, one standard error is about ±4 episodes, so the rising trend is suggestive, not established.

**Verdict at this budget:** shaping is the most promising of the three settings, but 300 k samples on 7 local cores is not enough to tell whether learning beats the scripted wriggle. Two things would decide it:

1. a longer shaped run (≥ 1–2 M samples; about 3–6 h locally, or the cluster);
2. a larger validation set (for example 3 seeds, or `per_class` 5) to separate checkpoints.

### 6.9 Skill-level RL with PPO (`skill_main`: no improvement)

**Why change the formulation.** Three residual-PPO runs (§6.4-§6.8) never beat checkpoint 0 on validation. The suspected cause is the learning signal: a 20 Hz residual over about 200 steps, a sparse ±1 reward and about 10 finished episodes per update. The skill-level run makes fewer, larger decisions and gets about 400 finished episodes per update (measured in the smoke run, below).

**Formulation** (`skill_env.py`, `train_skill.py`, `eval_skill.py`; all of `SocketInsertionEnv` is reused unchanged, including contexts, DR, brace conditions, cheap sensing, terminations and the §3.5 reward).

- *Approach is not a decision.* ScriptedBase align and descend run until first contact (the scripted phase flips to search or press).
- *Macro step.* From first contact, one decision = K = 10 env ticks (0.5 s, exactly one period of the 2 Hz rock). The policy outputs θ ∈ [-1, 1]^10. `SkillController` replaces `env.base` in the env's `"scripted"` mode, so the unchanged `env.step` asks it for the 9-D action every tick and then does the impedance control, sensing, reward and terminations. An episode has about 2–17 macro steps (200 ticks in all, of which about 45 are approach).
- *θ = 0 is the scripted controller.* Same spiral, rock, 8 N search press, seating-press ramp, release rule and post-release press. Checked: θ = 0 gives the same reason, step count and peak force as `ScriptedWriggle` on the same seed (test), and the 72-episode validation score of checkpoint 0 equals the scripted one (37/72, below).

| θ | channel | range (θ = -1 … 0 … +1) |
|---|---|---|
| 0, 1 | centre nudge x, y | ±0.6 mm per macro step, accumulated on the spiral centre; the centre is kept within 4 mm of the believed target; world frame (= socket frame, the believed yaw is under 2°) |
| 2, 3 | static tilt roll, pitch | ±3° about the believed pose, search phase only; commanded tilt (static + rock) capped at 5.5° |
| 4 | yaw offset | ±1.5°, search and press phases |
| 5 | rock amplitude | 0 … 1.5° … 3° at 2 Hz (roll; pitch 0.7×) |
| 6 | rock yaw mix | yaw rock = 0.5 · θ6 · roll rock (0 at θ = 0: the scripted rock only tilts) |
| 7 | press force | search press 2 … 8 … 24 N; seating press 0.4 … 1 … 1.2 × the scripted force, at most 0.9 of the budget (also used on the stud tops after a release) |
| 8 | lateral stiffness scale | 0.15 … 1 … 1 (θ > 0 is the same as 0) |
| 9 | spiral radius | 0 … 2 mm … 3 mm |

**Differences from the first sketch.** Ten channels, not eight: a spiral-radius channel (the scripted base searches in a 2 mm spiral, and without it θ = 0 could not match the base; it also lets the policy switch the lateral search off in a tight slot) and a separate rock-yaw channel (so that θ = 0 stays the pure-tilt scripted rock). The stiffness channel has no stiffer-than-scripted half. The nudge only acts in the search phase (the scripted press phase is laterally free). Release stays the scripted rule.

**Macro observation** (97 numbers, actor): depth progress and its change over the window; the cheap wrench estimate (mean, min, max over the window, hand frame); impedance tracking error (mean, last); hand linear and angular velocity (mean); hand xy, roll, pitch and yaw against the believed pose; controller phase (search, press, released); accumulated centre offset; tips closed; elapsed time and time in phase; the previous θ; the 32 context and 9 brace features. The critic adds the env's 28 privileged numbers (true offset, clutch state, contact count, neighbour utilisation, DR parameters, true wrench).

**Reward.** The sum of the env rewards over the K ticks, plus the potential-based depth shaping of §6.8 at the macro level: k · (γ_macro Φ(s') − Φ(s)), with k = 0.5, γ_macro = 0.97 (also the PPO discount) and Φ = 0 at termination (a test checks that the discounted shaping telescopes to −Φ(s₀)).

**PPO** (`train_skill.py`): Sakib's `ActorCritic` (2 × 256), zero-initialised mean (θ = 0 at the start), initial std 0.4 in pre-tanh space (θ = tanh(u)), exploration std clamped to [0.15, 0.6], entropy coefficient 0.001, lr 3e-4, gradient clip 1, 5 epochs × 4 minibatches with an early stop at approximate KL 0.05, GAE λ = 0.95. 7 workers × 3 envs, 50 macro steps per env per update = 1050 macro steps per update. The critic trains alone for the first 5,000 macro steps. Curriculum: the success-band buckets of `train_wriggle.Buckets`, unlock gate 0.4. One addition: 10 % of episodes are drawn uniformly over all 30 buckets (including locked ones; §3.6 lists this 10 % but `train_wriggle` did not implement it), so the policy sees SLOT, POCKET and ENCLOSED sockets and the locked buckets get success statistics. A checkpoint is written every 15 minutes of wall time, plus checkpoint 0 (θ = 0) and the last.

**Selection and evaluation** (`eval_skill.py`). Every checkpoint is run deterministically (θ = tanh(mean)) on the family validation split: 5 contexts per class, levels 0 and 2, braces none and wriggle, 1 seed = 120 episodes; `ScriptedWriggle` is run on the same 120 episodes and the same seeds, and recorded in `select.json`. The selected checkpoint (ties go to the earlier one) and checkpoint 0 are then run on the `scripted_cheap_v2` protocol (repo + family test + probe, levels 0 / 2 / 4, braces none + wriggle, 2 seeds), writing `results/eval/skill_main.{md,jsonl}` and `skill_main_ckpt0.{md,jsonl}` in the same table format. `run_skill_chain.sh` runs train, select and both evaluations unattended; the run is `skill_main` (`results/runs/skill_main/`).

**Measured so far (smoke run `skill_smoke`, 3 updates, 3,150 macro steps).**

| quantity | value |
|---|---|
| training throughput, 7 workers × 3 envs | about 7 macro steps/s (about 150 s per 1,050-step update, about 195 env ticks/s including approach ticks and env rebuilds) |
| finished episodes per update | 206–413 (the old residual runs: about 10) |
| validation, 72 episodes (3 per class, levels 0 and 2, braces none + wriggle) | scripted 37; checkpoint 0 (θ = 0) 37; smoke checkpoint after 3 updates 34 |

Three updates say nothing about learning; the smoke run only checked the plumbing and the speed.

**Launch and outcome.** `run_skill_chain.sh` started the run `skill_main` at 03:38 CST on 2026-10-23 (7 workers × 3 envs, ~140 s per update). The supervisor stopped it after 46 logged updates (48,300 macro steps; checkpoints kept, last `ckpt_00045150.pt`). Training log (`results/runs/skill_main/train.jsonl`):

| update | 1 | 6 | 16 | 26 | 36 | 46 |
|---|---|---|---|---|---|---|
| success, last 200 training episodes | 0.80 | 0.47 | 0.38 | 0.42 | 0.38 | 0.45 |
| buckets unlocked | 3 | 8 | 11 | 12 | 12 | 12 |
| exploration std (mean) | 0.400 | 0.399 | 0.397 | 0.395 | 0.394 | 0.393 |
| largest |mean θ| over a batch | 0.02 | 0.02 | 0.02 | 0.09 | 0.06 | 0.09 |

Diagnostic on the 72-episode validation set (`results/eval/diag_ckpt45k.md`): `ckpt_00045150` seated **28/72**, checkpoint 0 and the scripted controller **37/72**. The policy barely moved (std 0.400 to 0.393, mean θ within ±0.1) and the training success did not rise. This is the same outcome as the three residual runs: PPO does not find an improvement at this budget (about 50k macro steps, 12 buckets unlocked). One class stands out in the diagnostic: FLUSH seated 0/12 (FREE 10/12), and the scripted baseline is poor on FLUSH too (`scripted_cheap_v2`: 8/28 at level 0). §6.10 traces that. §6.11 replaces PPO by a paired evolution-strategy search.

**Environment note.** On this machine `python` on the PATH is `lego_env`, a venv with no packages. The numbers above and the chain use `~/.venv/bin/python` (3.11; numpy, torch, mujoco, scipy). `pytest` is not installed in either; the tests were run with a small stand-in runner (same test functions).

### 6.10 FLUSH sockets: the finger pads overhang onto the neighbour (pad shift)

Full trace and tables: `results/flush_trace.md` (`flush_trace.py`, `flush_padshift.py`).

- **Finding.** On FLUSH sockets the scripted descent stops 5–7.5 mm above the seat with a finger pad resting on the flush neighbour's studs. The pads are 17.5 mm wide and centred on the brick; a 1-wide brick is 7.8 mm and a 2-wide one 15.8 mm, so the pads overhang by 4.85 mm and 0.85 mm per side, over the neighbour. The audit's "free pinch axis" looks at the brick's end cells only. The scripted controller then keeps spiralling and pressing with the pad on the neighbour (timeouts, 70 N presses, neighbour joint breaks). The controller is not at fault, and no termination, budget, gate or joint strength is involved.
- **Fix.** `SocketInsertionEnv(pad_shift=True)` (default off, so older numbers stand): when exactly one face perpendicular to the pinch axis touches a neighbour, the pads sit off-centre along that face, away from the neighbour, with their edge 0.2 mm inside the brick's face. This is a grasp-point choice from the plan. `skill_env.WriggleSkillEnv`, `eval_skill.py --pad-shift` and the ES run use it.
- **Before / after** (scripted, no brace, one seed):

| sockets | pad centred | pad shifted |
|---|---|---|
| FLUSH family, 6 train + 6 validation contexts × levels 0, 2 | 9/24 | **16/24** |
| `pyramid_b_006_exec` (3 seeds × levels 0, 2) | 0/6 | **6/6** |
| `pyramid_b_008_exec` | 0/6 | **5/6** |
| CORNER, SLOT (24 episodes each) | 10/24, 7/24 | 10/24, 7/24 (the shift does not apply) |

Seven FLUSH family episodes changed, all to seated; none got worse. Tests: `tests/test_skill.py`. Everything before this section (including `scripted_cheap_v2` and `skill_main`) used centred pads.

### 6.11 Paired evolution-strategy search (`es_main`)

`es_skill.py`; chain `run_es_chain.sh`; run directory `results/runs/es_main/`.

- **Policy.** θ = clip(b[socket class] + W·z, −1, 1): b is 6 × 10 (per-class bias), W is 10 × 8, 140 parameters, initialised to 0 (θ = 0 exactly, i.e. the scripted controller; test). z is 8 features of the macro observation the actor already sees: depth progress, its change over the window, mean cheap-wrench z, mean lateral wrench magnitude, impedance tracking-error magnitude, elapsed time, released flag, brace active. No privileged information.
- **Search.** OpenAI-ES with mirrored sampling (8 pairs per generation, σ = 0.2 on b and 0.1 on W), centred-rank fitness, Adam (lr 0.05) on the mean. Fitness = mean episode return (+1 seated, −1 for a break, force violation, arm contact or lost brick, the utilisation penalty and 0.002 per tick; no shaping).
- **Common random numbers.** All candidates of a generation run on the same batch: 24 episodes, one per (socket class, aim-error level in {0, 1, 2, 3}), the brace alternating none / ideal over the grid, a random train-split context of the class, one env seed per episode; a fresh batch each generation (test). The current mean is scored on every batch and θ = 0 on every third, so improvement is a paired difference on the same episodes. Per generation: 8 pairs + mean (+ θ = 0) = 18–19 candidates × 24 episodes. Pad shift on.
- **Throughput** (7 workers): the first generation of the real run, 432 episodes, took 479 s (about 1.1 s per episode), so a 99-minute search gives about 12 generations. This is a small budget for 140 parameters.
- **Selection.** The last 4 saved means, their average and θ = 0 are run on the validation split (5 contexts per class, levels 0 and 2, braces none + wriggle, 120 episodes, the same seeds for all); ties go to θ = 0, then to the earlier mean.
- **Held-out evaluation.** The selected policy and θ = 0 (both with the pad shift) are run on the `scripted_cheap_v2` protocol: repo + family test + probe, levels 0 2 4, braces none + wriggle, 2 seeds → `results/eval/es_main.{md,jsonl}` and `es_main_zero.{md,jsonl}`. θ = 0 with the pad shift is the paired baseline; `scripted_cheap_v2` (centred pads) remains the historical one.
- **Results** (13 generations; `train.jsonl`, `select.json`, `results/eval/es_main*.md`).

Paired fitness difference of the current mean against θ = 0 on the same 24-episode batch (only generations where θ = 0 was scored):

| generation | 1 | 4 | 7 | 10 | 13 |
|---|---|---|---|---|---|
| mean − θ = 0 | 0.000 | +0.049 | −0.018 | −0.12 | −0.27 |

Validation (120 episodes, same seeds), `select.json`:

| candidate | θ = 0 | gen 11 | gen 12 | gen 13 | gen 14 mean | average of last four |
|---|---|---|---|---|---|---|
| seated / 120 | **67** | 50 | 53 | 52 | 50 | 50 |
| joint breaks | 25 | 27 | 29 | 27 | 29 | 28 |

**θ = 0 was selected**, so `es_main.md` equals `es_main_zero.md` (the pad-shifted scripted baseline). The ES search found no improvement over the scripted wriggle at this budget (13 generations × 24 episodes, 140 parameters).

Held-out evaluation, seated / episodes by aim error (≤ 0.5, 1–2, 3–4 mm), two seeds, counted from the `.jsonl` files:

| contexts, brace | centred pads (`scripted_cheap_v2`) | pad shift (`es_main_zero`) |
|---|---|---|
| repo, none | 33/68, 22/68, 9/68 | 38/68, 26/68, 8/68 |
| repo, wriggle (ideal) brace | 31/68, 22/68, 9/68 | 36/68, 26/68, 8/68 |
| family test, none | 32/60, 23/60, 3/60 | 30/60, 25/60, 2/60 |

Joint breaks over the 204 repo, brace-none episodes: 87 with centred pads, 82 with the pad shift. By class (all braces, 780 episodes): FLUSH 18/84 → 40/84, SLOT 140/360 → 136/360, CORNER 19/120 → 21/120, POCKET 23/96 → 20/96.

**Verdict.** No learned policy (five PPO runs, one ES run) beat the scripted wriggle. The measured gain came from the grasp diagnosis (§6.10). The 85 % target at 2 mm is far off (26/68 = 38 %), and joint breaks stay the dominant failure (82 of 204 repo, brace-none episodes).

**Recommended next steps.** (a) Trace the SLOT joint-break episodes as FLUSH was traced (SLOT is 360 of the 780 held-out episodes) before more learning. (b) ES or PPO with much larger batches (≥ 200 episodes per generation) or on the HPC cluster (5–20 M steps, §3.6). (c) Question the 3–4 mm levels: the calibrated aim error is 0.8 mm (1σ), so 4 mm is a 5σ case. (d) Port to Newton only after (a).

## 7. Risks and checks

- **Cheap wrench fidelity.** A real motor-torque estimate is worse than any noise model. Keep the "no force" ablation, so it is known whether the policy actually needs force at all.
- **Rigid neighbours.** The twin's placed bricks are welded until their joint breaks; they never shove sideways by a fraction of a millimetre. The 0.3 mm displacement termination only matters once the neighbours are clutched instead of welded. Check this in phase 1.
- **Contact model.** Walls do not collide with walls (only through top plates and studs). Check that a brick 0.3 mm off cannot pass *through* a neighbour's wall below its top plate (a penetration test in phase 1).
- **Sim to real.** LEGO interference varies by batch and age. Randomise the per-stud insertion and break forces (8–15 N) and friction, as `InsertionEnv` already does.
- **Small probe sample.** The §2.1 numbers are from 6 seeds each.
- **The brace stand-in is optimistic.** 10× joint strength is not an arm. The real brace in the v4 cell is compliant (about 1.6 N/mm at the hand, `plan_v4_rl_brace.md` fact 3) and has collided with arm B (fact 4: a 1019 N arm-to-arm contact on the arch). Phase 1 replaces the stand-in with a spring of that stiffness plus a keep-out volume.

## 8. What it would take to run on the Newton (v4) cell (not done yet)

The MuJoCo twin is the training simulator. Newton (`dual_arm_sim.py`, `robofab_sim/`) is where full dual-arm builds run. **The wriggle task cannot be trained there as the cell stands**, for these reasons (all from the code and Sakib's ledger, not run here):

| # | Newton today | why it breaks wriggle-fit | what is needed |
|---|---|---|---|
| 1 | A placement succeeds by a **one-shot pose check** after a timed insert move: within 2 mm, 1.5 mm and 5° → weld (`GATE`, `dual_arm_sim.py:159`, `snap()`). | A brick 1.9 mm off counts as seated, so being sloppy is rewarded. | A **continuous** gate checked every frame while pressed, tightened to about 0.3 mm laterally and 0.3 mm vertically. The cell already computes a stricter 1.2 mm "screen" for scoring. |
| 2 | **No stud interference.** A brick seats at about 0.5 N, and above about 71 N it can pass through the brick below (`plan_v4.md` §0 fact 6). | No "jammed vs. slipping in" signal to feel. Press force and force-budget terms mean nothing. | A per-brick axial resistance with the twin's clutch semantics: hold until n × 8.9 N, break at 11.3 N per stud, extraction harder than insertion. It can run as a Warp kernel adding to `body_f` after every `clear_forces()`, the same hook the brace load fixture uses (`plan_v4_rl_brace.md` §2.2). |
| 3 | Arm B is driven by **position waypoints (IK)**; the code says "position-controlled IK without a force loop". | No compliance, no press channel, no stiffness channel. The tilt and yaw wriggle cannot be commanded. | Task-space **impedance** for arm B near the target, with joint torques computed as Jᵀ·F per frame (Newton accepts joint forces). This also gives the motor-torque wrench estimate the policy reads (§3.2). |
| 4 | Brick colliders are shrunk inward by `COLLIDER_INSET` (0.1 mm; `plan_v4.md` §0 fact 6). | Slot play becomes about 0.3 mm per side instead of 0.1 mm: easier than the real brick. | Zero inset between same-layer neighbour pairs, or measure the actual play and widen the twin's curriculum to match. Validate with the slot probe (row 8). |
| 5 | Light bricks under a press can tunnel through each other (same source). | Pressing hard, which the policy will try, fails for the wrong reason. | Cap the press speed (the twin's 8 mm/s rule) or use the 2× brick scale Sakib already supports, where 100 N holds. |
| 6 | The RL interface (`rl_batched.BatchedReachEnv`) offers 10 Hz dx/dy/dz/dyaw/grip and ends when the 2 mm gate passes. One model holds identical worlds only (`plan_v4_rl_brace.md` fact 6). | Wrong start state, action space and termination; one socket layout per batch. | A `BatchedWriggleEnv`: start each world at the 25 mm pre-insertion pose of a `socket_family` context, with the 9-D action of §3.3, the observation of §3.2 and the §3.5 reward. Group worlds by context, or park unused bricks off the plate so all worlds share one model. |
| 7 | Arm A's brace is executed by the real arm (`--strategy`), and arm-to-arm contact is only monitored. | – | Use the real brace instead of the twin's stand-in. End episodes on the contact monitor's `arm_arm` event (the twin's `arm_contact`). |
| 8 | – | – | **Validation gate before any Newton training:** run `slot_probe.py`'s protocol (open vs. pocket, braced vs. unbraced, 0–4 mm) in Newton; it must match the twin's pattern within ±10 points. Then evaluate the twin-trained policy zero-shot on the repo contexts in Newton. Fine-tune in Newton only if that transfer gap is large. |

The observation (§3.2), action (§3.3) and reward (§3.5) are defined without reference to the simulator. The context features, wrench estimate and brace observation can be computed in Newton from the same plan and joint state, so a policy trained in the twin can be loaded there unchanged. Compute: Newton needs a CUDA GPU with room for the model. The team's runs used an RTX 5090D; the local GTX 1660 Ti (6 GB) is untested for this.

## 9. Reproduce

```bash
# Python 3.12+ with numpy, scipy, pillow, mujoco (3.3+), torch (CPU is enough), pytest
python Tanvir/tight_fit_audit.py                 # -> results/tight_fit_audit.{md,json}
python Tanvir/slot_probe.py --brace none all     # -> results/slot_probe.md            (~3 min)
python Tanvir/socket_family.py                   # -> results/socket_family.json       (repo + sample contexts)
python Tanvir/wriggle_brace.py                   # brace trigger on the probe and the repo contexts
python -m pytest Tanvir/tests -q                 # 15 tests, ~5 s
python Tanvir/eval_wriggle.py --controller scripted --contexts repo family probe --levels 0 2 4 \
       --braces none wriggle --seeds 2 --tag scripted_cheap               # baseline, ~15 min on 7 workers
python Tanvir/train_wriggle.py --name safe_short --mode residual --safe --samples 300000 --workers 7 --select
                                                 # ~45 min; 5-20 M samples needs the cluster
python Tanvir/eval_wriggle.py --controller Tanvir/results/runs/safe_short/best.pt --contexts repo family probe \
       --levels 0 2 4 --braces none wriggle --seeds 2
```

All scripts import `Sakib/brickassembly` read-only (with bytecode writing off, so nothing is written there) and write only under `Tanvir/results/`.
