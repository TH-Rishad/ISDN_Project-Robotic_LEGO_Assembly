# Kean: motion and insertion RL

Sakib owns environment provisioning, package versions, the Newton cell, build plans, grasp/brace execution, sensing, collision models and contact classification. Kean consumes those interfaces and adds asynchronous planned motion and camera/proprioception insertion RL.

## Run with Sakib's supplied environment

No Conda activation or Kean installer is required. The thin `scripts/run.sh` wrapper delegates to `Sakib/brickassembly/scripts/run.sh`, selecting Sakib's `.venv` through its existing `ROBOFAB_PY` interface. If that environment or a dependency is unavailable, send the diagnostic to Sakib.

```sh
bash Kean/scripts/run.sh Kean/scripts/check_env.py --gpu
bash Kean/scripts/test_m05.sh
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
| Build/brace plans | `planner`, `bracing`, `stability` | M0.5 experiment manifest and brace-all stress annotation |
| Motion planning | P3 probe `Rig`, `warm`, pose/world conversions, `plan_traj`, `resample` | Async requests, zone arbitration, final 30 mm Cartesian split |
| Collision models and brick geometry | P3 `PROFILES`, `TABLE`, `brick_box` | Dynamic scene inputs and geometry-query adapters |
| Clearance | P3 sphere/FK/gap routines | Measured-joint/finger-opening adapter and reporting |
| Contact events and recordings | `ContactMonitor`, `Recorder`, `Example.finish` | Job labels and M0.5 acceptance summary |
| Camera sensing and observation encoding | `cell.cameras`, `rl_env.Sim.observe`, `ObsSpec.encode` | `insertion.obs.from_shared` filters to actor cameras/proprioception |
| PPO utilities and physical force limits | `tasks.ppo.squash_logp`, joint-model capacity constants | Camera encoder, residual action contract and task-specific training adapter |
| Camera display | `Example.refresh_feeds` | Existing HTTP tile bridge only |

Import local modules as `Kean.*`; the top-level `motion` package belongs to Sakib. `shared.py` loads the existing P3 script once and redirects only report output to Kean. Model paths stay exactly as supplied by Sakib. Old `Kean/results/p3/fr3*.yml` copies are retained as historical artifacts and are no longer selected by the runtime. The shared vendor loader runs as Sakib defines it; Kean does not inject Conda packages or override that loader.

## Experiments and prototypes

Default M0.5 uses shared grasp execution and optimizer iterations. `P3_ITER_SCALE` explicitly opts into Kean's iteration-budget experiment; its default is now 1.0. The pickup-offset search stays opt-in (`m05.py --experimental-grasp-offset`). The recorded launcher passes that flag. Cube job 0 in `results/checks/20261021-191332-980676` placed 8/8, had no unintended contacts, and passed the gate; planned-playback clearance was 0.046 m. The same cube without the offset, `results/checks/20261021-171655-549530`, had 8 unintended finger contacts and no measured clearance. Arch, hollow_box, and S3 are not run. Contact acceptance thresholds are unchanged.

`p3.sh` runs the shared probe without automatic sphere repair, contact rescoring or summary-source rewriting. Historical extensions require explicit `--kean-extensions`. Collision-model generation (`--step 2` / `--all`) is delegated to Sakib and rejected by this wrapper. P3 report files continue to be written under Kean.

`memory/`, `support/hold.py`, Handoff and GripAdjust are isolated prototypes, not a second live perception/support stack. M0.5's no-op Store bookkeeping has been removed. Future live memory should adapt Sakib's `top_detect.remember`, `track_held` and sensing interfaces while enforcing the task's scan schedule; the local full-pose prototypes are retained for review, not claimed as integrated.

`KinematicInsert` is a CPU fixture with synthetic observations. `scripts/insert.py` is only a network/update smoke experiment. It is not a Newton inserter or evidence of a trained policy. A live cell adapter can supply `observe()` using the shared encoded observation; `from_shared` discards all fields except cameras and proprioception. Newton action, reset and reward integration is still required. The frozen MuJoCo twin's 121-D environment is not substituted for this task.

See [review.md](review.md) for the audit, remaining shared-interface needs and validation limits. Sync the complete source tree after these interface changes.
