# Richard — Minimal Dual-Arm LEGO Assembly Control Flow

Personal workspace for **Richard** (repo owner: TH-Rishad).

A self-contained control flow that meets the course feedback's **100-point**
criteria with the minimum moving parts. The heavy cell, planner and brace
generator are **imported read-only** from `Sakib/brickassembly` (via
`shared.py`, the single boundary); the motion adapter is copied from Kean's
certified acceptance path with all feedback-descoped machinery removed.

## Criteria and how each is met

| # | Criterion (100 pts) | Where it is met |
|---|---|---|
| 1 | Target structure as direct input; no image conversion | `main.py --plan-check` / `--live` take structure ids (`arch`, `S3`); `plans.py` never touches image input |
| 2 | Auto-generate assembly sequence + support locations | Sakib `planner.build_plan` + `bracing.assign` (`weakest_joint`), driven by `plans.generate` |
| 3 | Both arms autonomously assemble two different structures, each ≥8 bricks and ≥1 temporary-support section | `runner.run_job` executes **arch** (11 bricks, support step 10) and **S3** (16 bricks, supports 11/13/14) in the Newton cell: arm B places, arm A braces; `verify.py` independently recomputes all three criteria |

Collision detection runs throughout (60/80-pt requirement, retained):
cuRobo collision-aware planning → dense full-sphere path validation
(`path_check`) → 1 mm per-frame command guard (`tick`) → `ContactMonitor`
event audit. Events are recorded and reported.

## Layout

```
Richard/
├── main.py            # CLI: --plan-check | --live | --verify RUN_DIR
├── plans.py           # structure id -> sequenced, brace-annotated plan (criteria 1-2)
├── runner.py          # one-structure execution engine (Newton cell, HPC)
├── verify.py          # independent CPU verifier of all three criteria
├── shared.py          # single boundary: sys.path += Sakib/brickassembly; probe singleton
├── motion/            # copied from Kean's certified acceptance path (see below)
├── support/sequence.py# build-zone geometry used by the adapter
├── scripts/run.sh     # interpreter routing to Sakib/.venv (GPU host)
├── scripts/setup_pod.sh # provision that venv (pinned Newton + cuRobo)
├── scripts/main.sh    # convenience wrapper: bash Richard/scripts/main.sh ...
├── tests/test_cpu.py  # CPU self-checks (criteria 1-2 + import safety)
└── results/           # run packages and verification reports
```

`motion/` contents (14 modules): `plan_curobo` (cuRobo planner wrapper),
`tick` (joint playback + command guard), `contact_segment` (last-30 mm
Cartesian moves), `clearance` / `world` / `obstacles` / `brick_spheres`
(collision geometry), `path_check` (dense sampled validation), `wriggle`
(scripted tight-socket insert, e.g. the arch crown), `brace_hold` (freeze
measured joints while supporting), `brace_search` (retry generated brace
poses), `coordinated` (placer/helper handshake timing), `playback` (replay).

## Deliberately removed (feedback: not required)

| Removed | Why |
|---|---|
| `stud_fit.py`, `newton_fit.py` | Force simulation / impedance insertion — "Force simulation will not be required" |
| `gate.py` (M0.5), 235-placement jobs | Out of scope of the three criteria |
| `grasp_clearance.py`, `iter_budget.py` | Opt-in experiments (env-gated) |
| `drop_press.py` | Pyramid-only tight steps; arch/S3 do not need it |
| `reservations.py` | Concurrent dual-build motion; the acceptance path builds one structure at a time |
| `contact_policy.py` | Dual-build stud-mating evidence detail; the single-build verifier does not use it |
| `dual_build.py`, `dual_coordinator.py`, `verify_dual.py` | Concurrent two-structure builds (beyond the criteria) |
| RL / perception / image-to-structure | Descoped by the feedback |

Kean's originals remain untouched under `Kean/` (read-only reference).

## Run

CPU (any machine, proves criteria 1-2):

```bash
python Richard/main.py --plan-check --out Richard/results/plan_check_001
python -m unittest Richard.tests.test_cpu -v
```

GPU (any Ubuntu + CUDA 12.8 host, proves criterion 3). If `Sakib/.venv` does
not exist yet, provision it first — one interpreter holds Newton **and**
cuRobo, which the upstream two-interpreter `env.lock` splits:

```bash
bash Richard/scripts/setup_pod.sh           # pinned Newton/Warp/cuRobo; ends with probe step 1
bash Richard/scripts/main.sh --live --out Richard/results/acceptance_001
python Richard/main.py --verify Richard/results/acceptance_001 \
    --out Richard/results/verification_001     # exit 0 = full pass
```

`setup_pod.sh` pins exactly what `Sakib/brickassembly/env.lock` records:
Warp 1.13.0, Newton 1.2.1, MuJoCo-Warp 3.8.0.3, MuJoCo 3.8.0, cuRobo
0.8.0.post1.dev43 at `78fd485`, on Python 3.12 sharing the image's
torch 2.11.0+cu128. Verified on a RunPod RTX 5090 pod: probe step 1 reports
`franka_plan_ok: true`, ~24 ms per plan.

A partial run is diagnosable too: interrupt it and the cell still writes a
complete `episodes.jsonl` / `cell_rows.jsonl` / `.npz`, with
`cell_rows[-1].error == "KeyboardInterrupt: "` and an honest `placed` count.

A live run writes `plans/`, `episodes.jsonl`, `cell_rows.jsonl`, per-structure
`.npz` recordings, `provenance.json` and `summary.json`. The verifier ignores
the saved `ok` flags and recomputes placement, completion, support hold timing
(before + during + after each supported insertion) and provenance binding.

## Write boundary (hard rule)

| Path | Permission |
| --- | --- |
| `Richard/**` | **Write** — all work happens here |
| `Kean/**`, `Sakib/**`, `Tanvir/**`, `final_report/**` | **Read-only** |
| Repo-root files (`README.md`, `RUNNING_ON_THIS_PC.md`, `.gitignore`, …) | **Read-only** |

A `pre-commit` hook enforces this: any staged path outside `Richard/` aborts
the commit. Bypass only with an explicit maintainer instruction.
Local-only files (never pushed): `AGENTS.md`, `config/`, `.cursor/`, `richard.id`.

## Branch

Work happens on `feat/richard-workspace`, branched from `main`.

```bash
git switch feat/richard-workspace
```

## Environment

Two paths:

- **GPU pod / HPC** (Linux, CUDA 12.8): `scripts/setup_pod.sh` builds the
  single interpreter at `Sakib/.venv` with the pinned stack; `scripts/run.sh`
  selects it and delegates to `Sakib/brickassembly/scripts/run.sh`.
- **This Windows copy**: CPU checks only (plan generation + verification).
  `../RUNNING_ON_THIS_PC.md` documents the original four-interpreter HPC
  layout, which the single venv above deliberately collapses.

## Current status (2026-11-17)

CPU (both machines, Windows and the GPU pod):

- Criteria 1-2 verified: arch 11 bricks (support step 10), S3 16 bricks
  (supports 11, 13, 14), `strategy=weakest_joint`, `image_input=false`;
  8/8 self-checks pass (`results/plan_check_001/manifest.json`).

Newton cell on a RunPod RTX 5090 pod (`results/acceptance_pod/`):

- The full stack builds and plans: Newton 1.2.1 + Warp 1.13.0 + cuRobo
  0.8.0.post1.dev43 in one interpreter, probe step 1 `franka_plan_ok: true`.
  The dual-arm cell then placed 9 arch bricks before the run was stopped.
- Evidence: `episodes.jsonl` shows **9/11 arch bricks placed, every one
  `success: true`**, lateral error 0.4-0.6 mm, `dz ≤ 0.1 mm`, tilt ≤ 0.2°;
  `cell_rows.jsonl` records `placed: 9`, `error: "KeyboardInterrupt: "`
  (deliberate stop, not a failure).
- Not yet exercised on this pod: the braced step (arch step 10) and S3 — the
  run was stopped at 9/11 by request, so **criterion 3 is not certified by
  this package**; `--verify` on it will report criterion 3 as unverified. A
  complete `--live` (both structures, ~50 min at the observed ~2 min/brick)
  is what produces an exit-0 verification.

