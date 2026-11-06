# Running this repository on this PC

Checked and updated on 2026-11-06. `Sakib/.venv` is now provisioned for Kean
and the main Newton simulator, including CUDA PyTorch and pinned cuRobo.
The other examples can reuse the environments installed for the older
`~/Documents/ISDN_Robofab` checkout. This checkout does not contain
`Sakib/isdnenv`, `Sakib/mjenv`, or `Sakib/robofab_sim/.venv`;
`Sakib/BrickSim` is empty.

## Machine and available environments

Ubuntu 24.04.5, x86_64; NVIDIA RTX 5090 D, 32 GB VRAM; driver 595.99.02,
reporting CUDA support 13.2. `uv` and `/usr/bin/python3.12` are available.
The command `python3` currently selects **Python 3.14.2**, so use the explicit
interpreters below for these projects. No environment activation is necessary.

| Environment | Existing location | Relevant installed packages |
| --- | --- | --- |
| Shared Kean / Newton | `Sakib/.venv` in this checkout | Python 3.12.3; torch 2.10.0+cu128, Newton 1.2.1, Warp 1.13.0, cuRobo 0.8.0.post1.dev43; Kean launchers select this automatically |
| Newton / RL / Isaac | `~/Codes/CAIRSS/Issac` | Python 3.12.3; Newton 1.2.1, Warp 1.13.0, MuJoCo 3.8.0, MuJoCo-Warp 3.8.0.3, torch 2.10.0+cu128, Viser, RL trainers, Isaac Lab |
| Coursework / cuRobo | `~/Documents/ISDN_Robofab/isdnenv` | Python 3.11.16; cuRobo 0.8.0.post1.dev43, Viser, yourdfpy, torch, ipykernel, ruckig, stable-baselines3 |
| MuJoCo twin / Tanvir | `~/Documents/ISDN_Robofab/mjenv` | Python 3.11.16; MuJoCo 3.3.7, numpy, scipy, torch 2.14.0+cu130, py_trees, pytest |
| Historical BrickSim | `~/Documents/ISDN_Robofab/BrickSim/.venv` | Python 3.11.16; BrickSim source install and its separate dependencies |

The spelling **`Issac`** in the first path is intentional.

Set these variables in each new terminal before using the examples:

```bash
export LEGO_REPO="$HOME/Documents/ISDN_Project-Robotic_LEGO_Assembly"
export LEGO_OLD="$HOME/Documents/ISDN_Robofab"
export NEWTON_PY="$HOME/Codes/CAIRSS/Issac/bin/python"
export CUROBO_PY="$LEGO_OLD/isdnenv/bin/python"
export TWIN_PY="$LEGO_OLD/mjenv/bin/python"
cd "$LEGO_REPO"
```

## Sakib/Codes: course exercises and notebooks

Use the existing cuRobo environment. These commands start persistent browser
viewers; open **http://localhost:8080** and stop each script with Ctrl+C before
starting the next viewer.

```bash
cd "$LEGO_REPO/Sakib/Codes"
"$CUROBO_PY" urdf_viewer.py     # local planar 3-DOF URDF and joint sliders
"$CUROBO_PY" dual_franka.py     # two Panda models
"$CUROBO_PY" class3.py          # G1 model and kneeling pose
"$CUROBO_PY" hello_world.py     # cuRobo Panda viewer
"$CUROBO_PY" robot_game.py      # cube-picking game
```

Run one line at a time. `hello_world.py` and `robot_game.py` use CUDA kinematics.
`class3.py` hardcodes a G1 asset under the older `ISDN_Robofab/curobo` tree;
that asset currently exists on this PC. `IK_practice.py` is empty.

For `ik.ipynb` and `mp.ipynb`, open them in VS Code and select
`~/Documents/ISDN_Robofab/isdnenv/bin/python` as the notebook kernel. This
environment already has ipykernel and ruckig. Start from `Sakib/Codes` so
relative YAML/asset paths resolve. `lecture6.ipynb` is a separate Gym/SB3
exercise; review its `%pip install` cells before running them because they
install packages and pin numpy to 1.26.4.

## Sakib/brickassembly: main assembly simulator

Run from the repository root through its launcher, with an explicit
`ROBOFAB_PY`. The launcher then changes to `Sakib/brickassembly` itself.

```bash
cd "$LEGO_REPO"

# Available options; imports verified in the existing Newton environment.
ROBOFAB_PY="$NEWTON_PY" bash Sakib/brickassembly/scripts/run.sh dual_arm_sim.py --help

# Live browser view. Open http://localhost:8080.
ROBOFAB_PY="$NEWTON_PY" bash Sakib/brickassembly/scripts/run.sh dual_arm_sim.py \
  --shape cube --viewer viser --num-frames 40000

# Desktop window, including camera views.
ROBOFAB_PY="$NEWTON_PY" bash Sakib/brickassembly/scripts/run.sh dual_arm_sim.py \
  --shape cube --viewer gl --show-cameras --num-frames 40000

# Headless full-build check; --test checks completion.
ROBOFAB_PY="$NEWTON_PY" bash Sakib/brickassembly/scripts/run.sh dual_arm_sim.py \
  --shape cube --viewer null --test --num-frames 40000

# Generated temporary supports; the basic simulator defaults to --strategy none.
ROBOFAB_PY="$NEWTON_PY" bash Sakib/brickassembly/scripts/run.sh dual_arm_sim.py \
  --shape arch --strategy weakest_joint --viewer viser --num-frames 60000

# Direct structure input instead of a sample shape.
ROBOFAB_PY="$NEWTON_PY" bash Sakib/brickassembly/scripts/run.sh dual_arm_sim.py \
  --structure S3 --strategy weakest_joint --viewer viser --num-frames 60000
```

The main options are `--shape cube|arch|hollow_box`, `--structure S1|S2|S3|...`,
`--strategy none|nearest|weakest_joint`, and `--viewer gl|viser|null|usd|rerun`.
The default frame limit shown by `--help` is only 100; supply a larger value
for assembly runs. `--look --aim fk_vision` enables the camera-guided approach;
`--manual --viewer gl` enables manual controls. Press Ctrl+C to stop a viewer.

Normal runs write plans and append episode results under
`Sakib/brickassembly/`. Relative output paths are relative to that folder.
Use `--spec /absolute/path/episode.json` to control the run without rewriting
the normal plan files or appending to `results/proto_episodes.jsonl`.

Planner/status tools and the CPU twin use the existing MuJoCo environment:

```bash
cd "$LEGO_REPO"
"$TWIN_PY" Sakib/brickassembly/planner.py --list
"$TWIN_PY" Sakib/brickassembly/scripts/status.py

# Generates plans under Sakib/brickassembly/plans/.
"$TWIN_PY" Sakib/brickassembly/planner.py --structure S3 --strategy weakest_joint

# CPU controller checks: verified, 9 passed.
MJ_PY="$TWIN_PY" bash Sakib/brickassembly/scripts/run.sh \
  tests/test_control.py -q -p no:cacheprovider

# The wrapper invokes these package entry points as Python modules.
MJ_PY="$TWIN_PY" bash Sakib/brickassembly/scripts/run.sh \
  orchestration/twin_executor.py --structure S3 --strategy weakest_joint --seed 0
```

Do not globally set `ROBOFAB_PY` to the Newton interpreter when reproducing
the frozen twin: its MuJoCo version is different. `MJ_PY` selects the 3.3.7
twin environment. The twin downloads pinned Menagerie assets into
`Sakib/brickassembly/.cache` on first use; those were fetched during this check.
Several tests write JSON result files next to the tests.

Other entry points: `run_assembly.py` is the older cuRobo/Viser assembly
demonstration; run it with `ROBOFAB_PY="$CUROBO_PY"` through the same launcher.
`experiments/*.py` and `tasks/*.py` are batch experiments/trainers, normally
run through the wrapper with `MJ_PY="$TWIN_PY"` (v4 experiments need Newton).
`viz/play_viser.py` consumes a JSONL physics recording and uses the cuRobo
environment; `viz/to_usd.py` uses Newton's environment; `viz/open_usd.py`
requires the Isaac Lab environment. These are distinct recording formats
from Kean's NPZ playback.

## Sakib/robofab_sim: standalone simulator and RL training

This folder has its own launcher. It understands **`VENV`**, not `ROBOFAB_PY`.
Point it at the existing Newton environment to avoid installing another copy:

```bash
cd "$LEGO_REPO"
VENV="$HOME/Codes/CAIRSS/Issac" bash Sakib/robofab_sim/run.sh --help
VENV="$HOME/Codes/CAIRSS/Issac" bash Sakib/robofab_sim/run.sh \
  --shape cube --viewer viser --num-frames 40000

# Optional RL demos/training; these perform work and write their own outputs.
VENV="$HOME/Codes/CAIRSS/Issac" bash Sakib/robofab_sim/run.sh rl_env.py --demo
VENV="$HOME/Codes/CAIRSS/Issac" bash Sakib/robofab_sim/run.sh rl_batched.py --demo 16
VENV="$HOME/Codes/CAIRSS/Issac" bash Sakib/robofab_sim/run.sh \
  train_rl.py --iterations 3 --num-envs 16

# Run this folder's tests.
VENV="$HOME/Codes/CAIRSS/Issac" bash Sakib/robofab_sim/run.sh test -q
```

The launcher changes to `Sakib/robofab_sim`, so its outputs belong to that
folder. The Newton and RL packages required here are already installed;
`--help` was checked, but its full simulator/test/training runs were not run.
For a separate environment instead, use `cd Sakib/robofab_sim` followed by
`bash setup.sh --cuda cu128`, then `bash run.sh ...`.

## Tanvir: tight-fit / wriggle simulation and learning

Use the existing **MuJoCo 3.3.7** environment, from the repository root.
Tanvir's modules import the shared planner and twin from `Sakib/brickassembly`.

```bash
cd "$LEGO_REPO"

# Verified: all 27 tests pass.
PYTHONDONTWRITEBYTECODE=1 "$TWIN_PY" -m pytest Tanvir/tests -q -p no:cacheprovider

# Small scripted socket demo.
"$TWIN_PY" Tanvir/socket_env.py

# Generate the tight-fit audit, or inspect evaluation/training flags.
"$TWIN_PY" Tanvir/tight_fit_audit.py --corbel 0
"$TWIN_PY" Tanvir/eval_wriggle.py --help
"$TWIN_PY" Tanvir/train_skill.py --help

# Example bounded evaluation; results go into Tanvir/results/eval/.
"$TWIN_PY" Tanvir/eval_wriggle.py --controller scripted --contexts probe \
  --levels 0 --braces none --seeds 1 --workers 1 --tag local_probe
```

`train_wriggle.py`, `train_skill.py`, and `es_skill.py` are training/search
entry points. The unattended chains assume a different machine's
`~/.venv/bin/python` unless **`PY`** is overridden:

```bash
# Optional long-running search. Choose a new name to preserve existing runs.
PY="$TWIN_PY" RUN_NAME=local_es HOURS=0.1 bash Tanvir/run_es_chain.sh
# Skill chain: PY="$TWIN_PY" RUN_NAME=local_skill TRAIN_END=23:00 bash Tanvir/run_skill_chain.sh
```

The chain includes checkpoint selection and evaluation after training, so its
total runtime exceeds the training cap. The repository reports that scripted
wriggle outperformed the learned policies; training is optional for running
the assembly project. `report/make_figures.py` uses matplotlib;
`report/build_docx.py` additionally needs `python-docx`, which is absent from
the inventoried environments. Report generation overwrites report artifacts.

## Kean: acceptance plans, CPU checks, and strict dual-arm builds

The CPU paths work immediately with the existing MuJoCo environment:

```bash
cd "$LEGO_REPO"

# Verified: 120 tests pass.
PYTHONDONTWRITEBYTECODE=1 "$TWIN_PY" -m unittest discover -s Kean/tests -q

# Choose a new output directory for each invocation.
"$TWIN_PY" Kean/scripts/verify_acceptance.py --plan-check \
  --out Kean/results/local_plan_preview

"$TWIN_PY" Kean/scripts/verify_acceptance.py \
  --run Kean/results/instructor_run_001 \
  --out Kean/results/local_saved_run_verification
```

Plan-check deliberately exits **2**: it verifies plans, not execution. The
saved instructor run also exits **2 in this checkout**: its plans and episode
rows are present, but both per-frame `.npz` recordings are missing. They are
gitignored. Copy the original `arch_*.npz` and `S3_*.npz` recordings into that
run directory to enable complete evidence verification and saved playback.
Do not overwrite `instructor_run_001`.

A fresh local arch run is now available at
`Kean/results/local_arch_playback_001`: all 11 bricks placed, required support
executed, 9,139 recorded frames, no broken joints, and `accept: ok=True clean`.
It is a new run, separate from the original instructor evidence. Replay it with:

```bash
cd "$LEGO_REPO"
bash Kean/scripts/play_run.sh --run Kean/results/local_arch_playback_001 \
  --shape arch --port 8080 --loop
```

Open http://localhost:8080. The full recording is about 152 seconds at 1x.

Kean's shell launchers require **exactly `Sakib/.venv/bin/python`** and do not
honor a caller's `ROBOFAB_PY`. That environment is now installed and its GPU
preflight and cuRobo planning smoke check pass. Activation is optional:

```bash
cd "$LEGO_REPO"
source Sakib/.venv/bin/activate
```

For the repository-supported shared environment, install it separately:

```bash
cd "$LEGO_REPO"
bash Sakib/setup_env.sh --cuda cu128 --with-curobo

# Correct the setup script's runtime override to match torch's declared requirement.
uv pip install --python Sakib/.venv/bin/python nvidia-cuda-runtime-cu12==12.8.90

# Ensure the DAE mesh loader is present (already installed transitively on this PC).
uv pip install --python Sakib/.venv/bin/python pycollada==0.9.3

bash Kean/scripts/run.sh Kean/scripts/check_env.py --gpu
bash Kean/scripts/accept.sh --live --out Kean/results/local_acceptance_001

# After the original recordings have been supplied:
bash Kean/scripts/play_run.sh --run Kean/results/instructor_run_001 --shape arch
```

This setup was executed successfully. The setup self-check passed 18 tests;
Newton stepping and cuRobo motion planning passed on the RTX 5090 D.
`uv pip check` reports all installed packages compatible after the runtime
correction above. Re-running the setup script currently reinstalls its
12.9.79 runtime pin, so repeat the 12.8.90 correction when using cu128.
`--with-curobo` installs the repository's pinned cuRobo revision. The shared
setup's requirements do not include all RL trainer extras; use the existing
Newton environment or `robofab_sim/setup.sh` for RL.
Strict acceptance is a new simulated build, and its completion was not tested
here. `m05.py` is the older motion gate/diagnostic entry point.

## Sakib/BrickSim and historical Isaac scripts

`orchestration/run_plan.py`, `tests/test_joint.py`, `tests/test_gate.py`, and
`scripts/02_dynamic_calibration.py` use BrickSim / Isaac Sim 5.1. The shared
Newton setup does not cover them. Their launcher hardcodes
`Sakib/BrickSim/.venv/bin/python`; some scripts also hardcode the sibling
BrickSim source directory, so borrowing only the older interpreter is
insufficient.

To provision this separate path, initialize the submodule first:

```bash
cd "$LEGO_REPO"
git submodule update --init --recursive Sakib/BrickSim
cd Sakib/BrickSim
bash scripts/download_prebuilt_native.sh
uv sync --locked
```

These are setup instructions from the existing local BrickSim README, not
steps executed during this check. This project's force experiments also rely
on local BrickSim patches (`Sakib/bricksim-local.patch` and
`Sakib/brickassembly/patches/`); reproducing historical force-query results
requires those changes and the patched native `static_solve` build. Do not
assume a stock upstream installation reproduces them.

Once provisioned, the normal entry point is:

```bash
cd "$LEGO_REPO"
bash Sakib/brickassembly/scripts/run.sh orchestration/run_plan.py --structure S1 --headless
```

For just the static solver, its binary location can be overridden using
`BRICKSIM_STATIC_SOLVE`. The launcher defaults BrickSim runs to headless
because of the historically recorded RTX renderer problem. No BrickSim
simulation was launched in this investigation.

## Remaining folders and verification limits

`Richard/` contains only `.gitkeep`. `Docs/`, `graphify-out/`, `report/`, and
`results/` mainly contain documentation, evidence, and analysis artifacts.
Files under result directories are historical probes, not the normal entry
points. Most files under `motion/`, `sim/`, `cell/`, and `support/` are imported
library modules; start from the CLI entry points above.

Verified on this PC: a two-frame headless CUDA simulation of the main
`brickassembly` cell exited 0; both simulator help commands worked; coursework
viewer imports and referenced Franka/G1 assets were found; Tanvir 27/27,
Kean 120/120, and twin controller 9/9 tests passed. The two-frame check proves
GPU initialization and stepping, not full assembly completion. The subsequent
strict local arch run completed 11/11 with support and its playback server
returned HTTP 200. The two-structure acceptance run, interactive desktop
windows, notebooks, and training were not executed. Investigation outputs are
under `/tmp/lego-*`; no simulator source
was changed. The subsequent setup created `Sakib/.venv` and `Sakib/curobo`;
its planning smoke report is `/tmp/lego-venv-curobo-check/step1.json`.

The first GPU run compiles Warp kernels and can take minutes. Running GPU
checks inside Codex's sandbox can hide the NVIDIA driver; the successful GPU
check ran with permission outside the sandbox. Run the commands above in a
normal desktop terminal, or allow GPU/network/cache access when using Codex.
