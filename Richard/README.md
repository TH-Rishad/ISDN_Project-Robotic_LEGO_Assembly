# Richard — Workspace

Personal workspace for **Richard** (repo owner: TH-Rishad).

## Write boundary (hard rule)

| Path | Permission |
| --- | --- |
| `Richard/**` | **Write** — all work happens here |
| `Kean/**`, `Sakib/**`, `Tanvir/**`, `final_report/**` | **Read-only** |
| Repo-root files (`README.md`, `RUNNING_ON_THIS_PC.md`, `.gitignore`, …) | **Read-only** |

A `pre-commit` hook enforces this: any staged path outside `Richard/` aborts the
commit. Bypass only with an explicit maintainer instruction.

## Layout

```
Richard/
├── .gitignore      # personal ignores (mirrors Kean/.gitignore convention)
├── motion/         # motion / control logic
├── scripts/        # runnable entry points
├── tests/          # unit + acceptance tests
├── results/        # experiment outputs (JSON tracked, .npz ignored)
├── report/         # documents, figures, final write-ups
└── support/        # helper assets and reference material
```

Local-only files (never pushed): `AGENTS.md`, `config/`, `.cursor/`, `richard.id`.

## Branch

Work happens on `feat/richard-workspace`, branched from `main`.

```bash
git switch feat/richard-workspace
```

## Environment

See `../RUNNING_ON_THIS_PC.md` for the machine-specific Python interpreters and
launcher commands. Note that file describes a Linux checkout; this working copy
is on Windows, so use the interpreter paths that exist on the target host.
