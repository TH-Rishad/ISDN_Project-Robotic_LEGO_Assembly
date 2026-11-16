# Acceptance and dual-builder evidence inventory

Current packages stay at `results/` root. Failed and intermediate runs are under
`results/archive/` (see `archive/README.md`).

## Current packages (do not overwrite)

| Package | Role | Status |
|---|---|---|
| `instructor_run_001` + `verification_instructor_run_001` | Single-cell arch + S3 acceptance | verifier exit 0 |
| `dual_arch_s3_012` | Dual-builder arch + S3 (no stud-fit) | `verify_dual` exit 0 |
| `dual_pyramid_s3_006` | Dual-builder pyramid + S3 (no stud-fit) | `verify_dual` exit 0 |
| `dual_pyramid_s3_studfit_030` | Dual-builder pyramid + S3 **with** `--stud-fit` | `verify_dual` `ok`/`full_build` true (job 2313901, rev `6be3fc9`) |

## Legacy acceptance (archived)

`archive/acceptance_legacy/acceptance` holds both structures in one directory and
has no aggregation manifest. Plan hashes match the separate runs. Arch frames
match (9139) but S3 frames do not (14649 vs 14581). No provenance.json or source
revision is stored, so the combined summary is unverified.

### archive/acceptance_legacy/acceptance

- provenance: False
- cell rows: 2
- episodes: 27
- S3 supports [11, 13, 14] (plan brace_required_count 3, 16 bricks)
- arch supports [10] (plan brace_required_count 1, 11 bricks)
- summary arch 11/11 frames 9139 done True
- summary S3 16/16 frames 14649 done True

### archive/acceptance_legacy/acceptance_arch_wriggle2

- provenance: False
- cell rows: 1
- episodes: 11
- arch supports [10] (plan brace_required_count 1, 11 bricks)
- summary arch 11/11 frames 9139 done True

### archive/acceptance_legacy/acceptance_s3_wriggle

- provenance: False
- cell rows: 1
- episodes: 16
- S3 supports [11, 13, 14] (plan brace_required_count 3, 16 bricks)
- summary S3 16/16 frames 14581 done True

Combined versus separate runs: {"arch": {"same_frames": true, "combined_frames": 9139, "separate_frames": 9139, "same_supports": true}, "S3": {"same_frames": false, "combined_frames": 14649, "separate_frames": 14581, "same_supports": true}}

Plan hash matches the separate run: {"S3": true, "arch": true}

## Stud-fit trail (archived)

`archive/dual_pyramid_s3_studfit_failed/` — runs that did not pass `verify_dual`
(mostly full place with `displaced` on brace_search GUARD scrapes, or mid-run
`command_clearance` / `joint_break`). Diagnostics:
`archive/dual_studfit_diag/`. Job logs: `archive/studfit_logs/`.
