# Kean results layout

Current evidence stays at this directory root. Failed and intermediate runs live
under `archive/` (see `archive/README.md`).

## Current packages (do not overwrite)

| Directory | Role |
|---|---|
| `instructor_run_001` | Single-cell arch + S3 acceptance (gpu12) |
| `verification_instructor_run_001` | Independent verifier report for that package |
| `instructor_preview` | CPU plan-check dry run |
| `dual_arch_s3_012` | Dual-builder arch + S3 (no stud-fit) |
| `dual_pyramid_s3_006` | Dual-builder pyramid + S3 (no stud-fit) |
| `dual_pyramid_s3_studfit_030` | Dual-builder pyramid + S3 with `--stud-fit` (`verify_dual` ok; JSON evidence) |
| `dual_pyramid_s3_studfit_031` | Same path re-run with recording + 8x MP4 (job 2315555 / render 2315938; `verify_dual` ok) |

Index: `evidence_inventory.md` / `evidence_inventory.json`.

## New runs

Write to a fresh `Kean/results/<name>_NNN` directory. Do not `--replace` the
current packages above. Failed or superseded runs should be moved into
`archive/<category>/` rather than left at the root.
