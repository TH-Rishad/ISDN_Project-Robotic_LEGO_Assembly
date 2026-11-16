# Archived Kean runs

Intermediate, diagnostic, and failed packages moved here after
`dual_pyramid_s3_studfit_030` passed verification. Paths below are relative to
this directory.

| Subdirectory | Contents |
|---|---|
| `acceptance_legacy/` | Older acceptance dirs without provenance; `verification_acceptance` |
| `checks/` | Historical M0.5 / motion check timestamps |
| `dual_arch_s3_trail/` | Dual arch+S3 attempts other than certified `../dual_arch_s3_012` |
| `dual_pyramid_s3_trail/` | Pyramid+S3 attempts other than certified `../dual_pyramid_s3_006` |
| `dual_pyramid_s3_studfit_failed/` | Full `--stud-fit` pyramid+S3 runs that did not pass `verify_dual` (001–029 trail) |
| `dual_studfit_diag/` | Short / stepped stud-fit diagnostics |
| `dual_probes/` | Contact audits, late-support, pick probes, support-tie, etc. |
| `m05/` | M0.5 / collision-gate experiment outputs |
| `local/` | Local playback-only packages |
| `studfit_logs/` | Slurm stdout/stderr for stud-fit jobs |

Do not promote an archive directory back to a current package without
re-running `verify_dual` / `verify_acceptance` and updating the workspace docs.
