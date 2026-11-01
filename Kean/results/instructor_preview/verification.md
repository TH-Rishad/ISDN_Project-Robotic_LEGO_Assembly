# Acceptance verification

Mode: plan-check

Exit 0 is a complete pass. Exit 1 means a criterion FAIL. Exit 2 means invalid or incomplete evidence (UNVERIFIED) with no FAIL.
Saved `ok` and `full_acceptance` flags are ignored. Contact events do not change the three criteria.

## Criterion 1: Direct structure input — PASS

- PASS: direct structure ids, no image input

## Criterion 2: Generated sequence and supports — PASS

- PASS: arch: 11 bricks, supports [10]
- PASS: S3: 16 bricks, supports [11, 13, 14]

## Criterion 3: Two braced dual-arm assemblies — UNVERIFIED

- UNVERIFIED: execution was not run

## Contacts

- none recorded

## Hardware

Newton simulation only. A physical robot demonstration is not in this package.

Exit code: 2
