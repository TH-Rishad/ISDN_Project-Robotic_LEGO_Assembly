# Acceptance verification

Mode: evidence

Exit 0 is a complete pass. Exit 1 means a criterion FAIL. Exit 2 means invalid or incomplete evidence (UNVERIFIED) with no FAIL.
Saved `ok` and `full_acceptance` flags are ignored. Contact events do not change the three criteria.

## Criterion 1: Direct structure input — PASS

- PASS: direct structure ids, no image input

## Criterion 2: Generated sequence and supports — PASS

- PASS: S3: 16 bricks, supports [11, 13, 14]
- PASS: arch: 11 bricks, supports [10]

## Criterion 3: Two braced dual-arm assemblies — PASS

- PASS: bound to revision, dirty hashes, config, seed and simulator
- PASS: every planned brick placed once
- PASS: explicit completion and no execution failure
- PASS: step 11 held before and during insertion, then released
- PASS: step 13 held before and during insertion, then released
- PASS: step 14 held before and during insertion, then released
- PASS: every planned brick placed once
- PASS: explicit completion and no execution failure
- PASS: step 10 held before and during insertion, then released

## Contacts

- arch: {'neighbour_rub': 14, 'unintended': 6, 'penetration': 4}
- S3: {'neighbour_rub': 17, 'unintended': 11, 'penetration': 5}

## Hardware

Newton simulation only. A physical robot demonstration is not in this package.

Ignored summary flags: ok, full_acceptance.

Exit code: 0
