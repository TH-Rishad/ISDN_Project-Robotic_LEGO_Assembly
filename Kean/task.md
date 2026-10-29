# Kean's remaining tasks: instructor acceptance

Reviewed 2026-10-29 on Kean's Mac (CPU). Work inside `Kean/`; reuse Sakib's planner, bracing, cell, recorder and environment, and the existing Tanvir scripted-wriggle integration.

## Completion criteria and scope

1. Accept a target structure directly; image conversion is unnecessary.
2. Automatically generate the assembly sequence and determine temporary-support locations.
3. Autonomously assemble two different structures with both arms; each has at least eight LEGO bricks and at least one section requiring temporary support.

These override the broader master-plan priorities. Do not resume perception, RL training, the 235-placement gate, or all of plan_v4 merely to finish this assignment. Dedicated instructor verification scripts are part of the remaining deliverable.

## Current progress and evidence

Reviewed `../README.md`, `../Sakib/Docs/plan_v4.md`, `README.md`, `review.md`, `work_record.md`, `acceptance.py`, `scripts/accept.py`, the acceptance portions of `scripts/m05.py`, `tests/test_acceptance.py`, and saved acceptance summaries.

- Direct named-structure inputs and automatic planning already exist. `acceptance.generate` calls the shared planner and checks its generated sequence; live acceptance requests fresh plans through the existing executor. Preserve this path.
- Existing execution is `bash Kean/scripts/accept.sh --live`. The combined saved `results/acceptance/summary.json` reports `scope=all`, `full_acceptance=true`: arch 11/11, support step 10; S3 16/16, support steps 11, 13, 14. Both report `done=true`, no failure, and every required support executed.
- Separate successful arch and S3 summaries also exist. These are simulation records, not evidence of real hardware execution, and were not rerun on this Mac during review.
- Documentation is inconsistent: historical review paragraphs say acceptance is unfinished, parts of README say GPU verification is pending, and older descriptions count two S3 supports. The current combined summary records three. The work record describes a crashed combined attempt followed by separate successes; establish the provenance of the later combined summary before treating it as a reproducible certification package.
- Existing `judge` is useful but insufficient as an independent verifier: it does not enforce at least eight bricks, allows missing `done`, and trusts supplied required-support lists and execution booleans. A summary's `ok` flag alone must not certify completion.
- Saved records contain unintended-contact and penetration events. Report these honestly; passing the three course criteria does not imply passing every master-plan safety/physics gate.

## Prioritized work

### P0 — Reconcile and preserve evidence

- [ ] Inventory the combined and individual runs: generated plans, cell rows, event logs, recordings, source revision and available runtime metadata. Identify missing files and whether the combined result came from one execution or an explicit aggregation.
- [ ] Preserve existing evidence. Use a fresh output directory for each new run; never overwrite a successful run by default.
- [ ] Correct current-status sections in Kean's README/review, clearly labeling historical notes. Derive support counts from actual plans rather than hard-coding two or three.

### P1 — Dedicated instructor verification (CPU)

Implement `Kean/scripts/verify_acceptance.py` with a thin `verify_acceptance.sh` launcher if needed. These names and commands are proposed deliverables, not existing tools.

- [ ] Provide a plan-check mode that loads direct structure inputs and invokes existing generation. Save input definitions, generated order, support targets/poses and planner configuration. No image input, hand-authored order or manually inserted support steps.
- [ ] Provide an offline evidence mode accepting an explicit run directory. Recompute the verdict from plans and execution records; do not trust stored `ok` or `full_acceptance` flags.
- [ ] Check two distinct structure definitions; at least eight unique target bricks each; complete sequence coverage without duplicates; valid generated support locations; and at least one required support per structure.
- [ ] Cross-check each run against its saved plan and target: all expected bricks placed, no missing/extra placements, explicit completion and no execution failure. Require every generated support to have been executed. Use available per-frame/phase records to establish Arm A held support before and during Arm B insertion, then released/retracted. If the recorder lacks this evidence, extend Kean's adapter recording rather than fabricate it.
- [ ] Bind evidence to input/plan hashes, source revision plus dirty-file hashes, configuration, seed and simulator/runtime identity. Report missing provenance as unverified, not a pass. Hashes establish consistency, not independent proof of execution.
- [ ] Emit human-readable `verification.md` and machine-readable `verification.json`, with a separate PASS/FAIL/UNVERIFIED for each criterion and links to evidence. Exit zero only for a complete pass; distinguish invalid evidence and unmet criteria with documented nonzero codes.
- [ ] A dry run can verify input/planning only: criterion 3 must remain UNVERIFIED. A one-structure run must never pass the full assessment. Keep contact findings visible without silently redefining course requirements.
- [ ] Add focused negative tests: missing/duplicate structure, fewer than eight bricks, altered or missing plan, missing completion, missing support, support held at the wrong time, incomplete placement and misleading saved success flags. Test the verifier without GPU imports.

### P2 — Reproducible execution and evidence capture (HPC)

- [ ] Reuse `scripts/accept.sh` and `m05.run_job`; do not build another simulator or planner. Add only the orchestration/recording needed by the verifier.
- [ ] Run arch and S3 with fresh generated plans, strict motion and recording in a new directory using Sakib's supplied GPU environment. Capture stdout/stderr and process exit status, including abnormal termination.
- [ ] Make partial completion durable: save each structure's result before starting the next. If combining separate runs, explicitly record their provenance and compatibility rather than relabeling an individual summary.
- [ ] Run the independent verifier on the produced package. Fix only failures preventing these three criteria or trustworthy verification. Preserve existing collision guards and support handshakes.
- [ ] Include a viewable recording or documented replay command showing both arms and support timing. State explicitly that the evidence is from Newton simulation; if physical robots are required by the instructor, hardware demonstration remains a separate unmet requirement.

Existing execution commands, from the repository root:

```bash
python3 Kean/scripts/accept.py --dry-run --out Kean/results/instructor_preview
bash Kean/scripts/run.sh Kean/scripts/check_env.py --gpu
bash Kean/scripts/accept.sh --live --out Kean/results/instructor_run_001
```

Use a new run suffix if the directory already exists. GPU checks/live execution belong on HPC; report missing environment prerequisites to Sakib.

### P3 — Instructor handoff

- [ ] Document a short CPU-only verification command and a separate GPU execution command, prerequisites, expected outputs and exit codes in Kean's README.
- [ ] Package the two target definitions, plans, execution evidence and verification reports with usable relative paths. Ensure collaborators can obtain required artifacts; ignored local results are not a shared submission.
- [ ] Finish when both structures satisfy all three criteria and the dedicated verifier independently passes the evidence package. Further master-plan work is optional.
