# Acceptance evidence inventory

results/acceptance holds both structures in one directory and has no aggregation manifest. Plan hashes match the separate runs. Arch frames match (9139) but S3 frames do not (14649 vs 14581). No provenance.json or source revision is stored, so the combined summary is unverified.

## acceptance

- provenance: False
- cell rows: 2
- episodes: 27
- S3 supports [11, 13, 14] (plan brace_required_count 3, 16 bricks)
- arch supports [10] (plan brace_required_count 1, 11 bricks)
- summary arch 11/11 frames 9139 done True
- summary S3 16/16 frames 14649 done True

## acceptance_arch_wriggle2

- provenance: False
- cell rows: 1
- episodes: 11
- arch supports [10] (plan brace_required_count 1, 11 bricks)
- summary arch 11/11 frames 9139 done True

## acceptance_s3_wriggle

- provenance: False
- cell rows: 1
- episodes: 16
- S3 supports [11, 13, 14] (plan brace_required_count 3, 16 bricks)
- summary S3 16/16 frames 14581 done True

Combined versus separate runs: {"arch": {"same_frames": true, "combined_frames": 9139, "separate_frames": 9139, "same_supports": true}, "S3": {"same_frames": false, "combined_frames": 14649, "separate_frames": 14581, "same_supports": true}}

Plan hash matches the separate run: {"S3": true, "arch": true}
