# Output Storage Optimization Plan

## Current Size
- Total scanned size: `1.34 GiB` (`1434060165` bytes).
- Exact duplicate wasted size: `43.08 MiB`.
- Conservative reclaimable size: `41.95 MiB`.

## Largest Duplicate Sources
- `output`: `1.32 GiB`
- `output/topic_packets`: `451.40 MiB`
- `output/topic_packets/review_required`: `381.55 MiB`
- `output/topic_packets/review_required/p3`: `102.97 MiB`
- `output/p3`: `101.44 MiB`
- `output/p1`: `101.13 MiB`
- `output/p5`: `98.94 MiB`
- `output/topic_packets/review_required/p1`: `97.03 MiB`
- `output/p4`: `96.27 MiB`
- `output/topic_packets/review_required/p5`: `93.38 MiB`
- `output/review`: `91.20 MiB`
- `output/archive`: `90.85 MiB`
- `output/archive/generated_cleanup_20260513T233456Z`: `90.85 MiB`
- `output/topic_packets/review_required/p4`: `88.17 MiB`
- `output/run_status`: `76.19 MiB`

## Canonical Directories To Keep
- `output/json/question_bank.json`
- `output/json/asset_manifest.v1.json`
- `output/p*/<paper>/questions/*.png`
- `output/p*/<paper>/mark_scheme/*.png`
- `output/asterion/exports/latest/*.json` as lightweight references, not copied images

## Generated Or Rebuildable Candidates
- `output/candidates/ocr/*`
- `output/codex_text_extraction_candidate*`
- `output/codex_text_extraction_targeted/*`
- `output/archive/generated_cleanup_*`
- `output/topic_packets/*/topic_packet.pdf`
- `output/audits/*`
- `output/run_status/*`

## Exact Duplicate Groups
- `11504684e965` `report`: `6` files, `1.74 MiB` wasted, canonical `output/audits/asterion_student_runtime_safe_loop/iteration_003/runtime_safe_candidate_results.csv`
- `8399f0db1cfe` `report`: `6` files, `1.60 MiB` wasted, canonical `output/audits/asterion_content_lab_loop/iteration_002/mark_evidence_gap_report.csv`
- `d824282df5b2` `report`: `4` files, `859.21 KiB` wasted, canonical `output/audits/asterion_content_lab_loop/iteration_004/mark_evidence_gap_report.csv`
- `4d372d641053` `report`: `3` files, `711.06 KiB` wasted, canonical `output/audits/asterion_student_runtime_safe_loop/iteration_002/runtime_safe_candidate_results.csv`
- `6331e86043fa` `report`: `3` files, `710.88 KiB` wasted, canonical `output/audits/asterion_student_runtime_safe_loop/iteration_006/runtime_safe_candidate_results.csv`
- `820e39f361f9` `report`: `3` files, `710.65 KiB` wasted, canonical `output/audits/asterion_student_runtime_safe_loop/iteration_005/runtime_safe_candidate_results.csv`
- `5bc7bedc6db9` `report`: `3` files, `706.94 KiB` wasted, canonical `output/audits/asterion_student_runtime_safe_loop/iteration_004b/runtime_safe_candidate_results.csv`
- `9624186819c3` `report`: `3` files, `658.30 KiB` wasted, canonical `output/audits/asterion_content_lab_loop/iteration_002/p3_candidate_inventory.csv`
- `753869b192a0` `report`: `3` files, `650.46 KiB` wasted, canonical `output/audits/asterion_content_lab_loop/iteration_002/skill_mapping_gap_report.csv`
- `56ffa87cd062` `report`: `3` files, `607.44 KiB` wasted, canonical `output/audits/asterion_content_lab_loop/iteration_004/p3_candidate_inventory.csv`
- `0fb55f88a7db` `report`: `9` files, `359.16 KiB` wasted, canonical `output/audits/asterion_content_lab_loop/iteration_002/p3_sample_frame.csv`
- `b853e457be1b` `report`: `2` files, `331.16 KiB` wasted, canonical `output/audits/asterion_student_runtime_safe_loop/iteration_001/runtime_safe_classification.csv`
- `54f8483ae05d` `json`: `2` files, `325.49 KiB` wasted, canonical `output/review/asterion_runtime_safe_loop_005/auto_review_decisions.jsonl`
- `f0ab443cc19d` `report`: `2` files, `298.04 KiB` wasted, canonical `output/audits/asterion_content_lab_loop/iteration_003b/mark_evidence_gap_report.csv`
- `e9456d7b36c4` `json`: `3` files, `284.35 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12summer25/questions.json`
- `f072cd7d80b5` `json`: `3` files, `283.94 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_33autumn24/questions.json`
- `00052c3923ee` `report`: `2` files, `283.66 KiB` wasted, canonical `output/audits/asterion_content_lab_loop/iteration_004/skill_mapping_gap_report.csv`
- `474202972bf6` `json`: `3` files, `282.89 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_15autumn25/questions.json`
- `dc100048c8df` `json`: `3` files, `277.13 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_13autumn25/questions.json`
- `4d9cb82a219c` `json`: `3` files, `276.31 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_32autumn25/questions.json`
- `5bc1ff5e4b68` `json`: `3` files, `274.88 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12summer21/questions.json`
- `97bfd5069d59` `json`: `3` files, `273.65 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_32autumn24/questions.json`
- `18cdb92bb5d7` `json`: `3` files, `272.07 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_11summer23/questions.json`
- `f3f2aca083e8` `json`: `3` files, `270.21 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_33summer25/questions.json`
- `bbf31abed8ae` `json`: `3` files, `268.00 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_35summer25/questions.json`
- `7c6893b1defd` `json`: `3` files, `265.89 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_32summer25/questions.json`
- `fd204a68ac0d` `report`: `2` files, `265.17 KiB` wasted, canonical `output/audits/asterion_content_lab_loop/iteration_006/full_pool_final_candidate_results.csv`
- `768e27a95ae2` `json`: `3` files, `264.72 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12autumn21/questions.json`
- `2cf1ec12fc39` `json`: `3` files, `264.42 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_13summer25/questions.json`
- `5d71a2e85d80` `json`: `3` files, `264.02 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12autumn25/questions.json`
- `ecd8587b6a0f` `json`: `3` files, `263.09 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_13autumn24/questions.json`
- `2f0a1856e0e6` `json`: `3` files, `262.07 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12spring25/questions.json`
- `f7bdf60afc93` `json`: `3` files, `259.95 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_11autumn25/questions.json`
- `18656da8b498` `json`: `3` files, `258.34 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12autumn24/questions.json`
- `1a0a60487ed0` `json`: `3` files, `257.84 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_31summer25/questions.json`
- `76fade316593` `json`: `3` files, `257.51 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_13summer24/questions.json`
- `3f3b4319d533` `json`: `3` files, `257.36 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_33autumn25/questions.json`
- `d5fcb5368df8` `json`: `3` files, `256.75 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_33summer24/questions.json`
- `f93a417be0e9` `json`: `3` files, `256.56 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12autumn22/questions.json`
- `20a0bd2ad145` `json`: `3` files, `254.62 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_11summer24/questions.json`
- `80352dffa9d6` `json`: `3` files, `253.41 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_31autumn25/questions.json`
- `ce259ea9fa07` `json`: `3` files, `253.09 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12summer23/questions.json`
- `0d7ba5c30d98` `json`: `3` files, `252.83 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12summer22/questions.json`
- `26c8656f2ad0` `json`: `3` files, `251.43 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_35autumn25/questions.json`
- `e88f20735270` `json`: `3` files, `251.33 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12autumn23/questions.json`
- `ec980ac10742` `json`: `3` files, `249.92 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_11autumn24/questions.json`
- `d3919710c1e9` `json`: `3` files, `247.97 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_31autumn23/questions.json`
- `bc0a2e3fd819` `json`: `3` files, `247.66 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_32spring25/questions.json`
- `c1b8f078f9d5` `json`: `3` files, `247.07 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_11autumn23/questions.json`
- `de5d20e950a6` `json`: `3` files, `246.31 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_15summer25/questions.json`
- `a65e9438d393` `json`: `3` files, `245.57 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_31autumn24/questions.json`
- `a3c807f42676` `json`: `3` files, `244.58 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_13summer22/questions.json`
- `9bb88a6fe015` `json`: `3` files, `244.58 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_11autumn22/questions.json`
- `15fa93c03d26` `json`: `3` files, `244.41 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_32autumn23/questions.json`
- `be60ead0cd4d` `json`: `3` files, `244.05 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_11summer25/questions.json`
- `34837b71c303` `json`: `3` files, `243.71 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_32summer23/questions.json`
- `20b3de0e2d65` `json`: `3` files, `242.47 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_13summer21/questions.json`
- `1751c5e783d8` `json`: `3` files, `242.01 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_33summer22/questions.json`
- `97e58d3e5b8d` `json`: `3` files, `241.11 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_13autumn23/questions.json`
- `f915445727c0` `json`: `3` files, `240.47 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_11autumn21/questions.json`
- `692fab4241d7` `json`: `3` files, `240.39 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_33summer23/questions.json`
- `4b84d51738f1` `json`: `3` files, `240.19 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_32summer21/questions.json`
- `310d75179411` `json`: `3` files, `240.15 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12spring21/questions.json`
- `d0eeed6a6f0d` `json`: `3` files, `239.44 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_32spring24/questions.json`
- `9f382a3af3c1` `json`: `3` files, `239.39 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_32summer24/questions.json`
- `88dc2c96af99` `json`: `3` files, `239.07 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_13autumn22/questions.json`
- `d9455288c6db` `json`: `3` files, `238.55 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_32autumn22/questions.json`
- `9fbeaaa3a9b5` `json`: `3` files, `236.71 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12summer24/questions.json`
- `eaab97df8fdb` `json`: `3` files, `236.16 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_33autumn23/questions.json`
- `046e7e9d1647` `json`: `3` files, `235.99 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_32spring23/questions.json`
- `0e01451ab1ac` `json`: `3` files, `235.59 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_31autumn22/questions.json`
- `c2720bf6dadb` `json`: `3` files, `234.80 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_32autumn21/questions.json`
- `c81617c25e0b` `json`: `3` files, `234.76 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p3_33autumn21/questions.json`
- `58d175671712` `json`: `3` files, `234.24 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12spring24/questions.json`
- `8fad18833bd3` `json`: `3` files, `233.39 KiB` wasted, canonical `output/run_status/20260526T051640Z-standard-2c3ecc00/batch_artifacts/p1_12spring22/questions.json`
- Additional exact duplicate groups: `143`. See `reports/output_storage_duplicate_audit.v1.json`.

## Recommended Implementation Steps
1. Keep canonical images under `output/p*/...` as the source of truth.
2. Keep downstream JSON exports path-compatible but prefer `*_asset_id` fields and canonical relative paths.
3. Use `output/json/asset_manifest.v1.json` as an index for asset lookup and integrity checks.
4. Use hard-delete mode only for exact duplicate non-canonical files after reviewing `reports/output_storage_delete_manifest.v1.json`.
5. Regenerate topic packets and candidate outputs instead of storing copied image trees long term.

## Risks
- Some archive and candidate folders may still carry historical comparison evidence.
- Standalone PDFs intentionally embed images and should not be rewritten as path references.
- Canonical image duplicates can be real duplicate evidence across papers; do not remove them without explicit remap review.

## Regeneration Commands
- `.venv/bin/python -m exam_bank.cli asterion-export --input output/json/question_bank.json --artifact-root output`
- `.venv/bin/python -m exam_bank.cli asterion-content-lab-candidates --input output/json/question_bank.json --artifact-root output`
- `.venv/bin/python -m exam_bank.cli topic-packets --input output/json/question_bank.json --artifact-root output`
- `.venv/bin/python scripts/audit_output_storage.py --dry-run`
- `.venv/bin/python scripts/audit_output_storage.py --apply-delete`
- `.venv/bin/python scripts/validate_asset_references.py`
