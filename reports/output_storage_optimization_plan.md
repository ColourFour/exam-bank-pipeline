# Output Storage Optimization Plan

## Current Size
- Total scanned size: `1.16 GiB` (`1245251646` bytes).
- Exact duplicate wasted size: `785.98 KiB`.
- Conservative reclaimable size: `0 B`.

## Largest Duplicate Sources
- `output`: `1.15 GiB`
- `output/topic_packets`: `451.40 MiB`
- `output/topic_packets/review_required`: `381.55 MiB`
- `output/topic_packets/review_required/p3`: `102.97 MiB`
- `output/p3`: `101.44 MiB`
- `output/p1`: `101.13 MiB`
- `output/p5`: `98.94 MiB`
- `output/topic_packets/review_required/p1`: `97.03 MiB`
- `output/p4`: `96.27 MiB`
- `output/topic_packets/review_required/p5`: `93.38 MiB`
- `output/archive`: `90.85 MiB`
- `output/archive/generated_cleanup_20260513T233456Z`: `90.85 MiB`
- `output/topic_packets/review_required/p4`: `88.17 MiB`
- `output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate`: `70.89 MiB`
- `output/candidates`: `42.40 MiB`

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
- `48e5c6064df4` `image`: `2` files, `174.73 KiB` wasted, canonical `output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p1/12summer23/questions/q08.png`
- `6b00b7887fdb` `image`: `2` files, `138.21 KiB` wasted, canonical `output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p1/11autumn25/questions/q07.png`
- `7ea1edc49a0c` `image`: `2` files, `106.76 KiB` wasted, canonical `output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p1/12spring22/questions/q10.png`
- `7cb2a99b4ee5` `image`: `2` files, `98.47 KiB` wasted, canonical `output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p1/12spring22/questions/q06.png`
- `34bbcd109bde` `image`: `2` files, `95.56 KiB` wasted, canonical `output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p1/12spring22/questions/q08.png`
- `6f77114d0342` `image`: `2` files, `68.67 KiB` wasted, canonical `output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p1/12summer23/questions/q06.png`
- `829acffef2f2` `image`: `2` files, `57.62 KiB` wasted, canonical `output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p1/12summer23/questions/q05.png`
- `f4fc3c123565` `image`: `2` files, `42.68 KiB` wasted, canonical `output/archive/generated_cleanup_20260513T233456Z/output_ocr_candidate/p1/12spring22/questions/q07.png`
- `7abcfaf273bd` `json`: `2` files, `2.22 KiB` wasted, canonical `output/output_inventory.after_cleanup.json`
- `32a39a1dab95` `report`: `2` files, `1.05 KiB` wasted, canonical `output/output_inventory.after_cleanup.md`

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
