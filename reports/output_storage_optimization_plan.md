# Output Storage Optimization Plan

## Current Size
- Total scanned size: `979.54 MiB` (`1027118067` bytes).
- Exact duplicate wasted size: `30.90 MiB`.
- Conservative reclaimable size: `18.80 MiB`.

## Largest Duplicate Sources
- `output`: `979.54 MiB`
- `output/stats`: `291.46 MiB`
- `output/pm3`: `226.89 MiB`
- `output/pm1`: `226.24 MiB`
- `output/mechanics`: `96.63 MiB`
- `output/run_status`: `86.17 MiB`
- `output/json`: `52.09 MiB`
- `output/run_status/20260618T190242Z-standard-9ce75cdf`: `43.38 MiB`
- `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts`: `43.22 MiB`
- `output/run_status/canonical-regeneration-2026-06-19`: `42.79 MiB`
- `output/run_status/canonical-regeneration-2026-06-19/batch_artifacts`: `42.52 MiB`
- `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_12summer25`: `158.31 KiB`
- `output/run_status/canonical-regeneration-2026-06-19/batch_artifacts/pm1_12summer25`: `157.34 KiB`
- `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_33winter24`: `154.16 KiB`
- `output/run_status/canonical-regeneration-2026-06-19/batch_artifacts/pm3_33winter24`: `154.16 KiB`

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
- `b0b802df06e9` `image`: `2` files, `351.02 KiB` wasted, canonical `output/pm3/pm3_2010_w10_31_ms_q07_markscheme.png`
- `16a6a7dc00dd` `image`: `2` files, `337.70 KiB` wasted, canonical `output/pm3/pm3_2020_w20_31_ms_q11_markscheme.png`
- `1d7da3e146cc` `image`: `2` files, `313.25 KiB` wasted, canonical `output/pm3/pm3_2016_w16_31_ms_q08_markscheme.png`
- `f6fdb655bbc2` `image`: `2` files, `291.04 KiB` wasted, canonical `output/pm3/pm3_2020_w20_31_ms_q07_markscheme.png`
- `9684f1493498` `image`: `2` files, `259.58 KiB` wasted, canonical `output/pm3/pm3_2011_w11_31_ms_q07_markscheme.png`
- `c2372c461c93` `image`: `2` files, `251.56 KiB` wasted, canonical `output/pm3/pm3_2020_w20_31_ms_q09_markscheme.png`
- `926fc5e0558d` `image`: `2` files, `238.68 KiB` wasted, canonical `output/pm3/pm3_2010_w10_31_ms_q06_markscheme.png`
- `a3ad450e10fd` `image`: `2` files, `225.46 KiB` wasted, canonical `output/pm3/pm3_2020_w20_31_ms_q03_markscheme.png`
- `5ac1fb175c48` `image`: `2` files, `225.16 KiB` wasted, canonical `output/pm3/pm3_2010_w10_31_ms_q08_markscheme.png`
- `17c946d8f98a` `image`: `2` files, `221.24 KiB` wasted, canonical `output/pm3/pm3_2020_w20_31_ms_q10_markscheme.png`
- `7cacca97fff2` `image`: `2` files, `217.73 KiB` wasted, canonical `output/pm3/pm3_2018_w18_31_ms_q10_markscheme.png`
- `5268548bef1a` `image`: `2` files, `215.62 KiB` wasted, canonical `output/pm3/pm3_2011_w11_31_ms_q10_markscheme.png`
- `4db3b28c3c38` `image`: `2` files, `198.88 KiB` wasted, canonical `output/pm3/pm3_2017_w17_31_qp_q02_question.png`
- `7f37b53f25ef` `image`: `2` files, `196.44 KiB` wasted, canonical `output/pm3/pm3_2020_w20_31_ms_q01_markscheme.png`
- `2ba480be5847` `image`: `2` files, `191.41 KiB` wasted, canonical `output/pm3/pm3_2016_w16_31_ms_q04_markscheme.png`
- `1843c5542352` `image`: `2` files, `189.52 KiB` wasted, canonical `output/pm3/pm3_2016_w16_31_ms_q10_markscheme.png`
- `a1ec5266d835` `image`: `2` files, `179.18 KiB` wasted, canonical `output/pm3/pm3_2011_w11_31_ms_q06_markscheme.png`
- `04456e264295` `image`: `2` files, `177.98 KiB` wasted, canonical `output/pm3/pm3_2016_w16_31_ms_q09_markscheme.png`
- `e27b8c1f5528` `image`: `2` files, `165.44 KiB` wasted, canonical `output/pm3/pm3_2020_w20_31_ms_q05_markscheme.png`
- `0bdd9dfcf614` `image`: `2` files, `160.67 KiB` wasted, canonical `output/pm3/pm3_2011_w11_31_ms_q02_markscheme.png`
- `3bbb73bee9c5` `image`: `2` files, `160.36 KiB` wasted, canonical `output/pm3/pm3_2016_w16_31_ms_q03_markscheme.png`
- `fc8e2f66ba48` `image`: `2` files, `158.04 KiB` wasted, canonical `output/pm3/pm3_2011_w11_31_ms_q09_markscheme.png`
- `9b89fef60f03` `image`: `2` files, `155.39 KiB` wasted, canonical `output/pm3/pm3_2010_w10_31_qp_q10_question.png`
- `59bf1697d5b8` `image`: `2` files, `155.17 KiB` wasted, canonical `output/pm3/pm3_2016_w16_31_qp_q10_question.png`
- `ccceb843e873` `json`: `2` files, `154.16 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_33winter24/questions.json`
- `94df09c51826` `json`: `2` files, `152.48 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_15winter25/questions.json`
- `2871a464293a` `json`: `2` files, `152.30 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_12summer21/questions.json`
- `1e27a00bd7a6` `json`: `2` files, `151.92 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_11summer23/questions.json`
- `6ba39ac35c11` `json`: `2` files, `151.53 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_13winter25/questions.json`
- `94d2f92c44f6` `json`: `2` files, `150.29 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_32winter24/questions.json`
- `86a2b393e147` `json`: `2` files, `149.64 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_35summer25/questions.json`
- `6b2b4d42621c` `json`: `2` files, `148.58 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_33summer25/questions.json`
- `48e180e67ab6` `json`: `2` files, `146.25 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_13summer25/questions.json`
- `9e618604e1f7` `json`: `2` files, `144.91 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_32summer25/questions.json`
- `cb9e5a4bcca0` `json`: `2` files, `144.74 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_12winter25/questions.json`
- `dc7bfcc1f4b9` `image`: `2` files, `144.63 KiB` wasted, canonical `output/pm3/pm3_2015_w15_31_ms_q09_markscheme.png`
- `1fef4899a42c` `json`: `2` files, `144.12 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_13winter24/questions.json`
- `0a1777c73e38` `json`: `2` files, `143.56 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_12winter22/questions.json`
- `1ecdb72fc05e` `json`: `2` files, `142.99 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_12summer20/questions.json`
- `3569e707d955` `json`: `2` files, `142.74 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_12summer23/questions.json`
- `54f72e665f6a` `image`: `2` files, `142.62 KiB` wasted, canonical `output/pm3/pm3_2011_w11_31_ms_q05_markscheme.png`
- `3c8a4000d362` `json`: `2` files, `142.57 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_11winter20/questions.json`
- `a05dcee75a2b` `json`: `2` files, `142.33 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_12summer22/questions.json`
- `795ea7a82cf7` `json`: `2` files, `142.15 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_11summer19/questions.json`
- `3058a66eaba6` `json`: `2` files, `142.12 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_11winter25/questions.json`
- `d38ae3d5c309` `json`: `2` files, `141.71 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_12winter24/questions.json`
- `51d8704a96a6` `json`: `2` files, `141.61 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_33winter25/questions.json`
- `3100296f7b0c` `json`: `2` files, `141.43 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_12winter23/questions.json`
- `929de807ca7c` `json`: `2` files, `140.83 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_35winter25/questions.json`
- `4faf8e72b0b9` `json`: `2` files, `140.79 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_11summer24/questions.json`
- `8bea9487f5b4` `json`: `2` files, `139.79 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_11winter23/questions.json`
- `8e0119c6b65a` `json`: `2` files, `139.51 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_33summer24/questions.json`
- `d31ae45cc10a` `json`: `2` files, `139.14 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_31winter25/questions.json`
- `fe0585d0af20` `image`: `2` files, `139.14 KiB` wasted, canonical `output/pm3/pm3_2016_w16_31_ms_q05_markscheme.png`
- `8b72a863e233` `json`: `2` files, `138.77 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_11winter24/questions.json`
- `4716c1c96b09` `json`: `2` files, `138.48 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_13summer19/questions.json`
- `c74c58fc8542` `json`: `2` files, `137.76 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_11winter22/questions.json`
- `478eaa8a7aeb` `json`: `2` files, `137.30 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_32winter23/questions.json`
- `a2f7e5127b47` `json`: `2` files, `136.94 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_32summer23/questions.json`
- `ec9962d63e80` `json`: `2` files, `136.43 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_12winter20/questions.json`
- `7fc8d22e5e22` `json`: `2` files, `136.37 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_15summer25/questions.json`
- `8194bd53a8a7` `json`: `2` files, `136.32 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_13winter23/questions.json`
- `8fdd2c6530f9` `json`: `2` files, `135.87 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_13summer17/questions.json`
- `06f8fb8233d9` `json`: `2` files, `135.74 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_13summer22/questions.json`
- `7b0417874fbb` `json`: `2` files, `135.63 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_11summer25/questions.json`
- `11855341b5e0` `json`: `2` files, `135.19 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_31winter23/questions.json`
- `b76e8c8d4189` `json`: `2` files, `135.02 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_13summer21/questions.json`
- `6bb49e310f35` `json`: `2` files, `134.85 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_33summer23/questions.json`
- `64b0ae5c23e4` `json`: `2` files, `134.77 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_13summer24/questions.json`
- `8ecd96ad0270` `json`: `2` files, `134.74 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_13winter19/questions.json`
- `3c939d8e7fa3` `image`: `2` files, `134.62 KiB` wasted, canonical `output/pm3/pm3_2016_w16_31_ms_q06_markscheme.png`
- `87b24fed161b` `json`: `2` files, `134.56 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm3_32summer21/questions.json`
- `ba1105402f7f` `json`: `2` files, `133.94 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_11winter21/questions.json`
- `10c3b5fdec6c` `json`: `2` files, `133.65 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_11winter19/questions.json`
- `09c301f3a3bd` `json`: `2` files, `133.46 KiB` wasted, canonical `output/run_status/20260618T190242Z-standard-9ce75cdf/batch_artifacts/pm1_11winter18/questions.json`
- Additional exact duplicate groups: `223`. See `reports/output_storage_duplicate_audit.v1.json`.

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
