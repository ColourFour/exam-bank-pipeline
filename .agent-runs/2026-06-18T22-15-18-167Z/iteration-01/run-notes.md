# Iteration 01 Run Notes

## Initial repo state

`git status --short` at the start of the run showed pre-existing dirty state:

```text
 M src/exam_bank/asset_manifest.py
 M src/exam_bank/exporters.py
 M src/exam_bank/image_rendering.py
 M src/exam_bank/mark_scheme_models.py
 M src/exam_bank/mark_schemes.py
 M src/exam_bank/models.py
 M src/exam_bank/pdf_extract.py
 M src/exam_bank/pipeline.py
 M tests/test_image_rendering.py
 M tests/test_output_contract.py
 M tests/test_pdf_extract.py
 M tests/test_pipeline_reconciliation.py
 M tests/test_question_detection.py
?? .agent-loop/
?? agentic-loop-template/
?? src/exam_bank/image_alignment_controller.py
?? src/exam_bank/image_alignment_metrics.py
?? tests/test_image_alignment_controller.py
```

The selected slice is limited to asset-reference validation. It uses the already-dirty `src/exam_bank/asset_manifest.py` area and one existing test file, and does not revert or overwrite unrelated source/test changes.
