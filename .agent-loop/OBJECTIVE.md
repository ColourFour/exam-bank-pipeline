# Project Objective

Make the exam-bank pipeline produce trustworthy, image-first CAIE 9709 outputs.

Success means every exported question record, canonical question `.png`, and canonical mark-scheme `.png` is aligned to the correct exam item and contains only the content that belongs to that item. The loop should prioritize extraction integrity over new capability: prevent wrong-question images, wrong mark-scheme pairings, neighboring-question crop bleed, orphan images, missing images, duplicate mappings, stale generated artifacts, and cross-question contamination.

The canonical question and mark-scheme PNGs are the source of truth. Text, OCR, topic labels, AI sidecars, and downstream exports are advisory unless an explicit project contract promotes them. A loop iteration is valuable when it adds or tightens a deterministic check, focused repair, review pack, or small implementation fix that makes corrupted or misaligned outputs harder to produce and easier to catch.

Future agents should work in small, verified slices. They should prefer evidence-backed validation of output data and assets before changing extraction behavior, and they should never mark ambiguous output as clean without visual or deterministic evidence.
