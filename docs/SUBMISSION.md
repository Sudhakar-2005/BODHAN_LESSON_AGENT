# Submission status (evidence-based, current)

This report replaces the earlier stale submission notes. It describes the
repository as it stands **before** the outstanding uncommitted work is
committed. No claim below should be read as a clean-clone verification of the
current working tree.

## A. Repository state

- Branch: `assignment/narrated-lesson-Sudhakar-2005`
- HEAD: `f7d1e80792084be283f1ac4a21ac3cb3542b7987`
- Working tree is **dirty**: **12 modified files total**, with **no staged
  changes** and **no new commit containing these changes yet**. HEAD remains
  `f7d1e80792084be283f1ac4a21ac3cb3542b7987`.
- **Seven modified source/test files** (implementation and regression tests):
  - `src/bodhan/document/equations.py` (Phase 6K repeated-span validation and
    Phase 6N LaTeX text-command delimiter preservation)
  - `src/bodhan/pipeline/checkpoint.py` (Phase 6G empty-table checkpoint fix)
  - `src/bodhan/pipeline/orchestrator.py` (Phase 6I finalized export manifest)
  - `src/bodhan/pipeline/stages/export.py` (Phase 6I export manifest helper)
  - `tests/unit/test_checkpoint_integrity.py`
  - `tests/unit/test_math_safe_translation.py`
  - `tests/unit/test_pipeline_e2e.py`
- **Five modified documentation/configuration files**:
  - `.gitignore` (generated `out/` and `graphify-out/` are ignored)
  - `README.md`
  - `docs/ARCHITECTURE.md`
  - `docs/SUBMISSION.md`
  - `docs/VIDEO_SCRIPT.md`
- `graphify-out/` (generated graph output) and `out/` (generated assignment
  output copies) remain present locally and are intentionally ignored; no
  generated artifact has been deleted or moved.
- This report is being prepared **before** those changes are committed. A clean
  clone of HEAD contains **none** of the twelve modifications — neither the
  seven implementation/test fixes nor the five documentation/configuration
  updates.

## B. Implemented pipeline

The ordered, checkpointed pipeline is defined in
`src/bodhan/pipeline/orchestrator.py`:

1. `ingest` — rasterise PDF pages (or read page images) to PNG; never read the
   embedded PDF text layer; enforce the 5–30 page range in the assignment CLI.
2. `preprocess` — deskew/contrast-clean the page images.
3. `extract` — Bodhan Indic-OCR only; per-page blocks with IDs, bbox,
   confidence and reading order.
4. `structure` — deterministic label mapping; Gemini only for `UNKNOWN`
   labels; document intelligence (suppression, continuity, tables, figures).
5. `quality` — Gemini audit producing flags/reasons only, never rewritten text.
6. `translate` — Bodhan Indic-Translate only; math spans and LaTeX text-command
   delimiters are withheld from the provider and restored locally.
7. `compose` — English-only Gemini plan; every scaffold string is translated
   through Indic-Translate before entering the lesson.
8. `narrate` — Bodhan Indic-Speak only; WAV segments, resume at first missing
   segment, terminal budget errors are never retried.
9. `transcribe` — Bodhan Indic-Transcribe only; ≤30 s WAV splits, verbatim
   per-segment checkpoints.
10. `export` — local copy of lesson Markdown/HTML, section audio, quality
    report and the finalized manifest into `data/exports/<chapter>/`.

The assignment CLI (`python -m bodhan_lesson`) runs one language at a time and
copies the export into `out/<lang>/`. `--source-lang` is metadata only: the
Bodhan OCR and Translate endpoints take no source-language field.

## C. Verified language evidence

All three Chapter 6 runs used the same official 9-page English NCERT Class 9
Maths Chapter 6 PDF (`data/demo/inputs/ncert-class9-maths-ch6-lines-and-angles-official.pdf`).
Every run completed all ten stages and produced exported Markdown, HTML,
section audio, quality reports and manifests.

| Language | Source | Sections narrated | Sections transcribed | CER macro | CER micro |
|---|---|---:|---:|---:|---:|
| Hindi | NCERT Class 9 Maths Chapter 6, English source | 9 | 9 | 34.7211% | 50.8712% |
| Tamil | Same Chapter 6 source | 9 | 9 | 34.1256% | 47.5175% |
| Bengali | Same Chapter 6 source | 8 | 8 | 40.4301% | 52.7450% |

Bengali has eight narratable sections because the composed Bengali lesson has
eight bodied sections. This is the correct lesson structure, not missing audio.

For each language the following exist:
- `data/work/ncert-class9-maths-ch6-lines-and-angles-official-<lang>/`
  with all stage artifacts, `evaluation/cer/`, `audio/full_lesson.wav`,
  `audio/timeline.json`, `audio/assembly_manifest.json` and `report.json`;
- `data/exports/ncert-class9-maths-ch6-lines-and-angles-official-<lang>/`;
- `out/<lang>/lesson.md`, `lesson.html`, `audio/`, `quality_report.json` and
  `manifest.json`.

The Hindi report records commit `ecc3b37e…` (it was generated before the Phase
6F commit); the Tamil and Bengali reports record `f7d1e80…` and honestly mark
`working_tree_clean=false`.

## D. Second-source attempt (not complete)

- Candidate: official NCERT Class 9 Maths Chapter 5 Hindi-medium PDF,
  `data/demo/inputs/ncert-class9-maths-ch5-rekha-aur-kon-hindi-official.pdf`
  (14 pages).
- SHA-256:
  `3f6d594f3a6f91d2fd202758eb60b53acb93d9b51f2d91cf496ce351e669160e`.
- Ingest and preprocessing completed. Indic-OCR succeeded on the first five
  pages (61 blocks) but repeatedly returned provider-side HTTP 502 on page 6:
  *"the recognizer did not return a usable transcription for 1 of 10 blocks
  (order 12: repetitive); the page was not transcribed"*.
- The pipeline stopped before translation, narration, transcription and export.
  **No second-source target-language output exists.** One controlled resume
  reused the five cached pages and reproduced the same failure; no further
  retry is recommended without provider guidance.

This is **not** a successful second-source validation.

## E. Known limitations

- CER is inflated because the reference is the exact Indic-Speak input, which
  retains LaTeX commands, `$` delimiters and digits that are not necessarily
  spoken verbatim. It measures markup-to-speech divergence as well as STT
  quality; no reference text was modified to improve the score.
- Headerless STT 429 responses are terminal under the existing retry policy;
  controlled, spaced resume runs were required. Stage-level pacing for such
  rate limits is not implemented.
- Some provider telemetry is unavailable: request IDs, OCR/translate/latency
  details for some stages, usage for Indic-Speak/Indic-Transcribe, and any
  monetary cost. `report.json` records these as null with explicit reasons.
- The Chapter 5 OCR 502 remains unresolved.
- No demo video has been recorded; `docs/VIDEO_SCRIPT.md` is a script only.
- No pull request or submission email has been created.

## F. Verification evidence

- Latest verified suite: **437 tests passed** in Phase 6N. No source or test
  changes were made after that run.
- Static checks were green: Ruff, Ruff format, Mypy and `uv lock --check --offline`.
- Fresh-clone verification passed at HEAD `f7d1e80…` during Phase 6F-R
  (`uv sync --frozen`, tests, static checks, CLI help, math-safe tests).
  **The twelve uncommitted files (seven implementation/test changes and five
  documentation/configuration updates) made after that verification still
  require a new fresh-clone verification once they are committed.**
- Protected Chapter 6 hashes (hi/ta/bn outputs, reports and full-lesson WAVs)
  were rechecked after each later phase and remain unchanged.
