# Submission status (evidence-based)

This report reflects the repository after Phase 6AV. No claim here should be
read as a clean-clone verification of the current working tree.

## A. Repository state

- Branch: `assignment/narrated-lesson-Sudhakar-2005`
- HEAD: `ec22973da45f2aed78891d691e79a6b0ce9ddeae`
  (`fix: preserve math spans and align submission evidence`).
- Working tree is **dirty** with **no staged changes**: 14 modified tracked
  files and 3 untracked test files (17 total).
- Uncommitted implementation/test changes:
  - `src/bodhan/document/equations.py` (scientific-unit protection)
  - `src/bodhan/pipeline/stages/compose.py` (bounded missing-block recovery,
    deterministic placement, recovery provenance)
  - `src/bodhan/pipeline/stages/translate.py` (unit-span counting and metadata)
  - `src/bodhan/adapters/transcribe/indic_transcribe.py`, `src/bodhan/config.py`,
    `src/bodhan/wiring.py`, `src/bodhan/pipeline/stages/transcribe.py`
    (bounded headerless-429 backoff, `TRANSCRIBE_PACING_S`, per-chunk resume)
  - `src/bodhan/assembly/audio.py` (longest-prefix heading disambiguation)
  - `tests/unit/test_math_safe_translation.py`,
    `tests/unit/test_compose_missing_block_recovery.py` (untracked),
    `tests/unit/test_translation_checkpoint_metadata.py` (untracked),
    `tests/unit/test_transcribe_resilience.py` (untracked),
    `tests/unit/test_audio_assembly.py`
- Documentation updated in Phase 6AV: `README.md`, `docs/SUBMISSION.md`,
  `docs/ARCHITECTURE.md`, `docs/VIDEO_SCRIPT.md`.
- `graphify-out/` and `out/` remain present locally and are intentionally
  ignored; no generated artifact was deleted or moved.
- No Git remote is configured. No PR or submission email has been created.

## B. Implemented pipeline

The ordered pipeline in `src/bodhan/pipeline/orchestrator.py`:

1. `ingest` — rasterise PDF pages / read page images; embedded PDF text is
   never used; page range enforced by the assignment CLI.
2. `preprocess` — deskew/contrast-clean images.
3. `extract` — Bodhan Indic-OCR only; per-page block IDs, bbox, confidence,
   reading order and provider metadata.
4. `structure` — deterministic label mapping; Gemini only for `UNKNOWN` labels;
   suppression, continuity, tables and figure associations.
5. `quality` — Gemini flags/reasons only; never rewrites text.
6. `translate` — Bodhan Indic-Translate only; math spans, LaTeX delimiters and
   scientific units are protected, span counts persisted in checkpoints.
7. `compose` — English-only Gemini plan; at most one bounded recovery request
   for omitted IDs, inserted by authoritative source reading order; scaffold
   strings translated by Indic-Translate with provenance.
8. `narrate` — Bodhan Indic-Speak only (WAV); per-segment checkpoint/resume.
9. `transcribe` — Bodhan Indic-Transcribe only; ≤30 s splits, verbatim text,
   per-chunk checkpoints, configurable pacing and bounded 429 backoff.
10. `export` — local copy of lesson Markdown/HTML, section audio, quality
    report and finalized manifest.

`python -m bodhan_lesson` runs one target language at a time and copies the
export into `out/<lang>/`. `--source-lang` is metadata only.

## C. Completed demonstration — NCERT Class 9 Maths Chapter 6

All three Chapter 6 runs used the same official 9-page English NCERT Class 9
Maths Chapter 6 PDF. Every run completed all ten stages.

| Language | Source | Sections narrated/transcribed | Macro CER | Micro CER |
|---|---|---:|---:|---:|
| Hindi | NCERT Class 9 Maths Ch6, English source | 9 / 9 | 34.7211% | 50.8712% |
| Tamil | Same source | 9 / 9 | 34.1256% | 47.5175% |
| Bengali | Same source | 8 / 8 | 40.4301% | 52.7450% |

Bengali has eight narratable sections because the composed lesson has eight
bodied sections; this is correct structure, not missing audio.

For each language:
- `data/work/ncert-class9-maths-ch6-lines-and-angles-official-<lang>/`
  contains all stage artifacts, `evaluation/cer/`, `audio/full_lesson.wav`,
  `audio/timeline.json`, `audio/assembly_manifest.json` and `report.json`;
- `data/exports/ncert-class9-maths-ch6-lines-and-angles-official-<lang>/`
  and `out/<lang>/` contain the exported lesson, HTML, audio, quality report
  and manifest.

## D. Completed second source — NCERT Class 9 Science Chapter 2

Selected second source: official NCERT Class 9 Science Chapter 2, Hindi medium,
20 pages, at `D:\phase6w-temp\ncert-class9-science-hi-ch2-ihsc102.pdf`.
Target language: Tamil. Chapter ID: `ncert-class9-science-ch2-cell-hindi`.

- Completed: ingestion, OCR **20/20 pages (239 blocks)**, structure **224
  blocks**, Tamil translation **224/224 blocks (0 failed)**, composition
  (6 sections, provenance for recovered `p6-b11`), narration **10/10
  sections**, transcription **146/146 chunks**, CER, audio assembly and
  export. All ten stages are `done`.
- **Macro CER 16.3023%, micro CER 15.2483%** (3 Excellent, 4 Needs Review,
  3 Poor), normalization `nfc+ws-collapse-v1`, reference = exact Indic-Speak
  input in `narrate/segments.json`.
- Full lesson audio **4,287.744 s**, PCM16/24 kHz/mono, with a 10-section
  sidecar timeline and assembly manifest.
- The successful transcription run used `TRANSCRIBE_PACING_S=8` and
  `BODHAN_MAX_RETRIES=3` and recorded **zero HTTP 429 responses**. An earlier
  unbounded-burst attempt had been rate-limited; the permanent fix adds
  bounded backoff for headerless 429s, honours `Retry-After`, keeps
  auth/validation/budget failures terminal, and checkpoints each chunk.
- Artifacts: `data/work/ncert-class9-science-ch2-cell-hindi/` and
  `data/exports/ncert-class9-science-ch2-cell-hindi/`.

A separate earlier candidate, NCERT Class 9 Maths Chapter 5 Hindi-medium PDF
(14 pages, SHA-256 `3f6d594f…`), was blocked by a provider-side OCR 502 on
page 6. It is historical evidence only and is not the selected second source.

## E. Known limitations

- CER is inflated by LaTeX, `$` delimiters and digits in the exact TTS input
  that are not necessarily spoken verbatim; no reference was rewritten.
- `report.json` records unavailable provider request IDs, some usage/latency
  values and monetary cost as **null with explicit reasons**; null retry
  fields mean "not persisted", not zero. STT cost is an **estimate**
  (₹0.10 per audio minute), never provider-reported.
- The 224 existing second-source translation checkpoints retain historical
  zero `math_span_count`/`unit_span_count` values. Phase 6AJ fixes newly
  generated checkpoints; historical artifacts were not backfilled.
- `plan_sections` previously rejected overlapping lesson headings where one
  heading is a prefix of a longer one. The permanent fix selects the longest
  matching normalized heading and rejects only equal-length ties; existing
  duplicate-heading validation is unchanged.
- No demo video has been recorded; `VIDEO_SCRIPT.md` is a script only.
- No remote, PR or submission email exists; fresh-clone verification of the
  current uncommitted changes has not been performed.

## F. Verification evidence

- Latest verified suite: **512 tests passed** (Phase 6AV full offline run;
  earlier baseline **508 passed** before four assembler regression tests).
- Ruff, Ruff format, Mypy (`src`) and `uv lock --check` all passed.
- Both demonstrations were audited artifact-by-artifact: all stages `done`,
  CER reports present, audio timelines continuous and hashes consistent.
- Fresh-clone verification passed at earlier commits (`f7d1e80…`, `ec22973…`)
  for the then-committed 12-file state. **The current uncommitted changes
  have not yet been verified in a fresh clone after commit.**
- Protected baselines and source PDFs remained unchanged after every later
  phase.

## G. Requirement-evidence matrix

*No original assignment specification file exists locally; requirements below
are reconstructed from README/docs/tests.*

| Requirement | Evidence | Status | Missing work |
|---|---|---|---|
| CLI contract | pyproject, CLI source, prior `--help` checks | PASS | — |
| Bodhan-only language providers | wiring.py + adapters | PASS | — |
| LLM boundaries | English-only plan validation + tests | PASS | — |
| OCR/translate/narrate/transcribe stages | 4 completed runs | PASS | — |
| Retry/checkpoint/failure isolation | code + live resumes + tests | PASS | — |
| Headerless 429 backoff + configurable pacing | `indic_transcribe.py`, `config.py`, `wiring.py`, tests, zero-429 live run | PASS | — |
| CER / assembly / report | 4 complete sets | PASS | — |
| Multilingual same-chapter evidence | hi/ta/bn Chapter 6 | PASS | — |
| Second-source OCR + translation | 20/20 pages, 224/224 blocks | PASS | — |
| Second-source compose/narrate/transcript/CER/audio/report | all stages done, CER 16.3023/15.2483 | PASS | — |
| Assembler overlapping headings | longest-match fix + 4 regression tests | PASS | — |
| Fresh-clone reproducibility | passed at earlier commits | PARTIAL | Re-verify after committing current changes |
| Demo video | script only | FAIL | Record video |
| PR / submission email | no remote | NOT VERIFIED | Configure remote, PR, email |

## H. Remaining submission tasks

1. Commit the current uncommitted implementation/test and documentation changes.
2. Re-run the full offline suite and static checks at the new HEAD.
3. Repeat fresh-clone verification at the new HEAD.
4. Record the demo video from existing Chapter 6 and Chapter 2 artifacts.
5. Configure a remote and prepare the PR and submission email when authorized.
