# Architecture

This project is a **deterministic, ordered, multi-stage pipeline** with
**constrained LLM-assisted planning and quality auditing**. It is *not* a
fully autonomous agent: the stage order is fixed in code, the LLM is called
only inside three stages, and it never authors target-language text.

## Stage DAG

```
ingest → preprocess → extract → structure → quality → translate → compose
       → narrate → transcribe → export
```

| Stage | Provider / mechanism | Output |
|---|---|---|
| ingest | PyMuPDF (rasterise only), Pillow | page PNGs + `pages.json` |
| preprocess | OpenCV (deskew/CLAHE) | cleaned page images |
| extract | **Bodhan Indic-OCR only** (chat transport, `indic-ocr`) | `results.json` (blocks, bbox, confidence, reading order, raw provider metadata) |
| structure | deterministic 37-label map; Gemini only for `UNKNOWN` labels | `blocks.json` + document-intelligence artifacts |
| quality | Gemini audit (flags/reasons only) | `blocks.json`, `quality_report.json` |
| translate | **Bodhan Indic-Translate only** (per block/cell, math protected) | `translations.json`, per-block checkpoints, `tables.json` |
| compose | Gemini English-only plan + Indic-Translate scaffold | `lesson.json` (Markdown), `plan.json`, `assembly.json` |
| narrate | **Bodhan Indic-Speak only** (WAV, ≤30 s chunks) | `segments.json`, `status.json`, WAVs |
| transcribe | **Bodhan Indic-Transcribe only** (≤30 s splits, per-chunk checkpoints, configurable pacing) | `transcribe/segment-*.json`, `transcribe/chunks/`, `manifest.json` |
| export | local file copy | `data/exports/<chapter>/` |

## LLM boundaries (enforced in code)

- `structure.py` / `quality.py`: the LLM may only assign a `BlockKind` or
  `QualityFlag`; it may not rewrite, translate or drop text. Block IDs are
  validated for completeness/uniqueness (`_validate`).
- `compose.py`: the planner receives OCR block IDs + translations and returns
  an **English-only** plan. `_english()` + `contains_indian_script()` reject
  any Brahmic-script output; every authored string is then translated through
  Indic-Translate as a `ScaffoldItem` with provenance.
- No Indian-language text is ever hardcoded in production code.

## Document intelligence (Phase 3B-1, deterministic)

`bodhan/document/` implements page-furniture suppression, table grid parsing
and Markdown rendering, cross-page continuation merging, figure/caption
association and math protection. Results are recorded in `structure/` with
source block IDs and reasons; OCR artifacts are never mutated.

## Robustness (Phase 3B-2)

- Per-block/per-section checkpoints with source hashes; resume reuses only
  valid outcomes. Math spans are withheld from the provider and verified
  byte-for-byte after reconstruction (multiset count plus order); LaTeX
  text-command delimiters such as `\textbf{` and `}` are preserved locally.
- Scientific units (`μm`, `µm`, `mm`, `cm`, `km`, `nm`, `mg`, `kg`) are a
  separate protected segment kind. They are excluded from provider prose,
  restored byte-for-byte, and validated by multiset and order with exact
  Unicode identity (`μ` U+03BC and `µ` U+00B5 are never equated).
- Checkpoints persist `math_span_count` and `unit_span_count` for newly
  translated blocks. Historical checkpoints with zero values are not
  backfilled; legacy records without the fields remain reusable.
- Stage checkpoints validate artifact existence, JSON parseability and hashes
  (`checkpoint_verification_level`: `verified` / `legacy` / `invalid`).
- Input validation fails with actionable errors before any provider call.
- Provider calls are **intentionally serialized** (sequential awaits) to stay
  within documented RPM limits. `provider_max_concurrency` is documentation
  only — it is NOT an active semaphore.
- Transcription resilience: HTTP 429 without `Retry-After` is retried with
  bounded exponential backoff (`BODHAN_MAX_RETRIES`, default 3); a supplied
  `Retry-After` is honoured and capped at 60 s. Auth, validation and explicit
  budget/quota errors remain terminal. `TRANSCRIBE_PACING_S` (default 0) adds
  a delay before every request; `TRANSCRIBE_PACING_S=8` produced zero 429s in
  the second-source live run.
- Each successful ≤30 s transcription chunk is checkpointed under
  `transcribe/chunks/<section>/chunk-NN.json` with audio identity, chunk
  boundaries and language; a resume reuses valid chunks and retries only
  missing or failed ones.
- Structured, allowlisted JSON events via `emit_event` (no secrets/content).

## Lesson-plan recovery

`compose.py` validates the English-only Gemini plan for unknown, duplicate and
missing source block IDs. If IDs are missing, it makes **one** bounded recovery
request for only those IDs. Duplicate and unknown IDs are rejected outright and
are not hidden by recovery.

Recovered blocks are placed using authoritative `OCRBlock.page_index` and
`reading_order` metadata — never by sorting ID strings. A block is inserted
into an existing section when its source position lies within that section's
span (or before the first / after the last section); ambiguous placement fails
with an actionable diagnostic rather than inventing a section. Original
section headings and valid assignments are preserved. `plan.json` records
backward-compatible recovery provenance: `used`, `version`, `method`,
`requests`, `original_block_ids` and `recovered_block_ids`.

## Evaluation

- CER (`bodhan/evaluation/cer.py`): NFC + whitespace collapse, documented
  policy `nfc+ws-collapse-v1`, character-level Levenshtein with S/D/I,
  macro and micro averages.
- Full-lesson audio (`bodhan/assembly/audio.py`): byte-for-byte WAV concat,
  sidecar `timeline.json`; WAV has **no embedded chapters** (not claimed).
  `plan_sections` matches narration text to lesson headings using normalized
  prefixes; when one heading is a prefix of a longer one (for example
  `சுருக்கம்` inside `சுருக்கம் மற்றும் மதிப்பீட்டு கேள்விகள்`), the longest
  match is selected. Only equal-length top matches are rejected as ambiguous,
  and duplicate headings remain an error.
- Final report (`bodhan/reporting/report.py`): aggregates persisted evidence;
  unavailable values are null with explicit `null_value_reasons`.

## Source-language handling and environment notes

- `--source-lang` on the assignment CLI is pipeline metadata only. Bodhan
  Indic-OCR and Indic-Translate do not take a source-language request field;
  the source language is recorded for provenance and never causes source text
  to be rewritten. Target languages are validated against the documented
  provider enums before any API call.
- On Windows machines where the C: drive is nearly full, set `TEMP`, `TMP` and
  `UV_CACHE_DIR` to a directory on another drive (for example `D:\`) before
  running `uv` commands. This is an environment workaround only — dependency
  versions and `uv.lock` are unchanged.
- For read-only verification, `uv run pytest -p no:cacheprovider -q` avoids
  creating pytest cache files. Static checks are
  `uv run ruff check . && uv run ruff format --check . && uv run mypy src`.
- The selected second source (NCERT Class 9 Science Chapter 2, Hindi, 20 pages)
  completed all ten stages for Tamil: 224/224 translations, 10 narration
  sections, 146/146 transcription chunks, macro CER 16.3023% / micro CER
  15.2483%, and 4,287.744 s of assembled audio. No alternative provider or
  fallback is used; see `docs/SUBMISSION.md` for the evidence. An earlier Hindi
  Chapter 5 candidate was blocked by a provider OCR 502 on page 6.
