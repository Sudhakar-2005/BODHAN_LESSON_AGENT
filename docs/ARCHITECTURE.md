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
| transcribe | **Bodhan Indic-Transcribe only** (≤30 s splits) | `transcribe/segment-*.json`, `manifest.json` |
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
- Stage checkpoints validate artifact existence, JSON parseability and hashes
  (`checkpoint_verification_level`: `verified` / `legacy` / `invalid`).
- Input validation fails with actionable errors before any provider call.
- Provider calls are **intentionally serialized** (sequential awaits) to stay
  within documented RPM limits. `provider_max_concurrency` is documentation
  only — it is NOT an active semaphore.
- Structured, allowlisted JSON events via `emit_event` (no secrets/content).

## Evaluation

- CER (`bodhan/evaluation/cer.py`): NFC + whitespace collapse, documented
  policy `nfc+ws-collapse-v1`, character-level Levenshtein with S/D/I,
  macro and micro averages.
- Full-lesson audio (`bodhan/assembly/audio.py`): byte-for-byte WAV concat,
  sidecar `timeline.json`; WAV has **no embedded chapters** (not claimed).
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
- The optional second-source Hindi PDF remains blocked by a provider-side OCR
  502 on page 6; see `docs/SUBMISSION.md` for the exact disclosure. No
  alternative provider or fallback is used.
