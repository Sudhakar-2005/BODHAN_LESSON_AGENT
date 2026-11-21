# BODHAN LESSON AGENT

Deterministic stage pipeline (with a constrained LLM planner/auditor) that converts
scanned textbook chapters into translated lessons and narrated audio.

## Pipeline

```
ingest → preprocess → extract(Indic-OCR) → structure(LLM) → quality(LLM)
       → translate(Indic-Translate) → compose(LLM plan + Indic-Translate labels)
       → narrate(Indic-Speak) → transcribe(Indic-Transcribe) → export
```

### Provider architecture

- **OCR — Bodhan Indic-OCR (exclusive):** the *only* text-extraction engine.
  No Gemini Vision, no Tesseract, no fallback engines. Pages are rasterised
  to images and sent to Indic-OCR; embedded PDF text is never read.
- **Translation — Bodhan Indic-Translate (exclusive):** every target-language
  string in the lesson originates here — body blocks (per block id, equations
  verbatim) plus all labels, the title, objectives, summary, glossary terms,
  exercise questions and narration notes.
- **LLM — Gemini (structure, quality, lesson planning only):** classifies
  ambiguous OCR blocks, audits OCR quality, and plans the lesson *in English
  only* (section order, block grouping by id, English drafts of pedagogical
  scaffolding). It never extracts text, never translates, and never writes
  Hindi, Tamil, Bengali or any other Indian-language content — each English
  draft is routed through Indic-Translate before it can enter the lesson.
- **Narration — Bodhan Indic-Speak (the only engine, WAV):** the assignment
  restricts alternative TTS providers, so there is no fallback engine.
  Transient failures are retried with bounded backoff (honouring the
  documented `retry-after` window); if narration still fails, completed
  segments are preserved, the failure is recorded in `narrate/status.json`
  and the run resumes at the first missing segment.
- **Back-transcription — Bodhan Indic-Transcribe (the only STT):** every
  narrated segment is transcribed back with `POST /v1/audio/transcriptions`
  (model `indic-transcribe`) and stored verbatim under `transcribe/` —
  never translated, never rewritten, and kept separate from the narration
  script. Audio is split into ≤30 s chunks and each successful chunk is
  checkpointed under `transcribe/chunks/`, so a re-run never retranscribes a
  completed chunk; the stage cannot be marked done while any required chunk
  is missing.

## Setup (Windows / Linux)

Requires Python ≥3.11 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync                       # installs from uv.lock (exact versions)
copy .env.example .env        # fill in credentials (see below)
```

```bash
# Linux / macOS
uv sync
cp .env.example .env
```

Environment variables (all optional for running the offline test suite;
required only for a live pipeline run):

| Variable | Purpose |
|---|---|
| `BODHAN_API_BASE_URL` | gateway base URL (default `https://api.bodhan.ai`) |
| `BODHAN_OCR_KEY` | Indic-OCR credential |
| `BODHAN_TRANSLATE_KEY` | Indic-Translate credential |
| `BODHAN_SPEAK_KEY` | Indic-Speak credential |
| `BODHAN_TRANSCRIBE_KEY` | Indic-Transcribe credential |
| `BODHAN_MAX_RETRIES` | per-request attempt limit for Bodhan adapters (default `3`) |
| `TRANSCRIBE_PACING_S` | delay before each Indic-Transcribe request (default `0`; use `8` for long runs) |
| `LLM_API_KEY` / `GEMINI_API_KEY` | Gemini planner/auditor credential |

`.env` is gitignored; all credential fields are `repr=False`. Never commit a
real key. `BODHAN_ALLOW_SHORT_DOCUMENT=1` relaxes the assignment's 5–30 page
range for *internal synthetic fixtures only* (never needed for real inputs).

## Usage

```powershell
uv run bodhan --help
uv run bodhan run data/inbox/chapter-01.pdf
uv run bodhan run data/inbox/chapter-01.pdf --target-lang ta   # Tamil lesson

# Assignment interface (one lesson per target language, copied to --out):
uv run python -m bodhan_lesson --input data/inbox/chapter-01.pdf ^
    --langs hi,ta,bn --out ./out
# --source-lang is OPTIONAL and defaults to "en"; --glossary, --voices and
# --code-mixed are optional too.
```

`--input` accepts a PDF, a single page image, or a folder of page images. The
assignment CLI enforces the 5–30 page range; the internal override
`BODHAN_ALLOW_SHORT_DOCUMENT=1` is only for synthetic fixtures.

`--source-lang` is **metadata only**: Bodhan Indic-OCR and Indic-Translate do
not take a source-language request field. Set it to the real source language
(for example `hi` for a Hindi-medium PDF); it never rewrites source text.

## Verified evidence

### Completed demo — NCERT Class 9 Maths Chapter 6 (English source)

The official 9-page English NCERT Class 9 Maths Chapter 6 PDF has been run
end-to-end for three target languages. Each run completed all ten stages and
produced `lesson.md`, `lesson.html`, section audio, `quality_report.json` and
`manifest.json` under `out/<lang>/` and `data/exports/<chapter>-<lang>/`.

| Language | Sections narrated / transcribed | Macro CER | Micro CER |
|---|---:|---:|---:|
| Hindi (`hi`) | 9 / 9 | 34.7211% | 50.8712% |
| Tamil (`ta`) | 9 / 9 | 34.1256% | 47.5175% |
| Bengali (`bn`) | 8 / 8 | 40.4301% | 52.7450% |

Bengali has eight narratable sections because the composed lesson has eight
bodied sections; this is not missing audio.

### Completed second-source run — NCERT Class 9 Science Chapter 2 (Hindi source → Tamil)

The official 20-page Hindi-medium NCERT Class 9 Science Chapter 2 PDF
(`D:\phase6w-temp\ncert-class9-science-hi-ch2-ihsc102.pdf`) completed all ten
stages: OCR 20/20 pages (239 blocks), structure 224 blocks, Tamil translation
224/224 blocks, composition, 10 narration sections, 146/146 transcription
chunks, CER, audio assembly and export. Macro CER **16.3023 %**, micro CER
**15.2483 %**; full audio **4,287.744 s**. The successful paced transcription
used `TRANSCRIBE_PACING_S=8` and `BODHAN_MAX_RETRIES=3` with **zero HTTP 429
responses**. An earlier 14-page Hindi Chapter 5 candidate was blocked by a
provider-side OCR 502 on page 6 and was not used as the second source. Full
details are in `docs/SUBMISSION.md`.

## Layout

```
src/bodhan/
  config.py          # typed settings (pydantic-settings)
  domain/            # pure data models
  ports/             # provider interfaces (OCR, LLM, TTS, translation, artifact store)
  adapters/          # concrete provider implementations (Bodhan.AI + Gemini)
  pipeline/          # orchestrator + stages
  services/          # artifact store, manifest/checkpointing, atomic JSON
  evaluation/        # Phase 2B CER engine
  assembly/          # Phase 2C full-lesson audio assembly
  reporting/         # Phase 2C final report.json
  cli.py             # v1 entry point (Typer)
src/bodhan_lesson/   # assignment CLI: python -m bodhan_lesson
tests/               # unit + adapter contract + offline integration tests
data/                # inbox / work / exports / demo (gitignored)
docs/                # ARCHITECTURE.md, WORKFLOW.md, SUBMISSION.md, VIDEO_SCRIPT.md
```

## Robustness, testing and limitations

- **Checkpoint/resume:** per-stage artifact validation (`verified` / `legacy`
  / `invalid`) and per-block/per-segment checkpoints. See `docs/WORKFLOW.md`.
- **Concurrency:** provider calls are intentionally serialized (no semaphore);
  `provider_max_concurrency` is documentation only, not an active controller.
- **Failure handling:** terminal auth/validation/budget errors are not retried;
  HTTP 429 is retried with bounded exponential backoff, honouring
  `Retry-After` when supplied; per-chunk checkpoints preserve all successful
  work, and failed chunks are recorded, never silently dropped.
  `TRANSCRIBE_PACING_S` adds a configurable delay before each request.
- **Testing:** `uv run pytest` (offline, no credentials needed); use
  `uv run pytest -p no:cacheprovider -q` for read-only verification. The latest
  verified suite is **512 passed** (Phase 6AV; the earlier baseline was
  **508 passed** before four assembler regression tests were added), with Ruff,
  Ruff format, Mypy and the offline lock check green. On Windows machines with a nearly full C: drive,
  set `TEMP`, `TMP` and `UV_CACHE_DIR` to a directory on D: before running uv
  commands.
- **Known limitations:** document understanding is heuristic (tables/figures/
  cross-page continuity); CER is inflated by LaTeX/markup and digits that are
  not necessarily spoken verbatim; WAV has no embedded chapters (sidecar
  timeline only); some provider metrics (request IDs, usage, monetary cost) are
  not exposed — STT cost is an explicitly labelled estimate (₹0.10/audio-min),
  not provider-reported; historical second-source translation checkpoints
  retain zero span-count metadata (the fix applies only to newly generated
  checkpoints); no demo video has been recorded yet. The audio assembler's
  heading matcher now selects the longest matching heading when one heading is
  a prefix of another, with equal-length ties rejected. See
  `docs/SUBMISSION.md` for evidence and remaining tasks.
```
