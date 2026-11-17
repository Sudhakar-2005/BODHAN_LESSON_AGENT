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
  script. Records are checkpointed per segment: a re-run never retranscribes
  completed segments, and the stage cannot be marked done while any required
  segment is missing.

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
- **Failure handling:** terminal errors are not retried; 429+`Retry-After` is
  honoured with bounded backoff; failed blocks/segments are recorded, never
  silently dropped.
- **Testing:** `uv run pytest` (offline, no credentials needed).
  `uv run ruff check . && uv run ruff format --check . && uv run mypy`.
- **Known limitations:** document understanding is heuristic (tables/figures/
  cross-page continuity); CER is surface-level; WAV has no embedded chapters
  (sidecar timeline only); Tamil/Bengali live validation is not yet evidenced;
  see `docs/SUBMISSION.md` for the honest live-vs-offline matrix.
```
