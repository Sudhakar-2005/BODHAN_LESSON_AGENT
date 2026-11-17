# BODHAN LESSON AGENT

Agentic pipeline that converts scanned textbook chapters into translated lessons and narrated audio.

## Pipeline

```
ingest → preprocess → extract(Indic-OCR) → structure(LLM) → quality(LLM)
       → translate(LLM) → compose(LLM) → narrate(edge-tts) → export
```

- **OCR:** Bodhan AI Indic-OCR is the *exclusive* text-extraction engine. No Gemini Vision,
  no Tesseract, no alternative engines.
- **LLM:** used only for structural analysis, reading-order correction, quality evaluation,
  translation and lesson composition — never for text extraction.
- **Narration:** `edge-tts`.

## Setup

```powershell
uv sync
copy .env.example .env   # then fill in credentials
```

## Usage

```powershell
uv run bodhan --help
uv run bodhan run data/inbox/chapter-01.pdf
```

## Layout

```
src/bodhan/
  config.py          # typed settings (pydantic-settings)
  domain/            # pure data models
  ports/             # provider interfaces (OCR, LLM, TTS, artifact store)
  adapters/          # concrete provider implementations
  pipeline/          # orchestrator + stages
  services/          # artifact store, manifest/checkpointing
  api/               # reserved for a future HTTP interface
  cli.py             # v1 entry point (Typer)
tests/               # unit + adapter contract tests
data/                # inbox / work / exports (gitignored)
```
