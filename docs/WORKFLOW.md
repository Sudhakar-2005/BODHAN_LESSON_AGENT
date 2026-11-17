# Pipeline workflow / data flow

```
data/inbox/<source>.pdf
        │  (rasterise — text layer NEVER read)
        ▼
data/work/<chapter>/ingest/       page-0001.png … + pages.json
data/work/<chapter>/preprocess/   cleaned images
data/work/<chapter>/extract/      results.json (OCR blocks: id, page, order,
                                  kind, confidence, bbox, provider meta)
        │  deterministic label map → structure + document intelligence
        ▼
data/work/<chapter>/structure/    blocks.json, suppressed.json, continuity.json,
                                  tables.json, figures.json, document_intelligence.json
data/work/<chapter>/quality/      blocks.json (flags), quality_report.json
        │  per-block translation (math protected, tables cell-wise)
        ▼
data/work/<chapter>/translate/    translations.json, tables.json, blocks/*.json
        │  English-only LLM plan → translate scaffold
        ▼
data/work/<chapter>/compose/      plan.json, lesson.json, assembly.json, sections/*.json
        │  Indic-Speak (WAV, only TTS)
        ▼
data/work/<chapter>/narrate/      seg-NN.wav, segments.json, status.json
        │  Indic-Transcribe (only STT, ≤30 s split)
        ▼
data/work/<chapter>/transcribe/   segment-NN.json, manifest.json
        │  offline CER (reference = exact Indic-Speak input)
        ▼
data/work/<chapter>/evaluation/cer/  segment-NN.json, manifest.json
        │  offline WAV concat + sidecar timeline
        ▼
data/work/<chapter>/audio/        segments/, full_lesson.wav, timeline.json
        │
        ▼
data/work/<chapter>/report.json   comprehensive evidence report
        │  export copy
        ▼
data/exports/<chapter>/          lesson.md/html, audio/, manifest.json
```

## Resume / failure semantics

- Manifest records per-stage status, duration and artifact hashes.
- `done` stages with valid (verified or legacy) artifacts are skipped;
  corrupt/missing artifacts are regenerated.
- Translate and transcribe checkpoint per block/segment; a rerun reuses only
  successful, hash-matching outcomes.
- Narrate preserves completed segments and resumes at the first missing one.
- A failed stage is recorded in the manifest with an error and the pipeline
  fails loudly — no placeholder success statuses.
