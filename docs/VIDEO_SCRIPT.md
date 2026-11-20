# Demo video script (≤5 minutes) — **NOT YET RECORDED**

This is a **script only**. No demo video file exists in the repository.
Do not present this document as recorded footage. The narration below must only
describe artifacts that actually exist; for the second-source Chapter 5 attempt,
disclose the OCR failure explicitly.

## Sequence

### 0:00–0:30 — Objective and input
"Scanned textbook chapters must become accessible narrated lessons in Indian
languages. We use a deterministic ten-stage pipeline with a constrained LLM
planner/auditor, not a fully autonomous agent."
Show `data/demo/inputs/ncert-class9-maths-ch6-lines-and-angles-official.pdf`
(9 pages) and `data/demo/inputs/SOURCES.md`. State: pages are rasterised; the
embedded PDF text layer is never read.

### 0:30–1:10 — CLI and pipeline
Show the exact commands used for the completed Chapter 6 runs:

```powershell
uv run python -m bodhan_lesson --input data/demo/inputs/ncert-class9-maths-ch6-lines-and-angles-official.pdf --langs hi --out ./out --source-lang en
uv run python -m bodhan_lesson --input data/demo/inputs/ncert-class9-maths-ch6-lines-and-angles-official.pdf --langs ta --out ./out --source-lang en
uv run python -m bodhan_lesson --input data/demo/inputs/ncert-class9-maths-ch6-lines-and-angles-official.pdf --langs bn --out ./out --source-lang en
```

Point to the stage DAG in `docs/ARCHITECTURE.md`:
`ingest → preprocess → extract → structure → quality → translate → compose →
narrate → transcribe → export`. Explain that `--source-lang` is metadata only.

### 1:10–2:30 — Three language outputs
Open, one at a time:
- `out/hi/lesson.md` (Hindi, 9 narrated/transcribed sections)
- `out/ta/lesson.md` (Tamil, 9 sections)
- `out/bn/lesson.md` (Bengali, 8 sections — correct lesson structure)

Show `data/work/ncert-class9-maths-ch6-lines-and-angles-official-<lang>/translate/translations.json`
and state that every target-language string comes from Indic-Translate; LaTeX
text-command delimiters and math spans are restored locally and never sent to
the translator.

### 2:30–3:30 — HTML and audio
Open `out/hi/lesson.html`, then play `out/hi/audio/seg-01.wav` and
`data/work/ncert-class9-maths-ch6-lines-and-angles-official-hi/audio/full_lesson.wav`.
Show `audio/timeline.json` and `audio/assembly_manifest.json`. State that WAV
has no embedded chapters — the timeline sidecar is the marker source.

### 3:30–4:15 — Quality, CER, resume
Show `out/hi/quality_report.json` and
`data/work/ncert-class9-maths-ch6-lines-and-angles-official-hi/evaluation/cer/manifest.json`.
State the CER reference is the exact Indic-Speak input and that LaTeX/digits in
that reference inflate the score. Show `narrate/status.json` and
`transcribe/manifest.json` as evidence of per-segment checkpoint/resume.

### 4:15–5:00 — Architecture, limitations, second source
Show `docs/ARCHITECTURE.md`. Close with the honest limitations:

- Bodhan Indic-OCR, Indic-Translate, Indic-Speak and Indic-Transcribe are the
  only language providers; Gemini is limited to structure/quality/English plan.
- CER is markup-sensitive; provider request IDs, usage and cost are partly
  unavailable; headerless STT 429s required controlled resumes.
- The second official source,
  `data/demo/inputs/ncert-class9-maths-ch5-rekha-aur-kon-hindi-official.pdf`
  (14 pages, Hindi medium), completed ingest and preprocessing and OCR on five
  pages, then hit a provider-side HTTP 502 on page 6. There is no second-source
  target-language output; no further retry is recommended without provider
  guidance.
- This video has not been recorded; the script is provided for a future recording.

## Recording checklist (all currently unchecked)

- [ ] Record the video from the artifacts above.
- [ ] Confirm all Chapter 6 stages are `done` before narrating "generated live".
- [ ] Do not show `.env` contents or any credential on screen.
- [ ] Keep total runtime ≤5:00.
- [ ] State explicitly that Chapter 5 is blocked; do not imply success.
