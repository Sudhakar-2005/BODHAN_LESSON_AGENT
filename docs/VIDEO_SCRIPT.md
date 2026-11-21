# Demo video script (≤5 minutes) — **NOT YET RECORDED**

This is a **script only**. No demo video file exists in the repository.
Do not present this document as recorded footage. The narration below must only
describe artifacts that actually exist. Both demonstrations are complete:
Chapter 6 in hi/ta/bn and the second-source Science Chapter 2 in Tamil, with
CER reports and assembled audio for every run.

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
  unavailable — STT cost is an estimate, not provider-reported.
- Headerless STT 429s are now retried with bounded backoff; `TRANSCRIBE_PACING_S=8`
  and `BODHAN_MAX_RETRIES=3` produced zero 429s in the successful paced run,
  and every ≤30 s chunk is checkpointed for safe resume.
- The selected second source,
  `D:\phase6w-temp\ncert-class9-science-hi-ch2-ihsc102.pdf` (20 pages, Hindi
  medium), completed all ten stages for Tamil: OCR 20/20 pages (239 blocks),
  224/224 translations, 10 narration sections, 146/146 transcription chunks,
  macro CER 16.3023% / micro CER 15.2483%, and 4,287.744 s of full audio. Show
  `data/work/ncert-class9-science-ch2-cell-hindi/report.json` and
  `data/exports/ncert-class9-science-ch2-cell-hindi/lesson.html`.
  An earlier Chapter 5 candidate was blocked by an OCR 502 on page 6.
- This video has not been recorded; the script is provided for a future recording.

## Recording checklist (all currently unchecked)

- [ ] Record the video from the artifacts above.
- [ ] Confirm all Chapter 6 and Chapter 2 stages are `done` before narrating "generated live".
- [ ] Do not show `.env` contents or any credential on screen.
- [ ] Keep total runtime ≤5:00.
- [ ] State explicitly that the video is not yet recorded and that cost is estimated.
