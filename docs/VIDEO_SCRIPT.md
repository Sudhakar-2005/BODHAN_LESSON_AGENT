# Demo video script (≤5 minutes) and recording checklist

The narration must describe **only artifacts that actually exist**. Do not
present a pre-generated artifact as if it is being produced live.

| Time | Segment | Narration / on-screen action |
|---|---|---|
| 0:00–0:30 | Problem & motivation | "Scanned textbook chapters must become accessible narrated lessons in Indian languages. Manual conversion is slow and error-prone." Show a scanned textbook page. |
| 0:30–1:00 | Input | Show `data/demo/inputs/` and `SOURCES.md`: official NCERT sources, page counts 9 and 14. State: "the pipeline rasterises pages; the embedded text layer is never used." |
| 1:00–1:30 | CLI execution | Type the exact command: `python -m bodhan_lesson --input … --langs hi --out ./out`. Show the page-count enforcement and structured stage events in the console. |
| 1:30–2:15 | OCR & structure | Show `extract/results.json` (blocks with ids, bbox, confidence, reading order) and `structure/` artifacts (suppressed headers, tables grid, continuity merges). State Bodhan Indic-OCR is the only OCR provider. |
| 2:15–2:45 | Translated lesson | Open `lesson.md`; show translation provenance in `lesson.json` scaffold items and per-block translations. State: Indic-Translate is the only translator; the LLM wrote English only. |
| 2:45–3:30 | Narration & audio | Play one `seg-NN.wav`, then `full_lesson.wav`. Show `timeline.json`; state WAV has no embedded chapters — the sidecar is the marker source. |
| 3:30–4:15 | Transcription & CER | Show `transcribe/segment-NN.json` (verbatim text) and `evaluation/cer/manifest.json` (macro/micro CER, NFC policy). State the CER reference is the exact Indic-Speak input. |
| 4:15–4:45 | Final report | Show `report.json` sections, `null_value_reasons`, estimated-vs-actual cost distinction. |
| 4:45–5:00 | Architecture | Show `docs/ARCHITECTURE.md` DAG. Close: "deterministic pipeline with constrained LLM planning — not a fully autonomous agent." |

## Recording checklist

- [ ] Confirm live run actually succeeded (all stages `done`) before narrating "generated live".
- [ ] If a stage was blocked (e.g. LLM 503), say so explicitly; show the preserved checkpoints, not an invented success.
- [ ] Use a clean working tree snapshot of the demo inputs; do not rerun mid-recording.
- [ ] Keep total runtime ≤5:00.
- [ ] Do not show `.env` contents or any credential on screen.
