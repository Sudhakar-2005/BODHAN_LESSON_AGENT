# Submission preparation status (honest)

## Clean-clone verification plan

Run on a fresh clone of `assignment/narrated-lesson-Sudhakar-2005`:

1. `git status` — must be clean; all required files tracked.
2. `uv sync --frozen` — dependencies install from `uv.lock`.
3. `uv run pytest` — passes with **no** `.env` present (tests use fakes).
4. `uv run python -m bodhan_lesson --help` — CLI works.
5. `uv run ruff check . && uv run ruff format --check . && uv run mypy`.
6. `uv run python -m bodhan_lesson --input data/demo/inputs/… --langs hi --out ./out`.

**Current blocker (2026-11-17):** the working tree is intentionally uncommitted
per project instructions. A clean clone of HEAD `6f58961` would NOT contain
`bodhan_lesson/`, `document/`, `evaluation/`, `assembly/`, `reporting/`, the
transcribe adapter/stage, or most tests, and HEAD still contains the obsolete
Edge-TTS fallback files and old README. **Do not claim clean-clone success
until these are staged and committed.**

## Live vs offline verification matrix

| Capability | Offline (unit/integration) | Live evidence |
|---|---|---|
| 9-page NCERT OCR (English) | — | **Partial**: ingest/preprocess/extract/structure succeeded on 9 pages (111 OCR blocks); `quality` blocked by Gemini 503 (retried 5×). |
| English → Hindi | CLI/E2E fakes pass | **Blocked** at `quality` (Gemini 503) in Phase 3B-3; earlier Phase 2C Hindi artifacts remain valid live evidence for the 2-page smoke chapter. |
| English → Tamil / Bengali | Code + fake-based tests pass | **Not live-verified** in this project. |
| Hindi source → target | Fixtures/offline pass | **Not run live** (10-page official Hindi PDF prepared only). |
| Document intelligence | 40+ deterministic tests + offline integration | Not live-verified. |
| CER / full audio / report | Offline over Phase 2C artifacts | Phase 2C live Hindi artifacts verified. |

## Demo artifact inventory

- `data/demo/inputs/`: two official NCERT PDFs + `SOURCES.md` + `sha256sum.txt`.
- `data/work/ncert-class9-maths-ch6-lines-and-angles-official-hi/`:
  partial live run — ingest/preprocess/extract/structure `done`, `quality`
  `failed` (Gemini 503); preserved checkpoints are honest partial evidence.
- `data/work/phase1-verify`, `data/work/phase2a-verify`, `data/work/phase2c-verify`:
  historical Phase 1/2 evidence (untouched).
- `docs/VIDEO_SCRIPT.md`: ≤5-minute demo script + checklist.

## Remaining submission blockers

1. Commit all work on the assignment branch (currently uncommitted).
2. Re-run the full live demo when the Gemini 503 clears (or increase retries /
   switch to an available model) to obtain end-to-end hi/ta/bn artifacts.
3. Produce the actual demo video from a successful live run.
4. Repeat clean-clone verification after the commit.

These are reported, not hidden. The implementation is **not** submission-ready
until items 1–4 are resolved.
