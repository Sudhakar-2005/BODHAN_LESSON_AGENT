"""Assignment CLI tests: ``python -m bodhan_lesson`` (Task 2).

The pipeline itself is faked (``run_pipeline`` / ``build_providers`` are
monkeypatched) so these tests exercise the CLI contract only, offline:

- the required interface and ``--help``;
- validation errors (langs, input, glossary, voices) before any API work;
- one run per target language with unique chapter ids;
- output copying into ``<out>/<lang>/``;
- ``--source-lang`` -> chapter/lesson source language;
- ``--glossary`` seeds, ``--voices`` overrides, ``--code-mixed`` ->
  documented ``script=codemix`` field.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import bodhan_lesson.cli as cli
from bodhan.config import Settings
from bodhan.pipeline.stages.base import StageContext
from bodhan_lesson.cli import app
from tests.conftest import FakeLLM, FakeOCR, FakeTranscribe, FakeTranslate, FakeTTS

runner = CliRunner()


class _FakeProviders:
    """Offline provider set (never performs network I/O)."""

    def __init__(self) -> None:
        self.ocr = FakeOCR()
        self.llm = FakeLLM()
        self.translate = FakeTranslate()
        self.tts = FakeTTS()
        self.transcribe = FakeTranscribe()

    async def aclose(self) -> None:
        return None


@pytest.fixture
def recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[StageContext]:
    """Patch the CLI's pipeline entry points; return the recorded contexts."""
    contexts: list[StageContext] = []

    async def fake_run_pipeline(ctx: StageContext) -> dict[str, object]:
        contexts.append(ctx)
        return {"status": "done"}

    def fake_build_providers(settings: Settings, *, target_language: str | None = None):
        return _FakeProviders()

    monkeypatch.setattr(
        cli, "load_settings", lambda: Settings(project_root=tmp_path, data_dir="data")
    )
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(cli, "build_providers", fake_build_providers)
    return contexts


def _make_input(tmp_path: Path, name: str = "chapter.pdf") -> Path:
    source = tmp_path / name
    source.write_bytes(b"%PDF-1.4 fake for CLI validation")
    return source


def _seed_export(project_root: Path, chapter_id: str) -> None:
    """Pretend a run produced an export the CLI must copy to --out."""
    export = project_root / "data" / "exports" / chapter_id
    export.mkdir(parents=True, exist_ok=True)
    (export / "lesson.md").write_text(f"# Lesson {chapter_id}\n", encoding="utf-8")
    (export / "manifest.json").write_text("{}", encoding="utf-8")


def _invoke(args: list[str]) -> object:
    return runner.invoke(app, args)


# --- interface & help -----------------------------------------------------


def test_help_lists_the_required_interface() -> None:
    result = _invoke(["--help"])
    assert result.exit_code == 0, result.output
    for option in (
        "--input",
        "--langs",
        "--out",
        "--source-lang",
        "--glossary",
        "--voices",
        "--code-mixed",
    ):
        assert option in result.output, option


def test_no_arguments_shows_helpful_usage_error() -> None:
    """No args: a usage error (not a traceback) pointing at --help."""
    result = _invoke([])
    assert result.exit_code == 2
    assert "Usage" in result.output
    assert "--help" in result.output


# --- validation errors (all before any pipeline/API work) -----------------


def test_missing_required_option_is_a_usage_error(tmp_path: Path, recorded) -> None:
    source = _make_input(tmp_path)
    result = _invoke(
        ["--input", str(source), "--out", str(tmp_path / "out"), "--source-lang", "en"]
    )
    assert result.exit_code == 2
    assert "--langs" in result.output
    assert recorded == []  # no pipeline work happened


def test_unsupported_target_language_is_rejected(tmp_path: Path, recorded) -> None:
    source = _make_input(tmp_path)
    result = _invoke(
        [
            "--input",
            str(source),
            "--langs",
            "hi,xx",
            "--out",
            str(tmp_path / "out"),
            "--source-lang",
            "en",
        ]
    )
    assert result.exit_code == 2
    assert "unsupported target language" in result.output
    assert "xx" in result.output
    assert recorded == []


def test_empty_langs_is_rejected(tmp_path: Path, recorded) -> None:
    source = _make_input(tmp_path)
    result = _invoke(
        [
            "--input",
            str(source),
            "--langs",
            " , ",
            "--out",
            str(tmp_path / "out"),
            "--source-lang",
            "en",
        ]
    )
    assert result.exit_code == 2
    assert "at least one target language" in result.output
    assert recorded == []


def test_missing_input_file_is_rejected(tmp_path: Path, recorded) -> None:
    result = _invoke(
        [
            "--input",
            str(tmp_path / "nope.pdf"),
            "--langs",
            "hi",
            "--out",
            str(tmp_path / "out"),
            "--source-lang",
            "en",
        ]
    )
    assert result.exit_code == 2
    assert recorded == []


def test_invalid_glossary_yaml_is_rejected(tmp_path: Path, recorded) -> None:
    source = _make_input(tmp_path)
    glossary = tmp_path / "glossary.yaml"
    glossary.write_text("terms: [unclosed\n", encoding="utf-8")
    result = _invoke(
        [
            "--input",
            str(source),
            "--langs",
            "hi",
            "--out",
            str(tmp_path / "out"),
            "--source-lang",
            "en",
            "--glossary",
            str(glossary),
        ]
    )
    assert result.exit_code == 2
    assert "not valid YAML" in result.output
    assert recorded == []


def test_indic_glossary_term_is_rejected(tmp_path: Path, recorded) -> None:
    """Glossary terms must be English (they are translated per language)."""
    source = _make_input(tmp_path)
    glossary = tmp_path / "glossary.yaml"
    glossary.write_text('terms:\n  - term: "हिंदी"\n', encoding="utf-8")
    result = _invoke(
        [
            "--input",
            str(source),
            "--langs",
            "hi",
            "--out",
            str(tmp_path / "out"),
            "--source-lang",
            "en",
            "--glossary",
            str(glossary),
        ]
    )
    assert result.exit_code == 2
    assert "Indian-script" in result.output
    assert recorded == []


def test_unknown_voice_language_is_rejected(tmp_path: Path, recorded) -> None:
    source = _make_input(tmp_path)
    voices = tmp_path / "voices.yaml"
    voices.write_text("xx: SomeVoice\n", encoding="utf-8")
    result = _invoke(
        [
            "--input",
            str(source),
            "--langs",
            "hi",
            "--out",
            str(tmp_path / "out"),
            "--source-lang",
            "en",
            "--voices",
            str(voices),
        ]
    )
    assert result.exit_code == 2
    assert "unknown target language" in result.output
    assert recorded == []


@pytest.mark.parametrize("value", ["", "english", "!", "e"])
def test_invalid_source_language_is_rejected_before_any_pipeline_work(
    tmp_path: Path, recorded, value: str
) -> None:
    """Bad language tags fail locally, before providers are built or called."""
    source = _make_input(tmp_path)
    result = _invoke(
        [
            "--input",
            str(source),
            "--langs",
            "hi",
            "--out",
            str(tmp_path / "out"),
            "--source-lang",
            value,
        ]
    )
    assert result.exit_code == 2
    assert "invalid --source-lang" in result.output
    assert recorded == []


# --- the required interface end to end (pipeline faked) -------------------


def test_source_lang_is_optional_and_defaults_to_en(tmp_path: Path, recorded) -> None:
    """The assignment marks --source-lang optional; default must be explicit."""
    source = _make_input(tmp_path)
    _seed_export(tmp_path, "chapter-hi")

    result = _invoke(["--input", str(source), "--langs", "hi", "--out", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    assert recorded[0].chapter.language_code == "en"


def test_runs_once_per_language_and_copies_outputs(tmp_path: Path, recorded) -> None:
    source = _make_input(tmp_path)
    out = tmp_path / "out"
    _seed_export(tmp_path, "chapter-hi")
    _seed_export(tmp_path, "chapter-ta")

    result = _invoke(
        [
            "--input",
            str(source),
            "--langs",
            "hi,ta",
            "--out",
            str(out),
            "--source-lang",
            "en",
        ]
    )

    assert result.exit_code == 0, result.output
    # one full pipeline run per target language, unique chapter ids
    assert [ctx.chapter.target_language_code for ctx in recorded] == ["hi", "ta"]
    assert [ctx.chapter.chapter_id for ctx in recorded] == ["chapter-hi", "chapter-ta"]
    # source language handling
    assert all(ctx.chapter.language_code == "en" for ctx in recorded)
    # outputs copied to <out>/<lang>/
    assert (out / "hi" / "lesson.md").read_text(encoding="utf-8") == "# Lesson chapter-hi\n"
    assert (out / "ta" / "lesson.md").read_text(encoding="utf-8") == "# Lesson chapter-ta\n"
    assert (out / "hi" / "manifest.json").is_file()
    # without --code-mixed the documented script field stays untouched
    assert recorded[0].settings.translate_script is None


def test_image_folder_input_is_accepted(tmp_path: Path, recorded) -> None:
    folder = tmp_path / "pages"
    folder.mkdir()
    (folder / "01.png").write_bytes(b"\x89PNG fake page one")
    (folder / "02.png").write_bytes(b"\x89PNG fake page two")
    _seed_export(tmp_path, "pages-hi")

    result = _invoke(
        [
            "--input",
            str(folder),
            "--langs",
            "hi",
            "--out",
            str(tmp_path / "out"),
            "--source-lang",
            "en",
        ]
    )

    assert result.exit_code == 0, result.output
    assert recorded[0].chapter.source_path == str(folder.resolve())
    assert (tmp_path / "out" / "hi" / "lesson.md").is_file()


def test_code_mixed_sets_documented_translate_script(tmp_path: Path, recorded) -> None:
    source = _make_input(tmp_path)
    _seed_export(tmp_path, "chapter-hi")

    result = _invoke(
        [
            "--input",
            str(source),
            "--langs",
            "hi",
            "--out",
            str(tmp_path / "out"),
            "--source-lang",
            "en",
            "--code-mixed",
        ]
    )

    assert result.exit_code == 0, result.output
    # documented Indic-Translate option: script = "codemix"
    assert recorded[0].settings.translate_script == "codemix"


def test_glossary_seeds_flow_into_the_chapter(tmp_path: Path, recorded) -> None:
    source = _make_input(tmp_path)
    _seed_export(tmp_path, "chapter-hi")
    glossary = tmp_path / "glossary.yaml"
    glossary.write_text(
        "terms:\n  - term: Chlorophyll\n    description: Green pigment.\n",
        encoding="utf-8",
    )

    result = _invoke(
        [
            "--input",
            str(source),
            "--langs",
            "hi",
            "--out",
            str(tmp_path / "out"),
            "--source-lang",
            "en",
            "--glossary",
            str(glossary),
        ]
    )

    assert result.exit_code == 0, result.output
    assert recorded[0].chapter.glossary_seeds == [
        {"term": "Chlorophyll", "description": "Green pigment."}
    ]


def test_voices_yaml_overrides_only_the_listed_languages(tmp_path: Path, recorded) -> None:
    source = _make_input(tmp_path)
    _seed_export(tmp_path, "chapter-hi")
    _seed_export(tmp_path, "chapter-ta")
    voices = tmp_path / "voices.yaml"
    voices.write_text(
        "voices:\n  hi:\n    voice: CustomVoice\n    style: cheerful\n",
        encoding="utf-8",
    )

    result = _invoke(
        [
            "--input",
            str(source),
            "--langs",
            "hi,ta",
            "--out",
            str(tmp_path / "out"),
            "--source-lang",
            "en",
            "--voices",
            str(voices),
        ]
    )

    assert result.exit_code == 0, result.output
    by_lang = {ctx.chapter.target_language_code: ctx for ctx in recorded}
    assert by_lang["hi"].settings.bodhan_speak_voice == "CustomVoice"
    assert by_lang["hi"].settings.bodhan_speak_style == "cheerful"
    # untouched language keeps the configured default (Kavya)
    assert by_lang["ta"].settings.bodhan_speak_voice == "Kavya"
    assert by_lang["ta"].settings.bodhan_speak_style is None


def test_duplicate_languages_run_once(tmp_path: Path, recorded) -> None:
    source = _make_input(tmp_path)
    _seed_export(tmp_path, "chapter-hi")

    result = _invoke(
        [
            "--input",
            str(source),
            "--langs",
            "hi,hi",
            "--out",
            str(tmp_path / "out"),
            "--source-lang",
            "en",
        ]
    )

    assert result.exit_code == 0, result.output
    assert len(recorded) == 1
