"""Settings tests: environment-variable mapping and credential redaction."""

from __future__ import annotations

import os

import pytest

from bodhan.config import Settings


def test_bodhan_env_var_names_map_to_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The four required BODHAN_* variables must populate the settings."""
    values = {
        "BODHAN_OCR_KEY": "env-ocr-key",
        "BODHAN_TRANSLATE_KEY": "env-translate-key",
        "BODHAN_SPEAK_KEY": "env-speak-key",
        "BODHAN_TRANSCRIBE_KEY": "env-transcribe-key",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings()

    assert settings.bodhan_ocr_key == "env-ocr-key"
    assert settings.bodhan_translate_key == "env-translate-key"
    assert settings.bodhan_speak_key == "env-speak-key"
    assert settings.bodhan_transcribe_key == "env-transcribe-key"


def test_keys_never_appear_in_repr_or_str() -> None:
    """Keys must not be printable: repr/str of Settings must redact them."""
    secret = "sk-super-secret-value"
    settings = Settings(
        bodhan_ocr_key=secret,
        bodhan_translate_key=secret,
        bodhan_speak_key=secret,
        bodhan_transcribe_key=secret,
        llm_api_key=secret,
    )

    for printable in (repr(settings), str(settings), f"{settings}"):
        assert secret not in printable

    # Provider wiring still receives the real values; only representations
    # (logs, tracebacks, exception messages) are redacted.
    assert settings.llm_api_key == secret
    assert settings.bodhan_ocr_key == secret


def test_llm_key_redaction_survives_model_copy_and_validation_errors() -> None:
    """repr=False must survive model_copy (used for per-language settings)."""
    secret = "gemini-super-secret"
    settings = Settings(llm_api_key=secret)
    copied = settings.model_copy(update={"bodhan_speak_voice": "Kavya"})

    for printable in (repr(settings), repr(copied), str(copied)):
        assert secret not in printable
    assert copied.llm_api_key == secret

    try:
        Settings(cer_excellent_max_pct=10.0, cer_good_max_pct=5.0)
    except ValueError as exc:
        assert secret not in str(exc)
    else:  # pragma: no cover - the validator must reject this combination
        raise AssertionError("invalid CER thresholds were accepted")


def test_env_example_contains_placeholders_only() -> None:
    """Tracked configuration must never contain credential values."""
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / ".env.example"
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            assert not value.strip(), f"{key} has a non-empty value in .env.example"


def test_documented_endpoints_are_derived_from_base() -> None:
    settings = Settings(bodhan_api_base_url="https://api.bodhan.ai")

    # Paths taken verbatim from the official API reference.
    assert settings.bodhan_ocr_endpoint == "https://api.bodhan.ai/v1/chat/completions"
    assert settings.bodhan_translate_endpoint == "https://api.bodhan.ai/translate"
    assert settings.bodhan_speak_endpoint == "https://api.bodhan.ai/v1/audio/speech"
    assert settings.bodhan_transcribe_endpoint == "https://api.bodhan.ai/v1/audio/transcriptions"


def test_no_legacy_indicator_env_names_remain() -> None:
    """Conflicting INDIC_OCR_* credential names must be gone from settings."""
    field_names = set(Settings.model_fields)
    assert "indic_ocr_api_key" not in field_names
    assert "indic_ocr_base_url" not in field_names
    assert {
        "bodhan_ocr_key",
        "bodhan_translate_key",
        "bodhan_speak_key",
        "bodhan_transcribe_key",
    } <= field_names


def test_env_file_is_loaded_but_never_echoed(tmp_path, monkeypatch) -> None:
    """Sanity: a dotenv-sourced key loads without appearing in output."""
    env_file = tmp_path / ".env"
    env_file.write_text("BODHAN_OCR_KEY=sk-from-dotfile\n", encoding="utf-8")
    monkeypatch.delenv("BODHAN_OCR_KEY", raising=False)

    settings = Settings(_env_file=str(env_file))  # type: ignore[call-arg]

    assert settings.bodhan_ocr_key == "sk-from-dotfile"
    assert "sk-from-dotfile" not in repr(settings)
    assert os.environ.get("BODHAN_OCR_KEY") is None  # never exported to the env
