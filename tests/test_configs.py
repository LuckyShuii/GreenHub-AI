"""Unit tests for the Pydantic settings loader."""

import pytest
from pydantic import ValidationError

from configs import Settings, get_settings


def test_settings_read_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings are populated from environment variables."""
    monkeypatch.setenv("AI_PIPELINE_MODEL_NAME", "test/model")
    monkeypatch.setenv("AI_HOST", "127.0.0.1")
    monkeypatch.setenv("AI_PORT", "9000")

    settings = Settings(_env_file=None)  # type: ignore

    assert settings.ai_pipeline_model_name == "test/model"
    assert settings.ai_host == "127.0.0.1"
    assert settings.ai_port == 9000


def test_settings_apply_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional fields fall back to their defaults when unset."""
    monkeypatch.setenv("AI_PIPELINE_MODEL_NAME", "test/model")
    monkeypatch.delenv("AI_HOST", raising=False)
    monkeypatch.delenv("AI_PORT", raising=False)

    settings = Settings(_env_file=None)   # type: ignore

    assert settings.ai_host == "0.0.0.0"
    assert settings.ai_port == 8000


def test_settings_missing_required_field_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing mandatory model name aborts validation."""
    monkeypatch.delenv("PIPELINE_MODEL_NAME", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)   # type: ignore


@pytest.mark.parametrize("invalid_port", ["0", "70000", "-1"])
def test_settings_reject_out_of_range_port(
    monkeypatch: pytest.MonkeyPatch, invalid_port: str
) -> None:
    """Ports outside the valid TCP range are rejected."""
    monkeypatch.setenv("AI_PIPELINE_MODEL_NAME", "test/model")
    monkeypatch.setenv("AI_PORT", invalid_port)

    with pytest.raises(ValidationError):  # type: ignore
        Settings(_env_file=None)   # type: ignore


def test_get_settings_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_settings returns the same cached instance across calls."""
    monkeypatch.setenv("AI_PIPELINE_MODEL_NAME", "test/model")
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second
