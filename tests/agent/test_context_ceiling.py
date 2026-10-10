"""Effective context ceiling must not replace native auto-detection."""
import asyncio

import pytest

from agent import model_metadata
from hermes_cli import config


@pytest.mark.parametrize("native, expected", [(1_050_000, 356_000), (356_000, 356_000), (200_000, 200_000), (32_768, 32_768), (262_144, 262_144)])
def test_ceiling_preserves_smaller_auto_context(monkeypatch, native, expected):
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"context": {"ceiling_tokens": 356_000}})
    monkeypatch.setattr(model_metadata, "_detect_model_context_length", lambda *args, **kwargs: native, raising=False)
    assert model_metadata.get_model_context_length("test-model") == expected


@pytest.mark.parametrize("ceiling", [None, 0, -1, True, "356000"])
def test_unset_or_invalid_ceiling_keeps_detection(monkeypatch, ceiling):
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"context": {"ceiling_tokens": ceiling}})
    monkeypatch.setattr(model_metadata, "_detect_model_context_length", lambda *args, **kwargs: 1_050_000, raising=False)
    assert model_metadata.get_model_context_length("test-model") == 1_050_000


def test_auto_args_and_async_use_same_ceiling(monkeypatch):
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"context": {"ceiling_tokens": 356_000}})
    calls = []

    def detect(*args, **kwargs):
        calls.append((args, kwargs))
        return kwargs.get("config_context_length") or 1_050_000

    monkeypatch.setattr(model_metadata, "_detect_model_context_length", detect, raising=False)
    assert model_metadata.get_model_context_length("big", provider="custom:hollow") == 356_000
    assert calls[-1][1]["config_context_length"] is None
    assert model_metadata.get_model_context_length("small", config_context_length=100_000) == 100_000
    assert asyncio.run(model_metadata.get_model_context_length_async("big", config_context_length=1_050_000)) == 356_000
