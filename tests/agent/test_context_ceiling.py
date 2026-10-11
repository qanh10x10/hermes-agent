"""The effective budget survives process/config reloads without inflating native windows."""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent import model_metadata
from hermes_cli import config


@pytest.mark.parametrize("native,expected", [(1_050_000, 356_000), (356_000, 356_000), (262_144, 262_144), (200_000, 200_000), (32_768, 32_768)])
def test_ceiling_caps_resolved_window(monkeypatch, native, expected):
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"context": {"ceiling_tokens": 356_000}})
    assert model_metadata.get_model_context_length("test", config_context_length=native) == expected


@pytest.mark.parametrize("ceiling", [None, 0, -1, True, "356000"])
def test_invalid_or_unset_ceiling_keeps_native(monkeypatch, ceiling):
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"context": {"ceiling_tokens": ceiling}})
    assert model_metadata.get_model_context_length("test", config_context_length=1_050_000) == 1_050_000


@pytest.mark.parametrize("section", [None, [], "invalid", 42])
def test_invalid_context_section_keeps_native(monkeypatch, section):
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"context": section})
    assert model_metadata.get_model_context_length("test", config_context_length=1_050_000) == 1_050_000


def test_switch_models_and_async_preserve_auto_detection(monkeypatch):
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"context": {"ceiling_tokens": 356_000}})
    calls = []
    windows = {"big": 1_050_000, "small": 200_000, "tiny": 32_768}

    def detect(model, **kwargs):
        calls.append(kwargs)
        return windows[model]

    monkeypatch.setattr(model_metadata, "_detect_model_context_length", detect, raising=False)
    for model, expected in [("big", 356_000), ("small", 200_000), ("tiny", 32_768), ("big", 356_000)]:
        assert asyncio.run(model_metadata.get_model_context_length_async(model, provider="custom:hollow")) == expected
        assert calls[-1]["config_context_length"] is None


def test_config_reload_changes_budget_not_native_metadata(monkeypatch):
    cfg = {"context": {"ceiling_tokens": 356_000}}
    monkeypatch.setattr(config, "load_config_readonly", lambda: cfg)
    assert model_metadata.get_model_context_length("test", config_context_length=1_050_000) == 356_000
    cfg["context"]["ceiling_tokens"] = 128_000
    assert model_metadata.get_model_context_length("test", config_context_length=1_050_000) == 128_000
    assert model_metadata._detect_model_context_length("test", config_context_length=1_050_000) == 1_050_000


def test_ceiling_survives_fresh_process_reload(tmp_path):
    (tmp_path / "config.yaml").write_text("context:\n  ceiling_tokens: 356000\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    script = "import json; from agent.model_metadata import get_model_context_length as ctx; print(json.dumps([ctx('test', config_context_length=n) for n in [1050000,200000,32768]]))"
    env = dict(os.environ, HERMES_HOME=str(tmp_path))
    for _ in range(2):
        result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env, capture_output=True, text=True, timeout=30, check=True)
        assert json.loads(result.stdout) == [356_000, 200_000, 32_768]


def test_gateway_cache_rebuilds_when_ceiling_changes(monkeypatch):
    import ast
    from gateway.run_agent_cache import GatewayAgentCacheMixin

    # Read the real key list without booting a serving gateway inside the test sandbox.
    tree = ast.parse((Path(__file__).resolve().parents[2] / "gateway/run.py").read_text())
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GatewayRunner")
    keys = next(node.value for node in runner.body if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_CACHE_BUSTING_CONFIG_KEYS")

    class GatewayRunner(GatewayAgentCacheMixin):
        _CACHE_BUSTING_CONFIG_KEYS = ast.literal_eval(keys)

    monkeypatch.setattr(GatewayRunner, "_memory_provider_identity_signature", lambda *_: {})
    keys_a = GatewayRunner._extract_cache_busting_config({"context": {"ceiling_tokens": 356_000}})
    keys_b = GatewayRunner._extract_cache_busting_config({"context": {"ceiling_tokens": 200_000}})
    assert keys_a["context.ceiling_tokens"] == 356_000
    assert GatewayRunner._agent_config_signature("big", {}, [], "", cache_keys=keys_a) != GatewayRunner._agent_config_signature("big", {}, [], "", cache_keys=keys_b)
