from __future__ import annotations

import json

import pytest

from hermes_cli import claude_route_policy as policy


def _write_policy(tmp_path):
    token = tmp_path / "proxy-token"
    token.write_text("test-token", encoding="utf-8")
    token.chmod(0o600)
    route = {
        "version": 1,
        "provider": "claude-proxy",
        "provider_aliases": ["claude-cli-proxy"],
        "base_url": "http://127.0.0.1:4100/v1",
        "token_file": str(token),
        "default_model": "claude-sonnet-4-6",
        "models": {
            "claude-fable-5": "fable",
            "anthropic/claude-fable-5": "fable",
            "claude-sonnet-4-6": "sonnet",
            "sonnet": "sonnet",
        },
    }
    route_path = tmp_path / "route-policy.json"
    route_path.write_text(json.dumps(route), encoding="utf-8")
    return route_path, token, route


def test_classifier_captures_alias_native_and_claude_model():
    assert policy.is_claude_route(provider="claude-proxy", model="gpt-5.5")
    assert policy.is_claude_route(provider="claude-cli-proxy", model="claude-fable-5")
    assert policy.is_claude_route(provider="anthropic", model="claude-sonnet-4-6")
    assert policy.is_claude_route(provider="openrouter", model="anthropic/claude-opus-4.8")
    assert not policy.is_claude_route(provider="openai-codex", model="gpt-5.5")


def test_claude_provider_aliases_lock_without_a_model():
    assert policy.is_claude_route(provider="claude")
    assert policy.is_claude_route(provider="custom:anthropic")
    assert policy.is_claude_route(provider="custom:claude")


def test_vendor_qualified_known_claude_ids_stay_on_proxy(tmp_path, monkeypatch):
    route_path, _token, route = _write_policy(tmp_path)
    route["models"]["claude-opus-4-6"] = "opus"
    route_path.write_text(json.dumps(route), encoding="utf-8")
    monkeypatch.setattr(policy, "POLICY_PATH", route_path)
    assert policy.effective_model("global.anthropic.claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert policy.effective_model("us.anthropic.claude-opus-4-6") == "claude-opus-4-6"
    assert policy.cli_model_for("global.anthropic.claude-sonnet-4-6") == "sonnet"
    assert policy.cli_model_for("us.anthropic.claude-opus-4-6") == "opus"
    assert policy.is_claude_route(provider="openrouter", model="fable")


def test_endpoint_comparison_rejects_confusion_forms():
    assert policy.exact_proxy_url("http://127.0.0.1:4100/v1")
    assert not policy.exact_proxy_url("http://127.0.0.1:4100/v1?route=elsewhere")
    assert not policy.exact_proxy_url("http://127.0.0.1:4100/v1@evil.example")
    assert not policy.exact_proxy_url("http://127.0.0.1:4100/v1", {"base_url": "http://127.0.0.1:4101/v1"})
    assert policy.is_claude_route(
        provider="openrouter",
        model="claude-opus-4-8",
        base_url="https://api.anthropic.com.attacker.test/v1",
    )
    assert policy.is_claude_route(
        provider="openrouter",
        model="claude-opus-4-8",
        base_url="https://proxy.example.test/api.anthropic.com/v1",
    )


def test_runtime_is_locked_to_proxy_and_unknown_model_fails_closed(tmp_path, monkeypatch):
    route_path, _token, route = _write_policy(tmp_path)
    monkeypatch.setattr(policy, "POLICY_PATH", route_path)

    runtime = policy.resolve_claude_proxy_runtime(
        requested_provider="anthropic",
        target_model="anthropic/claude-fable-5",
    )
    assert runtime["provider"] == "claude-proxy"
    assert runtime["base_url"] == route["base_url"]
    assert runtime["claude_proxy_locked"] is True
    assert runtime["claude_cli_model"] == "fable"
    assert runtime["api_key"] == "test-token"

    with pytest.raises(policy.ClaudeRouteError) as exc_info:
        policy.resolve_claude_proxy_runtime(
            requested_provider="anthropic",
            target_model="claude-not-allowed",
        )
    assert exc_info.value.code == "claude_proxy_unknown_model"


def test_token_must_be_owner_only(tmp_path):
    token = tmp_path / "token"
    token.write_text("test-token", encoding="utf-8")
    token.chmod(0o640)
    with pytest.raises(policy.ClaudeRouteError) as exc_info:
        policy.read_proxy_token({"token_file": str(token)})
    assert exc_info.value.code == "claude_proxy_token_permissions"
