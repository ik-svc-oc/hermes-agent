"""Regression tests for Claude route locking during /model switch.

Claude selections are hard-locked to the OAuth mini-proxy regardless of the
provider selected in the model menu.  The switch must not leave a stale
Copilot Responses route behind.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def _run_copilot_switch(
    raw_input: str,
    current_provider: str = "copilot",
    current_model: str = "gpt-5.4",
    explicit_provider: str = "",
    runtime_api_mode: str = "codex_responses",
):
    """Run switch_model with Copilot mocks and return the result."""
    def _runtime(**kwargs):
        if kwargs.get("requested") == "claude-proxy":
            return {
                "api_key": "test-claude-proxy-token",
                "base_url": "http://127.0.0.1:4100/v1",
                "api_mode": "chat_completions",
                "model": kwargs.get("target_model"),
            }
        return {
            "api_key": "ghu_test_token",
            "base_url": "https://api.githubcopilot.com",
            "api_mode": runtime_api_mode,
        }

    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=_runtime,
        ),
        patch(
            "hermes_cli.models.validate_requested_model",
            return_value=_MOCK_VALIDATION,
        ),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
    ):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model=current_model,
            explicit_provider=explicit_provider,
        )


def test_same_provider_copilot_switch_recomputes_api_mode():
    """GPT-5 → Claude on copilot: api_mode must flip to chat_completions."""
    result = _run_copilot_switch(
        raw_input="claude-opus-4.6",
        current_provider="copilot",
        current_model="gpt-5.4",
    )

    assert result.success, f"switch_model failed: {result.error_message}"
    assert result.new_model == "claude-opus-4.6"
    assert result.target_provider == "claude-proxy"
    assert result.api_mode == "chat_completions"


def test_explicit_copilot_switch_uses_selected_model_api_mode():
    """Cross-provider switch to copilot: api_mode from new model, not stale runtime."""
    result = _run_copilot_switch(
        raw_input="claude-opus-4.6",
        current_provider="openrouter",
        current_model="anthropic/claude-sonnet-4.6",
        explicit_provider="copilot",
    )

    assert result.success, f"switch_model failed: {result.error_message}"
    assert result.new_model == "claude-opus-4.6"
    assert result.target_provider == "claude-proxy"
    assert result.api_mode == "chat_completions"


def test_copilot_gpt5_keeps_codex_responses():
    """GPT-5 → GPT-5 on copilot: api_mode must stay codex_responses."""
    result = _run_copilot_switch(
        raw_input="gpt-5.4-mini",
        current_provider="copilot",
        current_model="gpt-5.4",
        runtime_api_mode="codex_responses",
    )

    assert result.success, f"switch_model failed: {result.error_message}"
    assert result.new_model == "gpt-5.4-mini"
    assert result.target_provider == "copilot"
    # gpt-5.4-mini is a GPT-5 variant — should use codex_responses
    # (gpt-5-mini is the special case that uses chat_completions)
    assert result.api_mode == "codex_responses"
