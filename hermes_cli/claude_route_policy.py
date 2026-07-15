"""Fail-closed Claude OAuth mini-proxy routing policy.

This module is deliberately small and dependency-light.  It is the shared
policy boundary used by runtime resolution, fallback suppression, the route
doctor, and tests.  A Claude request is never allowed to proceed through the
normal provider resolver once this predicate matches.
"""

from __future__ import annotations

import json
import re
import stat
import unicodedata
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from hermes_cli.auth import AuthError


POLICY_PATH = Path("/Users/oc_runtime/.local/share/claude-proxy/route-policy.json")
# Kept as a compatibility-shaped metadata object for callers that import it,
# but intentionally contains no model map.  Classification must read the
# durable policy file through load_policy() so an upgrade or policy edit cannot
# leave an embedded alias table silently disagreeing with the proxy.
DEFAULT_POLICY = {
    "provider": "claude-proxy",
    "provider_aliases": ["claude-cli-proxy"],
    "base_url": "http://127.0.0.1:4100/v1",
    "token_file": "/Users/oc_runtime/.claude-proxy-token",
    "default_model": "claude-sonnet-4-6",
    "models": {},
}


class ClaudeRouteError(AuthError):
    """A Claude route policy failure that must never enter provider fallback."""

    def __init__(self, message: str, *, code: str = "claude_route_policy") -> None:
        super().__init__(message, provider="claude-proxy", code=code, relogin_required=False)


def _norm(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip().casefold()


def load_policy(path: Optional[Path] = None) -> dict[str, Any]:
    """Load the shared policy file; fail closed on missing or malformed bytes."""
    policy_path = Path(path or POLICY_PATH).expanduser()
    try:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClaudeRouteError(
            f"Claude route policy unavailable at {policy_path}: {exc}",
            code="claude_route_policy_missing",
        ) from exc
    if not isinstance(raw, dict):
        raise ClaudeRouteError("Claude route policy must be a JSON object", code="claude_route_policy_invalid")
    required = ("provider", "base_url", "models", "token_file")
    if any(not raw.get(key) for key in required) or not isinstance(raw.get("models"), dict):
        raise ClaudeRouteError("Claude route policy is incomplete", code="claude_route_policy_invalid")
    return raw


def canonical_provider(provider: Any, policy: Optional[dict[str, Any]] = None) -> str:
    value = _norm(provider)
    if value.startswith("custom:"):
        value = value.split(":", 1)[1]
    policy = policy or DEFAULT_POLICY
    canonical = _norm(policy.get("provider") or "claude-proxy")
    aliases = {_norm(item) for item in policy.get("provider_aliases", []) if item}
    if value == canonical or value in aliases:
        return canonical
    return value


def effective_model(model: Any) -> str:
    """Normalize provider/model menu strings to the model component."""
    value = _norm(model)
    if "/" in value:
        prefix, suffix = value.split("/", 1)
        if prefix in {
            "anthropic",
            "claude-proxy",
            "claude-cli-proxy",
            "custom:claude-proxy",
            "custom:claude-cli-proxy",
        }:
            value = suffix
    # Bedrock's cross-region Claude IDs are namespaced as
    # ``global.anthropic.<model>`` / ``us.anthropic.<model>``.  The namespace
    # is routing metadata, not a permission to use Bedrock; normalize it so a
    # known Claude ID still resolves to the OAuth CLI route.
    if ".anthropic." in value:
        value = value.split(".anthropic.", 1)[1]
    elif value.startswith("anthropic."):
        value = value.split(".", 1)[1]
    return value


def _model_map(policy: dict[str, Any]) -> dict[str, str]:
    models = policy.get("models")
    if not isinstance(models, dict):
        raise ClaudeRouteError("Claude route model map is not an object", code="claude_route_policy_invalid")
    normalized = {_norm(key): str(value).strip() for key, value in models.items() if key and value}
    if not normalized:
        raise ClaudeRouteError("Claude route model map is empty", code="claude_route_policy_invalid")
    return normalized


def cli_model_for(model: Any, policy: Optional[dict[str, Any]] = None) -> Optional[str]:
    policy = policy or load_policy()
    return _model_map(policy).get(effective_model(model))


def is_claude_route(
    *,
    provider: Any = None,
    model: Any = None,
    base_url: Any = None,
    policy: Optional[dict[str, Any]] = None,
) -> bool:
    """Return whether a request must be governed by the Claude proxy policy."""
    provider_norm = _norm(provider)
    model_norm = effective_model(model)
    policy = policy or DEFAULT_POLICY
    canonical = canonical_provider(provider_norm, policy)
    aliases = {_norm(item) for item in policy.get("provider_aliases", []) if item}
    if canonical == _norm(policy.get("provider") or "claude-proxy") or provider_norm in aliases:
        return True
    # Native Anthropic is never an alternate route, even when its model value
    # is empty or malformed. It must be forced to the proxy and then fail
    # closed if the model is not in the shared map.
    # ``canonical_provider`` strips the ``custom:`` prefix, so this also
    # covers custom:anthropic and custom:claude declarations before any
    # provider-specific resolver can reinterpret them.
    if canonical in {"anthropic", "claude", "claude-code"}:
        return True
    if policy.get("models") and cli_model_for(model_norm, policy) is not None:
        return True
    # These short menu aliases intentionally stay fail-closed even when a
    # caller supplies only a provider/model pair and classification has not
    # loaded the durable policy map yet. They are the sanctioned Claude CLI
    # families, never an alternate provider route.
    if model_norm in {"fable", "opus", "sonnet", "haiku"}:
        return True
    # Closed-shape detection catches unknown Claude/Anthropic IDs without
    # turning every arbitrary provider/model into a Claude route.
    if re.search(r"(^|[/@:._-])claude(?:[-./@:._-]|$)", model_norm):
        return True
    if re.search(r"(^|[/@:._-])anthropic(?:[-./@:._-]|$)", model_norm):
        return True
    if provider_norm in {"bedrock", "amazon-bedrock", "vertex", "vertex-ai", "google-vertex", "vertexai"}:
        return bool(re.search(r"claude|anthropic", model_norm))
    if base_url:
        try:
            if (urlsplit(str(base_url).strip()).hostname or "").casefold() == "api.anthropic.com":
                return True
        except ValueError:
            # A malformed endpoint is not an alternate safe route; let the
            # surrounding resolver fail closed rather than substring-match it.
            return True
    return False


def exact_proxy_url(candidate: Any, policy: Optional[dict[str, Any]] = None) -> bool:
    """Compare the proxy endpoint structurally, rejecting URL confusion forms."""
    if not candidate:
        return True
    policy = policy or load_policy()
    expected = urlsplit(str(policy.get("base_url") or ""))
    actual = urlsplit(str(candidate).strip())
    if actual.username or actual.password:
        return False
    if actual.query or actual.fragment:
        return False
    if (actual.scheme or "").casefold() != (expected.scheme or "").casefold():
        return False
    if (actual.hostname or "").casefold() != (expected.hostname or "").casefold():
        return False
    if actual.port != expected.port:
        return False
    return (actual.path or "").rstrip("/") == (expected.path or "").rstrip("/")


def read_proxy_token(policy: Optional[dict[str, Any]] = None) -> str:
    policy = policy or load_policy()
    path = Path(str(policy.get("token_file") or "")).expanduser()
    try:
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode):
            raise ClaudeRouteError(f"Claude proxy token is not a regular file: {path}", code="claude_proxy_token_invalid")
        if st.st_mode & 0o077:
            raise ClaudeRouteError(f"Claude proxy token is not owner-only: {path}", code="claude_proxy_token_permissions")
        token = path.read_text(encoding="utf-8").strip()
    except ClaudeRouteError:
        raise
    except OSError as exc:
        raise ClaudeRouteError(f"Claude proxy token unavailable at {path}: {exc}", code="claude_proxy_token_missing") from exc
    if not token or any(ch.isspace() for ch in token):
        raise ClaudeRouteError("Claude proxy token is empty or malformed", code="claude_proxy_token_invalid")
    return token


def resolve_claude_proxy_runtime(
    *,
    requested_provider: Any,
    target_model: Any = None,
    explicit_base_url: Any = None,
    model_cfg: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Return a locked proxy runtime, or ``None`` for non-Claude traffic."""
    policy = load_policy() if is_claude_route(
        provider=requested_provider,
        model=target_model,
        base_url=explicit_base_url,
    ) else None
    if policy is None:
        return None
    if explicit_base_url and not exact_proxy_url(explicit_base_url, policy):
        raise ClaudeRouteError(
            f"Claude requests may use only {policy['base_url']}; refusing {explicit_base_url}",
            code="claude_proxy_endpoint_rejected",
        )
    cfg_base_url = str((model_cfg or {}).get("base_url") or "").strip()
    cfg_provider = _norm((model_cfg or {}).get("provider"))
    if cfg_base_url and (cfg_provider == "anthropic" or _norm(requested_provider) == "anthropic") and not exact_proxy_url(cfg_base_url, policy):
        raise ClaudeRouteError(
            f"Native Anthropic base URL rejected for Claude route: {cfg_base_url}",
            code="claude_proxy_endpoint_rejected",
        )
    model_norm = effective_model(target_model or policy.get("default_model"))
    cli_model = cli_model_for(model_norm, policy) if model_norm else None
    if model_norm and not cli_model:
        raise ClaudeRouteError(
            f"Unknown Claude model {target_model!r}; no proxy fallback or default model is permitted",
            code="claude_proxy_unknown_model",
        )
    token = read_proxy_token(policy)
    runtime = {
        "provider": _norm(policy.get("provider") or "claude-proxy"),
        "api_mode": "chat_completions",
        "base_url": str(policy["base_url"]).rstrip("/"),
        "api_key": token,
        "source": "claude-proxy-policy",
        "requested_provider": requested_provider,
        "claude_proxy_locked": True,
        "claude_route_policy": policy,
    }
    if model_norm:
        runtime["model"] = model_norm
        runtime["claude_cli_model"] = cli_model
    return runtime


def is_locked_runtime(runtime: Optional[dict[str, Any]]) -> bool:
    return bool(isinstance(runtime, dict) and runtime.get("claude_proxy_locked"))
