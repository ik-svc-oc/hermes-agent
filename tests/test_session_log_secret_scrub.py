"""Session JSON snapshot: secret scrub + 0600 perms (J-P0b).

``AIAgent._save_session_log`` writes ``~/.hermes/sessions/session_<id>.json``.
These prove opaque + shape-based secrets are masked in the snapshot and that
the file is created 0600, without standing up a full agent.
"""

import stat
from datetime import datetime
from types import SimpleNamespace

import agent.redact as redact
from run_agent import AIAgent


GHP = "ghp_" + "A" * 30
OPAQUE = "Zx9Qw7Rt2LmN8Pv0kLwEeRt"


def _fake_agent(logs_dir):
    return SimpleNamespace(
        _session_json_enabled=True,
        _session_messages=[],
        logs_dir=logs_dir,
        session_id="sess1",
        model="test-model",
        base_url="http://x",
        platform="cli",
        session_start=datetime(2026, 7, 5, 12, 0, 0),
        _cached_system_prompt="you are a bot",
        tools=[],
        verbose_logging=False,
        _clean_session_content=AIAgent._clean_session_content,
        _redact_message_content=AIAgent._redact_message_content,
    )


def test_session_snapshot_masks_secrets_and_is_0600(tmp_path, monkeypatch):
    env_file = tmp_path / "seed.env"
    env_file.write_text(f"OPAQUE_TOKEN={OPAQUE}\n", encoding="utf-8")
    monkeypatch.setattr(redact, "_ENV_DENYLIST_FILES", (str(env_file),))
    monkeypatch.setattr(redact, "_REDACT_ENABLED", True)
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    redact.reset_denylist_cache()

    agent = _fake_agent(tmp_path)
    messages = [
        {"role": "user", "content": f"paste {OPAQUE} and {GHP}"},
        # B1: reasoning + reasoning_content must be scrubbed in the snapshot too,
        # not just content.
        {"role": "assistant", "content": "ok",
         "reasoning": f"thinking {OPAQUE}",
         "reasoning_content": f"more {GHP}"},
    ]

    AIAgent._save_session_log(agent, messages=messages)

    log_file = tmp_path / "session_sess1.json"
    assert log_file.exists()

    text = log_file.read_text(encoding="utf-8")
    assert OPAQUE not in text, "opaque secret leaked into session snapshot"
    assert GHP not in text, "PAT leaked into session snapshot"
    assert "paste" in text  # non-secret content preserved

    mode = stat.S_IMODE(log_file.stat().st_mode)
    assert mode == 0o600, f"session snapshot mode {oct(mode)} != 0600"

    redact.reset_denylist_cache()
