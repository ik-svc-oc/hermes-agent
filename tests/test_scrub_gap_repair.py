"""Regression tests for the scrub-gap repair (J-B5b).

Each test pins one previously-uncovered persistence / emission boundary that a
raw secret could reach:

  R1  sessions-table columns (system_prompt / model_config / title)
  R2  response_store.db (ResponseStore.put + legacy get re-emission)
  R3  bare-dict message content
  R4  SessionDB DB-file creation no longer mutates the process-global umask
  R5  memory snapshot withholds when the redactor import fails (fail CLOSED)
  R6  env-value denylist follows a relocated HERMES_HOME
  R7  --redact export refuses (raises) on a redactor fault
"""

import json
import os
import sqlite3
import stat

import pytest

import agent.redact as redact
from hermes_state import SessionDB


# Shape-based secret (always caught) + an opaque value only the env denylist
# can catch (no vendor prefix).
GHP = "ghp_" + "A" * 30
OPAQUE = "Zx9Qw7Rt2LmN8Pv0kLwEeRt"


@pytest.fixture()
def seeded_denylist(tmp_path, monkeypatch):
    """Point the env-value denylist at a temp .env holding the opaque secret."""
    env_file = tmp_path / "seed.env"
    env_file.write_text(f"OPAQUE_TOKEN={OPAQUE}\n", encoding="utf-8")
    monkeypatch.setattr(redact, "_ENV_DENYLIST_FILES", (str(env_file),))
    monkeypatch.setattr(redact, "_REDACT_ENABLED", True)
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    redact.reset_denylist_cache()
    yield
    redact.reset_denylist_cache()


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "scrub_state.db"
    session_db = SessionDB(db_path=db_path)
    session_db._db_path_for_test = db_path
    yield session_db
    session_db.close()


def _session_row(db_path, session_id):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT system_prompt, model_config, title FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()


# ── R1 ───────────────────────────────────────────────────────────────────────
class TestSessionsTableScrub:
    def test_create_session_masks_system_prompt_and_model_config(
        self, db, seeded_denylist
    ):
        db.create_session(
            session_id="s1",
            source="cli",
            system_prompt=f"system prompt leaking {OPAQUE} and {GHP}",
            model_config={"provider_key": OPAQUE, "nested": {"tok": GHP}},
        )
        system_prompt, model_config, _ = _session_row(db._db_path_for_test, "s1")
        for secret in (OPAQUE, GHP):
            assert secret not in (system_prompt or ""), f"{secret!r} leaked to system_prompt"
            assert secret not in (model_config or ""), f"{secret!r} leaked to model_config"
        # Structure of the JSON blob survives the scrub.
        assert "provider_key" in (model_config or "") and "nested" in (model_config or "")

    def test_update_system_prompt_masks_on_disk(self, db, seeded_denylist):
        db.create_session(session_id="s2", source="cli")
        db.update_system_prompt("s2", f"assembled {OPAQUE} {GHP}")
        system_prompt, _, _ = _session_row(db._db_path_for_test, "s2")
        assert OPAQUE not in (system_prompt or "")
        assert GHP not in (system_prompt or "")

    def test_update_session_meta_masks_model_config(self, db, seeded_denylist):
        db.create_session(session_id="s3", source="cli")
        db.update_session_meta("s3", json.dumps({"api_key": OPAQUE, "shape": GHP}))
        _, model_config, _ = _session_row(db._db_path_for_test, "s3")
        assert OPAQUE not in (model_config or "")
        assert GHP not in (model_config or "")

    def test_set_session_title_masks_on_disk(self, db, seeded_denylist):
        db.create_session(session_id="s4", source="cli")
        db.set_session_title("s4", f"title {OPAQUE}")
        _, _, title = _session_row(db._db_path_for_test, "s4")
        assert OPAQUE not in (title or ""), "secret leaked into session title"


# ── R2 ───────────────────────────────────────────────────────────────────────
class TestResponseStoreScrub:
    def _store(self):
        from gateway.platforms.api_server import ResponseStore

        return ResponseStore(max_size=10)

    def test_put_masks_secret_on_disk(self, seeded_denylist):
        store = self._store()
        store.put(
            "resp_1",
            {"output": [{"type": "text", "text": f"here is {OPAQUE} and {GHP}"}]},
        )
        raw = store._conn.execute(
            "SELECT data FROM responses WHERE response_id = ?", ("resp_1",)
        ).fetchone()[0]
        assert OPAQUE not in raw, "opaque secret persisted raw to response_store"
        assert GHP not in raw, "shape secret persisted raw to response_store"
        assert "output" in raw  # structure preserved

    def test_get_scrubs_legacy_unscrubbed_row(self, seeded_denylist):
        store = self._store()
        # Simulate a row written before the put()-side scrub existed.
        store._conn.execute(
            "INSERT OR REPLACE INTO responses (response_id, data, accessed_at) "
            "VALUES (?, ?, ?)",
            ("legacy", json.dumps({"text": f"legacy {OPAQUE}"}), 1.0),
        )
        store._conn.commit()
        blob = repr(store.get("legacy"))
        assert OPAQUE not in blob, "legacy row re-emitted a raw secret via get()"


# ── R3 ───────────────────────────────────────────────────────────────────────
class TestDictContentScrub:
    def test_redact_message_content_masks_bare_dict(self, seeded_denylist):
        out = redact.scrub_content_for_storage({"k": f"leak {OPAQUE}", "n": {"d": GHP}})
        blob = repr(out)
        assert OPAQUE not in blob and GHP not in blob
        assert isinstance(out, dict) and set(out.keys()) == {"k", "n"}

    def test_append_message_masks_dict_content_on_disk(self, db, seeded_denylist):
        db.create_session(session_id="d1", source="cli")
        db.append_message("d1", "user", content={"payload": f"secret {OPAQUE} {GHP}"})
        conn = sqlite3.connect(str(db._db_path_for_test))
        try:
            stored = conn.execute("SELECT content FROM messages").fetchone()[0]
        finally:
            conn.close()
        assert OPAQUE not in stored and GHP not in stored
        assert "payload" in stored  # dict structure survived


# ── R4 ───────────────────────────────────────────────────────────────────────
class TestUmaskNotMutated:
    def test_init_does_not_touch_process_global_umask(self, tmp_path, monkeypatch):
        """The DB is pre-created 0600 without any os.umask() call, so concurrent
        opens can't race on the process-global umask."""
        calls = []
        real_umask = os.umask

        def _tracking_umask(mask):
            calls.append(mask)
            return real_umask(mask)

        monkeypatch.setattr(os, "umask", _tracking_umask)
        s = SessionDB(db_path=tmp_path / "umask_state.db")
        try:
            assert calls == [], f"SessionDB.__init__ mutated the global umask: {calls}"
            mode = stat.S_IMODE((tmp_path / "umask_state.db").stat().st_mode)
            assert mode == 0o600, f"state.db mode {oct(mode)} != 0600"
        finally:
            s.close()


# ── R5 ───────────────────────────────────────────────────────────────────────
class TestMemorySnapshotImportFailClosed:
    def test_import_failure_withholds_entry(self, monkeypatch):
        from tools.memory_tool import MemoryStore

        # Make `from agent.redact import scrub_text_for_storage` fail exactly as
        # a broken/absent redactor module would.
        monkeypatch.delattr(redact, "scrub_text_for_storage", raising=False)
        out = MemoryStore._sanitize_entries_for_snapshot(
            [f"remember token {OPAQUE}"], "user.md"
        )
        assert len(out) == 1
        assert OPAQUE not in out[0], "raw secret injected despite redactor import failure"
        assert "REDACTION-ERROR" in out[0]


# ── R6 ───────────────────────────────────────────────────────────────────────
class TestDenylistFollowsHermesHome:
    def test_relocated_hermes_home_env_enters_denylist(self, tmp_path, monkeypatch):
        """A secret in <HERMES_HOME>/.env must be masked when HERMES_HOME is
        relocated — the previous hardcoded ~/.hermes/.env missed it."""
        home = tmp_path / "relocated_home"
        home.mkdir()
        (home / ".env").write_text(f"RELOCATED_TOKEN={OPAQUE}\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        # Auto-resolve mode (no explicit override) is what production uses.
        monkeypatch.setattr(redact, "_ENV_DENYLIST_FILES", None)
        monkeypatch.setattr(redact, "_REDACT_ENABLED", True)
        monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
        redact.reset_denylist_cache()
        try:
            out = redact.redact_for_storage(f"leak {OPAQUE} here", force=True)
            assert OPAQUE not in out, "relocated HERMES_HOME secret not masked"
        finally:
            redact.reset_denylist_cache()


# ── R7 ───────────────────────────────────────────────────────────────────────
class TestExportRedactRefusesOnFault:
    def test_scrub_fault_refuses_export(self, monkeypatch):
        from hermes_cli.session_export_md import (
            ExportRedactionError,
            redact_session_data,
        )

        def _boom(*a, **k):
            raise RuntimeError("redactor down")

        monkeypatch.setattr(redact, "redact_structured_for_storage", _boom)
        with pytest.raises(ExportRedactionError):
            redact_session_data({"messages": [{"role": "user", "content": "hi"}]})

    def test_opaque_secret_masked_in_export(self, seeded_denylist):
        """The storage-grade scrub catches opaque (denylist-only) secrets that
        the old shape-only redact_sensitive_text pass let through."""
        from hermes_cli.session_export_md import redact_session_data

        out = redact_session_data(
            {"messages": [{"role": "assistant", "content": f"key {OPAQUE}"}]}
        )
        assert OPAQUE not in json.dumps(out)
