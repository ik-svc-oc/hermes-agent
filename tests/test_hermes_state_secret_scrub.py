"""Write-boundary + read-path secret scrub for SessionDB (J-P0b).

These prove that raw credentials never reach the on-disk ``messages`` rows
(``append_message`` / ``replace_messages``) and cannot resurface through the
FTS search path, plus that the DB file is created 0600.
"""

import sqlite3
import stat

import pytest

import agent.redact as redact
from hermes_state import SessionDB


# Fake secrets covering every shape the scrub must catch, plus one opaque
# value that matches no vendor prefix and is only caught via the env denylist.
GHP = "ghp_" + "A" * 30
GLSA = "glsa_" + "b" * 30
PHC = "phc_" + "c" * 30
TELEGRAM = "123456:" + "D" * 36
OPAQUE = "Zx9Qw7Rt2LmN8Pv0kLwEeRt"  # 23 chars, no known prefix


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


def _raw_content_rows(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        return [r[0] for r in conn.execute("SELECT content FROM messages").fetchall()]
    finally:
        conn.close()


class TestAppendMessageScrub:
    def test_shape_and_opaque_secrets_masked_on_disk(self, db, seeded_denylist):
        db.create_session(session_id="s1", source="cli")
        secret_line = f"here are creds {GHP} {GLSA} {PHC} {TELEGRAM} and {OPAQUE}"
        db.append_message("s1", "user", content=secret_line)

        rows = _raw_content_rows(db._db_path_for_test)
        assert rows, "no message row persisted"
        stored = rows[0]
        # No raw secret survives on disk.
        for secret in (GHP, GLSA, PHC, OPAQUE):
            assert secret not in stored, f"{secret!r} leaked into DB"
        assert ":" + "D" * 36 not in stored, "telegram token leaked into DB"
        # Non-secret text is preserved.
        assert "here are creds" in stored

    def test_reasoning_fields_masked_on_disk(self, db, seeded_denylist):
        db.create_session(session_id="s2", source="cli")
        db.append_message(
            "s2", "assistant",
            content="ok",
            reasoning=f"thinking about {OPAQUE}",
            reasoning_content=f"more {GHP}",
        )
        conn = sqlite3.connect(str(db._db_path_for_test))
        try:
            row = conn.execute(
                "SELECT reasoning, reasoning_content FROM messages"
            ).fetchone()
        finally:
            conn.close()
        assert OPAQUE not in (row[0] or "")
        assert GHP not in (row[1] or "")

    def test_multimodal_parts_masked_on_disk(self, db, seeded_denylist):
        db.create_session(session_id="s3", source="cli")
        db.append_message(
            "s3", "user",
            content=[
                {"type": "text", "text": f"leak {OPAQUE}"},
                {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
            ],
        )
        stored = _raw_content_rows(db._db_path_for_test)[0]
        assert OPAQUE not in stored
        assert "image_url" in stored  # structure preserved


class TestReplaceMessagesScrub:
    def test_replace_masks_secrets(self, db, seeded_denylist):
        db.create_session(session_id="s4", source="cli")
        db.replace_messages("s4", [
            {"role": "user", "content": f"first {OPAQUE}"},
            {"role": "assistant", "content": f"second {GHP}",
             "reasoning": f"why {GLSA}"},
        ])
        rows = _raw_content_rows(db._db_path_for_test)
        joined = " ".join(rows)
        for secret in (OPAQUE, GHP):
            assert secret not in joined


class TestSearchRedaction:
    def test_search_snippet_does_not_resurface_legacy_secret(self, db, seeded_denylist):
        """A row written directly (pre-patch style) must not leak via search."""
        db.create_session(session_id="s5", source="cli")
        # Bypass append_message to simulate a historical unscrubbed row.
        with db._lock:
            db._conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?,?,?,?)",
                ("s5", "user", f"legacy secret {OPAQUE} needle", 1000.0),
            )
        results = db.search_messages("needle", limit=5)
        blob = repr(results)
        assert OPAQUE not in blob, "search re-emitted a raw secret"


class TestStructuredPayloadScrub:
    """B5: reasoning_details / codex_*_items serialized columns must be masked."""

    def test_reasoning_details_and_codex_json_masked(self, db, seeded_denylist):
        db.create_session(session_id="s6", source="cli")
        db.append_message(
            "s6", "assistant",
            content="ok",
            reasoning_details=[{"type": "text", "text": f"secret {OPAQUE}"}],
            codex_reasoning_items=[{"summary": f"tok {GHP}"}],
            codex_message_items=[{"note": f"grafana {GLSA}"}],
        )
        conn = sqlite3.connect(str(db._db_path_for_test))
        try:
            row = conn.execute(
                "SELECT reasoning_details, codex_reasoning_items, "
                "codex_message_items FROM messages"
            ).fetchone()
        finally:
            conn.close()
        blob = " ".join(c or "" for c in row)
        for secret in (OPAQUE, GHP, GLSA):
            assert secret not in blob, f"{secret!r} leaked into a *_json column"
        # Structure survives (still valid JSON with the shape keys).
        assert "type" in (row[0] or "") and "summary" in (row[1] or "")


class TestOptOutStillScrubsAtRest:
    """B3: HERMES_REDACT_SECRETS=false must NOT disable at-rest scrubbing."""

    def test_disabled_flag_still_masks_db(self, db, seeded_denylist, monkeypatch):
        monkeypatch.setattr(redact, "_REDACT_ENABLED", False)  # display opt-out ON
        db.create_session(session_id="s7", source="cli")
        db.append_message("s7", "user", content=f"still {OPAQUE} and {GHP}")
        stored = _raw_content_rows(db._db_path_for_test)[0]
        assert OPAQUE not in stored
        assert GHP not in stored


class TestFailClosed:
    """B4: a redactor fault must persist a placeholder, never the raw value."""

    def test_redactor_exception_persists_placeholder(self, db, seeded_denylist, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("redactor down")
        # Break the underlying redactor so the fail-closed layer engages.
        monkeypatch.setattr(redact, "redact_message_content", _boom)
        monkeypatch.setattr(redact, "redact_for_storage", _boom)
        db.create_session(session_id="s8", source="cli")
        db.append_message("s8", "user", content=f"secret {OPAQUE}")
        stored = _raw_content_rows(db._db_path_for_test)[0]
        assert OPAQUE not in stored
        assert "REDACTION-ERROR" in stored


class TestDbFilePerms:
    def test_state_db_created_0600(self, tmp_path):
        db_path = tmp_path / "perms_state.db"
        s = SessionDB(db_path=db_path)
        try:
            mode = stat.S_IMODE(db_path.stat().st_mode)
            assert mode == 0o600, f"state.db mode {oct(mode)} != 0600"
        finally:
            s.close()

    def test_wal_shm_sidecars_created_0600(self, tmp_path):
        """S6/R4: WAL/SHM sidecars end up 0600 via _ensure_db_file_perms.

        The main DB file is pre-created 0600 (no process-global umask mutation);
        the sidecars are narrowed by _ensure_db_file_perms on the read-write open.
        """
        db_path = tmp_path / "perms_wal_state.db"
        s = SessionDB(db_path=db_path)
        try:
            # Force a write so WAL/SHM materialize.
            s.create_session(session_id="w1", source="cli")
            s.append_message("w1", "user", content="hello")
            for suffix in ("-wal", "-shm"):
                sidecar = tmp_path / (db_path.name + suffix)
                if sidecar.exists():
                    mode = stat.S_IMODE(sidecar.stat().st_mode)
                    assert mode == 0o600, f"{suffix} mode {oct(mode)} != 0600"
        finally:
            s.close()
