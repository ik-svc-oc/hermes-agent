"""Tests for hermes -z --usage-file (per-run JSON usage report)."""

import json
import logging
from unittest.mock import MagicMock

from hermes_cli.oneshot import (
    _mark_oneshot_session_end,
    _oneshot_end_reason,
    _write_usage_file,
    run_oneshot,
)


def _result(**overrides):
    base = {
        "estimated_cost_usd": 0.1234,
        "cost_status": "estimated",
        "cost_source": "pricing-table",
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_tokens": 800,
        "cache_write_tokens": 0,
        "reasoning_tokens": 50,
        "total_tokens": 1250,
        "api_calls": 3,
        "model": "openai/gpt-5.5",
        "provider": "openrouter",
        "session_id": "abc123",
        "completed": True,
        "failed": False,
    }
    base.update(overrides)
    return base


class TestWriteUsageFile:
    def test_writes_report_with_cost_and_tokens(self, tmp_path):
        path = tmp_path / "usage.json"
        _write_usage_file(str(path), _result())
        report = json.loads(path.read_text())
        assert report["estimated_cost_usd"] == 0.1234
        assert report["input_tokens"] == 1000
        assert report["output_tokens"] == 200
        assert report["model"] == "openai/gpt-5.5"
        assert report["api_calls"] == 3
        assert report["failed"] is False
        assert "failure" not in report

    def test_none_path_is_noop(self, tmp_path):
        # Must not raise and must not create a report file.
        _write_usage_file(None, _result())
        assert not (tmp_path / "usage.json").exists()

    def test_failure_marks_failed_and_records_message(self, tmp_path):
        path = tmp_path / "usage.json"
        _write_usage_file(str(path), {}, failure="boom")
        report = json.loads(path.read_text())
        assert report["failed"] is True
        assert report["failure"] == "boom"
        # Missing result fields serialize as null, not KeyError.
        assert report["estimated_cost_usd"] is None

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "usage.json"
        _write_usage_file(str(path), _result())
        assert json.loads(path.read_text())["total_tokens"] == 1250

    def test_unwritable_path_never_raises(self):
        # Root-owned path — the write must be swallowed, not raised.
        _write_usage_file("/proc/definitely/not/writable/usage.json", _result())

    def test_result_failed_flag_carries_through(self, tmp_path):
        path = tmp_path / "usage.json"
        _write_usage_file(str(path), _result(failed=True))
        assert json.loads(path.read_text())["failed"] is True


class TestOneshotSessionEnd:
    def test_end_reason_classification(self):
        assert _oneshot_end_reason("ok", _result()) == "oneshot_complete"
        assert _oneshot_end_reason("ok", _result(failed=True)) == "oneshot_error"
        assert _oneshot_end_reason("", _result()) == "oneshot_error"
        assert _oneshot_end_reason("ok", _result(interrupted=True)) == "oneshot_interrupted"

    def test_mark_oneshot_session_end_is_best_effort(self):
        db = MagicMock()
        _mark_oneshot_session_end(db, "sess-1", "oneshot_complete")
        db.end_session.assert_called_once_with("sess-1", "oneshot_complete")

        # Missing db/session/reason are no-ops, not crashes.
        _mark_oneshot_session_end(None, "sess-1", "oneshot_complete")
        _mark_oneshot_session_end(db, None, "oneshot_complete")
        _mark_oneshot_session_end(db, "sess-1", "")

    def test_run_oneshot_marks_session_complete(self, monkeypatch, capsys):
        db = MagicMock()
        monkeypatch.setattr(
            "hermes_cli.oneshot._create_session_db_for_oneshot",
            lambda: db,
        )
        monkeypatch.setattr(
            "hermes_cli.oneshot._run_agent",
            lambda *args, **kwargs: (
                "ROUTE-OK",
                _result(session_id="sess-oneshot", failed=False, partial=False),
            ),
        )

        try:
            assert run_oneshot("probe") == 0
        finally:
            logging.disable(logging.NOTSET)

        assert capsys.readouterr().out == "ROUTE-OK\n"
        db.end_session.assert_called_once_with("sess-oneshot", "oneshot_complete")
