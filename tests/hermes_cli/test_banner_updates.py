import json
import time

from hermes_cli import banner


def test_check_for_updates_invalidates_local_git_cache_when_head_changes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    repo_dir = hermes_home / "hermes-agent"
    (repo_dir / ".git").mkdir(parents=True)
    cache_file = hermes_home / ".update_check"
    cache_file.write_text(json.dumps({"ts": time.time(), "behind": 712, "rev": None, "local_rev": "old-head"}))

    monkeypatch.setattr(banner, "get_hermes_home", lambda: hermes_home)
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    monkeypatch.setattr(banner, "_get_local_git_head", lambda path: "new-head", raising=False)
    monkeypatch.setattr(banner, "_check_via_local_git", lambda path: 0)

    assert banner.check_for_updates() == 0

    cached = json.loads(cache_file.read_text())
    assert cached["behind"] == 0
    assert cached["local_rev"] == "new-head"
