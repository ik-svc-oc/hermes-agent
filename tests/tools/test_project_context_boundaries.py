import json
from unittest.mock import MagicMock, patch

from hermes_cli.project_context import (
    ActiveProjectContext,
    DurableWriteIntent,
    clear_project_context,
    load_project_context,
    save_project_context,
    validate_durable_write_intent,
)
from run_agent import AIAgent
from tools.memory_tool import MemoryStore, memory_tool
from tools.skill_manager_tool import skill_manage


ACTIVE_PROJECT = ActiveProjectContext(
    project_id="W080-H",
    project_name="Project Boundary Hardening",
    capsule_path="Projects/080/PROJECT_IDENTITY.md",
)


class FakeSessionDB:
    def __init__(self):
        self.meta = {}

    def get_meta(self, key):
        return self.meta.get(key)

    def set_meta(self, key, value):
        self.meta[key] = value


def _decode(result):
    return json.loads(result)


def test_project_context_save_load_clear(monkeypatch):
    import hermes_cli.project_context as project_context

    db = FakeSessionDB()
    monkeypatch.setattr(project_context, "_get_session_db", lambda: db)

    assert save_project_context("s1", ACTIVE_PROJECT) is None
    loaded = load_project_context("s1")
    assert loaded is not None
    assert loaded.project_id == "W080-H"
    assert loaded.project_name == "Project Boundary Hardening"
    assert loaded.capsule_path == "Projects/080/PROJECT_IDENTITY.md"

    assert clear_project_context("s1") is True
    assert load_project_context("s1") is None


def test_memory_write_without_active_project_allows_legacy_path(tmp_path, monkeypatch):
    import hermes_cli.project_context as project_context
    import tools.memory_tool as memory_module

    monkeypatch.setattr(project_context, "load_project_context", lambda session_id: None)
    monkeypatch.setattr(memory_module, "get_memory_dir", lambda: tmp_path)

    store = MemoryStore()
    result = _decode(memory_tool(action="add", target="memory", content="Durable non-project fact.", store=store, session_id="s1"))

    assert result["success"] is True


def test_active_project_requires_scope(monkeypatch):
    import hermes_cli.project_context as project_context

    monkeypatch.setattr(project_context, "load_project_context", lambda session_id: ACTIVE_PROJECT)
    result = _decode(memory_tool(action="add", target="memory", content="Project fact", store=MemoryStore(), session_id="s1"))

    assert result["success"] is False
    assert "Provide an explicit `scope`" in result["error"]


def test_active_project_rejects_invalid_scope(monkeypatch):
    import hermes_cli.project_context as project_context

    monkeypatch.setattr(project_context, "load_project_context", lambda session_id: ACTIVE_PROJECT)
    result = _decode(
        memory_tool(
            action="add",
            target="memory",
            content="Project fact",
            store=MemoryStore(),
            session_id="s1",
            scope="planetary",
            source_reference="Projects/080/source.md",
        )
    )

    assert result["success"] is False
    assert "unknown scope" in result["error"]


def test_active_project_requires_source_reference(monkeypatch):
    import hermes_cli.project_context as project_context

    monkeypatch.setattr(project_context, "load_project_context", lambda session_id: ACTIVE_PROJECT)
    result = _decode(
        memory_tool(
            action="add",
            target="memory",
            content="Project fact",
            store=MemoryStore(),
            session_id="s1",
            scope="project",
        )
    )

    assert result["success"] is False
    assert "`source_reference`" in result["error"]


def test_project_scope_requires_global_approval(monkeypatch):
    import hermes_cli.project_context as project_context

    monkeypatch.setattr(project_context, "load_project_context", lambda session_id: ACTIVE_PROJECT)
    result = _decode(
        memory_tool(
            action="add",
            target="memory",
            content="Project fact",
            store=MemoryStore(),
            session_id="s1",
            scope="project",
            source_reference="Projects/080/source.md",
        )
    )

    assert result["success"] is False
    assert "approved_global=true" in result["error"]


def test_global_scope_requires_approval_in_active_project():
    rejection = validate_durable_write_intent(
        intent=DurableWriteIntent(
            tool_name="memory",
            action="add",
            destination="memory",
            scope="global",
            source_reference="Projects/080/source.md",
            approved_global=False,
        ),
        active_project=ACTIVE_PROJECT,
    )

    assert rejection is not None
    assert "approved_global=true" in rejection


def test_approved_global_memory_write_passes(tmp_path, monkeypatch):
    import hermes_cli.project_context as project_context
    import tools.memory_tool as memory_module

    monkeypatch.setattr(project_context, "load_project_context", lambda session_id: ACTIVE_PROJECT)
    monkeypatch.setattr(memory_module, "get_memory_dir", lambda: tmp_path)

    store = MemoryStore()
    result = _decode(
        memory_tool(
            action="add",
            target="memory",
            content="Approved reusable workflow fact.",
            store=store,
            session_id="s1",
            scope="global",
            source_reference="Projects/080/source.md",
            project_id="W080-H",
            approved_global=True,
        )
    )

    assert result["success"] is True


def test_skill_create_requires_scope(monkeypatch):
    import hermes_cli.project_context as project_context

    monkeypatch.setattr(project_context, "load_project_context", lambda session_id: ACTIVE_PROJECT)
    result = _decode(skill_manage(action="create", name="w080-test", content="---\nname: w080-test\ndescription: test\n---\n# Test", session_id="s1"))

    assert result["success"] is False
    assert "Provide an explicit `scope`" in result["error"]


def test_skill_create_approved_global_passes(monkeypatch):
    import hermes_cli.project_context as project_context
    import tools.skill_manager_tool as skill_module

    monkeypatch.setattr(project_context, "load_project_context", lambda session_id: ACTIVE_PROJECT)
    monkeypatch.setattr(skill_module, "_create_skill", lambda name, content, category=None: {"success": True, "message": "created"})

    result = _decode(
        skill_manage(
            action="create",
            name="w080-test",
            content="---\nname: w080-test\ndescription: test\n---\n# Test",
            session_id="s1",
            scope="global",
            source_reference="Projects/080/source.md",
            project_id="W080-H",
            approved_global=True,
        )
    )

    assert result["success"] is True


def test_non_mutating_intents_skip_project_boundary():
    assert validate_durable_write_intent(
        intent=DurableWriteIntent(tool_name="memory", action="read", destination="memory"),
        active_project=ACTIVE_PROJECT,
    ) is None


def _bare_agent_with_memory_manager():
    agent = object.__new__(AIAgent)
    agent._memory_store = MagicMock()
    agent._memory_manager = MagicMock()
    agent.session_id = "s1"
    agent._parent_session_id = ""
    agent.platform = "test"
    return agent


def test_rejected_memory_write_does_not_reach_external_provider():
    agent = _bare_agent_with_memory_manager()
    rejected = json.dumps({"success": False, "error": "Durable write rejected"})

    with patch("tools.memory_tool.memory_tool", return_value=rejected):
        result = agent._invoke_tool(
            "memory",
            {"action": "add", "target": "memory", "content": "Project fact"},
            effective_task_id="task1",
        )

    assert _decode(result)["success"] is False
    agent._memory_manager.on_memory_write.assert_not_called()


def test_successful_memory_write_reaches_external_provider():
    agent = _bare_agent_with_memory_manager()
    accepted = json.dumps({"success": True, "message": "Entry added."})

    with patch("tools.memory_tool.memory_tool", return_value=accepted):
        result = agent._invoke_tool(
            "memory",
            {"action": "add", "target": "memory", "content": "Approved fact"},
            effective_task_id="task1",
            tool_call_id="call1",
        )

    assert _decode(result)["success"] is True
    agent._memory_manager.on_memory_write.assert_called_once()
