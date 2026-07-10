"""Class-closing invariant: the persistence/emission scrub-boundary inventory.

Two consecutive security review rounds found the SAME defect class — a durable
write site that reached disk without passing through the storage-grade scrub.
Point fixes close individual holes; they do not stop the NEXT unannotated writer
from being added. This test does.

It statically enumerates every durable at-rest write site across the documented
persistence modules and asserts each is accounted for in ``ALLOWLIST`` with an
explicit disposition — either ``"scrubbed"`` (the payload passes through a
fail-closed scrubber before the write) or ``"exempt: <reason>"`` (the write
provably cannot carry an operator secret). A NEW write site — a fresh
``INSERT``/``UPDATE`` binding a secret-bearing column, or a new persistent-store
``.put()`` — appears with an unknown fingerprint and FAILS this test until a
human adds an annotated entry. A deleted/edited site leaves a stale entry that
also fails, so the inventory can never silently drift.

Scope (the two unambiguous durable-DB signals):
  * SQL ``INSERT``/``UPDATE`` binding a secret-bearing TEXT column, and
  * ``*store*.put(...)`` on a persistent store (asyncio queue ``.put`` excluded).

``json.dumps(...)`` feeding these sinks is covered transitively by the sink's
own entry (e.g. the reasoning/codex/model_config serializations feed the
``messages``/``sessions`` INSERT captured here; the response_store serialization
lives inside the captured ``.put``). Streaming/SSE/tool-arg serialization is not
an at-rest boundary and is deliberately out of scope.
"""

import hashlib
import re
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTED_MODULES = (
    "hermes_state.py",
    "gateway/platforms/api_server.py",
    "gateway/session.py",  # serializes the routing data (model_override.base_url,
                           # display_name, origin) persisted via hermes_state's
                           # gateway_routing/sessions writers; scanned so a direct
                           # durable writer added here can't bypass the inventory.
    "run_agent.py",
    "hermes_cli/session_export_md.py",
    "tools/memory_tool.py",
)

# Columns whose VALUE can hold an operator secret (credential pasted into a
# prompt, tool output, reasoning trace, serialized config, response payload).
SECRET_COLUMNS = frozenset({
    "content", "system_prompt", "model_config", "title", "reasoning",
    "reasoning_content", "reasoning_details", "codex_reasoning_items",
    "codex_message_items", "data", "entry_json", "origin_json", "display_name",
})

_INSERT_RE = re.compile(r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+(\w+)", re.IGNORECASE)
_UPDATE_RE = re.compile(r"UPDATE\s+(\w+)\s+SET", re.IGNORECASE)
_STORE_PUT_RE = re.compile(r"(\w*store\w*)\.put\s*\(", re.IGNORECASE)
_SQL_STMT_RE = re.compile(
    r"(INSERT\s+(?:OR\s+\w+\s+)?INTO\s+\w+.*?|UPDATE\s+\w+\s+SET\s+.*?)(?:\"\"\"|'''|\"|')",
    re.IGNORECASE | re.DOTALL,
)


def _fingerprint(s: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", s).strip().encode()).hexdigest()[:12]


def _scan_file(rel: str):
    """Yield (fingerprint, human_label) for each durable write site in *rel*."""
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    for m in _SQL_STMT_RE.finditer(text):
        stmt = m.group(1)[:600]
        cols = sorted(c for c in SECRET_COLUMNS if re.search(r"\b" + re.escape(c) + r"\b", stmt))
        if not cols:
            continue
        op = "INSERT" if stmt.upper().lstrip().startswith("INSERT") else "UPDATE"
        tbl_m = (_INSERT_RE if op == "INSERT" else _UPDATE_RE).search(stmt)
        tbl = tbl_m.group(1) if tbl_m else "?"
        lineno = text[: m.start()].count("\n") + 1
        yield _fingerprint(stmt), f"SQL {rel}:{lineno} {op}:{tbl}:{','.join(cols)}"
    for i, line in enumerate(text.splitlines(), 1):
        mm = _STORE_PUT_RE.search(line)
        if mm:
            yield _fingerprint(line.strip()), f"STORE_PUT {rel}:{i} {mm.group(1)}.put"


def _discover():
    sites = OrderedDict()
    for rel in DOCUMENTED_MODULES:
        for fp, label in _scan_file(rel):
            sites.setdefault(fp, label)
    return sites


# ── The inventory ────────────────────────────────────────────────────────────
# Each durable write site → its disposition. "scrubbed" == the payload passes a
# fail-closed scrubber before the write. "exempt: ..." == the write provably
# cannot carry an operator secret (the reason must say why). To (re)generate the
# label comments after an intended change, run _discover() and eyeball each site.
ALLOWLIST = {
    # -- scrubbed at the write boundary --------------------------------------
    "2cf267a72775": "scrubbed",  # SQL hermes_state.py INSERT sessions(model_config,system_prompt) — _insert_session_row scrubs both (R1)
    "1899d598f767": "scrubbed",  # SQL hermes_state.py UPDATE sessions.model_config — update_session_meta scrubs (R1)
    "aa0fddfa0756": "scrubbed",  # SQL hermes_state.py UPDATE sessions.system_prompt — update_system_prompt scrubs (R1)
    "d470aacc2ee4": "scrubbed",  # SQL hermes_state.py UPDATE sessions.title (clear-on-transfer) — set_session_title scrubs (R1)
    "896f2ab7161d": "scrubbed",  # SQL hermes_state.py UPDATE sessions.title — set_session_title scrubs (R1)
    "44f7e8809eab": "scrubbed",  # SQL hermes_state.py INSERT messages(content,reasoning,*_items) — _scrub_record_for_storage
    "dee043f3bae5": "scrubbed",  # SQL api_server.py INSERT responses.data — ResponseStore.put scrubs (R2)
    "6cefe6ad95a5": "scrubbed",  # STORE_PUT api_server.py _response_store.put — put() scrubs the payload (R2)
    "696330e946a6": "scrubbed",  # SQL hermes_state.py UPDATE sessions(display_name,origin_json) — record_gateway_session_peer scrubs (F2)
    "18a5acdc62c9": "scrubbed",  # SQL hermes_state.py UPDATE sessions(display_name,origin_json) backfill — scrubs display_name+origin (F2)
    "53f48b3b4ecf": "scrubbed",  # SQL hermes_state.py INSERT gateway_routing.entry_json — save_gateway_routing_entry scrubs base_url creds (F1)
    "b1fd76428a19": "scrubbed",  # SQL hermes_state.py INSERT gateway_routing.entry_json bulk replace — replace_gateway_routing_entries scrubs (F1)
    # -- exempt (provably no operator secret; reason MUST say why) ------------
    "6e6b641b88ee": "exempt: FTS health-probe writes the constant literal '_fts_health_probe' inside a ROLLBACK",
    "8879218d9472": "exempt: messages_fts shadow index mirrors already-scrubbed messages.content",
    "635d0145660f": "exempt: messages_fts_trigram shadow index mirrors already-scrubbed messages.content",
    "948c5232a4e7": "exempt: messages_fts rebuild mirrors already-scrubbed messages.content",
    "940a4e828f9e": "exempt: messages_fts_trigram rebuild mirrors already-scrubbed messages.content",
    "e4741b4f9f95": "exempt: json_set writes a lineage marker (_delegate_from/_branched_from), never user text",
    "f15e2d533ec0": "exempt: sets system_prompt = NULL (clears the cached snapshot), writes no value",
}


_MIN_EXEMPT_REASON_CHARS = 12


def _valid_disposition(value: str) -> bool:
    if value == "scrubbed":
        return True
    if value.startswith("exempt:"):
        # Reason-discipline: an exempt entry MUST cite why it is provably
        # non-secret. A bare "exempt:" (or a token gesture) is not a decision.
        return len(value[len("exempt:"):].strip()) >= _MIN_EXEMPT_REASON_CHARS
    return False


def test_every_durable_write_site_is_annotated():
    discovered = _discover()

    unknown = [(fp, label) for fp, label in discovered.items() if fp not in ALLOWLIST]
    assert not unknown, (
        "New durable persistence/emission write site(s) found with no scrub "
        "disposition. A raw secret could reach disk here. Route the payload "
        "through a fail-closed scrubber (scrub_text_for_storage / "
        "scrub_structured_for_storage / scrub_record_for_storage), then add the "
        "fingerprint to ALLOWLIST in tests/test_scrub_boundary_inventory.py as "
        '"scrubbed" (or "exempt: <why it cannot carry a secret>"):\n'
        + "\n".join(f'    "{fp}": "",  # {label}' for fp, label in unknown)
    )

    stale = [fp for fp in ALLOWLIST if fp not in discovered]
    assert not stale, (
        "Stale ALLOWLIST entries no longer match any write site (the code moved "
        "or changed). Re-run _discover() and update the fingerprints so the "
        f"inventory stays honest: {stale}"
    )

    bad = {fp: v for fp, v in ALLOWLIST.items() if not _valid_disposition(v)}
    assert not bad, (
        'Every ALLOWLIST disposition must be "scrubbed" or "exempt: <reason>" '
        "where the reason states why the site provably cannot carry an operator "
        f"secret (>= {_MIN_EXEMPT_REASON_CHARS} chars). Malformed entries: {bad}"
    )
