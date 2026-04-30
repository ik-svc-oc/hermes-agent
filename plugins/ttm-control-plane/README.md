# ttm-control-plane

PR-F-H1 of the [Hermes alignment plan](../../../developer-handbook/docs/control-plane). Receives runtime dispatches from TTM's `HermesAdapter`, validates the principal-scoped payload, binds the run to a Hermes session, and reports `run.dispatched` back to TTM ingress.

## Wire contract

Mounted at `/api/plugins/ttm-control-plane/` on the Hermes dashboard (`127.0.0.1:9119` by default).

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Plugin metadata + binding count (unauthenticated). |
| POST | `/runs/dispatch` | Initial run-spawn dispatch from TTM. Returns 202 + `runtime_run_ref`. |
| GET | `/runs/{ref}/status` | Last-known status for a previously-dispatched run. |
| POST | `/runs/{ref}/stop` | Tear down the binding so a follow-on dispatch can rebind. |

The dispatch body mirrors TTM's `RuntimeDispatchPayload` plus `runtime_id`, `ingress_base_url`, and `principal_token`. See [`RUNTIME-ADAPTER-CONTRACT.md`](https://github.com/you-kol/developer-handbook/blob/main/docs/control-plane/RUNTIME-ADAPTER-CONTRACT.md) and [`RUNTIME-PRINCIPAL-CONTRACT.md`](https://github.com/you-kol/developer-handbook/blob/main/docs/control-plane/RUNTIME-PRINCIPAL-CONTRACT.md).

## Auth

Shared-secret header `X-TTM-Control-Plane-Secret` whose value matches the `TTM_CONTROL_PLANE_SECRET` env var. The dashboard's general auth middleware deliberately bypasses `/api/plugins/*`; this plugin owns its own check.

If `TTM_CONTROL_PLANE_SECRET` is unset, the plugin runs unauthenticated (dev/CI fallback). Production deployment must set the env var on both sides:

```bash
# Hermes side — load on dashboard service start
echo 'TTM_CONTROL_PLANE_SECRET="<long-random-value>"' >> ~/.hermes/.env

# TTM side — Doppler
doppler secrets set TTM_CONTROL_PLANE_SECRET=<long-random-value> --project ttm
doppler secrets set HERMES_GATEWAY_URL=http://127.0.0.1:9119/api/plugins/ttm-control-plane --project ttm
```

## Service install

The plugin lives inside the dashboard FastAPI app, so the dashboard must run continuously. The provided launchd plist runs it as a per-user agent:

```bash
cp launchd/ai.hermes.dashboard.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/ai.hermes.dashboard.plist
launchctl list | grep ai.hermes.dashboard
curl -s http://127.0.0.1:9119/api/plugins/ttm-control-plane/health | jq .
```

The plist runs `hermes_cli.main dashboard --no-browser --port 9119` with `KeepAlive` on non-success and writes to `~/.hermes/logs/dashboard.{log,error.log}`.

## What this PR does NOT do (deferred to follow-ups)

- **Headless agent spawn**: `_spawn_headless_session` in `dashboard/plugin_api.py` is a logging stub. The follow-up wires `hermes chat --headless --session-id <run_id>` (or the in-process equivalent) so a real agent picks up the run.
- **Pause / resume / retry-slice routes**: the contract surface is broader than this PR. The next PR adds them once the spawn pathway is real.
- **Persistence**: the binding registry is in-memory. Dashboard restarts drop active bindings. Persisting to `~/.hermes/state.db` is a follow-up.
- **TTM rebind alignment**: TTM's `POST /control-plane/{run_id}/rebind` (PR-G) still returns plaintext tokens to the operator, mirroring the deviation [TTM PR #648](https://github.com/you-kol/ttm/pull/648) corrected for dispatch. The matching rebind contract fix is a separate TTM PR.

## Tests

```bash
cd ~/.hermes/hermes-agent
venv/bin/python -m pytest tests/plugins/test_ttm_control_plane_plugin.py -v
```
