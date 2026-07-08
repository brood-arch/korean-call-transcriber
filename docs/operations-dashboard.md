# Operations and dashboard status

_Last verified: 2026-07-08 09:14 KST._

This document records the production-oriented OpenClaw/Hermes operating mode that sits around the reusable `korean-call-transcriber` package. It is intentionally evidence-based: do not mark a run healthy from process existence alone; read the summary/state files and health-check output.

## Current production topology

| Area | Current state | Evidence / source |
|---|---|---|
| Active owner | Hermes/WSL active runner is the intended owner; Windows Task Scheduler remains rollback-only. | `scripts/hermes_transcription_active_run.py` checks `OpenClaw-CallRecordingsAutomation` and `OpenClaw-Pipeline` and skips unless those Windows tasks are `Disabled`. |
| Duplicate-run guard | Enabled. | Atomic lock directory: `state/hermes_transcription_pipeline.lock`. Stale lock defaults to 1800 seconds and validates PID liveness before reclaim. |
| Pipeline summary | JSON summary written every run. | `state/active_run_summary.json` via `SUMMARY_PATH`. |
| Shared-event emission | Enabled but non-fatal. | `task_started`, `task_completed`, `error`, and `blocked` events are emitted with idempotency keys; event-bus failures only append log rows. |
| Health alerting | Partial success is a failure for cron/checking. | Active runner now returns non-zero for any failed step, including `partial_success`. |
| Dashboard access | Local dashboard API may be protected by token. | The companion OpenClaw dashboard server reads `DASHBOARD_TOKEN` or `state/dashboard_access_token.txt`; token can be passed as `X-Dashboard-Token`, bearer auth, cookie, or query param. |
| Dashboard responsiveness | KIS/mock-broker state should be cached/timeout-bounded. | The companion dashboard uses a short dashboard-only KIS timeout and small in-process cache so `/api/state` does not freeze on slow broker reads. |

## Active-run step order

The Hermes active runner executes the following gates in order unless a skip flag disables a stage:

1. Windows owner-task state check.
2. VRAM/memory snapshot and optional reclaim at high pressure.
3. `call_recordings_automation.py --transcribe-only` through Windows Python.
4. `batch_diarize.py` through the WSL virtualenv.
5. `build_chroma_index.py` unless `--health-only`.
6. `extract_all.py --today --batch-size 1` with `KCT_DISABLE_TELEGRAM_NOTIFY=1`.
7. `verify_extraction_quality.py --today --re-extract`.
8. `obsidian_sync_wsl.py transcripts`.
9. `index_shared_events_wsl.sh`.
10. MemPalace transcript archive.
11. Naver Mail → MemPalace archive.
12. `pipeline_health_check.py`.
13. Post-run VRAM/memory snapshot and optional reclaim.

## 2026-07-08 field health snapshot

A fast health snapshot was generated outside the package repo and should be treated as an operational field note, not a deterministic unit test fixture.

| Metric | Value |
|---|---:|
| Recent operational status | `ok`, `PIPELINE QUICK HEALTHY: latest run succeeded` |
| Overall health summary | `watch` |
| Alert conditions | `none` |
| Audio files | 6,269 |
| Transcript files | 6,485 |
| Exact complete | 6,017 |
| Exact gap | 252 |
| Reprocess queue candidates | 0 |
| Entity pending | 0 |
| RAG pending | 0 |
| Obsidian pending | 0 |
| Action queue length | 0 |
| Main retry queue | 0 total / 0 pending |
| Local retry queue | 1 total / 0 pending / `resolved: 1` |
| Exhausted quality items | 3 |
| Low-verified quality items | 6 |

Interpretation: latest active run is healthy and there are no alert conditions, but the corpus remains in `watch` because there are historical exact gaps and quality review counts. Do not present this as a fully clean corpus.

## Dashboard integration notes

The dashboard server currently lives in the OpenClaw workspace as an operator-facing companion rather than inside this package. When documenting or reusing it, keep these constraints:

- It is local-first and should bind to loopback unless explicitly proxied.
- Do not commit token values; document only `DASHBOARD_TOKEN` and `state/dashboard_access_token.txt` as sources.
- API auth accepts header, bearer, cookie, or query token for mobile convenience.
- Embed/control UI links may need the OpenClaw gateway token from environment or local OpenClaw config; never publish that token.
- `/api/state` should stay fast; any slow external state such as KIS/mock-broker balance must have short timeouts and cache fallback.

## Verification checklist before declaring green

- [ ] `state/active_run_summary.json` exists and `status == "succeeded"`.
- [ ] No failed step in `steps[*].returncode`.
- [ ] `pipeline_health_check.py` body says healthy or the residual warning is explained.
- [ ] Retry queues are empty or resolved.
- [ ] Dashboard `/api/state` returns a body quickly; status code alone is not enough.
- [ ] If `partial_success` appears, alert/report it as a failure, not as a normal run.
