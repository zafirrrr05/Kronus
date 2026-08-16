# Frontend Guide

KRONUS ships a FastAPI layer (`api/main.py`, served by `api/serve.py`)
purpose-built for a frontend to sit on top of. This document covers the
API surface, the exact data shapes, and what the frontend actually needs
to show to do the system justice — this is the "Live Map" the
architecture refers to (`REMEMBER — always: Live Map + hash-chained
Logbook`): a live view over the Response Engine and Logbook, not a new
backend component.

## Starting the API

```bash
python -m api.serve
```

Serves on `http://localhost:8000`. CORS is not configured out of the box
(this is a backend-for-a-frontend reference, not a public API) — add
`fastapi.middleware.cors.CORSMiddleware` to `api/main.py` for your
frontend's origin before deploying one.

## Endpoint reference

| Method | Path | Returns | Notes |
|---|---|---|---|
| `GET` | `/health` | `{"status": "ok", "system_ready": bool}` | Poll this before anything else |
| `GET` | `/verdicts?limit=50` | List of `DetectionVerdict` payloads, most recent last | Every tier, unfiltered |
| `GET` | `/decisions?limit=50` | List of `PolicyDecision` payloads, most recent last | |
| `GET` | `/logbook?limit=100` | List of full `AuditLogEntry` objects | The Live Map's raw feed — every entry type |
| `GET` | `/logbook/verify` | `{"intact": bool, "broken_at_index": int \| null}` | Chain integrity, on demand |
| `GET` | `/blocks/{target_ip}` | `{"target_ip": str, "blocked": bool}` | Real-time TTL check |
| `POST` | `/corrections` | `{"status": "recorded"}` or 404 | Body: `{"target_ip", "is_false_positive", "operator"}` |
| `WS` | `/live` | A stream of `AuditLogEntry` JSON objects | Polls the Logbook every 500ms server-side and pushes new entries |

All response bodies are the same Pydantic schemas defined in
`libs/schemas.py` (spec.md §3), serialized with `model_dump(mode="json")`
— field names and types match that spec exactly. There is no separate
"frontend DTO" layer to keep in sync.

### `DetectionVerdict` shape (what `/verdicts` returns)

```json
{
  "verdict_id": "uuid",
  "window_id": "string",
  "tier": "bouncer | detective | decoy",
  "label": "flood | port_scan | lateral_movement | benign | uncertain | decoy_interaction",
  "confidence": 0.0,
  "evidence": {
    "node_ids": ["..."],
    "edge_ids": ["..."],
    "attribution_method": "gnn_explainer | attention_rollout | rate_threshold | honeypot_interaction"
  }
}
```

### `PolicyDecision` shape (what `/decisions` returns)

```json
{
  "decision_id": "uuid",
  "verdict_id": "uuid",
  "action": "block | dry_run | allowlist_exempt | throttled",
  "reason_codes": ["confidence_above_threshold | on_allowlist | gray_zone | circuit_breaker_exceeded | below_gray_zone | window_already_resolved"],
  "ttl_seconds": 900,
  "policy_version": "12-char hash of policy/response.rego"
}
```

## What the frontend needs

Five views cover the system end to end. In rough priority order:

### 1. Live feed (the "Live Map")

A scrolling/streaming list backed by the `/live` WebSocket, one row per
`AuditLogEntry`. Each row's `entry_type` determines its shape and how to
render it:

- `detection` — show tier (badge color per tier: Bouncer/Detective/Decoy
  should read as visually distinct, since "which lane caught this" is
  the whole architectural story), label, confidence as a percentage or a
  small bar.
- `policy_decision` — show the action prominently; color-code
  (`block`=red, `throttled`=orange, `dry_run`=yellow/gray,
  `allowlist_exempt`=blue). This is the row a human scans fastest.
- `enforcement` — the actual TTL countdown. A live-updating "expires in
  Xm" is more useful here than a static timestamp.
- `correction` — visually distinct (a human acted here, not the system) —
  show the operator and whether it reversed a block.
- `explanation` — the LLM Explainer's narrative; good as an expandable
  detail on the corresponding `detection` row rather than its own
  timeline entry, to avoid duplicating the same incident twice in the
  feed.

### 2. Verdict detail / evidence view

When a Detective verdict is selected, render its `evidence.node_ids` and
`evidence.edge_ids` as an actual small graph (nodes + highlighted edges) —
this is the single most demo-able piece of the whole system: "here's
exactly which hosts and connections the model weighted most heavily,"
not a black-box score. A force-directed layout with the evidence edges
drawn thicker/brighter than the rest of the window's graph communicates
this in about one second of looking at it.

### 3. Gray-zone review queue

Filter `/decisions` for `reason_codes` containing `"gray_zone"` and
surface those as an actionable queue — each with a "confirm attack" /
"mark false positive" action pair that calls `POST /corrections`. This
is the human-in-the-loop surface `docs/instructions.md` describes; it
deserves its own dedicated screen, not a filter buried in the main feed.

### 4. System health strip

A persistent header/sidebar showing:
- `GET /health`'s `system_ready` as a simple up/down indicator.
- Active block count and circuit-breaker load — these are exposed as
  Prometheus metrics (`kronus_active{component="response_engine",...}`),
  not REST endpoints; either scrape Prometheus directly from the
  frontend's backend-for-frontend layer, or add a thin `/metrics-summary`
  REST endpoint to `api/main.py` that reads the same in-process
  `ACTIVE_GAUGE` values `libs/observability.py` already tracks (a few
  lines — see that module's `ACTIVE_GAUGE.labels(...).set(...)` call
  sites for what's available).
- `GET /logbook/verify`'s result as an "audit intact ✓" badge, refreshed
  periodically (this doesn't need to be real-time; once a minute is
  plenty).

### 5. Correction / block management

A simple table view over currently-active blocks (there's no
`GET /blocks` list endpoint yet — only per-IP lookup; add one if the
frontend needs it, by iterating `ResponseEngine._active_blocks` in a new
route) with a one-click "release" action wired to `POST /corrections`.

## Visual design notes

- **Lane color-coding matters more than any other visual choice.** The
  entire architectural pitch is "two independent lanes plus a
  deterministic bypass" — if the UI doesn't make tier visually obvious at
  a glance, it's hiding the most interesting thing about the system.
- **Confidence should never render as a bare float.** A percentage plus
  a threshold marker (where does 0.85 sit?) communicates the gray-zone
  concept for free, without needing separate explanatory text.
- **Dark, data-dense, monitoring-tool aesthetic** fits the domain (this
  is a SOC-analyst-facing tool, not a consumer app) — treat it like a
  Grafana dashboard or a terminal, not a marketing page.
- **Don't poll the REST endpoints on a tight interval for the live feed**
  — use the WebSocket; it already exists for exactly this and avoids
  hammering `/logbook` every second from every open tab.

## Extending the API

The current `/live` WebSocket polls the Logbook server-side every 500ms
rather than pushing on genuine append events, since `LogbookStore`
(`services/logbook/base.py`) doesn't define a change-notification hook.
For a lower-latency live feed, the cleanest extension point is adding an
optional callback parameter to `LogbookStore.append()` — or, more simply,
having `KronusSystem._handle_verdict` (in `pipeline/kronus_system.py`)
publish each new entry onto its own EventBus topic alongside writing it
to the Logbook, and having `/live` subscribe to that topic instead of
polling. Either is a small, self-contained change; neither requires
touching the Logbook's storage backends.
