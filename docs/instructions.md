# Operating KRONUS

This document is for whoever runs KRONUS day to day — what each piece of
output means, what to do when something needs a human, and how the
self-healing loop actually behaves.

## The three response modes

Every verdict from every tier resolves to exactly one of four actions
(`PolicyAction` in `libs/constants.py`), decided by `policy/response.rego`:

| Action | Meaning | What happens |
|---|---|---|
| `block` | Confidence cleared the threshold, not allowlisted, breaker not tripped | The source IP is blocked for up to 15 minutes, reversible at any time |
| `dry_run` | Gray zone, below-threshold, or benign | Logged; **gray zone specifically also notifies a human** — see below |
| `allowlist_exempt` | Would otherwise have blocked, but the source is on the allowlist | Logged, never enforced |
| `throttled` | Would otherwise have blocked, but the circuit breaker is tripped | Logged as a near-miss; the system fails safe rather than over-acting |

`dry_run` is not one undifferentiated bucket. Check `reason_codes` on the
`PolicyDecision`:

- `reason_codes: ["gray_zone"]` — genuinely ambiguous (confidence between
  0.50 and 0.85, or the model's own "uncertain" call). **This is the one
  that needs a human.** Review the evidence in the Logbook entry and
  decide.
- `reason_codes: ["below_gray_zone"]` — confidence too low to even flag as
  ambiguous, or the traffic was benign. Logged for audit completeness,
  nothing to review.
- `reason_codes: ["window_already_resolved"]` — a different verdict on
  the same detection window already triggered an action (FR-12); this one
  is logged, not acted on.

## Reviewing a gray-zone call

1. Pull the detection from the Logbook (`GET /verdicts` or query the
   store directly) — look at `tier`, `label`, `confidence`, and, for
   Detective verdicts, the `evidence.node_ids`/`evidence.edge_ids`.
2. Decide: is this actually the attack it looks like, or a false
   positive?
3. Record the correction:

   ```python
   await system.record_correction(target_ip, is_false_positive=<bool>, operator="<your-id>")
   ```

   or via the API: `POST /corrections` with
   `{"target_ip": "...", "is_false_positive": true, "operator": "..."}`.

   If `is_false_positive=True` and the source happened to already be
   blocked (e.g. a later verdict on a different window did trigger a
   block), the block is reversed immediately — it does not wait for the
   TTL. Either way, the correction becomes a labeled training example for
   the next Drift Watcher cycle.

## The allowlist

`ResponseEngine.allowlist_add(ip)` adds an IP that should never be
auto-blocked (known-good infrastructure, monitoring systems, etc.).
Allowlisted sources still generate verdicts and still get logged
(`action: allowlist_exempt`) — the allowlist suppresses *enforcement*, not
*visibility*. There is no persistence layer for the allowlist in this
build; a production deployment should back it with the same store the
Logbook uses, keyed and audited the same way.

## The circuit breaker

Caps automated *blocking* actions at 20 per minute, per instance — not
detections, not logging, just enforcement. If traffic is generating
more than 20 legitimate blocks a minute, that is itself worth
investigating (a genuine large-scale event, or a miscalibrated
threshold) — the breaker's job is to make sure the system fails toward
"stop auto-acting and tell a human" rather than toward "keep blocking
things unsupervised at an unbounded rate."

When tripped, new would-be blocks resolve to `throttled` until the
60-second window rolls past. Nothing needs to be manually reset.

## Checking audit integrity

```bash
python -c "
import asyncio
from services.logbook.sqlite_store import SQLiteLogbookStore

async def main():
    store = SQLiteLogbookStore('data/demo_logbook.db')
    ok, broken_at = await store.verify_chain()
    print('intact' if ok else f'BROKEN at entry {broken_at}')

asyncio.run(main())
"
```

or via the API: `GET /logbook/verify`. A broken chain means a stored
entry's fields no longer match its recorded hash — investigate
immediately; this should never happen under normal operation, since the
store is append-only by both convention (SQLite) and grant (Postgres,
`services/logbook/schema.sql`'s comment on required privileges).

## The self-healing loop

`DriftWatcher.run_cycle(...)` is meant to run on a schedule (an
external cron/CronJob calls it — this repo doesn't ship a standing
scheduler, since the demo runs it on demand). Each cycle:

1. Compares live feature distributions against the training baseline via
   PSI (threshold 0.2 — the standard convention: <0.1 no shift, 0.1-0.2
   moderate, >0.2 significant).
2. Reports live precision/recall against the Digital Twin's continuously-
   injected, perfectly-labeled probes.
3. Rolls up corrections and Decoy sessions accumulated since the last
   cycle as additional training signal.
4. Sets `action: retrain_triggered` if PSI crosses the threshold.

When `retrain_triggered` fires:

```bash
python -m services.bouncer.train      # or services.detective.train
```

retrains on the current real data (extend these scripts to also fold in
accumulated corrections and Decoy sessions as additional real-world-
validated examples — the `services/*/train.py` scripts are the place to
do that; the synthetic Digital Twin data is a separate, additional
augmentation source, never a substitute for real data in this pipeline).

Then decide whether to promote:

```python
from services.drift_watcher.retrainer import should_promote
from services.drift_watcher.model_registry import ModelRegistry

if should_promote(current_metrics, candidate_metrics):
    registry.promote("bouncer", candidate_version.version_id)
```

**A worse candidate is never promoted, automatically or otherwise** —
`should_promote` requires a strict improvement on the comparison metric
(F1 by default). A model that doesn't clear its predecessor stays
available in the registry (for inspection) but never becomes `current`.

## Interpreting Detective evidence

Every non-benign Detective verdict carries `evidence.node_ids` and
`evidence.edge_ids` — the specific hosts and flows the attention
mechanism weighted most heavily (an attention-rollout sum across all
GAT layers, top-3 edges by default). This is not a post-hoc
approximation bolted onto an opaque model; it's the same attention
weights the model used to make the call, surfaced directly. When
reviewing a Detective verdict, start there before looking at raw traffic.

## Running with a live Claude API key

By default `LLMExplainer` runs offline (deterministic, templated, zero
cost) — good for the demo and for CI, since it needs no external
network. To use the real API:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Every explanation's real token cost is tracked
(`kronus_token_cost_usd_total`, a Prometheus counter — visible on the
Grafana dashboard's "LLM Explainer spend" panel) using the actual
`input_tokens`/`output_tokens` the API returns; the USD-per-1k-token
conversion is an approximate, clearly-labeled constant in
`services/llm_explainer/explainer.py` (Anthropic's published pricing
changes over time — check `docs.claude.com` for current rates before
relying on this for real budgeting).
