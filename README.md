# KRONUS

**Knowledge-driven Response & Orchestration for Network Understanding and Security**

KRONUS is a network intrusion detection and automated response system built around eleven components across four tiers. It combines a fast statistical detection lane, a graph-based detection lane, and a deterministic honeypot signal with a policy-controlled response layer, an LLM incident explainer, and a self-healing production layer.

The system is designed around a simple principle: detection can be learned, but enforcement should remain bounded and auditable. Every automated response passes through the policy engine and circuit breaker, while model drift is monitored separately from enforcement.

The project is built around real NSL-KDD data, real protocol implementations, real policy evaluation, and an end-to-end runnable demo.

For measured results, implementation details, design rationale, and repository structure, see [`ABOUT.md`](ABOUT.md).

---

## Architecture

```text
                     ┌────────┐              ┌─────────────┐
                     │ Decoy  │              │ Digital Twin│
                     └───┬────┘              └──────┬──────┘
                         │ (sensor)     (hosts Decoy)│
                         ▼                           ▼
                   ┌─────────────────────────────────────┐
                   │          Ingestion Pipeline          │
                   └───┬───────────────────┬──────────────┘
                       │                    │
             (fast lane)│                    │(deep lane)
                       ▼                    ▼
                 ┌──────────┐        ┌──────────────┐
                 │ Bouncer  │        │ Graph Builder │
                 └────┬─────┘        └───────┬──────┘
                      │                       ▼
                      │                ┌──────────┐
                      │                │Detective │
                      │                └────┬─────┘
                      │                     │
       (decoy bypass) │                     │
     ┌────────────────┼─────────────────────┘
     │                ▼
     │         ┌────────────────┐
     └────────▶│ Response Engine│◀── OPA + circuit breaker
               └───────┬────────┘
              ┌────────┴──────────┐
              ▼                   ▼
         ┌─────────┐       ┌─────────────┐
         │ Logbook │       │LLM Explainer│  (async, cold path)
         └────┬────┘       └─────────────┘
              ▼
       ┌────────────────┐
       │ Drift Watcher &│  (background)
       │ Auto-Retrainer │
       └────────────────┘

     Observability (Prometheus/OTel) instruments the system.
```

### The four tiers

| Tier | Components | Purpose |
|---|---|---|
| **Detection Core** | Digital Twin, Ingestion Pipeline, Graph Builder, Bouncer, Detective, Decoy | Provides two analytical detection lanes and a deterministic honeypot signal |
| **Trust & Safety** | Response Engine, Logbook | Keeps automated actions bounded, reversible, and auditable |
| **Intelligence** | LLM Explainer | Produces a grounded incident explanation after enforcement |
| **Production Maturity** | Drift Watcher & Auto-Retrainer, Observability Stack | Monitors model quality and supports controlled retraining and promotion |

### Two detection lanes

The Bouncer reads the Event Stream directly and runs independently of graph construction. This keeps the fast path from being tied to the deep lane's snapshot cadence.

The Graph Builder and Detective form the deep lane. If both lanes produce a verdict for the same detection window, the Response Engine acts on the first verdict that clears the configured confidence threshold and records the later verdict without executing the response twice.

The Decoy follows the same policy path as the model-driven signals. Its confidence is fixed at 1.0 because any interaction with the honeypot is itself a security signal.

---

## Service Level Objectives

| SLO | Target | Mechanism |
|---|---|---|
| Decoy interaction → block (p95) | < 200 ms | Event, rule match, policy evaluation, enforcement |
| Fast-lane detect → block (p95) | < 400 ms | Direct Event Stream consumption by Bouncer |
| Deep-lane detect → block (p95) | < 3 s | ~2 s snapshot window plus CPU inference |
| Explanation latency (p95) | < 8 s | Asynchronous Job Queue; never gates enforcement |
| Sustained ingest throughput | 3,000–5,000 events/sec | Redpanda partitioning and HPA on consumer lag |
| Recall, known attacks (twin) | > 95% / class | Perfect twin labels and class-balanced training |
| External-dataset F1 | > 0.85 | CIC-IDS2017-compatible validation harness |
| False positive rate | < 2 / hour baseline | Calibrated threshold and staged rollout |
| Auto-block blast radius | 1 IP, 15-min TTL | OPA policy with automatic expiry |
| Action throttle | ≤ 20 actions/min/replica | Circuit breaker switches to alert-only |
| Pod recovery | < 30 s to ready, 0 event loss | Kubernetes probes and event-stream offset replay |
| Audit completeness | 100%, 0 broken hash links | Append-only hash-chained Logbook |
| Drift-to-retrain cycle | detect < 15 min, decide < 2 hrs | Drift Watcher, twin probes, and shadow evaluation |

These targets describe the intended operating envelope. The measured benchmark results are documented separately in [`ABOUT.md`](ABOUT.md).

---

## Quick start

```bash
pip install -r requirements-dev.txt
python scripts/download_data.py
python -m services.bouncer.train
python -m services.detective.train
python -m pytest tests/ -q
python -m demo.run_demo
```

The commands download the real NSL-KDD dataset, train both detection models, run the complete test suite, and launch the end-to-end demonstration.

For the full setup, including the OPA binary and optional PostgreSQL/Redpanda backends, see [`docs/setup.md`](docs/setup.md).

For operating the running system, see [`docs/instructions.md`](docs/instructions.md).

For frontend integration, see [`docs/frontend_guide.md`](docs/frontend_guide.md).

---

## Production readiness

KRONUS is currently a laptop-scale, portfolio-scoped implementation rather than a production deployment. The core detection and response path runs as a single process, while the service boundaries are already separated under `services/` to support a future per-component deployment.

The current implementation uses CPU-based autoscaling rather than true per-lane consumer-lag scaling. The Detective's headline metric also has a deliberately narrow real-data scope. These limitations are documented rather than presented as production capabilities.

The intended production topology is documented separately in `infra/helm/kronus/values.yaml`.

---

## License

MIT — see [`LICENSE`](LICENSE).
