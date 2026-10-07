# KRONUS

**Knowledge-driven Response & Orchestration for Network Understanding and Security**

KRONUS is a network intrusion detection and automated response system built around eleven components across four tiers. It combines a fast statistical detection lane, a graph-based detection lane, and a deterministic honeypot signal with a policy-controlled response layer, an LLM incident explainer, and a self-healing production layer.

The system is designed around a simple principle: detection can be learned, but enforcement should remain bounded and auditable. Every automated response passes through the policy engine and circuit breaker, while model drift is monitored separately from enforcement.

The project is built around real NSL-KDD data, real protocol implementations, real policy evaluation, and an end-to-end runnable demo.

For measured results, implementation details, design rationale, and repository structure, see [`ABOUT.md`](ABOUT.md).

---

## Architecture

![KRONUS Architecture](docs/architecture/Kronus_architecture.svg)

The system is organized into four tiers: Detection Core, Trust & Safety,
Intelligence, and Production Maturity.

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

For full details on the independent ML experiments, data provenance, and benchmarking, see [`docs/experiments_guide.md`](docs/experiments_guide.md).

---

## ML Experiments

KRONUS provides four completely independent ML experiment pipelines with separate weights and metrics:

1. **Experiment A — Repo Data (NSL-KDD):**
   - Trains Bouncer and Detective independently using the NSL-KDD dataset already in the repository (`data/real/`).
   - Run command: `python scripts/run_repo_experiment.py`
   - Trained weights: `models/experiments/repo_data/bouncer/`, `models/experiments/repo_data/detective/`
   - Metrics: `results/experiments/repo_data_metrics.json` (Bouncer F1: 0.8179, Detective F1: 1.0000)

2. **Experiment B — CIC-IDS2017 (External Dataset):**
   - Trains Bouncer and Detective independently using the [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) external network intrusion detection dataset from the University of New Brunswick.
   - **Data must be obtained separately** — see `docs/experiments_guide.md` for instructions.
   - Run command: `python scripts/run_cic_experiment.py` (aborts cleanly, with no fabricated metrics, if data is absent)
   - Trained weights: `models/experiments/cic_ids2017/bouncer/`, `models/experiments/cic_ids2017/detective/`
   - Metrics: `results/experiments/cic_ids2017_metrics.json` (Bouncer F1: 0.9987, Detective F1: 0.9846, 2,828,563 flows)
   - **Note:** CIC-IDS2017 is an external dataset evaluated through the KRONUS inference path. It is NOT synthetic and NOT renamed NSL-KDD data.

3. **Experiment C — UNSW-NB15 (External Dataset):**
   - Trains Bouncer and Detective independently using the [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) external dataset from the Australian Centre for Cyber Security, UNSW Canberra.
   - Carries a genuine Reconnaissance class, so **both** KRONUS lanes train from one dataset: Bouncer on DoS-vs-normal, Detective on Reconnaissance-vs-normal.
   - **Data must be obtained separately** — see `docs/experiments_guide.md` for instructions.
   - Run command: `python scripts/run_unsw_nb15_experiment.py` (aborts cleanly, with no fabricated metrics, if data is absent)
   - Trained weights: `models/experiments/unsw_nb15/bouncer/`, `models/experiments/unsw_nb15/detective/`
   - Metrics: `results/experiments/unsw_nb15_metrics.json` (Bouncer F1: 0.9963, Detective F1: 1.0000, 257,673 flows)
   - **Honest disclosure:** the author-supplied pre-split CSVs carry no IP addresses, so hosts are reconstructed deterministically from each row's own connection-rate counters. Flow *features* are real; graph *topology* is reconstructed (`"synthetic_hosts": true` in the metrics).

4. **Experiment D — CIRA-CIC-DoHBrw-2020 (External Dataset):**
   - Trains the KRONUS **Detective** on DoH tunnel detection using the [CIRA-CIC-DoHBrw-2020](https://www.unb.ca/cic/datasets/dohbrw-2020.html) capture from the Canadian Institute for Cybersecurity, UNB.
   - **Detective-only, by design:** this dataset carries no DoS/flood traffic (every flow is benign DoH or a DNS tunnel), and the Bouncer's contract is strictly binary flood-vs-benign. Labelling tunnels as floods to make it train would be false, so the Bouncer is skipped and the metrics say so explicitly. This is the first experiment exercising the Detective's third class, `lateral_movement`.
   - **No reconstruction:** unlike Experiments A and C, this capture ships real IPs and real ports, so graph topology is the capture's own (`"synthetic_hosts": false`).
   - **Data must be obtained separately** — see `docs/experiments_guide.md` for instructions.
   - Run command: `python scripts/run_dohbrw2020_experiment.py` (aborts cleanly, with no fabricated metrics, if data is absent)
   - Trained weights: `models/experiments/dohbrw2020/detective/`
   - Metrics: `results/experiments/dohbrw2020_metrics.json` (Detective F1: 0.9697, 60,000 flows)

---

## Production readiness

KRONUS is currently a laptop-scale, portfolio-scoped implementation rather than a production deployment. The core detection and response path runs as a single process, while the service boundaries are already separated under `services/` to support a future per-component deployment.

The current implementation uses CPU-based autoscaling rather than true per-lane consumer-lag scaling. The Detective's headline metric also has a deliberately narrow real-data scope. These limitations are documented rather than presented as production capabilities.

The intended production topology is documented separately in `infra/helm/kronus/values.yaml`.

---

## License

MIT — see [`LICENSE`](LICENSE).
