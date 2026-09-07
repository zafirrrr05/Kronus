# KRONUS ML Experiments & Evaluation Guide

This document describes two completely independent machine learning experiments conducted on the KRONUS architecture:
1. **Repo-Data Experiment:** Reproducing the intended offline training pipeline using the repository dataset (NSL-KDD), generating newly trained weights, evaluating them, and comparing newly measured metrics against historical claims.
2. **API-Data Experiment:** Ingesting network telemetry genuinely transmitted over HTTP through the live KRONUS API (`POST /events`), benchmarking API latency percentiles and throughput, splitting API data into disjoint train/test sets, and independently training fresh Bouncer and Detective models from scratch (zero weight transfer/inheritance).

---

## 1. Experiment Definitions & Architectural Independence

The two experiments maintain strict separation at all stages:

```
[ REPO DATA ] (NSL-KDD KDDTrain+)
      │
      ▼
[ Independent Training ]
      │
      ├──> REPO WEIGHTS: models/experiments/repo_data/bouncer/, detective/
      │
      ▼
[ REPO METRICS ]: results/experiments/repo_data_metrics.json

-------------------------------------------------------------------------------

[ API DATA ] (Live HTTP Ingestion via POST /events)
      │
      ▼
[ Independent Training from Scratch ]
      │
      ├──> API WEIGHTS: models/experiments/api_data/bouncer/, detective/
      │
      ▼
[ API METRICS ]: results/experiments/api_data_metrics.json
```

### Critical Independence Guarantees
- **No Weight Sharing / Initialization:** API models are instantiated and trained entirely from scratch with zero transfer learning or parameter initialization from repo-trained weights.
- **Separate Artifact Storage:** Weights and checkpoints are stored in distinct directories (`models/experiments/repo_data/` and `models/experiments/api_data/`).
- **Separate Metric Registries:** Results are saved in separate JSON files (`results/experiments/repo_data_metrics.json` and `results/experiments/api_data_metrics.json`).
- **No Data Leakage:** API train and test splits have 0 sample overlap, verified by UUID checks.

---

## 2. Model Architectures

KRONUS employs a dual-lane detection core:

### Fast Lane: Bouncer Model
- **Model:** Gradient-Boosted Decision Trees (`XGBoostClassifier`) with Platt / Temperature Scaling (`TemperatureScaler`) fitted via L-BFGS on validation logits.
- **Inputs (6 Features):** `event_rate`, `byte_rate`, `dest_port_entropy`, `unique_dest_count`, `avg_duration_ms`, `same_dest_ratio`.
- **Target:** Binary classification — `flood` (1) vs `non_flood` / `benign` (0).
- **Artifacts:** `bouncer.json` (XGBoost booster) and `calibration.json` (temperature parameter $T$).

### Deep Lane: Detective Model
- **Model:** Graph Attention Network (GAT) implemented in NumPy/autograd. 3 layers, hidden dimension 32, edge embedding dimension 8, LeakyReLU non-linearities, and graph readout pooling.
- **Inputs:** `GraphSnapshot` containing node features (degrees, total bytes, port counts) and edge features (bytes, flow counts, entropy, duration).
- **Target:** Multi-class graph classification (`benign` vs `port_scan`).
- **Artifacts:** `detective.npz` (NumPy layer weights) and `detective.onnx` (ONNX export).

---

## 3. Experiment 1: Repo-Data Experiment

### Data Provenance
- **Dataset:** NSL-KDD (`KDDTrain+.txt` and `KDDTest+.txt`) downloaded from the University of New Brunswick repository.
- **Train Split:** 125,973 network connection records.
- **Test Split:** 22,544 held-out records.

### Reproduction & Training Methodology
1. **Bouncer:**
   - Featurized 125,973 train records into the 6 canonical rate and distribution features.
   - Performed an 80/20 train/calibration split.
   - Trained `XGBClassifier` with depth 4, 100 estimators, learning rate 0.1.
   - Fitted `TemperatureScaler` on calibration logits ($T \approx 0.9679$).
   - Saved weights to `models/experiments/repo_data/bouncer/`.
2. **Detective:**
   - Streamed records into `WindowedGraphBuilder` with 2.0-second time windows.
   - Constructed 200 training graph snapshots (normal vs probe attacks).
   - Trained GAT weights using batched Adam/momentum autograd over 4 epochs.
   - Saved weights to `models/experiments/repo_data/detective/`.

### Commands
```bash
# Run complete repo experiment (training, evaluation, comparison, weight check):
python scripts/run_repo_experiment.py
```

### Measured Metrics vs. Historical Comparison

| Component | Metric | Historical Claim | Measured Value | Delta | Status |
|---|---|---|---|---|---|
| **Bouncer** | Accuracy | 0.8840 | **0.8840** | 0.0000 | Match |
| | Precision | 0.8510 | **0.8509** | -0.0001 | Match |
| | Recall | 0.7870 | **0.7873** | +0.0003 | Match |
| | F1 Score | 0.8180 | **0.8179** | -0.0001 | Match |
| **Detective** | Accuracy | 1.0000 | **1.0000** | 0.0000 | Match |
| | Precision | 1.0000 | **1.0000** | 0.0000 | Match |
| | Recall | 1.0000 | **1.0000** | 0.0000 | Match |
| | F1 Score | 1.0000 | **1.0000** | 0.0000 | Match |

#### Discrepancy Analysis
- **Bouncer:** The metrics match the historical benchmark within $\pm 0.0003$. Variations are solely due to minor random seed differences in the 20% temperature calibration split and CPU floating-point rounding across XGBoost versions.
- **Detective:** The measured accuracy and F1 score on the held-out test windows are 1.0000, matching historical claims. Normal background traffic and port scan probe sweeps exhibit distinctly separate graph fan-out topologies, making structural classification cleanly separable.

---

## 4. Experiment 2: API-Data Experiment

### Data Provenance & Ingestion Path
- **Ingestion Path:** Data genuinely entered KRONUS via the HTTP `POST /events` endpoint on the running FastAPI application.
- **Traffic Generation:** A high-fidelity telemetry generator simulated realistic network subnets:
  - Internal clients: `10.0.1.0/24`
  - Internal servers: `10.0.2.0/24` (ports 80, 443, 53, 8080)
  - External attacker addresses: `198.51.100.0/24`
- **Traffic Patterns:**
  1. *Benign baseline:* Realistic web browsing, HTTPS requests, and DNS lookups.
  2. *Flood bursts:* High-rate volume surges from single external IPs targeting single internal servers.
  3. *Port scans:* Horizontal and vertical sweeps across sequential ports (20-1024).
- **Dataset Storage:**
  - `data/api/train_events.jsonl` (2,326 events)
  - `data/api/test_events.jsonl` (1,136 events)
  - Total: 3,462 live API requests.

### Live API Performance Benchmark
Measured during live HTTP transmission through the complete pipeline (Ingestion -> Bouncer -> Graph Builder -> Response Engine -> OPA Policy Client -> Logbook):

- **Total Requests:** 3,462
- **Successful Requests:** 3,462 (100%)
- **Failed Requests:** 0 (0.00% error rate)
- **HTTP Status Codes:** 2xx: 3,462, 4xx: 0, 5xx: 0
- **Throughput:** 49.60 requests/sec (single CPU worker)
- **Latency Percentiles:**
  - **Mean:** 20.111 ms
  - **Median (p50):** 19.004 ms
  - **p95:** 27.105 ms
  - **p99:** 37.356 ms
  - **Max:** 142.263 ms

### Live API Detection Correctness (Ground Truth)
Ground-truth attack labels were tracked to evaluate end-to-end API classification accuracy:
- **True Positives (TP):** 1,217
- **False Positives (FP):** 35
- **True Negatives (TN):** 1,196
- **False Negatives (FN):** 1,014
- **Precision:** 0.9720
- **Recall:** 0.5455
- **F1 Score:** 0.6988
- **Accuracy:** 0.6970

*Note:* In an online streaming API, events arriving at the start of an attack window before traffic rates cross the detection threshold contribute to false negatives until the sliding window accumulates sufficient evidence.

### Independent Training on API Data
Fresh instances of both models were trained exclusively on `data/api/train_events.jsonl` and tested on `data/api/test_events.jsonl`:

#### API Bouncer Model
- **Model Type:** XGBoost + Temperature Scaler (trained from scratch)
- **Samples:** 2,326 train / 1,136 test
- **Accuracy:** 0.9947
- **Precision:** 0.9961
- **Recall:** 0.9922
- **F1 Score:** 0.9941
- **Macro F1:** 0.9947
- **Confusion Matrix:** `[[624, 2], [4, 506]]`
- **Saved Weights:** `models/experiments/api_data/bouncer/bouncer.json`, `calibration.json`

#### API Detective Model
- **Model Type:** Graph Attention Network (GAT) in autograd (trained from scratch)
- **Snapshots:** 72 train graph windows / 34 test graph windows
- **Accuracy:** 1.0000
- **Precision:** 1.0000
- **Recall:** 1.0000
- **F1 Score:** 1.0000
- **Confusion Matrix:** `[[32, 0], [0, 2]]`
- **Saved Weights:** `models/experiments/api_data/detective/detective.npz`, `detective.onnx`

### Commands
```bash
# Run complete API experiment (traffic generation, API transmission, benchmarking, training, evaluation, weight check):
python scripts/run_api_experiment.py
```

---

## 5. Summary Comparison: REPO-TRAINED vs API-TRAINED

| Characteristic | REPO-TRAINED MODEL | API-TRAINED MODEL |
|---|---|---|
| **Data Source** | Offline NSL-KDD benchmark (`data/real/`) | Live HTTP API requests (`POST /events`, `data/api/`) |
| **Ingestion Mechanism** | Direct file parsing & batch featurization | Live HTTP client via FastAPI, full pipeline transit |
| **Sample Count** | 125,973 train / 22,544 test connections | 2,326 train / 1,136 test events (3,462 API calls) |
| **Weight Directory** | `models/experiments/repo_data/` | `models/experiments/api_data/` |
| **Bouncer F1** | **0.8179** | **0.9941** |
| **Detective F1** | **1.0000** | **1.0000** |
| **Live API Evaluation** | N/A (Offline batch evaluation) | **0.6988 F1** (TP=1217, FP=35, TN=1196, FN=1014) |
| **Weights Verified** | **YES** (Bouncer & Detective load & infer) | **YES** (Bouncer & Detective load & infer) |

---

## 6. Limitations

1. **Synthetic API Telemetry:** While genuinely sent over HTTP through the actual KRONUS API endpoints, the telemetry patterns are synthetically generated. Real enterprise networks feature far messier, jittery baseline traffic.
2. **Detective Scope on API Data:** Port scan sweeps are topologically obvious in graph representations. While the GAT achieves 1.00 F1 on distinguishing benign subnets from port scan fan-outs, more subtle stealth attacks (e.g. low-and-slow lateral movement) require multi-stage graph embeddings.
3. **Single-Worker HTTP Benchmarking:** The measured throughput (~50 req/s with ~20 ms latency) was evaluated synchronously using an in-memory client on a local machine. Production deployments with Uvicorn worker pools, Kafka/Redpanda partitioning, and Redis-backed state will exhibit significantly higher concurrent throughput.
