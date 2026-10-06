# KRONUS ML Experiments Guide

Two completely independent ML experiments with separate weights, separate metrics, and separate training data.

---

## Experiment A — Repo Data (NSL-KDD)

### Data

| Property | Value |
|---|---|
| Dataset | NSL-KDD (KDDTrain+.txt / KDDTest+.txt) |
| Source | University of New Brunswick, [NSL-KDD page](http://www.unb.ca/cic/datasets/nsl.html) |
| Download | `python scripts/download_data.py` |
| Train | `data/real/KDDTrain+.txt` — 125,973 connection records |
| Test | `data/real/KDDTest+.txt` — 22,544 connection records |
| Labels | Normal, DoS, Probe, R2L, U2R |

### Models Trained

Both KRONUS model architectures trained fresh from the repository dataset:

**Bouncer** (XGBoost + Temperature Scaler):
- 6 features: `event_rate`, `byte_rate`, `dest_port_entropy`, `unique_dest_count`, `avg_duration_ms`, `same_dest_ratio`
- Derived from raw connection records via `services/bouncer/features.py` + `services/telemetry_exporter/converters.py`
- Binary classification: flood (DoS/DDoS) vs benign
- 80/20 train/calibration split; evaluated on held-out KDDTest+

**Detective** (3-layer GAT in NumPy/autograd):
- Graph attention network, hidden_dim=32, edge_embed=8, 3 layers
- Input: `GraphSnapshot` objects from `WindowedGraphBuilder` (2-second windows)
- Binary classification: `benign` vs `port_scan` (graph window level)
- Trained on NSL-KDD Probe and Normal streams replayed through real graph builder

### Commands

```bash
python scripts/run_repo_experiment.py
```

### Measured Metrics

| Component | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Bouncer** | 0.8840 | 0.8509 | 0.7873 | **0.8179** |
| **Detective** | 1.0000 | 1.0000 | 1.0000 | **1.0000** |

### Historical Comparison (Bouncer)

| Metric | Historical | Measured | Δ | Explanation |
|---|---|---|---|---|
| Accuracy | 0.8840 | 0.8840 | 0.0000 | Exact match |
| Precision | 0.8510 | 0.8509 | -0.0001 | Float rounding in calibration split |
| Recall | 0.7870 | 0.7873 | +0.0003 | Float rounding in calibration split |
| F1 | 0.8180 | 0.8179 | -0.0001 | Float rounding in calibration split |

Discrepancies ≤ 0.0003 are caused by the random 20% temperature-calibration split seed and CPU floating-point rounding. Results were not altered to match historical numbers.

### Artifact Paths

```
models/experiments/repo_data/bouncer/bouncer.json
models/experiments/repo_data/bouncer/calibration.json
models/experiments/repo_data/detective/detective.npz
models/experiments/repo_data/detective/detective.onnx
results/experiments/repo_data_metrics.json
```

### Weight Verification

Both models load from disk and produce valid inference verdicts:

```
Bouncer load:   PASSED (verdict: benign, conf=0.933)
Detective load: PASSED (verdict: benign, conf=0.716)
```

---

## Experiment B — CIC-IDS2017 External Dataset

### Dataset Source

**CIC-IDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017  
Published by the University of New Brunswick (UNB).

- **Official page:** https://www.unb.ca/cic/datasets/ids-2017.html
- **Files used:** GeneratedLabelledFlows / MachineLearningCSV (eight `*.pcap_ISCX.csv` files)
- **Citation:**
  > Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani,
  > "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization",
  > 4th International Conference on Information Systems Security and Privacy (ICISSP),
  > Portugal, January 2018.

### Why CIC-IDS2017?

CIC-IDS2017 is a real-world network capture (not KDD 1999-era data). It was collected on a different network, three years after NSL-KDD, using a different tool (CICFlowMeter), by a different institution. Evaluating KRONUS on it is a genuine cross-dataset generalization test.

### Data Acquisition

> **The CIC-IDS2017 dataset requires completing a short registration form on the UNB website. It cannot be fetched with a single unauthenticated URL.**

To obtain it:

```bash
# Step 1: Visit and complete the form
open https://www.unb.ca/cic/datasets/ids-2017.html

# Step 2: After downloading, place the CSVs:
python scripts/download_cicids2017.py --from-local /path/to/your/downloaded.zip

# Step 3: Run the experiment
python scripts/run_cic_experiment.py
```

The script gracefully reports `NOT_RUN` (with no fabricated metrics) when data is absent.

### What Is NOT Done

This experiment does **NOT**:
- Generate synthetic traffic to substitute for real CIC data
- Rename or copy NSL-KDD data as "CIC-IDS2017"
- Fabricate any metrics
- Claim that CIC-IDS2017 events were naturally received by the KRONUS production API

### Feature Mapping (CIC → KRONUS)

CIC-IDS2017 flow records are converted to KRONUS features using the **same pipeline used in production and in the NSL-KDD training pipeline** — no separate feature extraction code:

```
CIC flow row
  → services/telemetry_exporter/converters.flow_row_to_event()
  → services/bouncer/features.FlowFeaturizer.features_for()
  → KRONUS Bouncer features
```

| KRONUS Feature | Source |
|---|---|
| `event_rate` | Events per second in FlowFeaturizer sliding window |
| `byte_rate` | Bytes per second in FlowFeaturizer sliding window |
| `dest_port_entropy` | Shannon entropy of destination ports in window |
| `unique_dest_count` | Unique destination IPs in window |
| `avg_duration_ms` | Mean flow duration from CIC `Flow Duration` field |
| `same_dest_ratio` | Fraction of events targeting the single most-targeted destination |

### CIC Label Mapping

| CIC Label | KRONUS Task | Notes |
|---|---|---|
| BENIGN | `benign` (normal) | Direct mapping |
| DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest | `flood` (Bouncer) | DoS/DDoS family |
| DDoS | `flood` (Bouncer) | DoS/DDoS family |
| Heartbleed | `flood` (Bouncer) | DoS-family CVE in CIC's own taxonomy |
| PortScan | `port_scan` (Detective) | Probe family |
| FTP-Patator, SSH-Patator, Web Attack (3 variants), Bot, Infiltration | Non-benign signal | No clean KRONUS label — used as non-flood negative signal in Bouncer only |

The original CIC label is preserved alongside the mapped label in all intermediate structures.

### Train/Test Split

| Property | Value |
|---|---|
| Method | Per-class stratified split |
| Train ratio | 67% |
| Seed | 42 (reproducible) |
| Leakage check | Split computed from disjoint index sets — verified structurally |
| Temporal leakage | None — split is label-stratified, not time-sliced |

### Artifact Paths (when experiment is run)

```
models/experiments/cic_ids2017/bouncer/bouncer.json
models/experiments/cic_ids2017/bouncer/calibration.json
models/experiments/cic_ids2017/detective/detective.npz
models/experiments/cic_ids2017/detective/detective.onnx
results/experiments/cic_ids2017_metrics.json
```

### Current Status

```
STATUS: COMPLETE
FILES:  eight *.pcap_ISCX.csv files in data/external/cicids2017/
FLOWS:  2,828,563 loaded (1,778,042 train / 875,754 test)
```

CIC-IDS2017 requires a manual registration step on the UNB website and cannot be
downloaded unattended, but once the CSVs are in place the experiment runs
end-to-end. The measured results live in
`results/experiments/cic_ids2017_metrics.json` — no fabricated metrics.

| Metric | Bouncer (flood vs benign) | Detective (port_scan vs benign) |
|---|---|---|
| Accuracy | 0.9996 | 0.9848 |
| Precision | 0.9990 | 1.0000 |
| Recall | 0.9985 | 0.9697 |
| F1 | **0.9987** | **0.9846** |
| Train seconds | 2.279 | 2.141 |

Loaded category distribution: `normal` 2,273,097 · `dos` 380,699 ·
`probe` 158,930 · `r2l` 15,801 · `u2r` 36.

Weight load verification: Bouncer **PASSED**, Detective **PASSED**.

---

## Experiment C — UNSW-NB15 External Dataset

### Dataset Source

**UNSW-NB15** — generated by the IXIA PerfectStorm tool at the Australian Centre
for Cyber Security (ACCS), UNSW Canberra.

- **Official page:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Files used:** the two author-supplied pre-split labelled-flow CSVs
  (`UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`)
- **Citation:**
  > Nour Moustafa and Jill Slay, "UNSW-NB15: a comprehensive data set for network
  > intrusion detection systems (UNSW-NB15 network data set)",
  > 2015 Military Communications and Information Systems Conference (MilCIS).

### Why UNSW-NB15?

Unlike CIC-IDS2017 — whose labelled days here are benign, DDoS and PortScan —
UNSW-NB15 carries a genuine **Reconnaissance** class, so **both** KRONUS lanes
are trained from one dataset: the Bouncer on DoS-vs-normal, the Detective on
Reconnaissance-vs-normal. It is also a different capture (2015, IXIA
PerfectStorm) from a different institution, making it a third independent
generalization test.

### Data Acquisition

The canonical ACCS portal is gated, so the CSVs are fetched by one of:

```bash
# Verified public mirror (byte-identical to the official release)
python scripts/download_unsw_nb15.py --from-mirror

# Or a base URL / .zip carrying the official filenames
python scripts/download_unsw_nb15.py --url https://example.org/unsw_nb15.zip

# Or files you already downloaded
python scripts/download_unsw_nb15.py --from-local /path/to/folder_or.zip

# Then:
python scripts/run_unsw_nb15_experiment.py
```

> **Mirror naming trap.** The verified mirror's filenames are *inverted*: its
> `test.csv` is the official **training** set (175,341 flows) and its `train.csv`
> is the official **testing** set (82,332 flows). The downloader maps them onto
> the official names by verified byte size (32,293,018 / 15,380,800 bytes), so
> the split can never be silently swapped — training on the test set would be a
> real correctness bug, since UNSW-NB15's official test set deliberately omits
> attack families the training set contains. Size and header are both checked;
> a mismatch is a hard error, never a silent pass.

The runner aborts cleanly (with no fabricated metrics) when the data is absent.

### What Is NOT Done

This experiment does **NOT**:
- Generate synthetic traffic to substitute for real UNSW-NB15 data
- Rename or copy NSL-KDD or CIC-IDS2017 data as "UNSW-NB15"
- Fabricate any metrics

### Host Reconstruction (honest disclosure)

The author-supplied pre-split CSVs carry **no IP addresses and no port numbers**.
Source and destination hosts are therefore reconstructed deterministically from
each row's own recorded connection-rate counters (`ct_srv_src`, `ct_dst_ltm`, …),
using the same category-aware rule as `twin/nsl_kdd.py`:

- attack categories whose real traffic concentrates (`dos`, `probe`) get
  `session = 0`, so they collapse onto few sources
- everything else is keyed by row index, so benign/r2l traffic disperses
- a probe's destination varies with the row index (a scan fans out); a flood's
  destination is pinned (a flood concentrates on a victim)

**The flow FEATURES are real; the graph TOPOLOGY is reconstructed.** The metrics
record this as `"synthetic_hosts": true`. `origin` remains `DataOrigin.REAL`,
because the flow measurements genuinely come from the dataset.

### Feature Mapping (UNSW → KRONUS)

```
UNSW-NB15 flow row
  → services/telemetry_exporter/converters.flow_row_to_event()
  → services/bouncer/features.FlowFeaturizer.features_for()
  → KRONUS Bouncer features
```

The identical production pipeline used in Experiments A and B — no UNSW-specific
feature code exists, because the loader emits the shared `NSLKDDRow` shape.

### UNSW Label Mapping

| `attack_cat` | KRONUS category | KRONUS label |
|---|---|---|
| Normal | `normal` | `benign` |
| DoS | `dos` | `flood` (Bouncer) |
| Reconnaissance | `probe` | `port_scan` (Detective) |
| Generic, Exploits, Fuzzers, Analysis, Backdoor, Shellcode, Worms | `r2l` | none — generic non-benign signal |

All ten documented `attack_cat` values are mapped explicitly, so an unrecognized
label is dropped and visible rather than silently coerced. The original label is
preserved alongside the mapped one as `raw_label`.

### Train/Test Split

| Property | Value |
|---|---|
| Method | Per-class stratified split |
| Train ratio | 67% |
| Seed | 42 (reproducible) |
| Bouncer pool | DoS (flood) vs Normal, capped at 20,000 rows/class |
| Detective pool | Reconnaissance vs Normal, capped at 10,000 rows/class |
| Leakage check | Disjoint index sets — verified structurally |

The two lanes are deliberately trained on **disjoint class pools** (flood vs
scan), matching the existing external-dataset runners.

### Artifact Paths

```
models/experiments/unsw_nb15/bouncer/bouncer.json
models/experiments/unsw_nb15/bouncer/calibration.json
models/experiments/unsw_nb15/detective/detective.npz
models/experiments/unsw_nb15/detective/detective.onnx
results/experiments/unsw_nb15_metrics.json
```

### Current Status

```
STATUS: COMPLETE
FLOWS:  257,673 loaded (175,341 official train + 82,332 official test)
```

| Metric | Bouncer (flood vs benign) | Detective (port_scan vs benign) |
|---|---|---|
| Accuracy | 0.9967 | 1.0000 |
| Precision | 0.9994 | 1.0000 |
| Recall | 0.9931 | 1.0000 |
| F1 | **0.9963** | **1.0000** |
| Train seconds | 0.169 | 1.843 |

Bouncer: 24,356 train / 11,997 test rows. Detective: 134 train / 66 test graph
windows. Loaded category distribution: `normal` 93,000 · `r2l` 134,333 ·
`dos` 16,353 · `probe` 13,987.

Weight load verification: Bouncer **PASSED** (verdict `uncertain`, conf 0.721),
Detective **PASSED** (verdict `benign`, conf 1.000).

Reproducibility: re-downloading from the mirror and re-running reproduces these
figures exactly — identical F1s, identical window counts, identical verdict
confidences.

---

## What Was Removed

A previous iteration of this repository included a second experiment ("API-data experiment") that used **synthetically generated network telemetry** (generated benign/flood/port-scan events, not real data). That experiment has been completely removed because:

1. The data was not real — it was programmatically generated by the experiment script itself.
2. Training on data you generated yourself is not a legitimate independent experiment.
3. Describing generated events as "API data" was misleading.

**Removed artifacts (no longer in repo or local working tree):**
- `data/api/train_events.jsonl` and `test_events.jsonl`
- `models/experiments/api_data/` (Bouncer and Detective weights trained on generated data)
- `results/experiments/api_data_metrics.json`
- The `scripts/run_api_experiment.py` generator-based experiment logic

**Preserved infrastructure (still in repo):**
- `api/main.py` — KRONUS FastAPI endpoints (`POST /events`, `POST /events/batch`)
- `services/response_engine/` — policy engine, OPA client
- `services/bouncer/`, `services/detective/` — model code unchanged

---

## Summary Comparison

| Property | Experiment A (NSL-KDD) | Experiment B (CIC-IDS2017) | Experiment C (UNSW-NB15) |
|---|---|---|---|
| Data source | NSL-KDD (`data/real/`) | CIC-IDS2017 from UNB/CIC | UNSW-NB15 from ACCS, UNSW Canberra |
| Data type | Benchmark dataset (1999, KDD-cup era) | External real-world capture (2017) | External capture (2015, IXIA PerfectStorm) |
| Records | 125,973 train | 2,828,563 flows | 257,673 flows |
| Bouncer F1 | **0.8179** | **0.9987** | **0.9963** |
| Detective F1 | **1.0000** | **0.9846** | **1.0000** |
| Weights | `models/experiments/repo_data/` | `models/experiments/cic_ids2017/` | `models/experiments/unsw_nb15/` |
| Metrics | `results/experiments/repo_data_metrics.json` | `results/experiments/cic_ids2017_metrics.json` | `results/experiments/unsw_nb15_metrics.json` |
| Weight load | PASSED | PASSED | PASSED |
| Script | `python scripts/run_repo_experiment.py` | `python scripts/run_cic_experiment.py` | `python scripts/run_unsw_nb15_experiment.py` |

---

## Reproducibility

### Experiment A
```bash
python scripts/download_data.py          # fetch NSL-KDD
python scripts/run_repo_experiment.py    # train, evaluate, verify
```

### Experiment B
```bash
# 1. Download CIC-IDS2017 from https://www.unb.ca/cic/datasets/ids-2017.html
# 2. Place CSVs via:
python scripts/download_cicids2017.py --from-local /path/to/downloaded_archive
# 3. Run experiment:
python scripts/run_cic_experiment.py
```

### Experiment C
```bash
# Fetch the two pre-split CSVs (verified mirror, or --url / --from-local)
python scripts/download_unsw_nb15.py --from-mirror
python scripts/run_unsw_nb15_experiment.py
```

### Recorded Seeds

| Experiment | Component | Seed |
|---|---|---|
| Repo (NSL-KDD) | Bouncer | 42 (via BouncerModel default) |
| Repo (NSL-KDD) | Detective | 0 (rng seed) |
| CIC-IDS2017 | Bouncer | 42 |
| CIC-IDS2017 | Detective | 42 |
| Split (CIC) | All | 42 (per-class, stratified) |
| UNSW-NB15 | Bouncer | 42 |
| UNSW-NB15 | Detective | 42 |
| Split (UNSW) | All | 42 (per-class, stratified) |
