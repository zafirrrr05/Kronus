# KRONUS ML Experiments Guide

Four completely independent ML experiments with separate weights, separate metrics, and separate training data.

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

## Experiment D — CIRA-CIC-DoHBrw-2020 External Dataset

### Dataset Source

**CIRA-CIC-DoHBrw-2020** — the DoH / DoH-tunnel capture from the Canadian
Institute for Cybersecurity, University of New Brunswick.

- **Official page:** https://www.unb.ca/cic/datasets/dohbrw-2020.html
- **Files used:** the four per-class CSVs `Benign-DoH.csv`, `DNSCat2-DoH.csv`,
  `dns2tcp-DoH.csv`, `iodine-DoH.csv`
- **Citation:**
  > Mohammadreza MontazeriShatoori, Logan Davidson, Gaya Dharmawansa,
  > Arash Habibi Lashkari, "Detection of DoH Tunnels using Time-series
  > Classification of Encrypted Traffic", 5th IEEE Cyber Science and
  > Technology Congress (CyberSciTech), 2020.

### Why DoHBrw2020?

This is the first external dataset here whose ground truth is neither
volumetric nor scanning: it is **tunnelling**. DNS-over-HTTPS tunnels
(dnscat2, dns2tcp, iodine) carry command-and-control and exfiltration traffic
that looks, at the packet level, like ordinary HTTPS to a public resolver.
It is also the first dataset to exercise the Detective's third class,
`lateral_movement` — Experiments B and C only ever reached `port_scan`.

### Data Acquisition

The officially-published direct links for this dataset are **dead, and they
fail in the worst possible way**: every one of them still answers HTTP 200,
with `Content-Type: text/html` and `content-length: 108784`, serving the UNB
"CIC | Datasets" web page. Verified, not assumed:

```
http://205.174.165.80/CICDataset/DoHBrw-2020/Dataset/BenignDoH-NonDoH-CSVs.zip
http://205.174.165.80/CICDataset/DoHBrw-2020/Dataset/MaliciousDoH-CSVs.zip
http://205.174.165.80/CICDataset/DoHBrw-2020/Dataset/Total-CSVs.zip
  -> all three: status 200, Content-Type: text/html, 108784 bytes
```

A downloader that trusted the status code would save 108 KB of HTML as a
named `.zip` and report success. The downloader therefore byte-checks every
file against its known size **and** requires the first line to be a real
header (`SourceIP`/`DestinationIP`/`Duration`), which no HTML error page is.

```bash
python scripts/download_dohbrw2020.py --from-mirror
# or: --url <base-or-zip>   /   --from-local <zip-or-dir>
python scripts/run_dohbrw2020_experiment.py
```

The four files total ~165 MB and are needed only while the experiment runs;
delete `data/external/dohbrw2020/` afterwards.

### Why Only the Detective Trains Here

KRONUS's two lanes have fixed contracts, and this dataset only fits one.

The **Bouncer is strictly binary flood-vs-benign**: `predict_verdict` can
emit only `Label.FLOOD` or `Label.BENIGN`, and its six features are
volumetric. This dataset contains **no DoS/flood traffic at all** — every
capture is either benign DoH or a DNS tunnel. A tunnel is not a flood.
Training the Bouncer here would mean labelling tunnel rows `flood`, which is
false and would corrupt what the model means. So the Bouncer is **not
trained**, and `dohbrw2020_metrics.json` records that as an explicit
`bouncer.skipped` with its reason — not as an empty section.

The **Detective's** contract fits exactly: `RAW_CLASSES` is already
`[benign, port_scan, lateral_movement]`, and a tunnelled / exfiltrating flow
is precisely `lateral_movement`.

### What Is NOT Done

This experiment does **NOT**:
- Train the Bouncer on a task it cannot honestly represent (see above)
- Generate synthetic traffic to substitute for real DoHBrw2020 data
- Rename or copy another dataset as "DoHBrw2020"
- Fabricate any metrics

### No Host Reconstruction (contrast with Experiments A and C)

This capture carries **real source/destination IPs, real ports, and a real
capture clock**, so nothing is synthesized. Graph topology here is the
capture's own — internal hosts (`192.168.20.x`) reaching public resolvers
(`8.8.8.8`, `1.1.1.1`, `9.9.9.11`, `176.103.130.x`). The metrics record
`"synthetic_hosts": false`.

Two details of the real data are preserved rather than normalised away:

- **Flow direction is left as recorded.** The iodine capture contains flows
  whose 443 endpoint is the *source* (server → client). The loader does not
  flip them.
- **Rows are replayed in the dataset's true chronological order** — the
  loader sorts by the capture's own `TimeStamp`. The absolute clock is not
  carried into the model: `NSLKDDRow` has no timestamp slot, and putting the
  capture epoch into `features` would leak the label outright (the benign
  capture is December 2019; the tunnel captures are March 2020, so a
  timestamp column would separate the classes by itself). Replay therefore
  uses the same synthetic clock as the other external experiments; only the
  *sequence* is the dataset's.

### Class Mapping (DoHBrw2020 → KRONUS)

The class comes from the **filename**, not the label column — and that is a
real trap, not a stylistic choice. The benign file's trailing column is
`Label` holding `"Benign"`, but each tunnel file's trailing column is `DoH`
holding `"True"`, which asserts only "this flow is DoH" and is true of every
row in the file, so it names no class at all.

| File | KRONUS category | KRONUS label |
|---|---|---|
| `Benign-DoH.csv` (benign DoH) | `normal` | `benign` |
| `DNSCat2-DoH.csv` (dnscat2 tunnel) | `lateral_movement` | `lateral_movement` |
| `dns2tcp-DoH.csv` (dns2tcp tunnel) | `lateral_movement` | `lateral_movement` |
| `iodine-DoH.csv` (iodine tunnel) | `lateral_movement` | `lateral_movement` |

`lateral_movement` is the KRONUS label for tunnelled/exfiltrating traffic: a
DNS tunnel is a host moving data out through an established channel, not a
volumetric flood.

The mirror's aggregate `Malicious-DoH.csv` **deliberately resolves to
nothing**: it does not say which tool produced a given flow, and the loader
will not invent that. An unresolvable file contributes no rows to training,
and is kept with a NaN `category` in the DataFrame view so the drop is
visible rather than silent.

### Feature Mapping (DoHBrw2020 → KRONUS)

```
DoHBrw2020 flow row
  → services/telemetry_exporter/converters.flow_row_to_event()
  → services/bouncer/features.FlowFeaturizer.features_for()
  → KRONUS features
```

The identical production telemetry pipeline used in every other experiment —
the loader emits the shared `NSLKDDRow` shape, so no DoHBrw-specific feature
code exists. `Duration` is recorded in **seconds** and converted to
milliseconds; `total_bytes` is `FlowBytesSent + FlowBytesReceived`. The
dataset carries no protocol column, so the transport is **derived** from the
well-known port on either endpoint (443/8443/80/8080 → TCP, 53/5353 → UDP,
otherwise `other`).

### Train/Test Split

| Property | Value |
|---|---|
| Method | Per-class stratified split |
| Train ratio | 67% |
| Seed | 42 (reproducible) |
| Row cap | 15,000 rows read per CSV, 10,000 rows per class |
| Leakage check | Disjoint index sets — verified structurally |

The row cap is applied as an evenly spaced **stride**, not a prefix. That
matters here: the `lateral_movement` class is three separate captures
concatenated in file order, so `[:limit]` would hand back one tool's traffic
and call it "tunnel traffic". A stride spans every source file and each
capture's full timeline, and it is deterministic, so the experiment
reproduces exactly.

### Artifact Paths

```
models/experiments/dohbrw2020/detective/detective.npz
models/experiments/dohbrw2020/detective/detective.onnx
results/experiments/dohbrw2020_metrics.json
```

No `models/experiments/dohbrw2020/bouncer/` — that lane is deliberately not
trained on this dataset.

### Current Status

```
STATUS: COMPLETE (Detective); Bouncer deliberately skipped
FLOWS:  60,000 loaded (15,000 normal / 45,000 lateral_movement)
```

| Metric | Detective (lateral_movement vs benign) |
|---|---|
| Accuracy | 0.9697 |
| Precision | 0.9697 |
| Recall | 0.9697 |
| F1 | **0.9697** |
| Train seconds | 1.147 |
| Train / test windows | 134 / 66 |
| Uncertain verdicts | 1 |

Confusion matrix over the model's full `RAW_CLASSES`
(`[benign, port_scan, lateral_movement]`):

```
[[32,  0,  1],
 [ 0,  0,  0],
 [ 1,  0, 32]]
```

`port_scan` has no representatives in this dataset, so it is never a training
target — but it *can* still be predicted, so it is kept as a visible column
rather than dropped from the matrix. It was never predicted here. Two of 66
test windows were misclassified, one in each direction.

As in Experiments B and C, a verdict landing in the gray zone counts toward
the attack class; the count is reported (`n_uncertain_verdicts: 1`) so the
effect is visible rather than buried.

Weight load verification: Detective **PASSED** (verdict `lateral_movement`,
conf 1.000). Bouncer: **not trained on this dataset**.

Reproducibility: re-running reproduces these figures exactly — identical F1,
identical window counts, identical confusion matrix. The only fields that
differ between runs are `timestamp` and `train_seconds`.

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

| Property | Experiment A (NSL-KDD) | Experiment B (CIC-IDS2017) | Experiment C (UNSW-NB15) | Experiment D (DoHBrw2020) |
|---|---|---|---|---|
| Data source | NSL-KDD (`data/real/`) | CIC-IDS2017 from UNB/CIC | UNSW-NB15 from ACCS, UNSW Canberra | CIRA-CIC-DoHBrw-2020 from UNB/CIC |
| Data type | Benchmark dataset (1999, KDD-cup era) | External real-world capture (2017) | External capture (2015, IXIA PerfectStorm) | External capture (2019–2020, DoH tunnels) |
| Lanes trained | Bouncer + Detective | Bouncer + Detective | Bouncer + Detective | **Detective only** (no flood class exists) |
| Attack class(es) | DoS → flood, Probe → port_scan | DoS/DDoS → flood, PortScan → port_scan | DoS → flood, Recon → port_scan | DoH tunnels → **lateral_movement** |
| Hosts | Reconstructed (no IPs in source) | Real | Reconstructed (no IPs in source) | **Real** |
| Records | 125,973 train | 2,828,563 flows | 257,673 flows | 60,000 flows |
| Bouncer F1 | **0.8179** | **0.9987** | **0.9963** | not trained |
| Detective F1 | **1.0000** | **0.9846** | **1.0000** | **0.9697** |
| Weights | `models/experiments/repo_data/` | `models/experiments/cic_ids2017/` | `models/experiments/unsw_nb15/` | `models/experiments/dohbrw2020/` |
| Metrics | `results/experiments/repo_data_metrics.json` | `results/experiments/cic_ids2017_metrics.json` | `results/experiments/unsw_nb15_metrics.json` | `results/experiments/dohbrw2020_metrics.json` |
| Weight load | PASSED | PASSED | PASSED | Detective PASSED · Bouncer n/a |
| Script | `python scripts/run_repo_experiment.py` | `python scripts/run_cic_experiment.py` | `python scripts/run_unsw_nb15_experiment.py` | `python scripts/run_dohbrw2020_experiment.py` |

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

### Experiment D
```bash
# Fetch the four per-class CSVs (~165 MB)
python scripts/download_dohbrw2020.py --from-mirror
python scripts/run_dohbrw2020_experiment.py
# Reclaim the space when done:
rm -rf data/external/dohbrw2020
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
| DoHBrw2020 | Detective | 42 |
| Split (DoHBrw) | All | 42 (per-class, stratified) |
