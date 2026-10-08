# KRONUS ML Experiments Guide

Seven completely independent ML experiments with separate weights, separate metrics, and separate training data. Two of them are deliberately honest about a negative result: Experiment E runs the pipeline but measures the dataset to be incapable of supporting a detection claim, and Experiment F reports a Detective score below the repo's SLO because the dataset's graph-granularity signal is genuinely faint. Experiment G is the opposite case — a near-perfect Bouncer score whose own section argues it is a weak generalisation test, because the release's flood comes from a single host and cannot speak to distributed attacks. In every case the metrics say what the data supports rather than quoting a flattering number.

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

## Experiment E — CIC-Bell-DNS-EXF-2021 External Dataset

**This is a deliberately negative result.** The experiment runs end to end —
it trains, saves, verifies, and records its metrics — but the metrics are
marked `reportable_as_detection: false` and the run's `verdict` is
`NO_VALID_DETECTION_METRIC`. The measured facts that force that verdict are
below, and they were measured, not assumed.

### Dataset Source

| Property | Value |
|---|---|
| Name | CIC-Bell-DNS-EXF-2021 |
| Publisher | Canadian Institute for Cybersecurity (CIC), University of New Brunswick |
| Authors | Mahdavifar & Ghorbani |
| URL | https://www.unb.ca/cic/datasets/dns-exf-2021.html |
| Contents | 16 per-class CSVs under three directories — `benign_labeled/` (4), `heavy_attack_labeled/` (6), `light_attack_labeled/` (6) |
| Size | ~53 MB |
| Records | 536,138 rows before the loader's guards |

`heavy_attack` and `light_attack` are both DNS exfiltration and map to the
Detective's `lateral_movement` class; `benign` maps to `normal`.

### Why This Dataset Is Worth Running Anyway

It is the only dataset in this group whose *labeling methodology* is the
finding. It fails in a way that is invisible unless you measure it, and it
fails twice over for independent reasons. Both failures are reproducible from
the released files by anyone, which is why the numbers below are in the
metrics rather than in a caveat.

### Data Acquisition

```bash
# Verified public mirror; every file is byte-size-checked against the release
python scripts/download_dnsexf2021.py --from-mirror
# or: --url <base-or-zip>   /   --from-local <zip-or-dir>
```

Without an argument the script prints manual instructions and exits 0, so a
network it cannot reach never becomes a pipeline failure.

### Failure 1 — The Labels Are Not Recoverable From The Features

The label is a **capture-level** annotation: "this capture contained an
exfiltration run" is stamped onto every row of that capture, including the
ordinary DNS lookups the monitored machine made while the attack ran. The
consequence, measured across all 536,138 rows:

| Measurement | Value |
|---|---|
| Distinct feature vectors | 46,462 (91.33% of rows are exact duplicates) |
| Vectors carrying two different KRONUS classes | 64 |
| Rows sitting on those vectors | 390,215 — **72.78% of the corpus** |
| Exfiltration rows on a label-ambiguous vector | 294,204 of 294,353 — **99.95%** |
| Hard accuracy ceiling for ANY model on these features | **0.8210** |

The smoking gun: the base32 exfiltration payload
`FHEPFCELEHFCEPFFFACACACACACACABN` carries *one identical feature vector*
under both labels. No function of those features can separate the two.

The ceiling is a bound, not a model limitation. On the dataset's own three-way
split (benign / heavy / light) it is lower still — 0.7414 — because vectors
shared between the heavy and light corpora collide there too; merging them
under KRONUS's single `lateral_movement` label is what lifts it to 0.8210.
Either way it sits **below this repository's own external-dataset SLO of F1 >
0.85**, so `slo.meets_slo` is `false` by construction, whatever a smoke test
scores.

The loader drops ambiguous vectors by default (`drop_ambiguous=True`) so the
experiment cannot score label noise. On this dataset that removes almost the
whole attack class:

| After the ambiguity guard | Rows | Distinct signatures |
|---|---|---|
| `lateral_movement` | 149 | **32** |
| `normal` | 145,774 | 46,366 |

De-duplication then collapses those 149 attack rows to the 32 the experiment
actually trains and evaluates on — against a bounded 10,000-row benign sample
(drawn from the 46,366 available, capped to keep the run comfortable on a
laptop). Every attack row is a signature seen exactly once. A model scoring
1.0 there has built a 32-entry lookup table, not a detector, and this
repository does not present that as performance.

The entry floor that enforces this is in the runner:
`MIN_REPORTABLE_MINORITY_VECTORS = 100` and `MIN_REPORTABLE_MINORITY_ROWS =
500`. Both are missed, so `detective.reportable_as_detection` is `false` and
`why_not_reportable` states why.

### Failure 2 — None Of The Lanes' Inputs Are In The Dataset

The dataset has **no IP address, no port, no byte count and no flow
duration.** Its columns are DNS query statistics plus the queried name:

```
timestamp, FQDN_count, subdomain_length, upper, lower, numeric, entropy,
special, labels, labels_max, labels_average, longest_word, sld, len,
subdomain, Label
```

So every quantity the KRONUS lanes consume is reconstructed. This is disclosed
in the metrics as `"synthetic_hosts": true` and `"features_are_real": false`,
with the exact derivation in `synthetic_hosts_note`:

| Field | Reconstruction |
|---|---|
| `source_ip` | A single constant monitored client (`10.30.0.1`). The dataset records no client identity, and a class-varying source would be a perfect label proxy. |
| `dest_ip` | Deterministic synthetic IPv4 derived from the row's **real** queried domain (`sld`) — see `_dest_ip_for`. Distinct-destination counts in a window are therefore the dataset's own distinct-domain counts. |
| `total_bytes` | Derived from the **real** query-name length (`len`) by a documented DNS packet-size formula (44 bytes overhead + name length). |
| `duration_ms` | `0` — none is recorded, so none is invented. |

Contrast with Experiments C (hosts reconstructed, features real) and D (both
real). Here neither is real, and the one genuinely discriminating real signal
in the file — `entropy`, the domain's character entropy — is not read by
either lane's feature contract.

### Why Only the Detective Trains

There is no DoS or flood traffic anywhere in this dataset: every row is either
benign DNS or an exfiltration channel. The Bouncer's contract is strictly
binary (`predict_verdict` emits only `Label.FLOOD` or `Label.BENIGN`), so
labelling exfiltration as a flood to make it train would be false. The Bouncer
is skipped and `bouncer.skipped` records the reason.

### Class Mapping (DNS-EXF-2021 → KRONUS)

| Source class | KRONUS category | KRONUS label | Files |
|---|---|---|---|
| `benign` | `normal` | `BENIGN` | 4 |
| `heavy_attack` | `lateral_movement` | `LATERAL_MOVEMENT` | 6 |
| `light_attack` | `lateral_movement` | `LATERAL_MOVEMENT` | 6 |

**The directory is the label.** The loader resolves each file's class from its
parent directory, not its filename — because the released benign corpus
contains `stateless_features-light_benign.csv`, whose name carries both
"light" and "benign". A first-token-wins filename scan would file real benign
traffic under an attack class. A name that contradicts itself resolves to
nothing and falls through to the directory instead.

### Train/Test Split

| Property | Value |
|---|---|
| Method | Per-class stratified split |
| Train ratio | 67% |
| Seed | 42 (reproducible) |
| Row cap | 10,000 benign rows, applied as an evenly spaced **stride**, not a prefix |
| Leakage check | Disjoint index sets; de-duplication means no vector can appear in both splits |

The stride matters for the same reason as in Experiment D: the benign corpus
is four separate capture files, so `[:limit]` would sample one capture's
traffic and call it the benign distribution.

### Artifact Paths

```
models/experiments/dnsexf2021/detective/detective.npz
models/experiments/dnsexf2021/detective/detective.onnx
results/experiments/dnsexf2021_metrics.json
```

No `models/experiments/dnsexf2021/bouncer/` — that lane is deliberately not
trained on this dataset.

### Current Status

```
STATUS:   COMPLETE — but VERDICT: NO_VALID_DETECTION_METRIC
FLOWS:    46,398 loaded (46,366 normal / 32 lateral_movement), from 536,138 raw rows
BOUNCER:  deliberately skipped (no flood traffic exists in this dataset)
```

| Metric | Detective (lateral_movement vs benign) |
|---|---|
| Accuracy | 1.0000 — **not reportable** |
| Precision | 1.0000 — **not reportable** |
| Recall | 1.0000 — **not reportable** |
| F1 | **1.0000 — NOT a detection result** |
| Train seconds | 0.83 |
| Train / test windows | 68 / 34 |
| Uncertain verdicts | 0 |
| Reportable as detection | **false** |

Confusion matrix over the model's full `RAW_CLASSES`
(`[benign, port_scan, lateral_movement]`):

```
[[33,  0, 0],
 [ 0,  0, 0],
 [ 0,  0, 1]]
```

Read this matrix with the numbers above in hand. Exactly **one** test window
carries `lateral_movement`, and it is classified correctly; 33 benign windows
are classified benign. That is the whole evaluation. A perfect score over 34
windows with one positive example is not evidence of detection — which is why
the runner prints the ceiling, the SLO and the non-reportable F1 alongside it
rather than the F1 alone. `port_scan` has no representatives here and was
never predicted.

Weight load verification: Detective **PASSED** (verdict `lateral_movement`,
conf 1.000). Bouncer: **not trained on this dataset**.

Reproducibility: re-running reproduces these figures. The only fields that
differ between runs are `timestamp` and `train_seconds`.

---

## Experiment F — CSE-CIC-IDS2018 External Dataset

This experiment trains **both** lanes, and they diverge: the Bouncer scores on a
real task, the Detective does not. The divergence is measured, not asserted —
the runner computes the Detective's ceiling *before* training it, so the weak
number is reported next to the evidence that explains it rather than excused
afterwards.

### Dataset Source

| Property | Value |
|---|---|
| Name | CSE-CIC-IDS2018 ("Intrusion Detection Evaluation Dataset") |
| Publisher | Canadian Institute for Cybersecurity (CIC), University of New Brunswick |
| Authors | Sharafaldin, Lashkari & Ghorbani (ICISSP 2018) |
| URL | https://www.unb.ca/cic/datasets/ids-2018.html |
| Source used | The official public AWS S3 bucket (`Processed Traffic Data for ML Algorithms`) — no form, no account, no mirror |
| Day files | 4 CSVs: `Friday-16-02-2018`, `Wednesday-21-02-2018`, `Wednesday-28-02-2018`, `Thursday-01-03-2018` |
| Size | ~980 MB on disk, ~1.95 M rows before thinning |
| Schema | 80 columns: `Dst Port, Protocol, Timestamp, Flow Duration, Tot Fwd Pkts, …, Label` |
| Raw labels seen | `Benign`, `DoS attacks-Hulk`, `DoS attacks-SlowHTTPTest`, `DDOS attack-HOIC`, `Infilteration`, plus brute-force/web/SQL/bot labels that are dropped |

Each file is byte-size-verified against the release on download, and its header
is checked against the real 80-column schema.

### Why Both Lanes Train Here

Unlike Experiments D and E, this dataset contains both flood traffic and
scanning traffic, so both lanes have a real task in it:

| Lane | Task | Source days |
|---|---|---|
| Bouncer | flood (DoS/DDoS) vs benign | `16-02` (Hulk, SlowHTTPTest) + `21-02` (HOIC) |
| Detective | benign vs `port_scan` | `28-02` + `01-03` (both `Infilteration` + Benign) |

Both lanes get **two** capture days rather than one. A single day lets a model
separate the classes by *capture date* instead of by traffic shape; two days
does not abolish that risk, so the Bouncer additionally carries a
leave-one-day-out block (§ Bouncer Results) which tests it directly.

### Data Acquisition

```bash
# The official public S3 bucket *is* the mirror: no form, no credentials
python scripts/download_cicids2018.py --from-mirror
# or: --url <base>   /   --from-local <dir>
python scripts/run_cicids2018_experiment.py
# Reclaim the space when done:
rm -rf data/external/cicids2018
```

Without an argument the script prints manual instructions and exits 0, so a
network it cannot reach never becomes a pipeline failure.

### The Capture Clock Is Real — A First For This Repository

Every other external loader here replays rows at **synthetic spacing**, because
its source carries no usable time. Equal spacing makes any *rate* feature
constant by construction, which quietly neuters the Bouncer's `event_rate` and
`byte_rate`. CSE-CIC-IDS2018 ships a real `Timestamp`, so the timed loader
returns it and the runner replays at true inter-arrival times: a 2-second
window is a real 2 seconds of capture, and a flood's burst rate appears as a
burst rate.

The clock is deliberately **not** placed in `features` — an attack day is a
different day from a benign day, so a timestamp would leak the label outright.

### Host Reconstruction (honest disclosure)

The ML-ready CSVs carry **no Source IP, no Destination IP and no Source Port**.
Only the destination *port* survives. So:

| Field | Reconstruction |
|---|---|
| `source_ip` | One constant monitored client (`172.31.69.1`). The release records no client identity, and a class-varying source would be a perfect label proxy. |
| `dest_ip` | A deterministic bijection of the **real** `Dst Port`. Distinct destinations in a window are therefore the capture's own distinct *services*. |
| `source_port` | `None` — genuinely absent from the release, never a stand-in. |
| `total_bytes` | Real: `TotLen Fwd Pkts + TotLen Bwd Pkts`. |
| `duration_ms` | Real: `Flow Duration`, microseconds → milliseconds. |
| `dest_port`, `protocol` | Real: `Dst Port`; `Protocol` IANA number → enum. |
| 74 features | Real CICFlowMeter measurements (the clock, label, port, protocol and byte totals are excluded so nothing is double-counted or leaked). |

Recorded in the metrics as `"synthetic_hosts": true`, with the derivation in
`synthetic_hosts_note`. Contrast with Experiment D (both hosts and features
real) and Experiment E (neither).

**A real limit this imposes on the graph lane:** because `dest_ip` is a
bijection of the destination port, each edge is a single flow, so `flow_count`
is identically 1 and `port_entropy` identically 0 — two of the four edge
features are constant. `degree_out` equals `unique_ports_contacted`. That is
stated in the metrics rather than left for a reader to discover.

### Label Mapping (CSE-CIC-IDS2018 → KRONUS)

| Raw label | KRONUS |
|---|---|
| `Benign` | `normal` (`Label.BENIGN`) |
| `DoS attacks-Hulk`, `-SlowHTTPTest`, `-GoldenEye`, `-Slowloris` | `dos` (`Label.FLOOD`) |
| `DDOS attack-HOIC`, `DDoS attacks-LOIC-HTTP` | `dos` (`Label.FLOOD`) |
| `Infilteration` | `probe` (`Label.PORT_SCAN`) |
| Brute force, web attacks, SQL injection, bot | **dropped** and counted, never force-fitted |

Matching is by prefix on a casefolded, dash-normalized label, because the
release is internally inconsistent — it ships `DDoS attacks-LOIC-HTTP`
alongside `DDOS attack-HOIC` (upper-case, and plural vs singular "attacks") in
the same vocabulary. Exact-string matching would silently drop real attacks.

`Infilteration` is the release's own typo, and it maps to `port_scan` rather
than `lateral_movement` on UNB's own description of the scenario — Nmap "IP
sweep, full port scan and service enumerations" — and the captured flows
measure that way: they touch ~1.7× more distinct destination ports and carry
~0.6× the packets per flow of the benign traffic in the same file. Note that
**this dataset has no `PortScan`-labelled flows at all**; the port scanning it
contains is rolled into `Infilteration`, making it the only source of
probe-class data here.

### Train/Test Split

| Lane | Split |
|---|---|
| Bouncer | Stratified 67/33 **per day** (61,224 train / 30,158 test), seeded. Splitting per day keeps each day's contribution proportional on both sides — splitting the pool would let one day land mostly in train and another mostly in test, converting a day difference into a test-set difference. |
| Detective | Stratified 67/33 per class over graph windows (11,187 train / 5,511 test), seeded, by an **evenly spaced stride** so neither the quota nor a time-of-day prefix biases the set. |

The Bouncer's stride is **global** across all four files: a flood row and a
benign row are decimated identically, so their rates stay comparable. A
per-file row budget would thin a 333 MB day harder than a 108 MB day and
manufacture a rate difference that is not in the data.

### Benign Rows Inside An Attack Are Held Out, Not Mislabelled

Because every row shares one reconstructed `source_ip`, the featurizer's
2-second aggregate is **global to the capture**, not per-host. A benign flow
arriving mid-flood sits in a window the flood dominates and its features are
the flood's. Labelling those rows "benign" would teach the model that a flood
is benign, so `attack_intervals()` grows each attack's span by a 3-second
guard and benign is drawn only from outside it.

The cost is real and is why benign comes from the infiltration days:

| Day | Rows loaded | Flood | Benign outside attack | Benign **held out** inside attack |
|---|---|---|---|---|
| `16-02` | 52,429 | 30,000 | 16 | **22,313** |
| `21-02` | 52,429 | 30,000 | 97 | **18,136** |
| `28-02` | 30,654 | 0 | 21,015 | 6,191 |
| `01-03` | 16,555 | 0 | 10,254 | 1,648 |

On the flood days, 99.9% of benign traffic is concurrent with the attack.

### Bouncer Results

| Metric | Value |
|---|---|
| Accuracy | 0.9902 |
| Precision | 0.9994 |
| Recall | 0.9857 |
| **F1** | **0.9925** |
| AUC | 0.9994 |
| Train / test rows | 61,224 (40,200 flood / 21,024 benign) / 30,158 |
| Confusion matrix (benign, flood) | `[[10347, 11], [284, 19516]]` |

Class-conditional medians, which show the separation is real traffic shape and
not an artifact:

| Feature | Flood | Benign |
|---|---|---|
| `event_rate` | 44.0 | 1.5 |
| `avg_duration_ms` | 12.38 | 2,505.7 |
| `dest_port_entropy` | 0.0 | 1.0 |
| `unique_dest_count` | 1.0 | 2.0 |
| `same_dest_ratio` | 1.0 | 0.5 |
| `byte_rate` | 1,967.5 | 1,440.0 |

A model that separates on traffic shape and one that separates on a *day
fingerprint* can both score 0.99, so the runner also fits on one flood day plus
one benign day and scores the two days it has never seen:

| Trained on (flood + benign) | Tested on (flood + benign) | F1 | AUC | Flood rows missed |
|---|---|---|---|---|
| `16-02` + `28-02` | `21-02` + `01-03` | **0.9985** | 0.9995 | 90 / 30,000 |
| `16-02` + `01-03` | `21-02` + `28-02` | **0.9977** | 0.9993 | 84 / 30,000 |
| `21-02` + `28-02` | `16-02` + `01-03` | **0.8684** | 0.9883 | 6,978 / 30,000 |
| `21-02` + `01-03` | `16-02` + `28-02` | **0.8681** | 0.9925 | 6,972 / 30,000 |

**Read the asymmetry honestly.** A model trained on the slow/rate-limited
family (Hulk, SlowHTTPTest) detects HOIC essentially perfectly. An
HOIC-trained model misses ~23% of Hulk/SlowHTTPTest traffic. But its AUC in
that direction is still 0.988–0.993 — the *ranking* holds, and what fails is
the calibrated 0.5 threshold sitting in the wrong place for a flood family
whose rate distribution differs from the one it was calibrated on. An operator
would recover most of it by re-thresholding; a deployer should re-calibrate per
attack family. This is the honest limit of a threshold-on-rate detector, and it
is in the metrics rather than in a footnote.

### The Detective's Ceiling, Measured Before Training

The `Infilteration` label is only weakly separable from benign at the
production 2-second cadence. Rather than train a model and explain a poor score
afterwards, the runner measures the ceiling first with a 3-fold cross-validated
linear probe over **the same windows the GAT is trained on** (all 32,777 of
them, not a quota):

| Measurement | Value |
|---|---|
| Windows at production cadence | 32,777 (8,542 probe / 24,235 benign) |
| Probe AUC | **0.6017** |
| Probe F1 (probe class) | 0.4033 |
| Majority-class accuracy | 0.7394 |
| Median edges per window (attack / benign) | 6 / 5 |

### Window-Cadence Sensitivity

Does a coarser window reveal the attack, or just measure the same small
difference more precisely? One pass over each day drives every cadence's
builder over the *same* event stream:

| Window | Windows | Probe AUC | Probe F1 | Median edges attack/benign | Ratio |
|---|---|---|---|---|---|
| **2 s** (production) | 32,777 | **0.6017** | 0.4033 | 6 / 5 | 1.20 |
| 5 s | 13,357 | 0.6327 | 0.4289 | 12 / 10 | 1.20 |
| 10 s | 6,742 | 0.6598 | 0.4438 | 22 / 17 | 1.29 |
| 30 s | 2,268 | 0.7395 | 0.4997 | 58 / 45 | 1.29 |
| 60 s | 1,136 | 0.7983 | 0.5741 | 110 / 85 | 1.29 |
| 120 s | 567 | **0.8518** | 0.6374 | 213 / 164 | 1.30 |

AUC climbs 0.60 → 0.85 across a 60× change in window length, and the
attack/benign activity ratio stays pinned between 1.20 and 1.30 the whole way.
**The rising AUC is precision of estimate on a constant ~25% activity
difference, not a port-scan signature.** Even at 120 seconds the F1 (0.6374)
remains below this repository's 0.85 SLO. The attack genuinely is faint in this
capture at graph granularity.

### Detective Results

| Metric | Value |
|---|---|
| Accuracy | 0.5075 |
| Precision (macro) | 0.4717 |
| Recall (macro) | 0.4986 |
| **F1 (macro)** | **0.3401 — below the 0.85 SLO** |
| Train / test windows | 11,187 (5,492 probe / 5,695 benign) / 5,511 |
| Train seconds | 89.7 (4 epochs, batch 16) |
| Predictions | 5,479 `benign`, 23 `port_scan`, 9 `uncertain` |
| Confusion matrix | `[[2787, 13, 5], [2692, 10, 4], [0, 0, 0]]` over `[benign, port_scan, uncertain]` |

The GAT learns essentially the majority class, and scores **below the linear
probe's ceiling** (F1 0.4033) — which is the expected outcome, not a training
bug: a 3-class head must apportion probability to `lateral_movement`, a class
this dataset never trains.

Abstentions are recorded **verbatim**. `predict_verdict` returns a label
post-`derive_label`, so it can be `uncertain`; those 9 verdicts are never
folded into the class they "probably" meant, and they count against recall
because a value equal to neither class can be no class's true positive.

### Why The Weak Score Is Not Label Noise

It would be convenient to blame the label. The measurement says otherwise:

| Measurement | Value |
|---|---|
| Rows sampled (stride 40, both infiltration days, features on) | 23,605 |
| Distinct feature vectors | 20,838 |
| Vectors carrying two different classes | 194 |
| Rows on those vectors | 1,762 — **7.46%** |
| Label-consistency ceiling for any function of the 74 features | **0.9778** |
| Rows by category | 19,537 normal / 4,068 probe |

Read that ceiling for what it is: because most rows carry a unique feature
vector, this figure is a **label-consistency** measure — it says only ~2.2% of
rows are irreducible, i.e. the `Infilteration` label is internally consistent
and *not* the DNS-EXF situation of Experiment E. It is **not** a claim that a
classifier could reach 0.98 here; the achievable figure is the ceiling probe's,
which is far lower.

So the honest conclusion is a **lane mismatch, not a broken label**: what
separates `Infilteration` from benign in this release lives in the 74
CICFlowMeter columns, and neither lane reads them — the Bouncer takes 6 rate
features, and the graph lane aggregates to 9 window statistics. This is the
same dataset that Experiment B (CIC-IDS2017) scores 0.9846 on, and the
difference is not that the traffic changed; it is that CIC-IDS2017's PortScan
class is dense and loud, while this release's scanning is folded into a
labelled window that is mostly ordinary background traffic.

### Artifact Paths

```
models/experiments/cicids2018/bouncer/bouncer.json
models/experiments/cicids2018/bouncer/calibration.json
models/experiments/cicids2018/detective/detective.npz
models/experiments/cicids2018/detective/detective.onnx
results/experiments/cicids2018_metrics.json
```

### Current Status

```
STATUS:   COMPLETE
VERDICT:  BOUNCER_REPORTABLE_DETECTIVE_BELOW_SLO
FLOWS:    151,867 rows loaded across four days at stride 20 (Bouncer);
          944,171 rows at stride 1 across two days (Detective windows)
WINDOWS:  32,777 graph windows at the production 2-second cadence
```

| Check | Result |
|---|---|
| Bouncer weights load + live inference | **PASSED** |
| Detective weights load + live inference | **PASSED** |
| Bouncer F1 | 0.9925 (cross-day 0.9985 … 0.8681) |
| Detective F1 | 0.3401 — below SLO, ceiling 0.6017 AUC |
| Elapsed | 172 s |

Reproducibility: re-running reproduces these figures. The only fields that
differ between runs are `timestamp`, `train_seconds` and `elapsed_seconds`.

---

## Experiment G — CIC-DDoS2019 External Dataset

### Dataset Source

| Property | Value |
|---|---|
| Dataset | CIC-DDoS2019 (DDoS Evaluation Dataset) |
| Source | Canadian Institute for Cybersecurity, University of New Brunswick ([dataset page](https://www.unb.ca/cic/datasets/ddos-2019.html)) |
| Paper | Sharafaldin, Lashkari, Hakak, Ghorbani, *Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy*, ICCST 2019, doi:10.1109/ccst.2019.8888419 |
| Download | `python scripts/download_cicddos2019.py` — **form-gated**, see below |
| Archives | `CSV-01-12.zip` (capture 2018-12-01), `CSV-03-11.zip` (capture 2018-11-03) |
| Rows | 38,282,659 + 20,364,525 = **58,647,184** across the two archives |
| Labels | `BENIGN`, plus 13 attack families written either as `DrDoS_<family>` or by bare name: LDAP, MSSQL, NetBIOS, Portmap, SNMP, SSDP, UDP, UDPLag, Syn, TFTP, WebDDoS, DNS, NTP |
| Schema | 88 columns, CICFlowMeter — the same family as CIC-IDS2017, which is the trap this loader guards against |

### Why CIC-DDoS2019, and Why Only the Bouncer Trains

Every labelled attack family in this release is volumetric: reflection/amplification
floods (LDAP, MSSQL, NetBIOS, Portmap, SNMP, SSDP, DNS, NTP, TFTP) and direct
floods (UDP, UDPLag, Syn, WebDDoS). There is no port-scan and no
lateral-movement class anywhere in it. The Bouncer's contract is
flood-vs-benign, which fits exactly; the Detective has no attack class to train
on and is skipped, with the reason recorded in
`dataset.detective_skipped_reason`. That makes this the **mirror image of
Experiment D**, which was Detective-only because a tunnel capture contains no
flood.

### Data Acquisition — Form-Gated, and an Interrupted Download

UNB gates these archives behind a registration form. The downloader posts it,
keeps the returned session cookie (without it `download.php` answers 403), then
fetches each archive. **No personal details live in the repository** — the form
values come from command-line flags or `KRONUS_CIC_*` environment variables,
because this repo is public and a committed name and email is a leaked one.

The transfer is also **not resumable**: `download_cicddos2019.py` issues a plain
GET with no `Range` header and deletes its `.part` file on failure, so an
interrupted fetch cannot continue. That happened. The `01-12` archive here is a
**salvage of an interrupted download**: the ZIP is a stream of
independently-compressed members, so the members whose bytes arrived complete
were carried into a rebuilt, valid archive and the incomplete tail was
discarded. **`CSV-01-12.zip` therefore holds 8 members, and members beyond the
transfer's stopping point are absent** — this is a partial day, not the
publisher's full manifest, and it is disclosed rather than presented as whole.
`CSV-03-11.zip` (7 members) downloaded complete. Those member counts are a
record of the one-time fetch, not of a committed artifact; `_verify_archives()`
checks only that each archive is a readable zip holding at least one CSV, so the
per-day `raw_labels` census below is the durable record of what arrived.

What was actually trained on is not asserted here: the metrics carry a per-day
`raw_labels` census, so the exact family mix is visible in
`results/experiments/cicddos2019_metrics.json`. At stride 32 the `01-12` day
supplies TFTP (627,575 rows), DrDoS_SNMP (161,246), DrDoS_NetBIOS (127,909),
DrDoS_UDP (97,957), DrDoS_SSDP (81,581), Syn (49,448), DrDoS_NTP (37,592) and a
15-row WebDDoS remnant; `03-11` supplies MSSQL (180,846), Syn (152,859), UDP
(120,842), NetBIOS (114,313), LDAP (59,852), Portmap (5,844) and UDPLag (63).

### What Measuring The Archive Changed

Three facts about this release were established by scanning the real archives
rather than by reading its documentation. All three changed the code.

#### 1. `Inbound` is the label wearing a column name

The loader excludes `Inbound` from `features`. The comment justifying that
originally claimed it was a *per-file constant* — a claim inherited, never
checked. It is **false**: both values appear inside every member. The exclusion
is still right, for a stronger reason. `Inbound` marks a flow arriving at the
monitored host, and a volumetric flood is inbound by construction, so the
column tracks the label. Read as a classifier (`attack iff Inbound == 1`) it
scores:

| Member | Accuracy |
|---|---|
| `03-11/Portmap.csv` | 99.35% |
| `01-12/UDPLag.csv` | 99.61% |
| `03-11/UDPLag.csv` | 99.73% |
| `01-12/Syn.csv` | 99.98% |

Keeping it would let the Bouncer separate attack from benign without looking at
a single packet count. The comment, and the two test docstrings that repeated
it, were corrected — the conclusion survived, the stated reason did not.

#### 2. The flood comes from one host

**Two source IPs carry a non-benign label in each archive, three across both
days** (`day_survey[*].flood_source_hosts` = 2; `host_overlap.attack_hosts` = 3,
the union over both days) — the attacker is a single host, `172.16.0.5`, on both
days, and the extra host is the victim replying. The benign traffic, by contrast,
comes from many distinct hosts (`day_survey[*].benign_source_hosts`: 132 and 205;
`host_overlap.benign_hosts`: 243 in union) on a real lab subnet. This is a lab
capture of a single-source flood, and it is the reason the scores below are
near-perfect — see the generalisation caveat further down.

#### 3. The victim host also sends benign traffic

The module docstring originally claimed that because the source IPs are real,
the "contaminated window" class of Experiment F was absent here. Measuring it
showed that is **nearly** true, and the near-miss matters: on each day exactly
one host — the victim, `192.168.50.1` on `01-12` and `192.168.50.4` on `03-11` —
appears as a source on both attack and benign rows. A one-time scan of the full
archives put roughly 0.7% of the benign class in the same 2-second windows as
that host's own flood; the committed metrics record the same overlap from the
strided load, where `shared_source_hosts_before_exclusion` is 1 on each day.

Rather than soften the claim, the runner **drops exactly those rows** before
building the pools, so "no benign window contains flood traffic" is true instead
of nearly true, and it counts the drop in the survey and prints it per day
(`benign_rows_dropped_on_flood_hosts`: 6 rows on `01-12`, 12 on `03-11` at
stride 32). The separate `host_overlap` block still reports the *raw,
pre-exclusion* overlap — 3 attack hosts, 243 benign hosts, 2 shared, naming
`192.168.50.1` and `192.168.50.4` — so the drop is auditable. The two blocks
disagree on purpose: one is the data, the other is the data after the fix.

### Label Mapping (CIC-DDoS2019 → KRONUS)

| Source label | KRONUS |
|---|---|
| `BENIGN` | `normal` (`Label.BENIGN`) |
| `DrDoS_*` and the bare family names (`LDAP`, `MSSQL`, `NetBIOS`, `Portmap`, `UDP`, `UDPLag`, `Syn`, `TFTP`, `WebDDoS`, `UDP-lag`) | `dos` (`Label.FLOOD`) |
| anything outside that vocabulary | **dropped and counted**, never assumed to be a flood |

The release mixes label vocabularies *within* single members — `03-11/LDAP.csv`
contains `NetBIOS` rows, `03-11/MSSQL.csv` contains `LDAP`, `03-11/UDP.csv`
contains `MSSQL`, `01-12/UDPLag.csv` contains `WebDDoS` — so the mapping is
applied per row, not per file.

### Feature Mapping — What Separates the Classes

The Bouncer's six features, with class-conditional medians over the training
pool. The separation is on **rate** and **destination-port spread**: the flood's
median event rate is 546× benign's, its byte rate 3,533×, and its destination
port entropy is 9.06 against 0.00 (a flood sprays ports; a benign host's window
is one conversation).

| Feature | Flood median | Benign median |
|---|---|---|
| `event_rate` | 273.0 | 0.5 |
| `byte_rate` | 374,513.5 | 106.0 |
| `dest_port_entropy` | 9.0607 | 0.0 |
| `unique_dest_count` | 1.0 | 1.0 |
| `avg_duration_ms` | 76.3646 | 32.0 |
| `same_dest_ratio` | 1.0 | 1.0 |

Two of the six carry no signal in this dataset at all: `unique_dest_count` and
`same_dest_ratio` are identical for both classes (1.0), because both a flooding
host and a benign host talk to a single destination inside a 2-second window.
They are left in the vector rather than dropped, since the feature set is fixed
across experiments and removing two features here would make this Bouncer a
different model from every other one in the repository.

### Train/Test Split

Flood and benign pools are built **per capture day** and split 67/33 within each
day by a seeded permutation, so the test set always contains both classes and
never a whole unseen day. Rows are thinned by a **single global stride** — 32 —
applied identically to both classes, because the rate features depend on the
replay stream's density and thinning one class differently would change the
ratio the model is asked to learn.

| Day | Rows loaded (stride 32) | Flood | Benign |
|---|---|---|---|
| `CSV-01-12` | 1,184,889 | 6,000 | 1,560 |
| `CSV-03-11` | 636,394 | 6,000 | 1,763 |

**The pool ratio is not the dataset's prevalence, and the difference is
disclosed.** Benign is genuinely rare here — 0.13% of the loaded `01-12` rows
and 0.28% of `03-11` (`day_survey[*].benign_fraction`: 0.001317 and 0.00277), a
natural flood:benign ratio of **758:1** and **360:1** (`flood_rows` /
`benign_rows`: 1,183,323/1,560 and 634,619/1,763). Training on that would
make a binary metric meaningless, so flood is capped at 6,000 rows per day
(`--bouncer-limit`), giving a **3.6:1** pool. Every precision and recall figure
below is measured at 3.6:1, not at natural prevalence. The consequence is
concrete: precision is exactly 1.0000 because the model made **zero false
positives across 1,097 benign test windows**, and a false-positive *rate* that
is 0.000 on 1,097 windows is not the same claim as one measured over 6M.

### Bouncer Results

| Metric | Value |
|---|---|
| Accuracy | 0.9986 |
| Precision | 1.0000 |
| Recall | 0.9982 |
| **F1** | **0.9991** |
| AUC | 0.9998 |

Confusion matrix (test set, 3,960 flood / 1,097 benign), rows = truth:

```
  benign   [1097    0]
  flood    [   7 3953]
```

Trained on 10,266 rows (8,040 flood / 2,226 benign) in 0.211 s.

**Cross-day.** Each day is held out entirely and the model is refit on the
other, which is the stronger test — nothing from the test day is in training:

| Trained on | Tested on | F1 | AUC |
|---|---|---|---|
| `CSV-03-11` | `CSV-01-12` | 0.9992 | 0.9996 |
| `CSV-01-12` | `CSV-03-11` | 0.9993 | 0.9993 |

Unlike Experiment F — where cross-day F1 collapsed from **0.9985** to **0.8681**
when the unseen day was the slow family (0.9925 is F's *within-day* figure, not
its cross-day baseline) — the score here does not degrade at all.
That is not a stronger model; it is an easier dataset, and the next section says
why.

### Why 0.9991 Is A Weak Generalisation Test

A near-perfect score should invite suspicion, so the evidence for and against it
ships with the number.

**For:** the label is clean. A separate feature-carrying load of 1,457,028 rows
found only 3 ambiguous feature vectors (25 rows, 0.00% of the load) — rows
carrying two different classes on one identical feature vector — so the
deterministic ceiling for this task is 1.0000 and there is no label noise to
inflate. The dataset's own clock drives the replay, so `event_rate` is a real
burst rate rather than a synthetic spacing. And no column in the input is the
label in disguise: `Inbound` is excluded precisely because it is.

**Against:** the task this measures is *easier than the one KRONUS faces in
production*, for two structural reasons.

1. **The flood is single-sourced.** All attack traffic comes from one host
   (`172.16.0.5`). The Bouncer's window is keyed by `source_ip`, so one host
   accumulates essentially the entire flood at a median 273 events per window
   against benign's 0.5. A real distributed denial-of-service — thousands of
   sources each sending a modest rate — is **not represented in this data** and
   would not present as a single extreme-rate host. The model has not been shown
   to detect that.
2. **Both days are the same lab.** The attacker, victim and topology are
   constant across the two captures, so the cross-day block tests generalisation
   across *time*, not across *environment* or *attack style*.

The honest summary: on this release the Bouncer separates a single-host
volumetric flood from real background traffic essentially perfectly, and the
result should be read as confirmation that the rate-based feature set works on
the case it was designed for — not as evidence that it handles distributed
floods at scale.

### Artifact Paths

```
models/experiments/cicddos2019/bouncer/bouncer.json
models/experiments/cicddos2019/bouncer/calibration.json
results/experiments/cicddos2019_metrics.json
```

The Detective has no artifacts here: it does not train.

### Current Status

```
STATUS:   COMPLETE
VERDICT:  BOUNCER_REPORTABLE
FLOWS:    1,821,283 rows loaded across two days at stride 32 (Bouncer)
WINDOWS:  n/a — the Bouncer reads flow vectors, not graph windows
```

| Check | Result |
|---|---|
| Bouncer weights load + live inference | **PASSED** |
| Bouncer F1 | 0.9991 (cross-day 0.9992 / 0.9993) |
| Label-quality ceiling | 1.0000 (0.00% of rows on an ambiguous vector) |
| Elapsed | 1,765 s |

Reproducibility: re-running reproduces these figures. The only fields that
differ between runs are `timestamp`, `train_seconds` and `elapsed_seconds`.

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

| Property | Experiment A (NSL-KDD) | Experiment B (CIC-IDS2017) | Experiment C (UNSW-NB15) | Experiment D (DoHBrw2020) | Experiment E (DNS-EXF2021) | Experiment F (CSE-CIC-IDS2018) | Experiment G (CIC-DDoS2019) |
|---|---|---|---|---|---|---|---|
| Data source | NSL-KDD (`data/real/`) | CIC-IDS2017 from UNB/CIC | UNSW-NB15 from ACCS, UNSW Canberra | CIRA-CIC-DoHBrw-2020 from UNB/CIC | CIC-Bell-DNS-EXF-2021 from UNB/CIC | CSE-CIC-IDS2018 from UNB/CIC (public S3 bucket) | CIC-DDoS2019 from UNB/CIC (form-gated; `01-12` is a partial salvage) |
| Data type | Benchmark dataset (1999, KDD-cup era) | External real-world capture (2017) | External capture (2015, IXIA PerfectStorm) | External capture (2019–2020, DoH tunnels) | External capture (2021, DNS exfiltration) | External capture (2018, four days) | External capture (2018, two days, volumetric floods) |
| Lanes trained | Bouncer + Detective | Bouncer + Detective | Bouncer + Detective | **Detective only** (no flood class exists) | **Detective only** (no flood class exists) | Bouncer + Detective (**Detective below SLO**) | **Bouncer only** (no port-scan or lateral-movement class exists) |
| Attack class(es) | DoS → flood, Probe → port_scan | DoS/DDoS → flood, PortScan → port_scan | DoS → flood, Recon → port_scan | DoH tunnels → **lateral_movement** | DNS exfiltration → **lateral_movement** | DoS/DDoS → flood, `Infilteration` → port_scan | DrDoS/reflection + direct floods → flood |
| Hosts | Reconstructed (no IPs in source) | Real | Reconstructed (no IPs in source) | **Real** | Reconstructed (**no IPs, ports, bytes or durations in source**) | Reconstructed (**no IPs, no source ports**; `dest_ip` is a bijection of the **real** dest port) | **Real** — but the attacker is one host (`172.16.0.5`) |
| Clock | Synthetic spacing | Synthetic spacing | Synthetic spacing | Synthetic spacing | Synthetic spacing | **Real capture clock** (a first here) | **Real capture clock** |
| Records | 125,973 train | 2,828,563 flows | 257,673 flows | 60,000 flows | 46,398 flows (from 536,138 rows; 99.95% of attack rows are label-ambiguous and dropped) | 61,224 Bouncer train / 30,158 test; 32,777 graph windows | 1,821,283 rows loaded at stride 32; 10,266 Bouncer train / 5,057 test |
| Bouncer F1 | **0.8179** | **0.9987** | **0.9963** | not trained | not trained | **0.9925** (cross-day 0.9985 … 0.8681) | **0.9991** (cross-day 0.9992 / 0.9993 — a weak test, see below) |
| Detective F1 | **1.0000** | **0.9846** | **1.0000** | **0.9697** | **not reportable** — see below | **0.3401 — below the 0.85 SLO** — see below | not trained |
| Weights | `models/experiments/repo_data/` | `models/experiments/cic_ids2017/` | `models/experiments/unsw_nb15/` | `models/experiments/dohbrw2020/` | `models/experiments/dnsexf2021/` | `models/experiments/cicids2018/{bouncer,detective}/` | `models/experiments/cicddos2019/bouncer/` |
| Metrics | `results/experiments/repo_data_metrics.json` | `results/experiments/cic_ids2017_metrics.json` | `results/experiments/unsw_nb15_metrics.json` | `results/experiments/dohbrw2020_metrics.json` | `results/experiments/dnsexf2021_metrics.json` | `results/experiments/cicids2018_metrics.json` | `results/experiments/cicddos2019_metrics.json` |
| Weight load | PASSED | PASSED | PASSED | Detective PASSED · Bouncer n/a | Detective PASSED · Bouncer n/a | Bouncer PASSED · Detective PASSED | Bouncer PASSED · Detective n/a |
| Script | `python scripts/run_repo_experiment.py` | `python scripts/run_cic_experiment.py` | `python scripts/run_unsw_nb15_experiment.py` | `python scripts/run_dohbrw2020_experiment.py` | `python scripts/run_dnsexf2021_experiment.py` | `python scripts/run_cicids2018_experiment.py` | `python scripts/run_cicddos2019_experiment.py` |

Experiment E is the only entry here whose Detective F1 is withheld. Its
trained score is 1.0000, but the dataset's label-ambiguity ceiling is 0.8210
and the ambiguity guard leaves 32 attack signatures across 32 rows, so the
score measures a lookup table rather than detection. The runner records it as
`reportable_as_detection: false` with `verdict: NO_VALID_DETECTION_METRIC`
instead of quoting it as a result.

Experiment F is the only entry here whose Detective F1 is **published and
below target**. It is reported rather than withheld because the dataset's
label is sound — only 7.46% of rows sit on a feature vector carrying two
classes, so `Infilteration` is internally consistent and the weak score is the
model's, not the label's. The runner measures the ceiling *before* training
(a 3-fold linear probe over the same windows scores AUC 0.6017 / F1 0.4033)
and sweeps the window cadence (AUC reaches 0.8518 at 120 s while the
attack/benign activity ratio stays flat at 1.20–1.30, so the gain is
measurement precision on a constant ~25% difference, not a port-scan
signature). The GAT's 0.3401 is below even the probe's ceiling, which is the
expected outcome of a 3-class head over a class that is faint at graph
granularity. The Bouncer, on the same data, is strong within-day (0.9925) and
strong cross-day in one direction (0.9985 / 0.9977), degrading to 0.8684 /
0.8681 when the unseen flood day is the slow/rate-limited family. Both numbers
ship; `verdict: BOUNCER_REPORTABLE_DETECTIVE_BELOW_SLO`.

Experiment G is the only entry whose strong number is flagged against itself. Its
Bouncer F1 of 0.9991 is not noise: the label is clean (0.00% of rows sit on an
ambiguous feature vector, deterministic ceiling 1.0000) and no input column is the
label in disguise (`Inbound` is excluded precisely because reading it as a
classifier scores 99.3–99.98%). What weakens it is structural — the release's
flood comes from a **single host**, so the per-host 2-second window sees one
extreme-rate source against ordinary background traffic, and the cross-day block
compares two captures of the same lab rather than two environments. It ships as
`verdict: BOUNCER_REPORTABLE`, with its section stating that it confirms the
rate-based feature set on the case it was designed for and says nothing about
distributed floods.

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

### Experiment E
```bash
# Fetch the sixteen per-class CSVs (~53 MB), byte-size-verified against the release
python scripts/download_dnsexf2021.py --from-mirror
python scripts/run_dnsexf2021_experiment.py
# Reclaim the space when done:
rm -rf data/external/dnsexf2021
```

Expect `VERDICT: NO_VALID_DETECTION_METRIC` and a printed F1 near 1.0 that is
explicitly marked non-reportable. That output is the intended result — see the
Experiment E section above. If the runner instead reports a valid detection
metric, the data is not what the loader was written against.

### Experiment F
```bash
# Fetch the four day CSVs (~980 MB) from the official public S3 bucket:
# no form, no account, no mirror. Byte-size-verified on download.
python scripts/download_cicids2018.py --from-mirror
python scripts/run_cicids2018_experiment.py
# Reclaim the space when done:
rm -rf data/external/cicids2018
```

Expect `VERDICT: BOUNCER_REPORTABLE_DETECTIVE_BELOW_SLO`. The Bouncer's F1 is
near 0.99 and the Detective's is near 0.34, well below the 0.85 external-dataset
SLO. That divergence is the intended, measured result — see the Experiment F
section above. The runner also prints a pre-training ceiling probe and a
window-cadence sweep; if either is missing, the run did not execute the
measurement path.

### Experiment G
```bash
# Form-gated. Pass your own registration details, or set KRONUS_CIC_FIRST_NAME,
# KRONUS_CIC_LAST_NAME, KRONUS_CIC_EMAIL, KRONUS_CIC_INSTITUTION,
# KRONUS_CIC_JOB_TITLE and KRONUS_CIC_COUNTRY. No personal details are committed.
# The downloader implements no resume (plain GET, .part deleted on failure):
# fetch it in one go or start over.
python scripts/download_cicddos2019.py \
  --first-name ... --last-name ... --email ... \
  --institution ... --job-title ... --country ...
python scripts/run_cicddos2019_experiment.py --stride 32 --bouncer-limit 6000
# Reclaim the space when done:
rm -rf data/external/cicddos2019
```

Expect `VERDICT: BOUNCER_REPORTABLE` with an F1 near 0.999 and a Detective that
does not train. If `CSV-01-12.zip` holds fewer than 8 members the download was
cut short: the ZIP is a stream of independently-compressed members, so the
complete ones remain usable, but that day is partial — the per-day `raw_labels`
census in the metrics is where to check which families actually arrived. The
runner prints both the survey (including
`benign_rows_dropped_on_flood_hosts`) and the label-quality ceiling; if either is
missing, the run did not execute the measurement path.

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
| DNS-EXF2021 | Detective | 42 |
| Split (DNS-EXF) | All | 42 (per-class, stratified) |
| CSE-CIC-IDS2018 | Bouncer | 42 |
| CSE-CIC-IDS2018 | Detective | 42 |
| Split (CIC-IDS2018) | Bouncer | 42 (per-day, stratified 67/33) |
| Split (CIC-IDS2018) | Detective | 42 (per-class, stratified 67/33, evenly spaced stride) |
| CIC-DDoS2019 | Bouncer | 42 |
| Split (CIC-DDoS2019) | Bouncer | 42 (per-day, stratified 67/33, global stride 32) |
