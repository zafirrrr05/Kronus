# KRONUS — Experimental Evidence Register

A single table-oriented view of every experiment in this repository: what it ran on,
what it measured, whether the result matched what the repository said to expect, and
why the number came out the way it did.

**Provenance.** This document was written by reading the committed artifacts —
`results/experiments/*.json`, `docs/experiments_guide.md`, `README.md` and the
runners that produced them. **Nothing was re-trained, re-downloaded or re-run to
produce it.** Every metric below is copied from the metrics artifact named in the
row it appears in; where a number is not in an artifact, that is stated. Where a
sentence is an interpretation rather than a recorded fact, it is marked
**[reading]**.

This register is a description of experimental work already in the repository. It
is not a claim of authorship of that work, and it adds no measurements of its own.

---

## 0. Scope, sources and how to read this

### 0.1 The artifacts covered

| # | Experiment | Metrics artifact | Tracked in git |
|---|---|---|---|
| 1 | A — Repo Data (NSL-KDD) | `results/experiments/repo_data_metrics.json` | yes |
| 2 | B — CIC-IDS2017 | `results/experiments/cic_ids2017_metrics.json` | yes |
| 3 | C — UNSW-NB15 | `results/experiments/unsw_nb15_metrics.json` | yes |
| 4 | D — CIRA-CIC-DoHBrw-2020 | `results/experiments/dohbrw2020_metrics.json` | yes |
| 5 | E — CIC-Bell-DNS-EXF-2021 | `results/experiments/dnsexf2021_metrics.json` | yes |
| 6 | F — CSE-CIC-IDS2018 | `results/experiments/cicids2018_metrics.json` | yes |
| 7 | G — CIC-DDoS2019 | `results/experiments/cicddos2019_metrics.json` | yes |
| 8 | H — CIC IoT 2023 | `results/experiments/ciciot2023_metrics.json` | yes |
| 9 | Lane substitution (A/B/C re-used) | `results/experiments/lane_substitution_metrics.json` | yes |
| 10 | CIC Datasense IoT (pilot) | `results/experiments/datasense_metrics.json` | **no — untracked** |

Experiments A–H are the ones `docs/experiments_guide.md` documents. Experiment 9
postdates the guide (`a1407ce feat(lane-substitution): route the Detective's data to
the trained Bouncer`) and has no section in it. Experiment 10 is a working-tree
artifact with no documentation anywhere.

### 0.2 The two lanes

Every experiment trains one or both of two independent models:

- **Bouncer** — XGBoost + temperature scaling over six rate/flow features
  (`event_rate`, `byte_rate`, `dest_port_entropy`, `unique_dest_count`,
  `avg_duration_ms`, `same_dest_ratio`). Binary contract: `flood` vs `benign`.
- **Detective** — a hand-rolled 3-layer graph attention network in NumPy/autograd
  over `WindowedGraphBuilder` graph windows. Multi-class head over
  `[benign, port_scan, lateral_movement]`, of which any given dataset usually
  supplies a subset.

A lane is *skipped* when the dataset does not contain its contract's classes — the
artifacts say so in a `skipped` field rather than training a mislabelled model.

### 0.3 Statistical conventions used below

| Term | Meaning in these artifacts |
|---|---|
| Precision / Recall / F1 | **Binary, positive class = the attack class** (flood for the Bouncer; the named attack class for the Detective). This is *not* macro-averaged unless the field is named `macro_*`. |
| `macro_*` | Unweighted mean over the model's classes. Present in A, B; absent from C–H. |
| AUC | Present only in F, G, H and the lane-substitution arms. Absent from A–E's lane blocks. |
| `n_uncertain_verdicts` | Detective verdicts landing in the abstention band, counted as errors for recall and **folded into the attack column** of the confusion matrix. |
| `deterministic_ceiling` | A hard upper bound on accuracy for *any* model reading that dataset's features — computed where one feature vector carries two different labels, since no function of those features can beat the majority count. |
| `synthetic_hosts` | `true` where the source carries no IP/port columns and hosts were reconstructed; `false` where addresses are the capture's own. |
| SLO | `README.md:51` — **External-dataset F1 > 0.85**. Applies to the external datasets, not to A (which is the repository's own benchmark data). |
| Majority baseline | Where recorded, the accuracy of always predicting the largest class. Given because a balanced pool makes a raw accuracy figure mean something different. |

### 0.4 How "expected?" is decided

The repository records its own expectations in exactly one place: the `## Reproducibility`
section of `docs/experiments_guide.md`, which states an `Expect VERDICT: ...` line for
Experiments E, F, G and H. Those lines are the basis for the verdict calls below. The
column therefore has four values:

| Value | Meaning |
|---|---|
| **Yes — declared** | The repository states the expected outcome in advance, and the measurement matches it. |
| **Yes — declared, discounted** | The repository states the outcome and *also* argues the number is weak evidence. The measurement matches; the repository itself declines to treat it as strong. |
| **Yes — documented result** | No `Expect` line exists, but the guide publishes these exact figures as the experiment's results, and the artifact's own `analysis` field explains the deviation from the prior baseline. |
| **Not assessable** | No documented expectation exists (the lane-substitution experiment and the untracked pilot). The judgement column carries **[reading]** instead. |

Nothing in this register was judged "unexpected" — because nothing measured came out
against a stated expectation. The interesting variance in this repository is not
surprise; it is **which numbers the repository itself refuses to stand behind.**

### 0.5 A note on three stale counts

While tabulating, three counting statements did not match the artifacts:

| Location | Says | Artifacts show |
|---|---|---|
| `docs/experiments_guide.md:3` | "**Seven** completely independent ML experiments" | eight documented sections, A–H |
| `README.md:88` | "**six** completely independent ML experiment pipelines" | eight enumerated items, A–H |
| `README.md:3`/guide intro | names E, F and G as the negative/caveated cases | E, F and G **and** the G-style caveats that also apply to H |

The guide's own summary table (`docs/experiments_guide.md:1641`) has eight columns, so
the table is authoritative and both prose counts are stale. Recorded here because it is
the kind of thing that makes a reader distrust the rest of the document.

---

## 1. Master tables

### 1.1 Headline result per experiment

| Experiment | Dataset | Lanes trained | Bouncer F1 | Detective F1 | Recorded verdict | Expected? |
|---|---|---|---|---|---|---|
| **A** | NSL-KDD (KDDTrain+/KDDTest+) | Bouncer + Detective | **0.8179** | **1.0000** | *(no `status`/`verdict` field)* | Yes — documented result |
| **B** | CIC-IDS2017 | Bouncer + Detective | **0.9987** | **0.9846** | `status: COMPLETE` | Yes — documented result |
| **C** | UNSW-NB15 | Bouncer + Detective | **0.9963** | **1.0000** | `status: COMPLETE` | Yes — documented result |
| **D** | CIRA-CIC-DoHBrw-2020 | Detective only | *not trained* | **0.9697** | `status: COMPLETE` | Yes — documented result |
| **E** | CIC-Bell-DNS-EXF-2021 | Detective only (not reportable) | *not trained* | **1.0000 — NOT a detection result** | `verdict: NO_VALID_DETECTION_METRIC` | Yes — declared |
| **F** | CSE-CIC-IDS2018 | Bouncer + Detective | **0.9925** (cross-day 0.9985 … 0.8681) | **0.3401 — below the 0.85 SLO** | `verdict: BOUNCER_REPORTABLE_DETECTIVE_BELOW_SLO` | Yes — declared |
| **G** | CIC-DDoS2019 | Bouncer only | **0.9991** (cross-day 0.9992 / 0.9993) | *not trained* | `verdict: BOUNCER_REPORTABLE` | Yes — declared, **discounted** |
| **H** | CIC IoT 2023 | Bouncer + Detective | **0.8036** (cross-flood-family) | **0.8884** (vs 0.5 baseline) | `status: COMPLETE` | Yes — declared |
| **9** | UNSW-NB15 + CIC-IDS2017 re-used | Bouncer substituted onto Detective windows | 0.9848 / **0.2747** | 1.0000 / 0.9848 | `status: COMPLETE` | Not assessable — **[reading]** |
| **10** | CIC Datasense IoT | Bouncer + Detective | 0.9694 | 0.4286 *(4 test windows — not citable)* | `status: COMPLETE` | Not assessable — **[reading]** |

### 1.2 Data and evaluation scale

| Experiment | Rows / flows | Bouncer train / test | Detective train / test windows | Window | Hosts | Clock |
|---|---|---|---|---|---|---|
| **A** | 125,973 / 22,544 records | 125,973 / 22,544 | 200 / 123 | 2 s | reconstructed *(not recorded in the artifact; stated in the guide)* | synthetic spacing |
| **B** | 2,828,563 flows | 1,778,042 / 875,754 | 268 / 132 | 2 s | real (`twin/cicids2017.py` reads `Source IP` / `Destination IP`) | synthetic spacing |
| **C** | 257,673 flows | 24,356 / 11,997 | 134 / 66 | 2 s | reconstructed (`synthetic_hosts: true`) | synthetic spacing |
| **D** | 60,000 flows | *skipped* | 134 / 66 | 2 s | **real** (`synthetic_hosts: false`) | synthetic spacing |
| **E** | 46,398 flows from 536,138 raw | *skipped* | 68 / 34 | 2 s | reconstructed (`synthetic_hosts: true`, `features_are_real: false`) | synthetic spacing |
| **F** | 151,867 rows @ stride 20 (Bouncer); 944,171 @ stride 1 (Detective) | 61,224 / 30,158 rows | 11,187 / 5,511 | 2 s | reconstructed (`synthetic_hosts: true`) | **real capture clock** |
| **G** | 1,821,283 rows @ stride 32 | 10,266 / 5,057 rows | *n/a* | 2 s | **real** (`synthetic_hosts: false`) | **real capture clock** |
| **H** | 442,422 flows from 34,581,737 packets | 15,404 / 81,348 rows | 498 / 488 | **30 s** | **real** (`synthetic_hosts: false`) | **real capture clock** |
| **9** | re-uses B and C streams | *n/a* — frozen artifacts | 66 / 66 (C) and 132 / 132 (B) | 2 s | inherited | synthetic spacing (`row_spacing_ms: 20`) |
| **10** | 1,000 flows | 669 / 331 | **7 / 4** | 2 s | reconstructed | synthetic spacing (`ROW_SPACING_MS = 20`) |

### 1.3 Label quality and ceilings

| Experiment | Rows measured | Distinct feature vectors | Rows on an ambiguous vector | Deterministic ceiling | Independent ceiling probe |
|---|---|---|---|---|---|
| A | *not recorded* | — | — | — | — |
| B | *not recorded* | — | — | — | — |
| C | *not recorded* | — | — | — | — |
| D | *not recorded* | — | — | — | — |
| E | 536,138 | 46,462 | 390,215 (**72.78%**) | **0.821** | — |
| F | 23,605 | 20,838 | 1,762 (7.46%) | 0.9778 | linear probe AUC **0.6017** / F1 0.4033 over 32,777 windows |
| G | 1,457,028 | 404,186 | 25 (0.00%) | **1.0000** | — |
| H | *not recorded* | — | — | — | cadence sweep (see §2.8) |
| 9 | *not recorded* | — | — | — | — |
| 10 | *not recorded* | — | — | — | — |

The `label_quality` block exists only from Experiment E onward. Experiments A–D carry
no label-ambiguity measurement, so **no deterministic ceiling is known for four of the
eight documented datasets.** That is a gap in the evidence, not a criticism of the
numbers — but it means a perfect score there cannot be checked against the bound the
repository applies to E and F.

---

## 2. Experiment-by-experiment detail

### 2.1 Experiment A — Repo Data (NSL-KDD)

**Source artifact:** `results/experiments/repo_data_metrics.json` · recorded
`2026-09-07T14:23:57Z` · no `status` field · dataset files
`data/real/KDDTrain+.txt` (125,973) and `data/real/KDDTest+.txt` (22,544).

**Bouncer** — XGBoost + Temperature Scaling, 6 features

| Metric | Value |
|---|---|
| Accuracy | 0.8840 |
| Precision (flood) | 0.8509 |
| Recall (flood) | 0.7873 |
| **F1 (flood)** | **0.8179** |
| Macro precision / recall / F1 | 0.8747 / 0.8595 / 0.8664 |
| Per-class F1 — non-flood / flood | 0.9149 (n=15,084) / 0.8179 (n=7,460) |
| Confusion matrix `[[tn,fp],[fn,tp]]` | `[[14055, 1029], [1587, 5873]]` |
| Temperature | 0.9679 |
| Train / eval seconds | 0.24 / 0.01 |

**Detective** — GAT, `hidden_dim=32`, `edge_embed_dim=8`, 3 layers, classes `[benign, port_scan]`

| Metric | Value |
|---|---|
| Accuracy / Precision / Recall / F1 | 1.0000 / 1.0000 / 1.0000 / 1.0000 |
| Macro precision / recall / F1 | 1.0000 / 1.0000 / 1.0000 |
| Train / test windows | 200 / 123 |
| Per-class support — benign / port_scan | 98 / 25 |
| Confusion matrix | `[[98, 0], [0, 25]]` |
| `limit` / epochs / train seconds | 20,000 / 4 / 2.92 |

**Weight verification:** `bouncer_loadable: true`, `detective_loadable: true`.

**Expected? Yes — documented result.** A is the only experiment whose target predates
the run: the artifact carries a `historical_comparison` block and it matches to within
0.0003 on every metric (accuracy Δ 0.0000, precision Δ −0.0001, recall Δ +0.0003,
F1 Δ −0.0001; the Detective's four deltas are all 0.0000). The recorded explanation is a
20% calibration-split seed plus CPU float rounding. This is a reproduction claim
succeeding, which is a different kind of evidence from the other seven.

**Why this result happened.**

- **Bouncer F1 0.8179.** NSL-KDD is a 1999 KDD-cup-era benchmark whose records were
  converted to the six live-pipeline features; the flow measured is DoS vs normal on
  aggregate connection statistics. 0.8179 *is* the historical figure being reproduced,
  so the number is the baseline rather than a finding. Note that it is **below the
  repository's 0.85 F1 SLO** — the SLO is scoped to external datasets (`README.md:51`)
  and the guide does not apply it to A, so this is not a recorded failure, but it is
  worth seeing in the same table as E and F.
- **Detective 1.0000.** The artifact's own analysis: normal and probe traffic form
  topologically distinct bipartite/fan-out graphs, and the GAT's attention layers
  separate them cleanly. **[reading]** The perfect score rests on **25 port_scan test
  windows** out of 123, and NSL-KDD's hosts are reconstructed, so the graph the lane
  sees is not a capture's own. Neither caveat is recorded in the artifact. The
  repository's own later work is the reason to weight this carefully: on
  CIC-IDS2017 (§2.9) a "perfect or near-perfect" graph lane turned out to be
  reproducible by a six-feature linear model.

**Artifact-path wrinkle.** A's recorded weight paths are
`/Users/manavjobanputra/kronos/**Kronus**/models/experiments/repo_data/bouncer/...`,
i.e. they assume a nested `Kronus/` directory that does not exist in this working tree.
The weights are actually at `/Users/manavjobanputra/kronos/models/experiments/repo_data/`.
Every experiment from B onward records the resolving path. A reader following A's
artifact paths by hand will not find the files.

---

### 2.2 Experiment B — CIC-IDS2017

**Source artifact:** `results/experiments/cic_ids2017_metrics.json` · `status: COMPLETE` ·
`2026-09-28T20:59:27Z` · eight `*.pcap_ISCX.csv` files · 2,828,563 rows
(1,778,042 train / 875,754 test) · loaded categories: `normal` 2,273,097 · `dos` 380,699 ·
`probe` 158,930 · `r2l` 15,801 · `u2r` 36.

**Bouncer**

| Metric | Value |
|---|---|
| Accuracy | 0.9996 |
| Precision / Recall (flood) | 0.9990 / 0.9985 |
| **F1 (flood)** | **0.9987** |
| Macro precision / recall / F1 | 0.9994 / 0.9992 / 0.9993 |
| Per-class F1 — non-flood / flood | 0.9998 (n=750,123) / 0.9987 (n=125,631) |
| Confusion matrix | `[[749997, 126], [188, 125443]]` |
| Temperature | 1.0 |
| Train / eval seconds | 2.279 / 0.219 |

**Detective**

| Metric | Value |
|---|---|
| Accuracy | 0.9848 |
| Precision / Recall (port_scan) | 1.0000 / 0.9697 |
| **F1 (port_scan)** | **0.9846** |
| Macro precision / recall / F1 | 0.9853 / 0.9848 / 0.9848 |
| Per-class F1 — benign / port_scan | 0.9851 (n=66) / 0.9846 (n=66) |
| Confusion matrix | `[[66, 0], [2, 64]]` |
| Train / test windows | 268 / 132 |
| `limit` / epochs / train seconds | 20,000 / 4 / 2.141 |

**Weight verification:** both `true`.

**Expected? Yes — documented result.** The guide publishes these exact figures
(`docs/experiments_guide.md:193`) and the artifact's own `analysis` field explains the
deviations from A.

**Why this result happened.**

- **The Bouncer jumps +0.1807 F1 over A** under an identical model and identical
  featurizer. The recorded explanation is that CIC-IDS2017's flows map cleanly onto the
  six features and that DoS/DDoS traffic is structurally distinct from benign at the
  flow-feature level. **[reading]** The mechanism is that A's inputs are 1999 connection
  records converted to pseudo-flows, whereas B's are 74-column real flow measurements
  from a live capture — the same six features are computed from far richer inputs.
- **The Detective falls 0.0154 F1 below A's perfect score**, and the artifact's `analysis`
  attributes it to CIC's PortScan traffic being more structurally varied than NSL-KDD's
  probe class: 2 misclassifications in 132 windows, both benign windows predicted
  `port_scan` (precision on `port_scan` is 1.0000; the loss is on benign precision,
  0.9706). **[reading]** This is the first hint in the register of the pattern that
  §2.9 later makes decisive: the Detective's near-perfect scores track how homogeneous
  the attack class is, not how hard the dataset is.

---

### 2.3 Experiment C — UNSW-NB15

**Source artifact:** `results/experiments/unsw_nb15_metrics.json` · `status: COMPLETE` ·
`2026-10-06T09:17:27Z` · 257,673 flows · categories `normal` 93,000 · `r2l` 134,333 ·
`probe` 13,987 · `dos` 16,353 · **`synthetic_hosts: true`** — the author-supplied CSVs
carry no IP or port columns, so hosts are reconstructed deterministically from each
row's own connection-rate counters.

**Bouncer** (DoS-flood vs benign)

| Metric | Value |
|---|---|
| Accuracy | 0.9967 |
| Precision / Recall | 0.9994 / 0.9931 |
| **F1** | **0.9963** |
| Confusion matrix | `[[6597, 3], [37, 5360]]` |
| Train / test rows | 24,356 (10,956 attack / 13,400 normal) / 11,997 (5,397 / 6,600) |
| Train seconds | 0.169 |
| AUC / macro metrics / temperature | *not recorded in this artifact* |

**Detective** (port_scan vs benign)

| Metric | Value |
|---|---|
| Accuracy / Precision / Recall / F1 | 1.0000 / 1.0000 / 1.0000 / 1.0000 |
| Train / test windows | 134 / 66 |
| Confusion matrix | `[[33, 0], [0, 33]]` |
| Train seconds | 1.843 |

**Weight verification:** `unsw_nb15_bouncer_loadable: true`, `unsw_nb15_detective_loadable: true`.
The guide additionally records the live-inference verdicts: Bouncer `uncertain`
(conf 0.721), Detective `benign` (conf 1.000).

**Expected? Yes — documented result.** Published at `docs/experiments_guide.md:341`
and `README.md:110`. The guide states the figures reproduce exactly on re-download.

**Why this result happened.**

- **Bouncer 0.9963** on DoS vs normal: same mechanism as B — real flow measurements
  feeding the six features; the `dos` class is a rate/volume signature the features are
  built to see.
- **Detective 1.0000** over 66 windows. This is the score most likely to be
  over-read, so it is worth stating what corroborates it and what does not.
  **Corroborating:** the lane-substitution experiment (§2.9) re-scored the *same*
  windows with two independent six-feature models — the shipped Bouncer reached
  `f1_macro` 0.9848 (max-aggregated) and a freshly-trained one reached 0.8611 — so a
  perfect score here is not what a trivial flow-level model gets. **Not corroborating:**
  `synthetic_hosts: true`, 33 positive test windows, and no `label_quality` block, so
  C carries neither a real-topology guarantee nor a measured ambiguity ceiling.
  **[reading]** The honest status is: a real 1.0000 on a small, reconstructed-graph
  evaluation whose size the artifact does not flag.

---

### 2.4 Experiment D — CIRA-CIC-DoHBrw-2020

**Source artifact:** `results/experiments/dohbrw2020_metrics.json` · `status: COMPLETE` ·
`2026-10-07T06:33:27Z` · 60,000 flows · `normal` 15,000 · `lateral_movement` 45,000 ·
**`synthetic_hosts: false`, `features_are_real: true`** — real IPs and real ports, rows
replayed in the dataset's true chronological order.

**Bouncer — skipped, by design.** `n_flood_rows: 0` over 60,000 considered. The
artifact's `skipped` field: DoHBrw2020 contains no DoS/flood traffic, and the Bouncer's
contract emits only `Label.FLOOD` or `Label.BENIGN`, so training it here would require
labelling tunnels as floods, which is false.

**Detective** (lateral_movement vs benign) — the first experiment to exercise the
third class

| Metric | Value |
|---|---|
| Accuracy / Precision / Recall / F1 | 0.9697 / 0.9697 / 0.9697 / 0.9697 |
| Train / test windows | 134 / 66 |
| Train / test flows per class | 6,700 benign + 6,700 tunnel / 3,300 + 3,300 |
| `n_uncertain_verdicts` | 1 |
| Confusion matrix over `[benign, port_scan, lateral_movement]` | `[[32, 0, 1], [0, 0, 0], [1, 0, 32]]` |
| Train seconds | 1.147 |

**Weight verification:** `dohbrw2020_bouncer_loadable: null` with the note "not trained
on this dataset"; `dohbrw2020_detective_loadable: true`.

**Expected? Yes — documented result.** Published at `docs/experiments_guide.md:541` and
`README.md:120`.

**Why this result happened.** 0.9697 is 64 of 66 windows. The `port_scan` column is
entirely zero because the dataset contains no port scans: the class is never a training
target, but it can still be predicted, so the guide keeps it as a visible column rather
than dropping it. The guide's account of the two errors is "one in each direction"
against a stated `n_uncertain_verdicts` of 1, with gray-zone verdicts counted toward the
attack class. **[reading]** The arithmetic reconciles as: 32 benign windows correct, 32
tunnel windows correct, 1 benign window classified as attack (which is the single
UNCERTAIN verdict, folded in), and 1 tunnel window classified benign — giving precision
32/33 and recall 32/33, both 0.9697.

This is the first experiment in the register where the graph topology is the capture's
own and where a third class is exercised, and it is one of only two Detective results
that stays off 1.0000 without being withheld or declared below SLO.

---

### 2.5 Experiment E — CIC-Bell-DNS-EXF-2021

**Source artifact:** `results/experiments/dnsexf2021_metrics.json` · `status: COMPLETE` ·
**`verdict: NO_VALID_DETECTION_METRIC`** · `2026-10-07T10:08:55Z` · 46,398 flows after
the ambiguity guard, from 536,138 raw · `normal` 46,366 · `lateral_movement` 32 ·
`verdict`-level flag: `detective.reportable_as_detection: false`.

**Label quality — the measurement that decides the verdict**

| Metric | Value |
|---|---|
| Rows / distinct feature vectors | 536,138 / 46,462 |
| Duplicate-row fraction | 0.9133 |
| Ambiguous vectors / rows on them | 64 / **390,215** |
| **Ambiguous row fraction** | **0.7278** |
| Ambiguous fraction — `lateral_movement` / `normal` | **0.9995** / 0.3971 |
| **Deterministic ceiling** | **0.821** |
| Distinct attack signatures, raw → after guard | 96 → **32** |
| Rows after guard — normal / `lateral_movement` | 145,774 / **149** |

**SLO block**

| Field | Value |
|---|---|
| `target_f1` | 0.85 |
| `measured_f1` | 1.0 |
| `ceiling` | 0.821 |
| `target_reachable` | **false** |
| `meets_slo` | **false** |

**Bouncer — skipped, by design** (`n_flood_rows: 0` over 46,398 considered).

**Detective** (lateral_movement vs benign)

| Metric | Value |
|---|---|
| Accuracy / Precision / Recall / F1 | 1.0000 / 1.0000 / 1.0000 / **1.0000 — not reportable** |
| Train / test windows | 68 / **34** |
| Train / test tunnel flows | 21 / **11** |
| `n_uncertain_verdicts` | 0 |
| Confusion matrix | `[[33, 0, 0], [0, 0, 0], [0, 0, 1]]` |
| Train seconds | 0.83 |
| `reportable_as_detection` | **false** |

**Weight verification:** `dnsexf2021_bouncer_loadable: null` (not trained);
`dnsexf2021_detective_loadable: true`.

**Expected? Yes — declared.** This is the most explicitly predicted result in the
repository: `docs/experiments_guide.md:1767` says *"Expect `VERDICT:
NO_VALID_DETECTION_METRIC` and a printed F1 near 1.0 that is explicitly marked
non-reportable. **That output is the intended result**"*, and adds that if the runner
instead reports a valid detection metric, the data is not what the loader was written
against.

**Why this result happened.** Two independent failures, both measured rather than
argued:

1. **The label is not recoverable from the features.** 72.78% of all rows sit on a
   feature vector carrying two different labels — 99.95% of the exfiltration rows. Where
   one vector carries two classes, no function of those features can be right on more
   than the majority count, which caps *any* model at 0.821. That is below the
   repository's own 0.85 SLO, so the SLO is unreachable by construction independent of
   the model. The runner composes `ceiling` and `meets_slo: false` into the artifact so
   a later reader cannot see the smoke-test 1.0 without them.
2. **The attack class collapses under the guard.** Removing ambiguous vectors leaves 32
   distinct attack signatures across 149 rows, against 46,366 distinct benign
   signatures. The artifact's `why_not_reportable` names the repository's stated floor —
   100 signatures / 500 rows — which this misses by more than 3×.
3. **Every lane input is reconstructed.** The dataset carries no IP, no port, no byte
   count and no flow duration, so `source_ip` is a single constant, `dest_ip` is a
   deterministic synthetic IPv4 derived from the real queried domain, `total_bytes` is
   derived from the real query-name length, and `duration_ms` is 0.
   `features_are_real: false` records it.

The printed 1.0000 is one correctly-classified positive window out of 34. The artifact
says plainly that it confirms the model trains, saves and loads, and "must NOT be cited
as detection performance." This register follows that instruction: **E's 1.0000 is not
a result.**

---

### 2.6 Experiment F — CSE-CIC-IDS2018

**Source artifact:** `results/experiments/cicids2018_metrics.json` ·
`verdict: BOUNCER_REPORTABLE_DETECTIVE_BELOW_SLO` · `2026-10-07T13:00:48Z` ·
`elapsed_seconds: 171.95` · four day files · `stride: 20` (Bouncer),
`detective_stride: 1` · **`synthetic_hosts: true`** with a detailed note: the ML-ready
CSVs carry no Source IP, Destination IP or Source Port; `source_ip` is one reconstructed
constant; `dest_ip` is a **deterministic bijection of the real destination port**; edge
features `port_entropy` and `flow_count` are constant by construction; `source_port` is
None. Flow measurements and the capture clock are real.

**Bouncer** (flood vs benign)

| Metric | Value |
|---|---|
| Accuracy | 0.9902 |
| Precision / Recall | 0.9994 / 0.9857 |
| **F1** | **0.9925** |
| AUC | 0.9994 |
| Confusion matrix | `[[10347, 11], [284, 19516]]` |
| Train / test rows | 61,224 (40,200 flood / 21,024 benign) / 30,158 (19,800 / 10,358) |
| Train seconds | 0.165 |

Feature medians, flood vs benign — the separation the model is reading:
`event_rate` 44.0 vs 1.5 · `byte_rate` 1,967.5 vs 1,440.0 · `dest_port_entropy` 0.0 vs 1.0 ·
`unique_dest_count` 1.0 vs 2.0 · `avg_duration_ms` 12.38 vs 2,505.7 · `same_dest_ratio` 1.0 vs 0.5.

**Bouncer cross-day (leave-one-day-out), four blocks**

| Trained on (flood / benign) | Tested on (flood / benign) | F1 | AUC | Confusion matrix |
|---|---|---|---|---|
| Fri-16 / Wed-28 | Wed-21 / Thu-01 | 0.9985 | 0.9995 | `[[10254, 0], [90, 29910]]` |
| Fri-16 / Thu-01 | Wed-21 / Wed-28 | 0.9977 | 0.9993 | `[[20959, 56], [84, 29916]]` |
| Wed-21 / Wed-28 | Fri-16 / Thu-01 | **0.8684** | 0.9883 | `[[10253, 1], [6978, 23022]]` |
| Wed-21 / Thu-01 | Fri-16 / Wed-28 | **0.8681** | 0.9925 | `[[20992, 23], [6972, 23028]]` |

**Detective** (port_scan vs benign), 2 s windows, 4 epochs, batch 16

| Metric | Value |
|---|---|
| Accuracy | 0.5075 |
| Precision / Recall | 0.4717 / 0.4986 |
| **F1 (port_scan)** | **0.3401 — below the 0.85 SLO** |
| Majority-class accuracy | 0.7394 on the window pool |
| Prediction breakdown | benign 5,479 · port_scan **23** · uncertain 9 |
| Confusion matrix over `[benign, port_scan, uncertain]` | `[[2787, 13, 5], [2692, 10, 4], [0, 0, 0]]` |
| Train / test windows | 11,187 (5,492 probe / 5,695 benign) / 5,511 (2,706 / 2,805) |
| Train seconds | 89.733 |

**Pre-training ceiling probe** (a cross-validated linear probe over the same windows,
measured *before* the GAT is trained) and the **cadence sweep**:

| Cadence | Windows | Attack fraction | AUC | F1 (attack) | Median edges attack/benign | Attack/benign activity ratio |
|---|---|---|---|---|---|---|
| **2 s** | 32,777 | 0.2606 | **0.6017** | 0.4033 | 6 / 5 | **1.20** |
| 5 s | 13,357 | 0.2578 | 0.6327 | 0.4289 | 12 / 10 | 1.20 |
| 10 s | 6,742 | 0.2563 | 0.6598 | 0.4438 | 22 / 17 | 1.29 |
| 30 s | 2,268 | 0.2540 | 0.7395 | 0.4997 | 58 / 45 | 1.29 |
| 60 s | 1,136 | 0.2535 | 0.7983 | 0.5741 | 109.5 / 85 | 1.29 |
| 120 s | 567 | 0.2557 | **0.8518** | 0.6374 | 213 / 164 | 1.30 |

`max_port_entropy` is `null` in every cadence block — the feature is constant, exactly as
the `synthetic_hosts` note predicts.

**Label quality:** 23,605 rows · 20,838 distinct vectors · duplicate-row fraction 0.1172 ·
194 ambiguous vectors covering 1,762 rows (**7.46%**) · **deterministic ceiling 0.9778** ·
ambiguous fraction by category: normal 0.06, probe 0.145.

**Weight verification:** `cicids2018_bouncer_loadable: true`, `cicids2018_detective_loadable: true`.

**Expected? Yes — declared.** `docs/experiments_guide.md:1782` says *"Expect `VERDICT:
BOUNCER_REPORTABLE_DETECTIVE_BELOW_SLO`. The Bouncer's F1 is near 0.99 and the
Detective's is near 0.34, well below the 0.85 external-dataset SLO. **That divergence is
the intended, measured result**"*, and adds that the ceiling probe and cadence sweep are
part of the expected output.

**Why this result happened.**

- **Bouncer 0.9925 within-day.** F is the first experiment with a real capture clock, so
  its `event_rate` is a genuine rate rather than an artefact of even replay spacing — and
  the feature medians show it doing real work (44.0 events/2 s for flood against 1.5 for
  benign, a 29× separation). The Bouncer was designed for volumetric floods and this is a
  volumetric flood.
- **Bouncer cross-day collapse to 0.8684 / 0.8681 is a calibration failure, not a
  discrimination failure.** In both weak blocks the model predicts the unseen flood day
  correctly for 23,022 / 23,028 of its rows but misses 6,978 / 6,972 — while AUC stays at
  0.988 / 0.993. The ranking holds; the calibrated 0.5 threshold lands in the wrong place
  when the unseen flood is the slow/rate-limited Hulk or SlowHTTPTest family, whose rate
  shape differs from the HOIC floods it trained on. Both numbers ship rather than being
  smoothed into an average.
- **Detective 0.3401 is below its own ceiling, and the ceiling is what explains it.**
  The probe scores AUC 0.6017 / F1 0.4033 on the identical windows *before any GAT
  training*, so the graph-granularity signal is near chance for a linear model too, and a
  3-class attention head doing worse than that is the expected outcome rather than a
  training failure. The model collapsed to the majority class: 5,479 of 5,511 windows
  predicted benign, accuracy 0.5075 against a 0.5090 majority baseline — i.e. at chance,
  slightly below it.
- **This is not label noise.** Only 7.46% of rows sit on an ambiguous vector and the
  label's deterministic ceiling is 0.9778, so `Infilteration` is internally consistent.
  What separates it lives in the 74 CICFlowMeter columns, and neither lane reads them.
  **[reading]** The graph lane is additionally handicapped by construction here: with
  `source_ip` a single constant and `dest_ip` a bijection of the destination port, the
  graph measures service fan-out rather than host fan-out, and `max_port_entropy` comes
  back `null` in every cadence block. The lane is being asked to find a scan in a graph
  that has had the host dimension removed from it.
- **The cadence sweep rules out the obvious rescue.** AUC climbs 0.6017 → 0.8518 across a
  60× change in window length, but the attack/benign median activity ratio stays flat at
  1.20–1.30 the whole way. Rising AUC at a constant ratio is the longer window estimating
  the same ~25% difference more precisely, not uncovering a port-scan signature. Even at
  120 s, F1 0.6374 remains below the 0.85 SLO.

---

### 2.7 Experiment G — CIC-DDoS2019

**Source artifact:** `results/experiments/cicddos2019_metrics.json` ·
`verdict: BOUNCER_REPORTABLE` · `2026-10-08T06:02:53Z` · `elapsed_seconds: 1764.63` ·
two day archives (`CSV-01-12`, `CSV-03-11`) at `stride: 32` ·
**`synthetic_hosts: false`** — Source IP, Source Port, Destination IP, Destination Port
and a wall-clock microsecond Timestamp are all real in this release.

**Bouncer** (flood vs benign)

| Metric | Value |
|---|---|
| Accuracy | 0.9986 |
| Precision | **1.0000** |
| Recall | 0.9982 |
| **F1** | **0.9991** |
| AUC | 0.9998 |
| Confusion matrix | `[[1097, 0], [7, 3953]]` |
| Train / test rows | 10,266 (8,040 flood / 2,226 benign) / 5,057 (3,960 / 1,097) |
| Train seconds | 0.211 |

Feature medians, flood vs benign: `event_rate` **273.0 vs 0.5** · `byte_rate` **374,513.5 vs 106.0** ·
`dest_port_entropy` 9.06 vs 0.0 · `unique_dest_count` 1.0 vs 1.0 · `avg_duration_ms` 76.36 vs 32.0 ·
`same_dest_ratio` 1.0 vs 1.0.

**Cross-day (leave-one-day-out)**

| Trained on | Tested on | F1 | AUC | Confusion matrix |
|---|---|---|---|---|
| CSV-03-11 | CSV-01-12 | 0.9992 | 0.9996 | `[[1560, 0], [9, 5991]]` |
| CSV-01-12 | CSV-03-11 | 0.9993 | 0.9993 | `[[1763, 0], [8, 5992]]` |

**Label quality:** 1,457,028 rows · 404,186 distinct vectors · **3 ambiguous vectors
covering 25 rows (0.00%)** · **deterministic ceiling 1.0000** ·
`by_category`: `dos` 1,454,352 / `normal` 2,676.

**Host overlap:** attack hosts 3 · benign hosts 243 · **shared hosts 2** —
`192.168.50.1` and `192.168.50.4` appear as sources on both attack and benign rows, and
the day survey records the resulting drops (`benign_rows_dropped_on_flood_hosts`: 6 on
01-12, 12 on 03-11) rather than claiming no benign window contains flood traffic.

**Day survey (the imbalance the pool had to correct for)**

| Day | Rows loaded | Flood rows | Benign rows | Benign fraction | Flood source hosts | Benign source hosts |
|---|---|---|---|---|---|---|
| CSV-01-12 | 1,184,889 | 1,183,323 | 1,560 | 0.001317 | 2 | 132 |
| CSV-03-11 | 636,394 | 634,619 | 1,763 | 0.00277 | 2 | 205 |

Per-day pool: 6,000 flood against 1,560 / 1,763 benign (`min_rows_per_class_per_day: 200`).

**Bouncer — no Detective block exists in this artifact.** The recorded reason:
every labelled attack family in CIC-DDoS2019 is volumetric, so there is no port-scan or
lateral-movement class for the Detective to train on. The artifact calls it "the mirror
image of Experiment D".

**Weight verification:** `cicddos2019_bouncer_loadable: true`. No Detective entry.

**Expected? Yes — declared, and then discounted by the source itself.**
`docs/experiments_guide.md:1804`: *"Expect `VERDICT: BOUNCER_REPORTABLE` with an F1 near
0.999 and a Detective that does not train."* The number matched. G is nonetheless the one
experiment in the register where the repository argues **against** its own strongest
result: the guide's section (§"Why 0.9991 Is A Weak Generalisation Test") and
`README.md:146` both state the score confirms the rate feature set *on the case it was
designed for* and says nothing about distributed attacks. The expected/unexpected column
is therefore "yes", and the evidential weight is deliberately low.

**Why this result happened.**

- **The separation is enormous and real.** `event_rate` medians of 273 vs 0.5 events/2 s
  and `byte_rate` medians of 374,513 vs 106 bytes/s are not marginal differences; on this
  data the classes are trivially separable, which is what a 0.9991 F1 on clean labels
  (ceiling 1.0000) looks like.
- **The known confound was measured, not assumed.** `Inbound` was documented as a per-file
  constant, but read as a classifier (`attack iff Inbound == 1`) it scores 99.3–99.98% —
  i.e. it is the label in disguise; excluding it was correct for a reason the original
  comment had wrong. The victim hosts appearing on both attack and benign rows were
  likewise handled by dropping those rows and counting the drops.
- **Why the number does not generalise.** Per `README.md:146`, every attack row in the
  release comes from a **single host** (`172.16.0.5`), so the per-host 2-second window
  sees one extreme-rate source against ordinary background traffic. A distributed flood
  from thousands of modest-rate sources is not represented in the data, and both days are
  captures of the same lab — so the cross-day block compares two captures of one
  environment, not two environments. Its 0.9992 / 0.9993 therefore shows *no* cross-day
  degradation, which the guide reads as a sign the test is weak rather than a sign the
  model is robust.
- **Precision 1.0000 has a small denominator.** It is zero false positives across **1,097**
  benign test windows, not a false-positive rate measured over millions. Benign is
  genuinely rare in this release (0.13% / 0.28% of loaded rows), which is why the flood
  pool was capped at 6,000/day.
- **The data is known-incomplete.** `CSV-01-12.zip` is a salvage of an interrupted
  download holding 8 members rather than the full day; the per-day `raw_labels` census in
  the artifact is what makes that visible.

---

### 2.8 Experiment H — CIC IoT 2023

**Source artifact:** `results/experiments/ciciot2023_metrics.json` · `status: COMPLETE` ·
`2026-10-08T09:29:57Z` · **442,422 flows extracted from 34,581,737 packets across 4
captures** · categories `normal` 197,194 · `dos` 48,376 · `probe` 196,852 ·
**`synthetic_hosts: false`, `features_are_real: true`**.

H is the first experiment whose **rows this repository builds itself**: the loader reads
the publisher's packet captures and groups them into bidirectional 5-tuple flows
(`idle_timeout_s: 30.0`, `active_timeout_s: 120.0`, source = first-packet sender,
timestamp = first-packet time). Addresses, ports and times are the capture's; the flow
definition is the repository's choice.

| Capture | Family | Packets parsed | Flows | Drops (IPv6 / linktype / non-IP) |
|---|---|---|---|---|
| `BenignTraffic.pcap` | Benign_Final | 3,534,248 | 197,194 | 18,428 / 1 / 111,488 |
| `DDoS-UDP_Flood.pcap` | DDoS-UDP_Flood | 26,622,523 | 40,674 | 884 / 1 / 30,144 |
| `Mirai-udpplain.pcap` | Mirai-udpplain | 3,622,666 | 7,702 | 1,537 / 1 / 18,066 |
| `Recon-PortScan.pcap` | Recon-PortScan | 802,300 | 196,852 | 3,915 / 1 / 25,641 |

**Bouncer** (flood vs benign) — capture-holdout split

| Metric | Value |
|---|---|
| Accuracy | 0.7973 |
| Precision / Recall | 0.7793 / 0.8295 |
| **F1** | **0.8036** |
| AUC | 0.8563 |
| Confusion matrix | `[[31116, 9558], [6934, 33740]]` |
| Train / test rows | 15,404 (7,702 / 7,702) / 81,348 (40,674 / 40,674) |
| `class_balanced` | **true** |
| Pre-balance train flood / benign | 7,702 / **131,463** |
| Pre-balance test flood / benign | 40,674 / 65,731 |
| `split_strategy_flood` | `capture_holdout(train=1,test=1)` |
| `split_strategy_benign` | `flow_stride_within_single_capture` |
| Train seconds | 0.322 |

**Detective** (port_scan vs benign), 30 s windows, 20 epochs

| Metric | Value |
|---|---|
| Accuracy | 0.8893 |
| **Majority-class accuracy** | **0.5000** |
| Precision / Recall | 0.9037 / 0.8893 |
| **F1** | **0.8884** |
| Train / test windows | 498 (249 / 249) / 488 (244 / 244) |
| Pool pre-balance (train) | 1,101 benign / 249 probe |
| `n_uncertain_verdicts` | **39 of 488** |
| Confusion matrix over `[benign, port_scan, lateral_movement]` | `[[240, 4, 0], [50, 194, 0], [0, 0, 0]]` |
| `trained_labels` / `never_trained_labels` | `[benign, port_scan]` / `[lateral_movement]` |
| Train seconds | 40.718 |

**Cadence sweep** — the measurement behind the 30 s window choice

| Cadence | Windows | AUC | F1 (attack) | Majority acc. | Median edges attack/benign | Max-unique-ports ratio |
|---|---|---|---|---|---|---|
| 2 s | 17,900 | 0.8300 | 0.5091 | 0.8139 | 9 / 9 | **1.00** |
| 5 s | 7,785 | 0.8965 | 0.6310 | 0.8140 | 18 / 18 | 1.67 |
| 10 s | 3,974 | 0.9608 | 0.7788 | 0.8158 | 32 / 32 | 3.00 |
| **30 s** | **1,352** | **0.9915** | **0.9280** | 0.8151 | 73 / 78 | **11.80** |
| 60 s | 689 | 0.9612 | 0.9164 | 0.8055 | 138 / 135 | 16.75 |
| 120 s | 358 | 0.9610 | 0.9317 | 0.7765 | 207.5 / 196 | 94.75 |
| 300 s | 156 | 0.9953 | 0.9647 | 0.7179 | 248 / 246.5 | 66.67 |
| 600 s | 81 | 0.9964 | 0.9600 | 0.6914 | 284 / 300.5 | 43.48 |

**Weight verification:** `ciciot2023_bouncer_loadable: true`, `ciciot2023_detective_loadable: true`.

**Expected? Yes — declared.** `docs/experiments_guide.md:1828`: *"Expect `STATUS:
COMPLETE` with a Bouncer F1 near 0.80 (cross-flood-family) and a Detective F1 near 0.89
against a 0.5 majority baseline."* Both matched.

**Why this result happened.**

- **Bouncer 0.8036 is a cross-flood-family score, not a within-capture one.** The
  capture-holdout split puts `Mirai-udpplain` in training and `DDoS-UDP_Flood` in test —
  a different flood family, 5× larger, with a different rate shape. The split refuses to
  average that difference away, which is precisely why this is the weakest Bouncer number
  in the register.
- **The pre-balance run is the informative part.** Trained as split, the model had 7,702
  flood rows against **131,463** benign (17:1) and answered by calling almost everything
  benign: recall 0.133, F1 0.2344. Striding the majority down to the minority — after
  featurizing the full stream, so windows stay intact — produced 0.8295 / 0.8036 with AUC
  essentially **unchanged** (0.8603 → 0.8563). Near-equal AUC across a 3.4× F1 change is
  the diagnosis: the imbalance was a threshold artefact, not a shortage of signal. Both
  lanes record `class_balanced: true` and their pre-balance counts, so the subsampling is
  visible in the artifact rather than done quietly.
- **Detective 0.8884 at a 30 s window.** The choice is measured, not assumed: at the
  production 2 s cadence the median number of ports a host contacts is **identical**
  between benign and scan windows (ratio 1.00), so there is no signal at that granularity
  to find. The reason is in the capture: `Recon-PortScan` is an 802,300-packet sweep
  spread over 24.75 hours, so a source appears mid-scan with only ~4 of its own flows
  beside it and the fan-out that *is* a port scan never assembles inside one 2 s bucket.
  At 30 s the port ratio is 11.8× and AUC 0.9915.
- **The honesty check on the sweep.** Edge counts stay *matched* between classes while the
  port ratio climbs (73 vs 78 at 30 s, 248 vs 246 at 300 s), which is what distinguishes
  revealing a scan signature from estimating a constant difference more precisely — the
  opposite of the F finding (§2.6), where the ratio stayed flat and the AUC rise was
  precision alone. The GAT's own numbers agree with the probe's choice: 30 s is best on
  both axes.
- **The 0.5 majority baseline matters.** The pools are balanced, so 0.8893 is 0.39 above
  chance rather than the 0.11 it would have been against the earlier 81.5%-skewed test
  set. The guide notes the unbalanced run reported a *higher* accuracy (0.9231) that said
  less.
- **20 epochs, not the siblings' 4, for a measured reason.** At 4 epochs the loss was
  still descending steeply (0.6231 → 0.4600 with no sign of flattening) and F1 was
  **0.6874** against 0.8884 at 20. The guide discloses that the epoch count was raised on
  the convergence check and then confirmed against test F1, so "the reported figures carry
  a little of that selection"; `--epochs 4` restores the sibling setting.
- **A correction the experiment encodes.** The runner originally thinned each capture to
  40,000 rows by an evenly spaced stride. Because both lanes read features off a 2 s
  window and these captures span 9–25 hours, that stride did not sample traffic — it
  punched holes in it: **89.5% of benign graph windows held a single edge and 392 of 393
  test verdicts came back `UNCERTAIN`.** Loading at full density and capping the *derived*
  windows fixed it. The un-strided load is deliberately not exposed as a convenience flag
  (`--sample-per-capture` reproduces the collapse).

---

### 2.9 Cross-cutting: the lane-substitution experiment

**Source artifact:** `results/experiments/lane_substitution_metrics.json` ·
`status: COMPLETE` · `2026-10-13T17:56:15Z` · added at HEAD (`a1407ce`) ·
**no section in `docs/experiments_guide.md`**.

**Question it asks** (from the artifact): route the exact event stream that formed each
baseline experiment's Detective test windows to the *already-trained* Bouncer, score both
lanes on those identical windows, and report the delta. Its `retrained_variant` sub-block
then isolates a second question: *"'the fast features cannot see a scan' vs 'the shipped
Bouncer was never asked to'."*

**Arm 1 — UNSW-NB15, 66 windows (33 benign / 33 attack), 6,600 events**

| Configuration | Accuracy | `f1_macro` | `precision_port_scan` | `recall_port_scan` | AUC | Δ `f1_macro` vs Detective |
|---|---|---|---|---|---|---|
| Detective (baseline) | 1.0000 | **1.0000** | 1.0000 | 1.0000 | — | — |
| Shipped Bouncer, max-aggregated | 0.9848 | 0.9848 | 0.9706 | 1.0000 | 0.9821 | **−0.0152** |
| Shipped Bouncer, mean-aggregated | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Shipped Bouncer, flow level | 0.9711 | 0.9710 | 0.9994 | 0.9427 | 0.9943 | −0.0290 |
| **Retrained** Bouncer, max-agg | 0.8636 | 0.8611 | 0.7857 | 1.0000 | 1.0000 | **−0.1389** |
| Retrained Bouncer, flow level | 0.9938 | 0.9938 | 0.9948 | 0.9927 | 0.9997 | — |
| Ordering ablation (source_ip-sorted) | 0.9697 | 0.9697 | 0.9429 | 1.0000 | 0.9467 | −0.0303 |

Event-level scores: benign median 0.0006 (mean 0.0033) vs port_scan median **0.7119**
(mean 0.6980) — the flood-trained model fires readily on this dataset's scan events.

**Arm 2 — CIC-IDS2017, 132 windows (66 / 66), 13,200 events**

| Configuration | Accuracy | `f1_macro` | `precision_port_scan` | `recall_port_scan` | AUC | Δ `f1_macro` vs Detective |
|---|---|---|---|---|---|---|
| Detective (baseline) | 0.9848 | **0.9848** | 1.0000 | 0.9697 | — | — |
| Shipped Bouncer, max-aggregated | 0.3788 | 0.2747 | **0.0000** | **0.0000** | **0.0466** | **−0.7101** |
| Shipped Bouncer, mean-aggregated | 0.5000 | 0.3333 | 0.0000 | 0.0000 | 0.0909 | −0.6515 |
| Shipped Bouncer, flow level | 0.4979 | 0.3324 | 0.0000 | 0.0000 | 0.6889 | −0.6524 |
| **Retrained** Bouncer, max-agg | 0.9924 | **0.9924** | 0.9851 | 1.0000 | 1.0000 | **+0.0076** |
| Retrained Bouncer, flow level | 0.9980 | 0.9980 | 0.9997 | 0.9964 | 0.9999 | — |
| Ordering ablation (source_ip-sorted) | 0.4924 | 0.3299 | 0.0000 | 0.0000 | 0.5473 | −0.6549 |

Event-level scores: benign mean 0.0042 vs port_scan median **0.0**, p95 **0.0**,
mean **0.0** — the shipped model never fires on this dataset's scan traffic at all.

**Cost block:** Detective 0.7962 ms/window (UNSW) and 0.9611 ms/window (CIC), graph
inference included; Bouncer 0.004977 ms/event (UNSW) and 0.023798 ms/event (CIC), no
graph. Observed 100 events per window in both arms. Wall-clock, CPU only, whole test set.

**Expected? Not assessable — no documented expectation exists.** This experiment postdates
the guide and has no section, no `Expect` line and no README entry. **[reading]** What
follows is my assessment against the experiment's own stated question, not the
repository's.

**Why this result happened.**

- **The two arms answer the experiment's question in opposite directions, and that
  disagreement is the finding.**
  - **On CIC-IDS2017** the shipped Bouncer's scores on `port_scan` windows are exactly
    zero (median, p95 and mean all 0.0), giving `f1_macro` 0.2747 and an **AUC of 0.0466 —
    below chance**, because some benign windows score high (p95 0.9952) while every attack
    window scores 0. The `retrained_variant` — identical six features, identical XGBoost
    recipe, trained on benign-vs-port_scan instead — reaches **0.9924**. That pins the
    failure on the task, not the features: the six rate features *can* see a scan when
    asked to; the shipped Bouncer simply was never asked, because B's Bouncer trained on
    `dos` against `normal` and `probe` was never its positive class.
  - **On UNSW-NB15** the same substitution barely degrades (0.9848 vs 1.0000, and exactly
    1.0000 mean-aggregated) because the shipped Bouncer *does* fire on that dataset's scan
    events (median event score 0.7119). **[reading]** Why the two arms differ is not
    established by any artifact I read. The most likely candidate is C's
    `synthetic_hosts: true` — hosts reconstructed from each row's own connection-rate
    counters — which may push the attack streams toward a flood-like event signature. That
    is a hypothesis, not a measurement, and it is the single largest unexplained
    discrepancy in this register.
- **An uncomfortable corollary on CIC-IDS2017:** the retrained six-feature Bouncer at
  0.9924 **beats the Detective's 0.9848** on the Detective's own windows, needing no graph
  — at 0.0238 ms/event against 0.9611 ms/window with 100 events per window. On this
  dataset the graph lane buys nothing and costs ~400× per window. The experiment does not
  state this conclusion, but it is what the numbers in the file show, and it sharpens the
  question §2.2 raised about what B's 0.9846 actually demonstrates.
- **The ordering ablation supports the exchangeability of the comparison.** The
  substitution deliberately does *not* re-sort rows by `source_ip` (the Bouncer's own
  training path does), on the stated grounds that re-ordering is changing the data. The
  ablation measures the size of that effect: re-sorting moves the UNSW arm from 0.9848 to
  0.9697 (−0.0151) and the CIC arm from 0.2747 to 0.3299 (still chance-level). Neither
  moves the conclusion, so the substitution's result is not an artefact of stream order.

---

### 2.10 Uncommitted pilot — CIC Datasense IoT

**Source artifact:** `results/experiments/datasense_metrics.json` · internal
`"experiment": "datasense_iot"` · `status: COMPLETE` · `2026-09-29T13:10:57Z` ·
**untracked in git**.

Companion untracked files in the working tree: `twin/datasense.py`,
`scripts/run_datasense_experiment.py`, `scripts/download_tabulariot2024.py`,
`tests/unit/test_datasense_loader.py`, `scratch/sample.pcap`.
`models/experiments/datasense/` exists locally. **Neither `README.md` nor
`docs/experiments_guide.md` mentions this experiment at all.**

| Lane | Metric | Value |
|---|---|---|
| Bouncer | Accuracy / Precision / Recall / **F1** | 0.9819 / 0.9596 / 0.9794 / **0.9694** |
| Bouncer | Confusion matrix | `[[230, 4], [2, 95]]` |
| Bouncer | Train / test rows | 669 (196 attack / 473 normal) / 331 (97 / 234) |
| Bouncer | Train seconds | 0.084 |
| Detective | Accuracy / Precision / Recall / **F1** | 0.75 / 0.375 / 0.50 / **0.4286** |
| Detective | Confusion matrix | `[[3, 0], [1, 0]]` |
| Detective | **Train / test windows** | **7 / 4** |
| Detective | Train seconds | 0.064 |

Dataset: 1,000 flows, `normal` 707 / `r2l` 293. Both weights verify as loadable.

**Expected? Not assessable.** There is no documented expectation to compare against, and
this artifact should not be cited as an experiment result at all. **[reading]** my
assessment only:

- **The Bouncer number is a plausible pilot result on a 331-row test set.** 0.9694 F1 on
  97 attack flows is thin but not meaningless; it is at least in the range the other
  flood-vs-benign runs occupy.
- **The Detective number is not interpretable and must not be quoted.** The evaluation is
  **4 test windows containing 1 positive example**, and the model predicted `benign` for
  all four — hence exactly zero true positives, recall 0.50 (the `r2l` window counted as
  the single positive by whatever the pooling rule is), precision 0.375 and F1 0.4286.
  Those three figures are arithmetic on one example. For scale: Experiment E's own
  single-positive window over 34 is the case the repository **refuses to report**, on the
  grounds that it measures a lookup table — and this pilot is an order of magnitude
  smaller than that.
- **It misses the repository's own stated floor by a wide margin.** E's
  `why_not_reportable` names a 100-signature / 500-row floor; a 1,000-flow sample with 293
  attack rows cannot form enough 2 s windows to approach it, which is the structural
  reason the window count is 7/4 rather than a training setting.
- **Two naming inconsistencies are worth fixing before this is committed.** The artifact
  keys the experiment `datasense_iot` while the file is `datasense_metrics.json` and the
  weights live under `models/experiments/datasense/`; every other experiment uses one
  name across artifact, filename and weight directory. And a `download_tabulariot2024.py`
  sits beside `run_datasense_experiment.py`, which suggests a second dataset is in
  progress under the same pilot.

---

## 3. Where the numbers do not line up

Findings from tabulating, recorded because a reader checking this register against the
repository will hit them. None of these is a fabricated number; all are gaps between what
the artifacts measure and what the prose says.

1. **Three stale experiment counts** — guide says "Seven", README says "six", both
   document eight (§0.5).
2. **No `label_quality` block before Experiment E.** A, B, C and D carry no measured
   ambiguity ceiling. For E the ceiling (0.821) is what *disqualifies* the result; for A,
   B, C and D there is no ceiling to check a perfect or near-perfect score against.
   **[reading]** The two Detective 1.0000 results (A, C) sit in exactly that unmeasured
   region.
3. **`event_rate` may be uninformative in the synthetic-spacing experiments.**
   `services/bouncer/features.py:60` computes `event_rate = n / 2.0` where `n` counts a
   source's events in the trailing 2 seconds, and each synthetic-spacing runner replays
   rows at strictly even wall-clock spacing (`ROW_SPACING_MS = 20`,
   `ts = T0 + i × 20 ms`, in the B, D, E, pilot and lane-substitution runners). For any
   source appearing at a regular stride in that replay, the window count is fixed and the
   feature carries no information. The guide states this mechanism for Experiment F
   (`docs/experiments_guide.md:137`, "equal spacing makes `event_rate` constant by
   construction") but no other section says whether it applies to A–E, and no metrics
   artifact records the per-feature variance that would settle it. **This is an open
   question, not a claim.**
4. **Experiment A's recorded weight paths do not resolve** — they assume a nested
   `Kronus/` directory (§2.1).
5. **`docs/experiments_guide.md` documents no lane-substitution section**, although the
   experiment is committed at HEAD and reports the single most consequential comparison in
   the repository: on CIC-IDS2017 a six-feature flow model matches the graph lane on the
   graph lane's own windows, while on UNSW-NB15 the same substitution nearly matches too.
   Neither dataset's guide section mentions it.
6. **The lane-substitution arms disagree with each other** (§2.9) — 0.2747 on
   CIC-IDS2017 against 0.9848 on UNSW-NB15 for the same procedure — and no artifact
   explains why.
7. **A's Bouncer is below the 0.85 SLO** (0.8179) while the SLO's scope
   (`README.md:51`, "External-dataset F1") appears to exclude A. E and F are both flagged
   for missing the SLO; A is not. This may be correct scoping, but the register notes it
   because a reader scanning the F1 column will see a sub-SLO number with no annotation.

---

## 4. Conclusion

### 4.1 What the eight documented experiments collectively establish

**The Bouncer's contract holds, and its limits are measured rather than asserted.**
Across eight datasets the flood-vs-benign lane scores 0.8036–0.9991, and in every case the
repository reports the number *with* the condition that qualifies it:

| Bouncer result | The condition attached to it |
|---|---|
| G 0.9991 | Single-host attacker, one lab, two captures, 1,097 benign test windows |
| H 0.8036 | **Cross-flood-family** — trained on Mirai, tested on a differently-shaped flood |
| F 0.9925 | Cross-day drops to 0.8684/0.8681 on the slow-flood family (calibration, not ranking) |
| B 0.9987 | 2.83M real flows, real IPs and ports, 2.2M:380k class skew |
| C 0.9963 | Reconstructed topology (features real) |
| A 0.8179 | A reproduction of a historical figure on 1999 benchmark data |

The range is not noise. It tracks three things the repository measures directly: **how
many hosts generate the attack** (G's single host is why 0.9991 is discounted), **whether
the test attack family was seen in training** (H's 0.8036 is the honest cost of refusing
to average that away), and **whether the clock is real** (F is the first dataset where
`event_rate` is not an artefact of replay spacing).

**The Detective's numbers split into three groups, and the repository sorts them
correctly.**

| Group | Experiments | Status |
|---|---|---|
| Scores that hold up | D 0.9697, H 0.8884 | Both on data with real topology or real capture clock; H measured against a 0.5 baseline |
| Scores withheld | E 1.0000 | Refused: ceiling 0.821 < SLO, 32 attack signatures, 1 positive window |
| Scores published below target | F 0.3401 | Published: the label is sound (7.46% ambiguity), so the weakness is the model's and the number should be seen |
| Perfect scores on reconstructed graphs | A 1.0000, C 1.0000 | Published, with small positive-window counts (25, 33) and no measured ceiling |

That E is withheld and F is published is the single most defensible judgement in the
repository: **the deciding factor is which side owns the failure.** E's failure is the
dataset's (labels unrecoverable, no IPs/ports/bytes/durations), so no number from it means
anything. F's failure is the model's on a dataset whose label is internally consistent and
whose ceiling was measured *before* training — so withholding it would be the dishonest
choice. The distinction is not "good result vs bad result"; it is "the evidence is
unusable vs the evidence is real and unflattering."

**The lane-substitution experiment is the most important result here and is the least
documented.** It shows that on CIC-IDS2017 a six-feature flow classifier trained for the
job matches or beats the graph lane on the graph lane's own windows (0.9924 vs 0.9848),
while the *shipped* Bouncer — trained only on `dos` vs `normal` — scores AUC 0.0466, below
chance, because it had never been asked to recognise a scan. This reframes Experiments B
and C: their Detective scores are not evidence that graphs are necessary, only that a
graph model can reach a given number. It is committed, it is well-designed (with its own
ordering ablation and a retrained control), and nothing in the guide or README discusses
it. **It deserves a guide section more than any other single artifact in the repository.**

### 4.2 What the evidence does not support

- **No distributed-attack claim anywhere.** G's own section rules it out; F's flooded
  days are a lab; the largest real-topology flood capture in the register is G's
  single-host one.
- **No claim that the graph lane beats a well-chosen flow model.** On the one dataset
  where both were trained for the same task on the same windows (§2.9, CIC-IDS2017), it
  does not.
- **No claim of production performance.** `README.md:168` already scopes the project to
  "laptop-scale, portfolio-scoped", and the register is consistent with that: the largest
  single training run is 89.7 s, the whole CIC-DDoS2019 experiment took 1,765 s
  wall-clock, and every evaluation is a held-out split of one or two captures rather than
  a live deployment.
- **Four of the eight datasets have no measured ambiguity ceiling** (§3.2), so the
  repository's own strongest honesty mechanism is not applied uniformly.
- **No experiment reports a false-positive rate at scale.** G's precision of 1.0000 rests
  on 1,097 benign windows; F's on 10,358 benign flows; nobody has measured over the
  millions of benign rows the raw captures actually contain.

### 4.3 The one-line summary of each experiment

| # | Experiment | One line |
|---|---|---|
| A | NSL-KDD | Reproduces its historical baseline to ±0.0003 — the control, not a result. |
| B | CIC-IDS2017 | The Bouncer's real-data ceiling (+0.18 F1 over A); the Detective's first non-perfect score. |
| C | UNSW-NB15 | Strong Bouncer on reconstructed hosts; a perfect Detective over 33 positive windows with no measured ceiling. |
| D | DoHBrw2020 | The honest middle: real topology, first third class, 0.9697 on 64/66 windows, Bouncer correctly refused. |
| E | DNS-EXF2021 | A deliberately negative result that measures the *dataset* as incapable (ceiling 0.821 < SLO) and refuses to report its own 1.0000. |
| F | CIC-IDS2018 | The clearest negative finding: the Bouncer's 0.9925 holds while the graph lane collapses to chance (0.3401) on a graph with the host dimension removed, and the pre-training probe (AUC 0.6017) proves the signal was never there. |
| G | CIC-DDoS2019 | The strongest number in the repository, argued down by its own section: one attacker, one lab, 1,097 benign windows. |
| H | CIC IoT 2023 | The most methodologically honest run: rows built from raw packets, class imbalance diagnosed as a threshold artefact, the 30 s window chosen by measurement, and the weakest Bouncer score shipped rather than hidden. |
| 9 | Lane substitution | Answers "can the fast features see a scan?" with **yes** — and shows the shipped Bouncer was simply never asked. |
| 10 | Datasense pilot | Untracked, undocumented, 4 test windows with 1 positive: not evidence, and must not be cited as any. |

### 4.4 The three things to fix first

1. **Document the lane-substitution experiment** in `docs/experiments_guide.md`, including
   the UNSW/CIC disagreement and the cost comparison. It is the strongest evidence in the
   repository and currently has no prose at all.
2. **Correct the counts** — "Seven" at `docs/experiments_guide.md:3`, "six" at
   `README.md:88`, against eight documented experiments.
3. **Resolve the `event_rate` question** (§3.3) by recording per-feature variance in the
   metrics, and either commit or delete the untracked Datasense pilot (§3, §2.10).

Underneath those, the repository's experimental practice is sound in a way that is
unusual to find: it measures the ceiling before training (F), verifies the labels
(E, G), discloses its splits and balancing (H), argues against its own best number (G),
and withholds a score that would look excellent and mean nothing (E). The failure modes
in this register are documentation drift and a missing section, not inflated numbers.
