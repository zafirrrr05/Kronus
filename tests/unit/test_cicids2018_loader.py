"""Unit tests for the CSE-CIC-IDS2018 loader (twin/cicids2018.py).

These do NOT require the real ~980 MB download: they build small
CSE-CIC-IDS2018-shaped CSVs in a tmp dir using the *real* 80-column header and
the release's own label spellings, then assert the loader resolves classes,
preserves the real measurements, reconstructs hosts deterministically, and
returns the real capture clock.

Four things here get deliberate extra attention, because each one is a way
this loader could be silently wrong:

1. THE SCHEMA GUARD. A loader for this dataset already exists in the tree that
   reads CIC-IDS2017's column names while being named for CIC-IDS2018, maps
   CIC-DDoS2019's label vocabulary, and against the real files returns an
   all-zero dataset while reporting success. `test_rejects_a_cicids2017_schema`
   is the regression test for that class of failure; it is the reason the
   guard exists at all.
2. THE CLOCK IS REAL AND MUST STAY OUT OF `features`. Every other external
   loader here replays rows at synthetic spacing. This one has a real
   timestamp, so a test pins that it comes back (via the timed loader) and a
   second pins that it does NOT leak into the model's feature dict.
3. THE STRIDE, NOT A PREFIX. Both row caps must thin the file evenly so a
   bounded sample still spans the attack window.
4. THE RECONSTRUCTION IS A BIJECTION OF THE REAL PORT. `dest_ip` is derived
   from `Dst Port`, so the tests assert distinct ports give distinct hosts and
   the same port gives the same host — the property the graph lane's
   `unique_ports_contacted` depends on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from libs.constants import DataOrigin, Label, Protocol
from libs.schemas import TelemetryEvent
from services.telemetry_exporter.converters import flow_row_to_event
from twin.cicids2018 import (
    COL_LABEL,
    COL_TIMESTAMP,
    MONITORED_CLIENT_IP,
    _dest_ip_for,
    attack_intervals,
    in_any_interval,
    label_quality_report,
    load_cicids2018,
    load_cicids2018_timed,
    resolve_class,
    summarize,
)

# The real 80-column header of the released ML-ready CSVs, verbatim. Using the
# actual schema rather than a subset is the point: it is what distinguishes
# this dataset from CIC-IDS2017 in the tree.
_REAL_HEADER = [
    "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts",
    "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max",
    "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max",
    "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s",
    "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Len", "Bwd Header Len",
    "Fwd Pkts/s", "Bwd Pkts/s", "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean",
    "Pkt Len Std", "Pkt Len Var", "FIN Flag Cnt", "SYN Flag Cnt",
    "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    "CWE Flag Count", "ECE Flag Cnt", "Down/Up Ratio", "Pkt Size Avg",
    "Fwd Seg Size Avg", "Bwd Seg Size Avg", "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg", "Subflow Fwd Pkts", "Subflow Fwd Byts",
    "Subflow Bwd Pkts", "Subflow Bwd Byts", "Init Fwd Win Byts",
    "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min", "Label",
]

# CIC-IDS2017's naming, which shares "Flow Duration" and "Timestamp" with the
# 2018 schema but renames every column the loader actually reads.
_CICIDS2017_HEADER = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Source IP", "Source Port", "Timestamp",
    "Label",
]


def _row(
    dport: int = 443,
    proto: int = 6,
    ts: str = "28/02/2018 10:00:00",
    dur_us: int = 250_000,
    fwd_bytes: int = 500,
    bwd_bytes: int = 700,
    label: str = "Benign",
    **feature_overrides: float,
) -> dict:
    """One CSV row: the seven columns the loader reads, plus real features.

    Every other column defaults to 0 so the only thing a test varies is the
    thing it is about.
    """
    row = dict.fromkeys(_REAL_HEADER, 0)
    row["Dst Port"] = dport
    row["Protocol"] = proto
    row["Timestamp"] = ts
    row["Flow Duration"] = dur_us
    row["TotLen Fwd Pkts"] = fwd_bytes
    row["TotLen Bwd Pkts"] = bwd_bytes
    row["Label"] = label
    row.update(feature_overrides)
    return row


def _write_day(path, rows, header=None) -> None:
    pd.DataFrame(rows, columns=header or _REAL_HEADER).to_csv(path, index=False)


# --- label resolution -------------------------------------------------------

@pytest.mark.parametrize(
    "raw_label",
    [
        "DoS attacks-Hulk",
        "DoS attacks-SlowHTTPTest",
        "DoS attacks-GoldenEye",
        "DoS attacks-Slowloris",
        "DDOS attack-HOIC",
        "DDoS attacks-LOIC-HTTP",
        "DoS attacks-SlowHTTPTest ",  # trailing space in the release
    ],
)
def test_flood_labels_resolve_across_the_release_spellings(raw_label):
    """The vocabulary is internally inconsistent ('DDoS attacks-LOIC-HTTP' and
    'DDOS attack-HOIC' in the same release); prefix matching must absorb that
    rather than silently dropping real attacks."""
    assert resolve_class(raw_label) == ("dos", Label.FLOOD)


@pytest.mark.parametrize("raw_label", ["Infilteration", "Infiltration", "infilteration"])
def test_infiltration_maps_to_port_scan(raw_label):
    """The typo is the release's own. It maps to PORT_SCAN because UNB's own
    scenario description is an Nmap IP sweep and full port scan, and because
    this dataset has no PortScan-labelled flows at all — so this label is the
    only source of probe-class data here."""
    assert resolve_class(raw_label) == ("probe", Label.PORT_SCAN)


def test_benign_maps_to_normal():
    assert resolve_class("Benign") == ("normal", Label.BENIGN)


@pytest.mark.parametrize(
    "raw_label",
    ["FTP-BruteForce", "SSH-Bruteforce", "Bot", "Web Attack - Brute Force",
     "Web Attack – XSS", "SQL Injection", "Label", "", None],
)
def test_labels_with_no_kronus_counterpart_resolve_to_none(raw_label):
    """Dropped and counted, never force-fitted into a class. 'Label' is the
    release's own embedded header row."""
    assert resolve_class(raw_label) is None


# --- the schema guard -------------------------------------------------------

def test_rejects_a_cicids2017_schema(tmp_path):
    """The regression test for the retired mislabeled loader.

    CIC-IDS2017 shares 'Flow Duration' and 'Timestamp' with this schema, so a
    plausible-looking CSV can still be the wrong dataset. Reading it here
    would produce an all-zero dataset and report success; it must raise.
    """
    path = tmp_path / "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"
    _write_day(path, [{c: 0 for c in _CICIDS2017_HEADER}], header=_CICIDS2017_HEADER)
    with pytest.raises(ValueError, match="Dst Port|missing"):
        load_cicids2018(path)


# --- real measurements survive ---------------------------------------------

def test_real_measurements_are_preserved_and_hosts_reconstructed(tmp_path):
    path = tmp_path / "day.csv"
    _write_day(path, [_row(dport=8080, proto=17, dur_us=250_000,
                           fwd_bytes=500, bwd_bytes=700)])
    rows = load_cicids2018(path, sample_per_file=None)

    assert len(rows) == 1
    row = rows[0]
    assert row.total_bytes == 1200                 # 500 + 700, summed
    assert row.duration_ms == 250                  # microseconds -> ms
    assert row.dest_port == 8080                   # real, from Dst Port
    assert row.protocol is Protocol.UDP            # IANA 17
    assert row.source_ip == MONITORED_CLIENT_IP    # reconstructed, constant
    assert row.source_port is None                 # absent from the release
    assert row.dest_ip == _dest_ip_for(8080)       # bijection of the real port
    assert row.origin is DataOrigin.REAL


@pytest.mark.parametrize(
    ("number", "expected"),
    [(6, Protocol.TCP), (17, Protocol.UDP), (1, Protocol.ICMP), (99, Protocol.OTHER)],
)
def test_protocol_numbers_map_to_the_enum(tmp_path, number, expected):
    path = tmp_path / "day.csv"
    _write_day(path, [_row(proto=number)])
    assert load_cicids2018(path, sample_per_file=None)[0].protocol is expected


def test_dest_ip_is_a_bijection_of_the_real_dest_port(tmp_path):
    """Distinct destinations in a window must be the capture's own distinct
    *services* — the graph lane's unique_ports_contacted depends on it."""
    path = tmp_path / "day.csv"
    _write_day(path, [_row(dport=80), _row(dport=443), _row(dport=80)])
    rows = load_cicids2018(path, sample_per_file=None)

    by_port: dict[int, str] = {}
    for row in rows:
        by_port.setdefault(row.dest_port, row.dest_ip)
    assert by_port[80] != by_port[443]         # distinct ports, distinct hosts
    assert sum(r.dest_ip == by_port[80] for r in rows) == 2   # same port, same host


def test_a_row_becomes_a_valid_telemetry_event(tmp_path):
    """The loader's output has to survive the schema boundary the pipeline
    actually uses, not just look right as a dataclass."""
    path = tmp_path / "day.csv"
    _write_day(path, [_row(dport=1433, proto=6)])
    event = flow_row_to_event(load_cicids2018(path, sample_per_file=None)[0])
    assert isinstance(event, TelemetryEvent)
    assert event.dest_port == 1433
    assert event.bytes == 1200
    assert event.source_port is None


# --- the clock --------------------------------------------------------------

def test_the_real_capture_clock_is_returned_in_order(tmp_path):
    """This dataset has a real clock and the runner replays at true
    inter-arrival times, so the times must come back — parsed, not inferred."""
    path = tmp_path / "day.csv"
    _write_day(path, [
        _row(ts="28/02/2018 10:00:05"),
        _row(ts="28/02/2018 10:00:01"),
        _row(ts="28/02/2018 10:00:03"),
    ])
    rows, times = load_cicids2018_timed(path, sample_per_file=None)
    assert len(rows) == len(times) == 3
    assert list(times) == sorted(times)                       # ascending
    assert [t.second for t in times] == [1, 3, 5]             # real seconds


def test_the_clock_does_not_leak_into_features(tmp_path):
    """A timestamp in `features` would hand the model the label outright: an
    attack day is a different day from a benign one."""
    path = tmp_path / "day.csv"
    _write_day(path, [_row()])
    row = load_cicids2018(path, sample_per_file=None)[0]
    assert COL_TIMESTAMP not in row.features
    assert COL_LABEL not in row.features


def test_with_features_false_leaves_the_feature_dict_empty(tmp_path):
    """The dense path for the graph lane: the 74 CICFlowMeter columns cost
    ~2.5 KB per row as a dict and neither detection lane reads them."""
    path = tmp_path / "day.csv"
    _write_day(path, [_row(), _row()])
    rows = load_cicids2018(path, sample_per_file=None, with_features=False)
    assert len(rows) == 2
    assert all(row.features == {} for row in rows)
    assert rows[0].dest_port == 443          # the real fields are still there


# --- malformed rows ---------------------------------------------------------

def test_embedded_header_rows_are_dropped_not_parsed(tmp_path):
    """The release really does contain repeated header rows mid-file (13 in
    one 8 MB slice of 01-03). Parsed as a flow, one becomes a bogus
    'Label'-labelled row."""
    path = tmp_path / "day.csv"
    header_row = dict.fromkeys(_REAL_HEADER, 0)
    header_row.update({c: c for c in _REAL_HEADER})
    _write_day(path, [_row(), header_row, _row()])

    rows = load_cicids2018(path, sample_per_file=None)
    assert len(rows) == 2
    assert all(r.raw_label != "Label" for r in rows)


def test_unmapped_rows_are_dropped_and_the_rest_kept(tmp_path):
    path = tmp_path / "day.csv"
    _write_day(path, [
        _row(label="Benign"),
        _row(label="SSH-Bruteforce"),
        _row(label="DoS attacks-Hulk"),
    ])
    rows = load_cicids2018(path, sample_per_file=None)
    assert sorted(r.raw_label for r in rows) == ["Benign", "DoS attacks-Hulk"]
    assert summarize(rows) == {"normal": 1, "dos": 1}


# --- the row cap is a stride, not a prefix ---------------------------------

def test_a_bounded_cap_still_spans_the_attack_window(tmp_path):
    """Every day file is written in capture order with its attack as a
    time-bounded window inside hours of ordinary traffic, so a *prefix* would
    sample the quiet start and can return an attack-free file while reporting
    success — the tail of the real 01-03 is entirely Benign. Both caps must
    thin evenly instead."""
    path = tmp_path / "day.csv"
    benign = [_row(ts=f"28/02/2018 10:00:{s:02d}", label="Benign") for s in range(40)]
    attack = [_row(ts=f"28/02/2018 11:00:{s:02d}", label="DoS attacks-Hulk")
              for s in range(40)]
    _write_day(path, benign + attack)

    for kwargs in ({"sample_per_file": 10}, {"sample_per_file": None, "stride": 8}):
        cats = summarize(load_cicids2018(path, **kwargs))
        assert cats.get("dos", 0) > 0, f"cap {kwargs} returned no attack at all"
        assert cats.get("normal", 0) > 0, f"cap {kwargs} returned no benign"


# --- attack intervals -------------------------------------------------------

def test_attack_intervals_merge_nearby_bursts_and_widen_by_the_guard(tmp_path):
    """The guard exists because a benign flow sharing a 2-second window with
    an attack flow has the attack's features: every row here shares one
    reconstructed source, so the window aggregate is global to the capture."""
    path = tmp_path / "day.csv"
    _write_day(path, [
        _row(ts="28/02/2018 10:00:00", label="Benign"),          # well clear
        _row(ts="28/02/2018 10:00:10", label="Infilteration"),
        # A benign flow one second into the attack: not an attack row itself,
        # but it shares the attack's 2-second window, so its features are the
        # attack's and it must be covered by the interval.
        _row(ts="28/02/2018 10:00:11", label="Benign"),
        # 2s after the first burst, so the two merge into one span.
        _row(ts="28/02/2018 10:00:12", label="Infilteration"),
        _row(ts="28/02/2018 12:00:00", label="Infilteration"),   # far away
    ])
    rows, times = load_cicids2018_timed(path, sample_per_file=None)
    intervals = attack_intervals(rows, times, guard_seconds=3.0)

    assert len(intervals) == 2                      # the pair merged
    # The merged span runs from the first attack minus the guard to the last
    # attack plus it — 10:00:10-3s through 10:00:12+3s, so 8s wide, not the 5s
    # a single unmerged burst would give.
    start, end = intervals[0]
    assert start == pd.Timestamp("2018-02-28 10:00:07")
    assert end == pd.Timestamp("2018-02-28 10:00:15")
    assert intervals[1][0] == pd.Timestamp("2018-02-28 11:59:57")

    assert in_any_interval(pd.Timestamp("2018-02-28 10:00:11"), intervals)   # benign, covered
    assert not in_any_interval(pd.Timestamp("2018-02-28 10:00:00"), intervals)
    assert not in_any_interval(pd.Timestamp("2018-02-28 09:00:00"), intervals)


def test_attack_intervals_are_empty_for_an_all_benign_day(tmp_path):
    path = tmp_path / "day.csv"
    _write_day(path, [_row(ts="28/02/2018 10:00:00")])
    rows, times = load_cicids2018_timed(path, sample_per_file=None)
    assert attack_intervals(rows, times) == []


# --- label quality ----------------------------------------------------------

def test_label_quality_flags_a_vector_shared_by_two_classes(tmp_path):
    """This is the measurement that decides whether a weak score is the
    model's fault or the label's. Two rows with byte-identical features but
    different labels are irreducible — no function of the features can get
    both right."""
    path = tmp_path / "day.csv"
    _write_day(path, [
        _row(label="Benign"),
        _row(label="Infilteration"),        # same features, different class
        _row(label="Benign", **{"Pkt Len Mean": 42.0}),   # its own vector
    ])
    report = label_quality_report(load_cicids2018(path, sample_per_file=None))

    assert report["rows"] == 3
    assert report["ambiguous_vectors"] == 1
    assert report["rows_in_ambiguous_vectors"] == 2
    assert report["ambiguous_row_fraction"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["deterministic_ceiling"] < 1.0


def test_label_quality_on_distinct_vectors_has_no_ambiguity(tmp_path):
    path = tmp_path / "day.csv"
    _write_day(path, [
        _row(label="Benign", **{"Pkt Len Mean": 1.0}),
        _row(label="Infilteration", **{"Pkt Len Mean": 2.0}),
    ])
    report = label_quality_report(load_cicids2018(path, sample_per_file=None))
    assert report["ambiguous_vectors"] == 0
    assert report["deterministic_ceiling"] == 1.0


def test_label_quality_on_no_rows_reports_a_zero_row_count():
    assert label_quality_report([]) == {"rows": 0}


# --- the directory form -----------------------------------------------------

def test_a_directory_of_day_files_loads_together(tmp_path):
    """The runner passes one day path at a time, but the loader is documented
    to accept a directory, and the summariser is what the survey reads."""
    _write_day(tmp_path / "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
               [_row(label="Benign")])
    _write_day(tmp_path / "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
               [_row(label="Infilteration")])
    rows = load_cicids2018(tmp_path, limit=2, sample_per_file=1)
    assert summarize(rows) == {"normal": 1, "probe": 1}
