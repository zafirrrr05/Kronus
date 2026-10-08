"""Unit tests for the CIC-DDoS2019 loader (twin/cicddos2019.py).

These do NOT require the real download: they build small archives in a tmp dir
using the *real* 88-column header and the release's own label spellings, then
assert the loader resolves families, preserves the real socket identity, keeps
the real clock, and refuses a file that is a different dataset.

Five things get deliberate extra attention, because each is a way this loader
could be silently wrong:

1. THE SCHEMA GUARD. This release mixes CIC-IDS2017's column names with its own
   label vocabulary — which is exactly the shape a retired loader in this tree
   was written for while being named for a different dataset. Against real
   files it returned an all-zero dataset and reported success.
   `test_rejects_a_cicids2017_schema` is the regression test for that class of
   failure.
2. THE HOSTS ARE REAL AND MUST STAY REAL. Unlike the 2018 loader next door,
   nothing here is reconstructed from the port. The tests pin that a row's own
   Source IP / Destination IP / Source Port survive, and that two rows sharing
   a destination port keep distinct sources — the property that makes a
   per-host 2-second window meaningful.
3. THE LEAK COLUMNS STAY OUT OF `features`. `Flow ID` contains the IPs and
   ports concatenated, the timestamp identifies the day, and `Inbound` tracks
   the label outright (it marks a flow arriving at the victim, and a volumetric
   flood is inbound by construction). Any of them in the feature dict would
   hand over the label.
4. THE STRIDE, NOT A PREFIX. Both row caps must thin a file evenly so a
   bounded sample still spans the attack window.
5. READ STRAIGHT FROM THE ARCHIVE. The release ships as zips of per-attack
   CSVs; the loader streams members in place so peak disk use stays at the
   archive size. The tests exercise the zip path, not just the directory path.
"""

from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from libs.constants import DataOrigin, Label, Protocol
from libs.schemas import TelemetryEvent
from services.telemetry_exporter.converters import flow_row_to_event
from twin.cicddos2019 import (
    COL_INBOUND,
    COL_LABEL,
    COL_SIMILAR_HTTP,
    COL_TIMESTAMP,
    host_overlap_report,
    label_quality_report,
    load_cicddos2019,
    load_cicddos2019_report,
    load_cicddos2019_timed,
    resolve_class,
    summarize,
)

# The real 88-column header of the released CSVs, verbatim — leading spaces
# included, because the release is internally inconsistent about them and the
# loader has to tolerate that. Using the actual schema rather than a subset is
# the point: it is what distinguishes this dataset from its neighbours.
_REAL_HEADER = [
    "Unnamed: 0", "Flow ID", " Source IP", " Source Port", " Destination IP",
    " Destination Port", " Protocol", " Timestamp", " Flow Duration",
    " Total Fwd Packets", " Total Backward Packets",
    "Total Length of Fwd Packets", " Total Length of Bwd Packets",
    " Fwd Packet Length Max", " Fwd Packet Length Min", " Fwd Packet Length Mean",
    " Fwd Packet Length Std", "Bwd Packet Length Max", " Bwd Packet Length Min",
    " Bwd Packet Length Mean", " Bwd Packet Length Std", "Flow Bytes/s",
    " Flow Packets/s", " Flow IAT Mean", " Flow IAT Std", " Flow IAT Max",
    " Flow IAT Min", "Fwd IAT Total", " Fwd IAT Mean", " Fwd IAT Std",
    " Fwd IAT Max", " Fwd IAT Min", "Bwd IAT Total", " Bwd IAT Mean",
    " Bwd IAT Std", " Bwd IAT Max", " Bwd IAT Min", "Fwd PSH Flags",
    " Bwd PSH Flags", " Fwd URG Flags", " Bwd URG Flags", " Fwd Header Length",
    " Bwd Header Length", "Fwd Packets/s", " Bwd Packets/s", " Min Packet Length",
    " Max Packet Length", " Packet Length Mean", " Packet Length Std",
    " Packet Length Variance", "FIN Flag Count", " SYN Flag Count",
    " RST Flag Count", " PSH Flag Count", " ACK Flag Count", " URG Flag Count",
    " CWE Flag Count", " ECE Flag Count", " Down/Up Ratio",
    " Average Packet Size", " Avg Fwd Segment Size", " Avg Bwd Segment Size",
    " Fwd Header Length.1", "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk",
    " Fwd Avg Bulk Rate", " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", " Subflow Fwd Bytes",
    " Subflow Bwd Packets", " Subflow Bwd Bytes", "Init_Win_bytes_forward",
    " Init_Win_bytes_backward", " act_data_pkt_fwd", " min_seg_size_forward",
    "Active Mean", " Active Std", " Active Max", " Active Min", "Idle Mean",
    " Idle Std", " Idle Max", " Idle Min", "SimillarHTTP", " Inbound", " Label",
]

# CIC-IDS2017's naming, which shares "Flow Duration" and "Timestamp" with this
# schema — so a plausible-looking CSV can still be the wrong dataset.
_CICIDS2017_HEADER = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Source IP", "Source Port", "Timestamp",
    "Label",
]

_REAL_ROW_COUNT = len(_REAL_HEADER)


def _row(
    src: str = "172.16.0.5",
    dst: str = "192.168.50.1",
    sport: int = 64670,
    dport: int = 64670,
    proto: int = 6,
    ts: str = "2018-12-01 13:34:27.403713",
    dur_us: int = 250_000,
    fwd_bytes: int = 500,
    bwd_bytes: int = 700,
    label: str = "BENIGN",
    **feature_overrides: float,
) -> dict:
    """One CSV row: the columns the loader reads, plus real features.

    Every other column defaults to 0 so the only thing a test varies is the
    thing it is about.
    """
    row = dict.fromkeys(_REAL_HEADER, 0)
    row["Unnamed: 0"] = 1
    row["Flow ID"] = f"{src}-{dst}-{sport}-{dport}-{proto}"
    row[" Source IP"] = src
    row[" Source Port"] = sport
    row[" Destination IP"] = dst
    row[" Destination Port"] = dport
    row[" Protocol"] = proto
    row[" Timestamp"] = ts
    row[" Flow Duration"] = dur_us
    row["Total Length of Fwd Packets"] = fwd_bytes
    row[" Total Length of Bwd Packets"] = bwd_bytes
    row[" Label"] = label
    row.update(feature_overrides)
    return row


def _write_zip(path, member_rows: dict[str, list[dict]], header=None) -> None:
    """Write a CIC-DDoS2019-shaped archive: one CSV member per attack family."""
    with zipfile.ZipFile(path, "w") as archive:
        for member, rows in member_rows.items():
            frame = pd.DataFrame(rows, columns=header or _REAL_HEADER)
            archive.writestr(member, frame.to_csv(index=False))


@pytest.fixture
def archive(tmp_path):
    def _build(member_rows: dict[str, list[dict]], name: str = "CSV-01-12.zip"):
        path = tmp_path / name
        _write_zip(path, member_rows)
        return path

    return _build


# --- label resolution -------------------------------------------------------

@pytest.mark.parametrize(
    "raw_label",
    [
        "DrDoS_LDAP", "DrDoS LDAP", "LDAP",
        "DrDoS_MSSQL", "MSSQL",
        "DrDoS_NetBIOS", "NetBIOS",
        "DrDoS_Portmap", "Portmap",
        "DrDoS_SNMP", "SNMP", "DrDoS_SSDP", "SSDP",
        "DrDoS_UDP", "UDP", "UDPLag", "Syn", "TFTP", "WebDDoS",
        "DrDoS_DNS", "DrDoS_NTP", "drdos_ldap",
    ],
)
def test_flood_families_resolve_across_the_release_spellings(raw_label):
    """The release writes the same family three ways — 'DrDoS_LDAP',
    'DrDoS LDAP' and bare 'LDAP' — so resolution must normalize separators
    rather than match exact strings and silently drop real attacks."""
    assert resolve_class(raw_label) == ("dos", Label.FLOOD)


@pytest.mark.parametrize("raw_label", ["BENIGN", "Benign", "benign", "Normal"])
def test_benign_maps_to_normal(raw_label):
    assert resolve_class(raw_label) == ("normal", Label.BENIGN)


@pytest.mark.parametrize(
    "raw_label",
    ["Label", "", None, "Infiltration", "SSH-Bruteforce", "DoS attacks-Hulk",
     "some_unknown_family"],
)
def test_labels_with_no_kronus_counterpart_resolve_to_none(raw_label):
    """Dropped and counted, never force-fitted into flood. An unknown family
    must not be assumed volumetric — that is how a wrong mapping hides."""
    assert resolve_class(raw_label) is None


def test_a_drdos_prefixed_family_is_not_double_normalized():
    """'DrDoS_' stripped once must leave a family the set recognizes — the
    prefix list is ordered so the underscore form wins over the bare one."""
    assert resolve_class("DrDoS_UDPLag") == ("dos", Label.FLOOD)


# --- the schema guard -------------------------------------------------------

def test_rejects_a_cicids2017_schema(tmp_path):
    """The regression test for the retired mislabeled loader.

    CIC-IDS2017 shares 'Flow Duration', 'Timestamp', 'Total Length of Fwd
    Packets' and 'Source IP' with this schema — so the check is written against
    the columns that actually differ (' Destination Port' vs 'Destination
    Port', ' Flow ID'). Reading a 2017 file here would resolve every row to
    None and report an empty dataset as a success; it must raise instead.
    """
    path = tmp_path / "CSV-01-12.zip"
    _write_zip(
        path,
        {"01-12/LDAP.csv": [{c: 0 for c in _CICIDS2017_HEADER}]},
        header=_CICIDS2017_HEADER,
    )
    with pytest.raises(ValueError, match="Dst|missing|not a CIC-DDoS2019"):
        load_cicddos2019(path, sample_per_file=None)


# --- real socket identity survives -----------------------------------------

def test_real_socket_identity_is_preserved_not_reconstructed(archive):
    path = archive({"01-12/LDAP.csv": [_row(src="172.16.0.5", dst="192.168.50.1",
                                            sport=64670, dport=389, proto=17,
                                            dur_us=250_000, fwd_bytes=500,
                                            bwd_bytes=700)]})
    rows = load_cicddos2019(path, sample_per_file=None)

    assert len(rows) == 1
    row = rows[0]
    assert row.source_ip == "172.16.0.5"            # real, not synthesized
    assert row.dest_ip == "192.168.50.1"            # real, not a port bijection
    assert row.source_port == 64670                 # present in this release
    assert row.dest_port == 389                     # real
    assert row.total_bytes == 1200                  # 500 + 700, summed
    assert row.duration_ms == 250                   # microseconds -> ms
    assert row.protocol is Protocol.UDP             # IANA 17
    assert row.origin is DataOrigin.REAL


def test_two_flows_sharing_a_dest_port_keep_distinct_source_hosts(archive):
    """The property the whole per-host window depends on: with real IPs, the
    Bouncer's 2-second aggregate is per source host. A loader that derived
    hosts from the port would collapse these two into one."""
    path = archive({"01-12/LDAP.csv": [
        _row(src="172.16.0.5", dport=389),
        _row(src="172.16.0.6", dport=389),
    ]})
    rows = load_cicddos2019(path, sample_per_file=None)
    assert {r.source_ip for r in rows} == {"172.16.0.5", "172.16.0.6"}


@pytest.mark.parametrize(
    ("number", "expected"),
    [(6, Protocol.TCP), (17, Protocol.UDP), (1, Protocol.ICMP), (99, Protocol.OTHER)],
)
def test_protocol_numbers_map_to_the_enum(archive, number, expected):
    path = archive({"01-12/LDAP.csv": [_row(proto=number)]})
    assert load_cicddos2019(path, sample_per_file=None)[0].protocol is expected


def test_a_row_becomes_a_valid_telemetry_event(archive):
    """The loader's output has to survive the schema boundary the pipeline
    actually uses, not just look right as a dataclass."""
    path = archive({"01-12/LDAP.csv": [_row(dport=1433, proto=6)]})
    event = flow_row_to_event(load_cicddos2019(path, sample_per_file=None)[0])
    assert isinstance(event, TelemetryEvent)
    assert event.dest_port == 1433
    assert event.bytes == 1200
    assert event.source_port == 64670


# --- the clock --------------------------------------------------------------

def test_the_real_capture_clock_is_returned_in_order(archive):
    """This dataset has a real clock with microsecond resolution and the runner
    replays at true inter-arrival times, so the times must come back parsed
    rather than inferred."""
    path = archive({"01-12/LDAP.csv": [
        _row(ts="2018-12-01 13:34:29.500000"),
        _row(ts="2018-12-01 13:34:27.403713"),
        _row(ts="2018-12-01 13:34:28.100000"),
    ]})
    rows, times = load_cicddos2019_timed(path, sample_per_file=None)
    assert len(rows) == len(times) == 3
    assert list(times) == sorted(times)                 # ascending
    assert times[0].microsecond == 403713               # real, not fabricated


def test_the_clock_does_not_leak_into_features(archive):
    """A timestamp in `features` would hand the model the label outright: an
    attack file is a different day from a benign one."""
    path = archive({"01-12/LDAP.csv": [_row()]})
    row = load_cicddos2019(path, sample_per_file=None)[0]
    assert COL_TIMESTAMP not in row.features
    assert COL_LABEL not in row.features


def test_identity_and_artifact_columns_stay_out_of_features(archive):
    """`Flow ID` is the source and destination IPs and ports concatenated, so
    it is the label twice over; `Unnamed: 0` is row identity; `Inbound` carries
    the label at 99.3-99.98% accuracy and `SimillarHTTP` is a constant zero.
    None may be a feature, or the Bouncer learns the label without the flow."""
    path = archive({"01-12/LDAP.csv": [_row()]})
    row = load_cicddos2019(path, sample_per_file=None)[0]
    for leak in ("Flow ID", "Unnamed: 0", "Source IP", "Destination IP",
                 "Source Port", "Destination Port", "Protocol",
                 COL_INBOUND, COL_SIMILAR_HTTP):
        assert leak not in row.features
    assert "Total Length of Fwd Packets" in row.features   # a real measurement
    assert len(row.features) == _REAL_ROW_COUNT - 11


def test_with_features_false_leaves_the_feature_dict_empty(archive):
    """The dense path for the graph lane: the ~79 CICFlowMeter columns cost
    real memory as a dict per row."""
    path = archive({"01-12/LDAP.csv": [_row(), _row()]})
    rows = load_cicddos2019(path, sample_per_file=None, with_features=False)
    assert len(rows) == 2
    assert all(row.features == {} for row in rows)
    assert rows[0].dest_port == 64670          # the real fields are still there


# --- malformed rows ---------------------------------------------------------

def test_embedded_header_rows_are_dropped_not_parsed(archive):
    """The release contains repeated header rows mid-file. Parsed as a flow,
    one becomes a bogus 'Label'-labelled row."""
    path = archive({"01-12/LDAP.csv": [
        _row(),
        {c: c for c in _REAL_HEADER},
        _row(),
    ]})
    rows = load_cicddos2019(path, sample_per_file=None)
    assert len(rows) == 2
    assert all(r.raw_label != "Label" for r in rows)


def test_unmapped_rows_are_dropped_and_counted(archive):
    path = archive({"01-12/LDAP.csv": [
        _row(label="BENIGN"),
        _row(label="Infiltration"),      # not a family in this release
        _row(label="DrDoS_LDAP"),
    ]})
    rows, report = load_cicddos2019_report(path, sample_per_file=None)
    assert sorted(r.raw_label for r in rows) == ["BENIGN", "DrDoS_LDAP"]
    assert summarize(rows) == {"normal": 1, "dos": 1}
    assert report["unmapped_labels"] == {"Infiltration": 1}


# --- the row cap is a stride, not a prefix ---------------------------------

def test_a_bounded_cap_still_spans_the_attack_window(archive):
    """Each file is written in capture order with its attack as a time-bounded
    window inside ordinary traffic, so a *prefix* would sample the quiet start
    and can return an attack-free file while reporting success. Both caps must
    thin evenly instead."""
    path = archive({"01-12/LDAP.csv":
                    [_row(label="BENIGN") for _ in range(40)]
                    + [_row(label="DrDoS_LDAP") for _ in range(40)]})
    for kwargs in ({"sample_per_file": 10}, {"sample_per_file": None, "stride": 8}):
        cats = summarize(load_cicddos2019(path, **kwargs))
        assert cats.get("dos", 0) > 0, f"cap {kwargs} returned no attack at all"
        assert cats.get("normal", 0) > 0, f"cap {kwargs} returned no benign"


# --- the directory forms ----------------------------------------------------

def test_a_directory_of_archives_loads_together(tmp_path):
    """The runner points at the download directory, which holds one archive per
    day; both must be read, and their members must not collide."""
    _write_zip(tmp_path / "CSV-01-12.zip", {"01-12/LDAP.csv": [_row(label="DrDoS_LDAP")]})
    _write_zip(tmp_path / "CSV-03-11.zip", {"03-11/Syn.csv": [_row(label="Syn")]})
    rows = load_cicddos2019(tmp_path, sample_per_file=None)
    assert summarize(rows) == {"dos": 2}
    assert {r.raw_label for r in rows} == {"DrDoS_LDAP", "Syn"}


def test_a_path_that_is_not_this_dataset_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_cicddos2019"):
        load_cicddos2019(tmp_path / "nope")


# --- measurements the runner reports ---------------------------------------

def test_host_overlap_measures_whether_attack_and_benign_share_a_host(archive):
    """The 2018 loader needed an attack-interval benign holdout because every
    row shared one reconstructed source. With real hosts that becomes an
    empirical question, and this is the measurement of it."""
    path = archive({"01-12/LDAP.csv": [
        _row(src="172.16.0.5", label="BENIGN"),
        _row(src="10.0.0.9", label="DrDoS_LDAP"),
        _row(src="10.0.0.9", label="BENIGN"),      # attack host also benign
    ]})
    report = host_overlap_report(load_cicddos2019(path, sample_per_file=None))
    assert report["attack_hosts"] == 1
    assert report["benign_hosts"] == 2
    assert report["shared_hosts"] == 1
    assert report["attack_hosts_that_also_send_benign"] == ["10.0.0.9"]


def test_label_quality_flags_a_vector_shared_by_two_classes(archive):
    """This is the measurement that decides whether a weak score is the
    model's fault or the label's. Two rows with byte-identical features but
    different labels are irreducible — no function of the features can get
    both right."""
    path = archive({"01-12/LDAP.csv": [
        _row(label="BENIGN"),
        _row(label="DrDoS_LDAP"),                      # same features, other class
        _row(label="BENIGN", **{" Min Packet Length": 42.0}),   # its own vector
    ]})
    report = label_quality_report(load_cicddos2019(path, sample_per_file=None))

    assert report["rows"] == 3
    assert report["ambiguous_vectors"] == 1
    assert report["rows_in_ambiguous_vectors"] == 2
    assert report["ambiguous_row_fraction"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["deterministic_ceiling"] < 1.0


def test_label_quality_on_distinct_vectors_has_no_ambiguity(archive):
    path = archive({"01-12/LDAP.csv": [
        _row(label="BENIGN", **{" Min Packet Length": 1.0}),
        _row(label="DrDoS_LDAP", **{" Min Packet Length": 2.0}),
    ]})
    report = label_quality_report(load_cicddos2019(path, sample_per_file=None))
    assert report["ambiguous_vectors"] == 0
    assert report["deterministic_ceiling"] == 1.0


def test_label_quality_on_no_rows_reports_a_zero_row_count():
    assert label_quality_report([]) == {"rows": 0}
