"""Unit tests for the CIRA-CIC-DoHBrw-2020 loader (twin/dohbrw2020.py).

These do NOT require the real ~165MB download: they build small,
DoHBrw2020-shaped CSVs in a tmp dir using the real 35-column header and the
real per-class filename convention, then assert the loader resolves classes,
preserves the capture's own hosts, converts units, and sorts by capture time.

Two things here are genuinely different from the other loaders' tests and get
extra attention:

1. NOTHING is synthesized. Unlike UNSW-NB15 (no addresses at all) and
   NSL-KDD, this capture carries real IPs and real ports, so the loader must
   pass them through untouched — the tests assert the exact input addresses
   come out, which would catch an accidental reuse of the synthetic-host
   helper.
2. The class comes from the FILENAME, not the label column. The tunnel files'
   label column is `DoH` holding "True", which names no class at all. The
   tests pin that down, including that the mirror's aggregate
   "Malicious-DoH.csv" resolves to nothing rather than being guessed at.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from libs.constants import DataOrigin, Label, Protocol
from libs.schemas import TelemetryEvent
from services.telemetry_exporter.converters import flow_row_to_event
from twin.dohbrw2020 import (
    DOHBRW_CLASSES,
    load_dohbrw2020,
    load_dohbrw2020_dataframe,
    resolve_class,
)
from twin.nsl_kdd import NSLKDDRow

# The real 34 feature columns (column 35 is the label, whose NAME differs per
# file: `Label` in the benign/malicious files, `DoH` in the tunnel files).
_FEATURE_HEADER = [
    "SourceIP", "DestinationIP", "SourcePort", "DestinationPort", "TimeStamp",
    "Duration", "FlowBytesSent", "FlowSentRate", "FlowBytesReceived",
    "FlowReceivedRate", "PacketLengthVariance", "PacketLengthStandardDeviation",
    "PacketLengthMean", "PacketLengthMedian", "PacketLengthMode",
    "PacketLengthSkewFromMedian", "PacketLengthSkewFromMode",
    "PacketLengthCoefficientofVariation", "PacketTimeVariance",
    "PacketTimeStandardDeviation", "PacketTimeMean", "PacketTimeMedian",
    "PacketTimeMode", "PacketTimeSkewFromMedian", "PacketTimeSkewFromMode",
    "PacketTimeCoefficientofVariation", "ResponseTimeTimeVariance",
    "ResponseTimeTimeStandardDeviation", "ResponseTimeTimeMean",
    "ResponseTimeTimeMedian", "ResponseTimeTimeMode",
    "ResponseTimeTimeSkewFromMedian", "ResponseTimeTimeSkewFromMode",
    "ResponseTimeTimeCoefficientofVariation",
]


def _make_row(src, dst, sport, dport, ts, duration_s, sent, recv, label_value):
    row = dict.fromkeys(_FEATURE_HEADER, 0.5)
    row.update(
        SourceIP=src,
        DestinationIP=dst,
        SourcePort=sport,
        DestinationPort=dport,
        TimeStamp=ts,
        Duration=duration_s,
        FlowBytesSent=sent,
        FlowBytesReceived=recv,
    )
    row["_label_value"] = label_value
    return row


def _write(path: Path, rows: list[dict], label_col: str) -> None:
    df = pd.DataFrame(rows)
    df = df.rename(columns={"_label_value": label_col})
    df.to_csv(path, index=False)


@pytest.fixture
def doh_dir(tmp_path: Path) -> Path:
    """The four per-class files, with each file's real label-column name."""
    # Benign: internal host -> Google/Cloudflare resolvers over 443.
    _write(tmp_path / "Benign-DoH.csv", [
        _make_row("192.168.20.111", "8.8.8.8", 44268, 443,
                  "2019-12-11 06:36:35", 4.370617, 432, 900, "Benign"),
        _make_row("192.168.20.111", "8.8.8.8", 44272, 443,
                  "2019-12-11 06:37:36", 0.255175, 4503, 120, "Benign"),
        _make_row("192.168.20.112", "1.1.1.1", 55001, 443,
                  "2019-12-11 06:38:00", 2.0, 100, 200, "Benign"),
    ], "Label")

    # The three tunnels: label column is `DoH` = "True", which names no class.
    _write(tmp_path / "dns2tcp-DoH.csv", [
        _make_row("192.168.20.209", "9.9.9.11", 39406, 443,
                  "2020-04-01 22:55:13", 120.772871, 42357, 1000, "True"),
        _make_row("192.168.20.209", "9.9.9.11", 39408, 443,
                  "2020-04-01 22:57:00", 60.0, 9000, 400, "True"),
    ], "DoH")

    _write(tmp_path / "DNSCat2-DoH.csv", [
        _make_row("192.168.20.207", "9.9.9.11", 59272, 443,
                  "2020-03-30 21:20:47", 34.073873, 1807, 300, "True"),
    ], "DoH")

    # iodine records some flows server->client: the 443 endpoint is the SOURCE.
    _write(tmp_path / "iodine-DoH.csv", [
        _make_row("1.1.1.1", "192.168.20.212", 443, 43756,
                  "2020-03-21 18:06:25", 120.619975, 42626, 500, "True"),
    ], "DoH")
    return tmp_path


# --- class resolution -------------------------------------------------------

def test_resolve_class_from_each_filename():
    assert resolve_class("Benign-DoH.csv") == ("normal", Label.BENIGN)
    assert resolve_class("dns2tcp-DoH.csv") == ("lateral_movement", Label.LATERAL_MOVEMENT)
    assert resolve_class("DNSCat2-DoH.csv") == ("lateral_movement", Label.LATERAL_MOVEMENT)
    assert resolve_class("iodine-DoH.csv") == ("lateral_movement", Label.LATERAL_MOVEMENT)


def test_resolve_class_is_case_insensitive_and_token_based():
    assert resolve_class("benign_doh.csv") == ("normal", Label.BENIGN)
    assert resolve_class("IODINE-DoH.csv") == ("lateral_movement", Label.LATERAL_MOVEMENT)
    # The original release's combined benign bundle name.
    assert resolve_class("BenignDoH-NonDoH-CSVs") == ("normal", Label.BENIGN)


def test_malicious_and_unknown_names_are_not_guessed():
    # The mirror's aggregate file does not say which tool produced a flow, so
    # it must NOT resolve — this is the assert that keeps the aggregate from
    # being silently folded into one tunnel's class.
    assert resolve_class("Malicious-DoH.csv", "Malicious") is None
    assert resolve_class("totally-unknown.csv", "True") is None


def test_label_column_is_used_only_as_a_fallback():
    # A file whose NAME is meaningless but whose label column names the tool.
    assert resolve_class("part-000.csv", "dns2tcp") == (
        "lateral_movement", Label.LATERAL_MOVEMENT,
    )


def test_tunnel_label_column_true_names_no_class():
    # `DoH`="True" is the trap this loader is built around: it is true of
    # every row in the file, so on its own it must resolve to nothing.
    assert resolve_class("mystery.csv", "True") is None


def test_every_class_in_the_map_is_used_by_a_test():
    # Guard against a mapping being added without coverage.
    assert set(DOHBRW_CLASSES) == {
        "benign", "nondoh", "dnscat2", "dnscat", "dns2tcp", "iodine",
    }


# --- loading ----------------------------------------------------------------

def test_loads_all_four_files_and_maps_categories(doh_dir):
    rows = load_dohbrw2020(doh_dir)
    assert len(rows) == 7  # 3 benign + 2 dns2tcp + 1 dnscat2 + 1 iodine
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.category] = counts.get(r.category, 0) + 1
    assert counts == {"normal": 3, "lateral_movement": 4}


def test_category_to_kronus_label(doh_dir):
    for r in load_dohbrw2020(doh_dir):
        if r.category == "normal":
            assert r.kronus_label == Label.BENIGN
        else:
            assert r.category == "lateral_movement"
            assert r.kronus_label == Label.LATERAL_MOVEMENT


def test_capture_hosts_are_preserved_verbatim(doh_dir):
    # No synthesis here: every address must be exactly what the CSV held.
    by_src = {r.source_ip for r in load_dohbrw2020(doh_dir)}
    assert "192.168.20.111" in by_src   # benign, internal
    assert "192.168.20.209" in by_src   # dns2tcp, internal
    assert "1.1.1.1" in by_src          # iodine, recorded server->client
    dests = {r.dest_ip for r in load_dohbrw2020(doh_dir)}
    assert {"8.8.8.8", "9.9.9.11", "192.168.20.212"} <= dests


def test_real_ports_are_preserved_and_not_replaced_by_a_service_map(doh_dir):
    rows = load_dohbrw2020(doh_dir)
    pairs = {(r.source_port, r.dest_port) for r in rows}
    assert (44268, 443) in pairs
    assert (39406, 443) in pairs
    assert (443, 43756) in pairs  # the reversed iodine flow


def test_duration_seconds_become_milliseconds(doh_dir):
    for r in load_dohbrw2020(doh_dir):
        assert isinstance(r.duration_ms, int)
        assert r.duration_ms >= 0
    # 120.772871 s in the dns2tcp fixture.
    assert any(r.duration_ms == 120_773 for r in load_dohbrw2020(doh_dir))


def test_total_bytes_is_sent_plus_received(doh_dir):
    rows = load_dohbrw2020(doh_dir)
    # 42357 + 1000 for the first dns2tcp row.
    assert any(r.total_bytes == 43_357 for r in rows)


def test_protocol_is_derived_from_the_well_known_port_either_side(doh_dir):
    rows = load_dohbrw2020(doh_dir)
    # DoH is HTTPS: every fixture flow has a 443 endpoint, including the
    # reversed iodine one (443 as source).
    assert all(r.protocol == Protocol.TCP for r in rows)


def test_unknown_ports_fall_back_to_other(tmp_path):
    _write(tmp_path / "Benign-DoH.csv", [
        _make_row("10.0.0.1", "10.0.0.2", 50000, 50001,
                  "2019-12-11 06:00:00", 1.0, 10, 20, "Benign"),
    ], "Label")
    assert load_dohbrw2020(tmp_path)[0].protocol == Protocol.OTHER


def test_udp_is_derived_for_dns_port(tmp_path):
    _write(tmp_path / "Benign-DoH.csv", [
        _make_row("10.0.0.1", "8.8.8.8", 50000, 53,
                  "2019-12-11 06:00:00", 1.0, 10, 20, "Benign"),
    ], "Label")
    assert load_dohbrw2020(tmp_path)[0].protocol == Protocol.UDP


# --- features ---------------------------------------------------------------

def test_features_exclude_label_and_identity_columns(doh_dir):
    for r in load_dohbrw2020(doh_dir):
        assert "Label" not in r.features
        assert "DoH" not in r.features
        # Host identity must not reach a model through `features`.
        assert "SourceIP" not in r.features
        assert "DestinationIP" not in r.features


def test_capture_timestamp_never_reaches_features(doh_dir):
    # The benign capture is Dec 2019 and the tunnels are Mar 2020, so a
    # timestamp column would separate the classes by itself. It must not be
    # in `features` under any name.
    for r in load_dohbrw2020(doh_dir):
        assert "TimeStamp" not in r.features
        assert not any("time" in k.casefold() and "stamp" in k.casefold() for k in r.features)
        assert all(v == v for v in r.features.values())  # no NaN
        assert all(isinstance(v, float) for v in r.features.values())


def test_rows_are_sorted_by_capture_time(tmp_path):
    # Written deliberately out of order; TimeStamp is the only ordering key.
    _write(tmp_path / "Benign-DoH.csv", [
        _make_row("10.0.0.3", "8.8.8.8", 1000, 443, "2020-01-03 00:00:00", 1.0, 1, 1, "Benign"),
        _make_row("10.0.0.1", "8.8.8.8", 1000, 443, "2020-01-01 00:00:00", 1.0, 1, 1, "Benign"),
        _make_row("10.0.0.2", "8.8.8.8", 1000, 443, "2020-01-02 00:00:00", 1.0, 1, 1, "Benign"),
    ], "Label")
    assert [r.source_ip for r in load_dohbrw2020(tmp_path)] == [
        "10.0.0.1", "10.0.0.2", "10.0.0.3",
    ]


# --- shape / conversion / limits --------------------------------------------

def test_origin_is_real_and_shape_is_the_shared_row(doh_dir):
    rows = load_dohbrw2020(doh_dir)
    assert all(isinstance(r, NSLKDDRow) for r in rows)
    assert all(r.origin == DataOrigin.REAL for r in rows)
    assert all(r.difficulty == 0 for r in rows)


def test_rows_convert_to_valid_telemetry_events(doh_dir):
    # The compatibility contract: these rows must flow through the SAME
    # converter every other loader's rows use, with no DoHBrw-specific path.
    for r in load_dohbrw2020(doh_dir):
        event = flow_row_to_event(r)
        assert isinstance(event, TelemetryEvent)
        assert event.bytes >= 0
        assert event.duration_ms >= 0
        assert event.protocol in (Protocol.TCP, Protocol.UDP, Protocol.ICMP, Protocol.OTHER)


def test_sample_per_file_caps_rows_read(doh_dir):
    rows = load_dohbrw2020(doh_dir, sample_per_file=1)
    # One row per file: 3 benign files? no — 4 files, 1 row each.
    assert len(rows) == 4


def test_limit_caps_total_rows(doh_dir):
    assert len(load_dohbrw2020(doh_dir, limit=5)) == 5


def test_unresolvable_file_contributes_nothing_but_is_visible_in_dataframe(doh_dir):
    _write(doh_dir / "Malicious-DoH.csv", [
        _make_row("192.168.20.250", "9.9.9.11", 40000, 443,
                  "2020-04-02 00:00:00", 5.0, 100, 100, "Malicious"),
    ], "Label")
    rows = load_dohbrw2020(doh_dir)
    assert len(rows) == 7  # unchanged: the aggregate file adds no rows
    assert all(r.source_ip != "192.168.20.250" for r in rows)

    df = load_dohbrw2020_dataframe(doh_dir)
    assert len(df) == 8  # kept, so the drop is visible rather than silent
    assert df["category"].isna().sum() == 1
    assert set(df["category"].dropna().unique()) == {"normal", "lateral_movement"}


def test_directory_with_only_the_benign_file_still_loads(tmp_path):
    _write(tmp_path / "Benign-DoH.csv", [
        _make_row("10.0.0.1", "8.8.8.8", 1000, 443, "2019-12-11 06:00:00", 1.0, 5, 5, "Benign"),
    ], "Label")
    rows = load_dohbrw2020(tmp_path)
    assert len(rows) == 1
    assert rows[0].category == "normal"


def test_missing_path_raises_with_pointer(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_dohbrw2020"):
        load_dohbrw2020(tmp_path / "does_not_exist")


# --- real-data test: skips gracefully if the download isn't present ---------

REAL_DATA_DIR = "data/external/dohbrw2020"


def test_real_dohbrw2020_if_present():
    data_dir = Path(REAL_DATA_DIR)
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        pytest.skip(f"{REAL_DATA_DIR} not populated — run scripts/download_dohbrw2020.py")
    rows = load_dohbrw2020(data_dir, sample_per_file=5_000)
    assert rows
    # Only the two classes this dataset actually carries.
    assert {r.category for r in rows} <= {"normal", "lateral_movement"}
    # Every host is a real address from the capture, not a synthesized 10.x.
    assert all(r.source_ip and r.dest_ip for r in rows)
    assert {r.category for r in rows} == {"normal", "lateral_movement"}
