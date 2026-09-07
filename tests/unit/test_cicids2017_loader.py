"""Unit tests for the CIC-IDS2017 loader (twin/cicids2017.py) and the
external-validation harness's data contracts.

These do NOT require the real ~500MB CIC-IDS2017 download: they build small,
CIC-IDS2017-shaped CSV frames in a tmp dir (real column names, including the
leading-space headers and the Inf/NaN Flow Bytes/s quirk this loader exists to
clean) and assert the loader maps, cleans, and converts them correctly. A
separate, sk-if-absent test exercises the real data when it's present.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path

import pandas as pd
import pytest

from libs.constants import DataOrigin, Label, Protocol
from libs.schemas import TelemetryEvent
from services.telemetry_exporter.converters import flow_row_to_event
from twin.cicids2017 import (
    CIC_ATTACK_CATEGORY,
    load_cicids2017,
    load_cicids2017_dataframe,
)

# Real CIC-IDS2017 column names, with the leading spaces the actual files
# carry on most headers (the quirk _clean_columns strips).
_COLUMNS = [
    " Source IP", " Source Port", " Destination IP", " Destination Port",
    " Protocol", " Flow Duration", "Total Length of Fwd Packets",
    " Total Length of Bwd Packets", "Flow Bytes/s", " Flow Packets/s", " Label",
]


def _make_row(src_ip, dst_ip, sport, dport, proto, duration_us, fwd_b, bwd_b,
              bytes_s, label):
    return {
        " Source IP": src_ip, " Source Port": sport,
        " Destination IP": dst_ip, " Destination Port": dport,
        " Protocol": proto, " Flow Duration": duration_us,
        "Total Length of Fwd Packets": fwd_b,
        " Total Length of Bwd Packets": bwd_b,
        "Flow Bytes/s": bytes_s, " Flow Packets/s": 10.0, " Label": label,
    }


@pytest.fixture
def cic_csv(tmp_path: Path) -> Path:
    rows = []
    # A benign flow.
    rows.append(_make_row("192.168.10.5", "8.8.8.8", 51000, 53, 17, 1200, 100, 200, 250.0, "BENIGN"))
    # A DDoS flood (maps to FLOOD).
    rows.append(_make_row("172.16.0.1", "192.168.10.50", 40000, 80, 6, 5000, 6000, 0, 1_200_000.0, "DDoS"))
    # A DoS Hulk (also FLOOD), with the classic zero-duration Inf byte-rate quirk.
    rows.append(_make_row("172.16.0.1", "192.168.10.50", 40001, 80, 6, 0, 500, 0, "Infinity", "DoS Hulk"))
    # A PortScan from one source to many dests (maps to PORT_SCAN).
    for i in range(6):
        rows.append(_make_row("172.16.0.99", f"192.168.10.{20 + i}", 44000 + i, 22, 6, 30, 40, 0, 900.0, "PortScan"))
    # An FTP-Patator (no clean KRONUS label -> r2l category, kronus_label None).
    rows.append(_make_row("172.16.0.7", "192.168.10.50", 45000, 21, 6, 800, 300, 300, 750.0, "FTP-Patator"))
    # A Web Attack with the en-dash variant CIC actually ships.
    rows.append(_make_row("172.16.0.8", "192.168.10.51", 46000, 80, 6, 900, 400, 400, 900.0, "Web Attack – Brute Force"))
    # A NaN byte-rate row (the other half of the quirk).
    rows.append(_make_row("192.168.10.6", "8.8.4.4", 51001, 53, 17, 0, 0, 0, "NaN", "BENIGN"))

    df = pd.DataFrame(rows, columns=_COLUMNS)
    path = tmp_path / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    df.to_csv(path, index=False)
    return path


def test_loads_and_maps_labels(cic_csv):
    rows = load_cicids2017(cic_csv)
    # 1 benign + 1 ddos + 1 hulk + 6 portscan + 1 patator + 1 webattack + 1 benign = 12
    assert len(rows) == 12
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r.category, []).append(r)
    assert len(by_cat["normal"]) == 2
    assert len(by_cat["dos"]) == 2
    assert len(by_cat["probe"]) == 6
    assert len(by_cat["r2l"]) == 2  # FTP-Patator + Web Attack Brute Force


def test_category_to_kronus_label_mapping(cic_csv):
    rows = load_cicids2017(cic_csv)
    for r in rows:
        if r.category == "normal":
            assert r.kronus_label == Label.BENIGN
        elif r.category == "dos":
            assert r.kronus_label == Label.FLOOD
        elif r.category == "probe":
            assert r.kronus_label == Label.PORT_SCAN
        else:  # r2l/u2r — deliberately no KRONUS label
            assert r.kronus_label is None


def test_inf_and_nan_are_cleaned_not_propagated(cic_csv):
    # The Inf/NaN Flow Bytes/s rows must load with finite feature values and
    # a schema-valid (bytes >= 0, duration >= 0) TelemetryEvent — the whole
    # reason _coerce_numeric/_safe_float exist.
    rows = load_cicids2017(cic_csv)
    for r in rows:
        assert r.total_bytes >= 0
        assert r.duration_ms >= 0
        for v in r.features.values():
            assert v == v  # not NaN
            assert v not in (float("inf"), float("-inf"))


def test_real_ips_are_preserved_not_reconstructed(cic_csv):
    # Unlike NSL-KDD, CIC-IDS2017 has real IPs — the loader must pass them
    # through verbatim, not synthesize new ones.
    rows = load_cicids2017(cic_csv)
    src_ips = {r.source_ip for r in rows}
    assert "172.16.0.99" in src_ips  # the portscan source, verbatim
    for r in rows:
        ipaddress.IPv4Address(r.source_ip)  # raises if malformed
        ipaddress.IPv4Address(r.dest_ip)


def test_rows_convert_to_valid_telemetry_events(cic_csv):
    # The compatibility contract: a CIC row must flow through the SAME
    # converter NSL-KDD rows use and produce a schema-valid TelemetryEvent,
    # so no downstream service needs a CIC-specific path.
    rows = load_cicids2017(cic_csv)
    for r in rows:
        event = flow_row_to_event(r)
        assert isinstance(event, TelemetryEvent)
        assert event.bytes >= 0
        assert event.duration_ms >= 0
        assert event.protocol in (Protocol.TCP, Protocol.UDP, Protocol.ICMP, Protocol.OTHER)


def test_protocol_numbers_map_to_enum(cic_csv):
    rows = load_cicids2017(cic_csv)
    # proto 6 -> TCP, 17 -> UDP in the fixture
    tcp = [r for r in rows if r.protocol == Protocol.TCP]
    udp = [r for r in rows if r.protocol == Protocol.UDP]
    assert tcp and udp


def test_portscan_fans_out_from_fewer_sources_than_dests(cic_csv):
    # The graph-shape property the Detective relies on — same claim the
    # NSL-KDD loader test makes (test_nsl_kdd_loader.py), here on real CIC IPs.
    rows = load_cicids2017(cic_csv)
    probe = [r for r in rows if r.category == "probe"]
    assert len({r.source_ip for r in probe}) < len({r.dest_ip for r in probe})


def test_origin_is_real(cic_csv):
    rows = load_cicids2017(cic_csv)
    assert all(r.origin == DataOrigin.REAL for r in rows)


def test_unknown_labels_are_dropped_not_coerced(tmp_path):
    df = pd.DataFrame(
        [_make_row("1.1.1.1", "2.2.2.2", 1, 2, 6, 10, 10, 10, 5.0, "SomeFutureAttackType")],
        columns=_COLUMNS,
    )
    path = tmp_path / "x.csv"
    df.to_csv(path, index=False)
    rows = load_cicids2017(path)
    assert rows == []  # dropped, never force-fit onto a wrong category


def test_directory_of_csvs_is_concatenated(tmp_path, cic_csv):
    # load_cicids2017 accepts a directory and reads every CSV in it.
    # cic_csv is requested for its side effect: it writes the first
    # (12-row) CSV into tmp_path, which this test adds a second CSV to.
    assert cic_csv.parent == tmp_path
    df = pd.DataFrame(
        [_make_row("9.9.9.9", "8.8.8.8", 1, 2, 6, 10, 10, 10, 5.0, "BENIGN")],
        columns=_COLUMNS,
    )
    (tmp_path / "Monday-WorkingHours.pcap_ISCX.csv").write_text(df.to_csv(index=False))
    rows = load_cicids2017(tmp_path)
    assert len(rows) == 13  # 12 from cic_csv + 1 here


def test_dataframe_loader_adds_category_column(cic_csv):
    df = load_cicids2017_dataframe(cic_csv)
    assert "category" in df.columns
    assert "Label" in df.columns  # cleaned (no leading space)
    assert set(df["category"].dropna().unique()) <= set(CIC_ATTACK_CATEGORY.values())


def test_missing_path_raises_with_pointer(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_cicids2017"):
        load_cicids2017(tmp_path / "does_not_exist")


# --- real-data test: skips gracefully if the download isn't present ---------

REAL_DATA_DIR = "data/external/cicids2017"


def test_real_cicids2017_if_present():
    data_dir = Path(REAL_DATA_DIR)
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        pytest.skip(f"{REAL_DATA_DIR} not populated — run scripts/download_cicids2017.py")
    rows = load_cicids2017(REAL_DATA_DIR)
    assert len(rows) > 0
    cats = {r.category for r in rows}
    # The real dataset contains benign, DoS/DDoS, and portscan traffic.
    assert "normal" in cats
    assert "dos" in cats or "probe" in cats
