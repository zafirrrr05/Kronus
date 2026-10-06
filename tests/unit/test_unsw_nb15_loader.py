"""Unit tests for the UNSW-NB15 loader (twin/unsw_nb15.py).

These do NOT require the real ~46MB download: they build small, UNSW-NB15-shaped
CSV frames in a tmp dir (real pre-split column names, a UTF-8 BOM on the header
exactly as the published files carry, and all ten `attack_cat` families) and
assert the loader maps, cleans, reconstructs hosts, and converts them
correctly. A separate, skip-if-absent test exercises the real data when present.

The host-reconstruction tests are the interesting ones: unlike CIC-IDS2017
(real IPs, see test_cicids2017_loader.py), the UNSW-NB15 pre-split release has
no addresses, so the loader synthesizes them from the row's own connection-rate
counters. Those tests pin down the *shape* the Detective depends on -- a scan
fans out from few sources onto many destinations, a flood collapses onto few
victims -- and that the synthesis is deterministic rather than random.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path

import pandas as pd
import pytest

from libs.constants import DataOrigin, Label, Protocol
from libs.schemas import TelemetryEvent
from services.telemetry_exporter.converters import flow_row_to_event
from twin.nsl_kdd import NSLKDDRow
from twin.unsw_nb15 import (
    UNSW_ATTACK_CATEGORY,
    load_unsw_nb15,
    load_unsw_nb15_dataframe,
)

# Real UNSW-NB15 pre-split columns (a representative subset; the loader treats
# everything outside {id, attack_cat, label} as a feature).
_COLUMNS = [
    "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes",
    "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sinpkt", "dinpkt",
    "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports", "attack_cat", "label",
]

# The ten documented `attack_cat` families and the KRONUS category each maps to
# (None = no clean KRONUS label, the same generic-non-benign treatment NSL-KDD's
# r2l/u2r get).
_EXPECTED_CATEGORY = {
    "Normal": "normal",
    "DoS": "dos",
    "Reconnaissance": "probe",
    "Generic": "r2l",
    "Exploits": "r2l",
    "Fuzzers": "r2l",
    "Analysis": "r2l",
    "Backdoor": "r2l",
    "Shellcode": "r2l",
    "Worms": "r2l",
}


def _make_row(idx, attack_cat, label, proto="tcp", service="-", state="FIN",
              dur=0.5, sbytes=100, dbytes=200, ct_srv_src=1):
    return {
        "id": idx, "dur": dur, "proto": proto, "service": service, "state": state,
        "spkts": 2, "dpkts": 2, "sbytes": sbytes, "dbytes": dbytes,
        "rate": 10.0, "sttl": 31, "dttl": 29, "sload": 1.0, "dload": 2.0,
        "sinpkt": 0.1, "dinpkt": 0.1, "smean": 50.0, "dmean": 50.0,
        "trans_depth": 0, "response_body_len": 0, "ct_srv_src": ct_srv_src,
        "ct_state_ttl": 1, "ct_dst_ltm": 1, "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 1, "is_ftp_login": 0,
        "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 1,
        "ct_srv_dst": 1, "is_sm_ips_ports": 0,
        "attack_cat": attack_cat, "label": label,
    }


@pytest.fixture
def unsw_csv(tmp_path: Path) -> Path:
    rows = [
        # Two benign flows on well-known services (dns/udp, http/tcp).
        _make_row(1, "Normal", 0, service="dns", proto="udp"),
        _make_row(2, "Normal", 0, service="http"),
        # A DoS flood: four rows that should collapse onto ONE victim.
        _make_row(3, "DoS", 1, state="INT", sbytes=5000),
        _make_row(4, "DoS", 1, state="INT", sbytes=5001),
        _make_row(5, "DoS", 1, state="INT", sbytes=5002),
        _make_row(6, "DoS", 1, state="INT", sbytes=5003),
        # A Reconnaissance scan: six rows that should fan out to many victims.
        *[_make_row(10 + i, "Reconnaissance", 1) for i in range(6)],
        # One row per family with no clean KRONUS label.
        _make_row(20, "Generic", 1, service="http"),
        _make_row(21, "Exploits", 1, service="http"),
        _make_row(22, "Fuzzers", 1),
        _make_row(23, "Analysis", 1, service="http"),
        _make_row(24, "Backdoor", 1),
        _make_row(25, "Shellcode", 1),
        _make_row(26, "Worms", 1),
    ]
    df = pd.DataFrame(rows, columns=_COLUMNS)
    path = tmp_path / "UNSW_NB15_training-set.csv"
    # utf-8-sig writes the BOM the published files begin with.
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_loads_and_maps_categories(unsw_csv):
    rows = load_unsw_nb15(unsw_csv)
    # 2 normal + 4 dos + 6 probe + 7 r2l = 19
    assert len(rows) == 19
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r.category, []).append(r)
    assert len(by_cat["normal"]) == 2
    assert len(by_cat["dos"]) == 4
    assert len(by_cat["probe"]) == 6
    assert len(by_cat["r2l"]) == 7


def test_every_documented_attack_family_is_mapped():
    # All ten values must be listed explicitly -- a gap should be a visible
    # omission, never a silent default.
    assert set(UNSW_ATTACK_CATEGORY) == {
        "normal", "dos", "reconnaissance", "generic", "exploits",
        "fuzzers", "analysis", "backdoor", "shellcode", "worms",
    }
    for family, expected in _EXPECTED_CATEGORY.items():
        assert UNSW_ATTACK_CATEGORY[family.casefold()] == expected


def test_category_to_kronus_label_mapping(unsw_csv):
    for r in load_unsw_nb15(unsw_csv):
        if r.category == "normal":
            assert r.kronus_label == Label.BENIGN
        elif r.category == "dos":
            assert r.kronus_label == Label.FLOOD
        elif r.category == "probe":
            assert r.kronus_label == Label.PORT_SCAN
        else:  # r2l -- deliberately no KRONUS label
            assert r.kronus_label is None


def test_attack_cat_matching_is_case_and_whitespace_insensitive(tmp_path):
    # The published files capitalize ("DoS", "Reconnaissance"); be tolerant of
    # padding and of mirrors that lower-case.
    df = pd.DataFrame(
        [_make_row(1, "  DoS  ", 1), _make_row(2, "reconnaissance", 1)],
        columns=_COLUMNS,
    )
    path = tmp_path / "x.csv"
    df.to_csv(path, index=False)
    rows = load_unsw_nb15(path)
    assert {r.category for r in rows} == {"dos", "probe"}


def test_bom_header_is_stripped(unsw_csv):
    # A surviving '﻿id' column would be treated as a numeric feature and
    # silently break the featurizer, so assert it never reaches the output.
    rows = load_unsw_nb15(unsw_csv)
    assert rows  # BOM handling must not drop every row
    assert not any(k.startswith("﻿") for k in rows[0].features)
    df = load_unsw_nb15_dataframe(unsw_csv)
    assert "id" in df.columns
    assert "﻿id" not in df.columns
    assert "attack_cat" in df.columns


def test_duration_and_bytes_are_non_negative_and_finite(unsw_csv):
    for r in load_unsw_nb15(unsw_csv):
        assert r.total_bytes >= 0
        assert r.duration_ms >= 0
        for v in r.features.values():
            assert v == v  # not NaN
            assert v not in (float("inf"), float("-inf"))


def test_source_port_absent_and_dest_port_from_service(unsw_csv):
    rows = load_unsw_nb15(unsw_csv)
    # The pre-split release has no port columns, so the source port is
    # genuinely None -- not a stand-in number.
    assert all(r.source_port is None for r in rows)
    ports = {r.dest_port for r in rows}
    assert 53 in ports   # dns
    assert 80 in ports   # http
    assert None in ports  # service '-' -- no well-known port, never invented


def test_synthetic_hosts_are_valid_and_deterministic(unsw_csv):
    rows = load_unsw_nb15(unsw_csv)
    for r in rows:
        ipaddress.IPv4Address(r.source_ip)  # raises if malformed
        ipaddress.IPv4Address(r.dest_ip)
        assert r.source_ip.startswith("10.10.")
        assert r.dest_ip.startswith("10.20.")
    # No RNG: the same input must produce identical hosts.
    again = load_unsw_nb15(unsw_csv)
    assert [(r.source_ip, r.dest_ip) for r in rows] == [
        (r.source_ip, r.dest_ip) for r in again
    ]


def test_probe_fans_out_while_dos_concentrates(unsw_csv):
    # The graph-shape asymmetry the Detective learns, here on synthesized hosts:
    # a scan spreads from few sources onto many destinations, a flood collapses
    # onto few victims.
    rows = load_unsw_nb15(unsw_csv)
    probe = [r for r in rows if r.category == "probe"]
    dos = [r for r in rows if r.category == "dos"]

    assert len({r.source_ip for r in probe}) < len({r.dest_ip for r in probe})
    assert len({r.source_ip for r in dos}) < len(dos)
    assert len({r.dest_ip for r in dos}) < len(dos)


def test_unknown_attack_cat_is_dropped_not_coerced(tmp_path):
    df = pd.DataFrame([_make_row(1, "SomeFutureAttack", 1)], columns=_COLUMNS)
    path = tmp_path / "x.csv"
    df.to_csv(path, index=False)
    assert load_unsw_nb15(path) == []


def test_unknown_protocol_falls_back_to_other(tmp_path):
    # UNSW-NB15's `proto` holds 133 IANA names; only tcp/udp/icmp are expressible.
    df = pd.DataFrame([_make_row(1, "Normal", 0, proto="unas")], columns=_COLUMNS)
    path = tmp_path / "x.csv"
    df.to_csv(path, index=False)
    assert load_unsw_nb15(path)[0].protocol == Protocol.OTHER


def test_protocol_names_map_to_enum(unsw_csv):
    rows = load_unsw_nb15(unsw_csv)
    assert any(r.protocol == Protocol.TCP for r in rows)
    assert any(r.protocol == Protocol.UDP for r in rows)


def test_origin_is_real_and_shape_is_the_shared_row(unsw_csv):
    rows = load_unsw_nb15(unsw_csv)
    # DataOrigin.REAL: the flow *features* are genuinely from the dataset; only
    # the host addresses are reconstructed (disclosed in the experiment metrics).
    assert all(r.origin == DataOrigin.REAL for r in rows)
    assert all(isinstance(r, NSLKDDRow) for r in rows)
    assert all(r.difficulty == 0 for r in rows)  # no NSL-KDD difficulty level


def test_rows_convert_to_valid_telemetry_events(unsw_csv):
    # The compatibility contract: a UNSW row must flow through the SAME
    # converter every other loader's rows use and produce a schema-valid
    # TelemetryEvent, so no downstream service needs a UNSW-specific path.
    for r in load_unsw_nb15(unsw_csv):
        event = flow_row_to_event(r)
        assert isinstance(event, TelemetryEvent)
        assert event.bytes >= 0
        assert event.duration_ms >= 0
        assert event.protocol in (Protocol.TCP, Protocol.UDP, Protocol.ICMP, Protocol.OTHER)


def test_directory_of_csvs_is_concatenated(tmp_path, unsw_csv):
    # unsw_csv is requested for its side effect: it writes the 19-row training
    # CSV into tmp_path, which this test then adds a testing CSV to.
    assert unsw_csv.parent == tmp_path
    extra = pd.DataFrame([_make_row(99, "Normal", 0)], columns=_COLUMNS)
    extra.to_csv(tmp_path / "UNSW_NB15_testing-set.csv", index=False, encoding="utf-8-sig")
    rows = load_unsw_nb15(tmp_path)
    assert len(rows) == 20  # 19 + 1


def test_dataframe_loader_adds_category_column(unsw_csv):
    df = load_unsw_nb15_dataframe(unsw_csv)
    assert "category" in df.columns
    assert set(df["category"].dropna().unique()) <= set(UNSW_ATTACK_CATEGORY.values())


def test_missing_path_raises_with_pointer(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_unsw_nb15"):
        load_unsw_nb15(tmp_path / "does_not_exist")


# --- real-data test: skips gracefully if the download isn't present ---------

REAL_DATA_DIR = "data/external/unsw_nb15"


def test_real_unsw_nb15_if_present():
    data_dir = Path(REAL_DATA_DIR)
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        pytest.skip(f"{REAL_DATA_DIR} not populated — run scripts/download_unsw_nb15.py")
    rows = load_unsw_nb15(data_dir, limit=20_000)
    assert len(rows) == 20_000
    # Every row must land on a documented category; the official corpus carries
    # normal, DoS and Reconnaissance traffic (among the r2l families).
    assert {r.category for r in rows} <= set(UNSW_ATTACK_CATEGORY.values())
