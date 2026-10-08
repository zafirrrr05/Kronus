"""Unit tests for the CIC IoT 2023 loader (twin/ciciot2023.py).

These do NOT require the real download. They build small captures in a tmp dir
with the same byte layout the publisher ships and assert the properties the
loader's honesty rests on.

Six things get deliberate extra attention, because each is a way this loader
could be silently wrong:

1. THE LABEL COMES FROM THE FOLDER AND NOWHERE ELSE. Nothing inside a capture
   says which family it is, so a capture in the wrong directory must be a
   *counted drop*, never a row under a guessed label.
2. THE ADDRESSES ARE THE CAPTURE'S OWN. No host is reconstructed here, unlike
   the NSL-KDD and UNSW-NB15 loaders next door. The tests pin that a packet's
   real source and destination survive into the row, and that both directions
   of a conversation land in one flow.
3. WHAT CANNOT BE PARSED IS COUNTED, NOT GUESSED. A non-initial fragment has
   no ports on the wire; an IPv6 frame has no IPv4 header. Both must be
   dropped and reported rather than keyed on a made-up 5-tuple.
4. THE TIMEOUTS BOUND A FLOW. A sustained flood must be cut into
   active-timeout-long flows rather than becoming one flow for the whole
   capture, and a long silence must start a new flow.
5. THE STRIDE, NOT A PREFIX. Both caps must thin evenly: the first N flows of a
   capture are all from its first moments.
6. THE CLOCK IS REAL. Rows come back with their true first-packet time, sorted,
   so the runner can replay at real inter-arrival spacing.
"""

from __future__ import annotations

import socket
import struct

import pytest

from libs.constants import DataOrigin, Label, Protocol
from services.telemetry_exporter.converters import flow_row_to_event
from twin.ciciot2023 import (
    FLOW_ACTIVE_TIMEOUT_S,
    FLOW_IDLE_TIMEOUT_S,
    PcapFormatError,
    _evenly_spaced,
    load_ciciot2023,
    load_ciciot2023_report,
    load_ciciot2023_timed,
    resolve_family,
)

ETHERTYPE_IPV4 = 0x0800
ETHERTYPE_IPV6 = 0x86DD
PROTO_TCP = 6
PROTO_UDP = 17
PROTO_ICMP = 1


# --- building captures ------------------------------------------------------

def _ipv4(src: str, dst: str, proto: int, payload: bytes,
          frag_offset: int = 0, more_fragments: bool = False) -> bytes:
    total = 20 + len(payload)
    flags_frag = (0x2000 if more_fragments else 0) | (frag_offset & 0x1FFF)
    header = struct.pack(
        "!BBHHHBBH4s4s", 0x45, 0, total, 0, flags_frag, 64, proto, 0,
        socket.inet_aton(src), socket.inet_aton(dst),
    )
    return header + payload


def _tcp(src: str, dst: str, sport: int, dport: int, payload: bytes = b"x") -> bytes:
    return _ipv4(src, dst, PROTO_TCP, struct.pack("!HH", sport, dport) + payload)


def _udp(src: str, dst: str, sport: int, dport: int, payload: bytes = b"y") -> bytes:
    return _ipv4(src, dst, PROTO_UDP, struct.pack("!HH", sport, dport) + payload)


def _icmp(src: str, dst: str, payload: bytes = b"z") -> bytes:
    return _ipv4(src, dst, PROTO_ICMP, struct.pack("!BBH", 8, 0, 0) + payload)


def _ethernet(datagram: bytes, ethertype: int = ETHERTYPE_IPV4) -> bytes:
    return b"\xaa" * 6 + b"\xbb" * 6 + struct.pack("!H", ethertype) + datagram


def _write_pcap(path, records, linktype: int = 1, magic: bytes = b"\xa1\xb2\xc3\xd4"):
    """records: (timestamp_seconds, frame_bytes). Classic little-endian pcap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(magic + struct.pack("<HHIIII", 2, 4, 0, 0, 65535, linktype))
        for ts, frame in records:
            seconds = int(ts)
            micros = int(round((ts - seconds) * 1_000_000))
            handle.write(struct.pack("<IIII", seconds, micros, len(frame), len(frame)))
            handle.write(frame)


def _eth_records(pairs):
    """(ts, datagram) pairs -> (ts, ethernet frame) pairs."""
    return [(ts, _ethernet(dg)) for ts, dg in pairs]


# --- family resolution ------------------------------------------------------

def test_resolve_family_maps_the_three_groups_both_lanes_use():
    assert resolve_family("Benign_Final") == ("normal", Label.BENIGN)
    assert resolve_family("Recon-PortScan") == ("probe", Label.PORT_SCAN)
    assert resolve_family("VulnerabilityScan") == ("probe", Label.PORT_SCAN)
    assert resolve_family("DDoS-UDP_Flood") == ("dos", Label.FLOOD)
    assert resolve_family("DoS-SYN_Flood") == ("dos", Label.FLOOD)
    assert resolve_family("Mirai-udpplain") == ("dos", Label.FLOOD)


def test_resolve_family_refuses_the_out_of_scope_families():
    """Backdoor/injection/spoofing/brute-force are neither volumetric nor a
    scan, so neither lane has a shape for them. They must resolve to None and
    be counted, not squeezed into the nearest label."""
    for name in ("Backdoor_Malware", "BrowserHijacking", "CommandInjection",
                 "DNS_Spoofing", "DictionaryBruteForce", "MITM-ArpSpoofing",
                 "SqlInjection", "Uploading_Attack", "XSS", ""):
        assert resolve_family(name) is None, name


def test_a_capture_in_an_unknown_folder_is_dropped_and_counted(tmp_path):
    _write_pcap(tmp_path / "XSS" / "x.pcap",
                _eth_records([(1.0, _tcp("10.0.0.1", "10.0.0.2", 1111, 80))]))
    _write_pcap(tmp_path / "Benign_Final" / "b.pcap",
                _eth_records([(1.0, _tcp("10.0.0.3", "10.0.0.4", 2222, 80))]))

    rows, report = load_ciciot2023_report(tmp_path)

    assert len(rows) == 1
    assert rows[0].source_ip == "10.0.0.3"
    assert report["dropped_families"] == {"XSS": 1}
    assert report["rows_by_family"] == {"Benign_Final": 1}


# --- the flow contract ------------------------------------------------------

def test_both_directions_of_a_conversation_are_one_flow(tmp_path):
    """The key is the sorted endpoint pair, so a request and its reply are the
    same flow. `source_ip` is whoever sent the first packet."""
    _write_pcap(tmp_path / "Benign_Final" / "b.pcap", _eth_records([
        (1.0, _tcp("10.0.0.1", "10.0.0.2", 5000, 80, payload=b"a" * 10)),
        (1.5, _tcp("10.0.0.2", "10.0.0.1", 80, 5000, payload=b"b" * 20)),
        (2.0, _tcp("10.0.0.1", "10.0.0.2", 5000, 80, payload=b"c" * 5)),
    ]))

    rows = load_ciciot2023(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert (row.source_ip, row.source_port) == ("10.0.0.1", 5000)
    assert (row.dest_ip, row.dest_port) == ("10.0.0.2", 80)
    assert row.protocol is Protocol.TCP
    # 20 (hdr) + 4 (ports) + payload, both directions.
    assert row.features["packets_fwd"] == 2
    assert row.features["packets_bwd"] == 1
    assert row.total_bytes == row.features["bytes_fwd"] + row.features["bytes_bwd"]
    assert row.duration_ms == 1000


def test_distinct_conversations_stay_distinct(tmp_path):
    _write_pcap(tmp_path / "Benign_Final" / "b.pcap", _eth_records([
        (1.0, _tcp("10.0.0.1", "10.0.0.2", 5000, 80)),
        (1.1, _tcp("10.0.0.1", "10.0.0.2", 5000, 443)),
        (1.2, _udp("10.0.0.5", "10.0.0.6", 5000, 5000)),
    ]))

    rows = load_ciciot2023(tmp_path)

    assert len(rows) == 3
    assert sorted(r.dest_port for r in rows) == [80, 443, 5000]
    by_proto = {r.protocol for r in rows}
    assert by_proto == {Protocol.TCP, Protocol.UDP}


def test_icmp_without_ports_is_still_one_flow(tmp_path):
    _write_pcap(tmp_path / "Recon-PingSweep" / "p.pcap", _eth_records([
        (1.0, _icmp("10.0.0.1", "10.0.0.9")),
        (1.2, _icmp("10.0.0.9", "10.0.0.1")),
    ]))

    rows = load_ciciot2023(tmp_path)

    assert len(rows) == 1
    assert rows[0].protocol is Protocol.ICMP
    assert rows[0].source_port is None and rows[0].dest_port is None
    assert rows[0].kronus_label is Label.PORT_SCAN


def test_rows_feed_the_telemetry_event_contract(tmp_path):
    """The whole point of returning NSLKDDRow: the converter downstream is
    unchanged and gets real identity and real ports out of it."""
    _write_pcap(tmp_path / "DDoS-UDP_Flood" / "d.pcap", _eth_records([
        (100.0, _udp("192.168.1.10", "192.168.1.20", 40000, 53)),
    ]))

    rows = load_ciciot2023(tmp_path)
    event = flow_row_to_event(rows[0], ts=100.0)

    assert event.source_ip == "192.168.1.10"
    assert event.dest_ip == "192.168.1.20"
    assert event.source_port == 40000
    assert event.dest_port == 53
    assert event.protocol == Protocol.UDP
    assert rows[0].origin is DataOrigin.REAL
    assert rows[0].kronus_label is Label.FLOOD


# --- what cannot be parsed is counted, not guessed ---------------------------

def test_non_initial_fragments_are_dropped_and_counted(tmp_path):
    """A later fragment carries no ports, so it cannot be keyed. Dropping it
    into a made-up 5-tuple would invent identity."""
    _write_pcap(tmp_path / "Benign_Final" / "b.pcap", _eth_records([
        (1.0, _ipv4("10.0.0.1", "10.0.0.2", PROTO_TCP,
                    struct.pack("!HH", 5000, 80) + b"x")),
        (1.1, _ipv4("10.0.0.1", "10.0.0.2", PROTO_TCP, b"y" * 8,
                    frag_offset=2, more_fragments=False)),
    ]))

    rows, report = load_ciciot2023_report(tmp_path)

    assert len(rows) == 1
    assert report["per_capture"][0]["drops"]["fragments_and_malformed"] == 1


def test_a_first_fragment_with_more_to_come_is_still_parsed(tmp_path):
    """Offset 0 carries the transport header even when MF is set, so it is a
    usable packet — dropping it would lose real flows."""
    _write_pcap(tmp_path / "Benign_Final" / "b.pcap", _eth_records([
        (1.0, _ipv4("10.0.0.1", "10.0.0.2", PROTO_TCP,
                    struct.pack("!HH", 5000, 80) + b"x", more_fragments=True)),
    ]))

    rows = load_ciciot2023(tmp_path)
    assert len(rows) == 1


def test_ipv6_and_non_ip_frames_are_counted(tmp_path):
    ipv6 = b"\x60" * 40
    arp = b"\x00" * 28
    _write_pcap(tmp_path / "Benign_Final" / "b.pcap", [
        (1.0, _ethernet(ipv6, ETHERTYPE_IPV6)),
        (1.1, _ethernet(arp, 0x0806)),
        (1.2, _ethernet(_tcp("10.0.0.1", "10.0.0.2", 5000, 80))),
    ])

    rows, report = load_ciciot2023_report(tmp_path)

    assert len(rows) == 1
    drops = report["per_capture"][0]["drops"]
    assert drops["ipv6_frames"] == 1
    assert drops["non_ip_frames"] == 1


def test_a_non_pcap_file_fails_loudly(tmp_path):
    bad = tmp_path / "Benign_Final" / "b.pcap"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"GET / HTTP/1.1\r\n" + b"\x00" * 64)

    with pytest.raises(PcapFormatError):
        load_ciciot2023(tmp_path)


def test_pcapng_is_named_rather_than_mis_parsed(tmp_path):
    bad = tmp_path / "Benign_Final" / "b.pcapng"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"\x0a\x0d\x0d\x0a" + b"\x00" * 64)

    with pytest.raises(PcapFormatError, match="pcapng"):
        load_ciciot2023(tmp_path)


# --- the timeouts -----------------------------------------------------------

def test_a_long_silence_starts_a_new_flow(tmp_path):
    gap = FLOW_IDLE_TIMEOUT_S + 1
    _write_pcap(tmp_path / "Benign_Final" / "b.pcap", _eth_records([
        (1.0, _tcp("10.0.0.1", "10.0.0.2", 5000, 80)),
        (1.0 + gap, _tcp("10.0.0.1", "10.0.0.2", 5000, 80)),
    ]))

    rows = load_ciciot2023(tmp_path)
    assert len(rows) == 2


def test_a_sustained_flood_is_cut_by_the_active_timeout(tmp_path):
    """The mirror of the test above: traffic that never pauses must still be
    bounded, or a flood capture becomes one flow for its whole duration."""
    step = 10.0
    records = [(i * step, _udp("10.0.0.1", "10.0.0.2", 40000, 53))
               for i in range(int(FLOW_ACTIVE_TIMEOUT_S / step) + 4)]

    _write_pcap(tmp_path / "Mirai-udpplain" / "m.pcap", _eth_records(records))

    rows = load_ciciot2023(tmp_path)

    assert len(rows) > 1
    for row in rows:
        assert row.duration_ms <= FLOW_ACTIVE_TIMEOUT_S * 1000
    assert all(r.kronus_label is Label.FLOOD for r in rows)


def test_a_pause_between_packets_does_not_split_the_flow(tmp_path):
    _write_pcap(tmp_path / "Benign_Final" / "b.pcap", _eth_records([
        (1.0, _tcp("10.0.0.1", "10.0.0.2", 5000, 80)),
        (1.0 + FLOW_IDLE_TIMEOUT_S - 0.5, _tcp("10.0.0.1", "10.0.0.2", 5000, 80)),
    ]))

    rows = load_ciciot2023(tmp_path)
    assert len(rows) == 1


# --- the stride -------------------------------------------------------------

def test_evenly_spaced_thins_across_the_whole_range_not_a_prefix():
    picked = _evenly_spaced(1000, 10)
    assert len(picked) == 10
    assert picked == sorted(set(picked)), "indices must be distinct and ordered"
    assert picked[0] == 0
    assert picked[-1] >= 900, "the sample must reach the end of the capture, not stop at 10"


def test_evenly_spaced_is_a_no_op_when_nothing_is_dropped():
    assert _evenly_spaced(5, None) == [0, 1, 2, 3, 4]
    assert _evenly_spaced(5, 5) == [0, 1, 2, 3, 4]
    assert _evenly_spaced(0, 10) == []


def test_the_per_capture_cap_keeps_a_spread_of_flows(tmp_path):
    records = [(float(i), _tcp("10.0.0.1", "10.0.0.2", 5000 + i, 80))
               for i in range(20)]
    _write_pcap(tmp_path / "Benign_Final" / "b.pcap", _eth_records(records))

    rows, report = load_ciciot2023_report(tmp_path, sample_per_capture=4)

    assert len(rows) == 4
    # Not the first four: the last kept flow must come from well into the file.
    assert rows[-1].features["flow_packets"] == 1
    kept_ports = sorted(r.source_port for r in rows)
    assert kept_ports[-1] >= 5000 + 10
    assert report["per_capture"][0]["flows"] == 20
    assert report["per_capture"][0]["sampled_to"] == 4


def test_the_row_limit_is_applied_after_merging(tmp_path):
    for index in range(2):
        records = [(float(i), _tcp(f"10.0.{index}.1", "10.0.0.9", 5000 + i, 80))
                   for i in range(10)]
        _write_pcap(tmp_path / "Benign_Final" / f"b{index}.pcap", _eth_records(records))

    rows = load_ciciot2023(tmp_path, limit=6)
    assert len(rows) == 6


# --- the clock --------------------------------------------------------------

def test_timed_rows_come_back_sorted_with_their_real_times(tmp_path):
    _write_pcap(tmp_path / "Benign_Final" / "b.pcap", _eth_records([
        (30.5, _tcp("10.0.0.1", "10.0.0.2", 5001, 80)),
        (31.5, _tcp("10.0.0.2", "10.0.0.1", 80, 5001)),   # same flow, a second on
        (10.25, _tcp("10.0.0.3", "10.0.0.4", 5002, 80)),
        (20.0, _tcp("10.0.0.5", "10.0.0.6", 5003, 80)),
    ]))

    rows, times = load_ciciot2023_timed(tmp_path)

    assert times == [10.25, 20.0, 30.5]
    assert [r.source_ip for r in rows] == ["10.0.0.3", "10.0.0.5", "10.0.0.1"]
    # The time is the flow's *first* packet: the 30.5 flow ends a second later,
    # so a row carrying its end time would be off by its own duration.
    assert rows[-1].duration_ms == 1000
    assert times[-1] + rows[-1].duration_ms / 1000 == pytest.approx(31.5)
    assert rows[0].duration_ms == 0


def test_the_clock_is_not_a_feature(tmp_path):
    """An attack capture is a different capture from a benign one, so a
    timestamp in the feature dict would hand over the label."""
    epoch = 1234567.0
    _write_pcap(tmp_path / "Benign_Final" / "b.pcap", _eth_records([
        (epoch, _tcp("10.0.0.1", "10.0.0.2", 5000, 80)),
    ]))

    rows = load_ciciot2023(tmp_path)
    features = rows[0].features

    # No column names the clock, and the capture's own time is nowhere in the
    # values — `duration_ms` is an elapsed span, not a time of day.
    assert not any("time" in key or "timestamp" in key for key in features)
    assert epoch not in features.values()


# --- the report -------------------------------------------------------------

def test_the_report_discloses_the_extraction(tmp_path):
    _write_pcap(tmp_path / "Benign_Final" / "b.pcap",
                _eth_records([(1.0, _tcp("10.0.0.1", "10.0.0.2", 5000, 80))]))

    _, report = load_ciciot2023_report(tmp_path)

    assert report["synthetic_hosts"] is False
    assert "extracted by us" in report["extraction"]
    assert report["flow_definition"]["idle_timeout_s"] == FLOW_IDLE_TIMEOUT_S
    assert report["flow_definition"]["active_timeout_s"] == FLOW_ACTIVE_TIMEOUT_S
    assert report["rows_loaded"] == 1


def test_an_empty_directory_reports_rather_than_raises(tmp_path):
    rows, report = load_ciciot2023_report(tmp_path)
    assert rows == []
    assert report["rows_loaded"] == 0
    assert "no captures" in report["reason"]


def test_a_big_endian_capture_reads_the_same(tmp_path):
    """The release could plausibly ship either byte order; the reader keys on
    the magic word, so both work."""
    path = tmp_path / "Benign_Final" / "b.pcap"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"\xd4\xc3\xb2\xa1" + struct.pack(">HHIIII", 2, 4, 0, 0, 65535, 1))
        frame = _ethernet(_tcp("10.0.0.1", "10.0.0.2", 5000, 80))
        handle.write(struct.pack(">IIII", 1, 500000, len(frame), len(frame)))
        handle.write(frame)

    rows = load_ciciot2023(tmp_path)

    assert len(rows) == 1
    assert rows[0].source_ip == "10.0.0.1"
    assert rows[0].duration_ms == 0
