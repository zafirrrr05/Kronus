"""CIC IoT 2023 external-validation data source.

CIC IoT 2023 ("A real-time dataset and benchmark for large-scale attacks in
IoT environment", Canadian Institute for Cybersecurity, UNB) is a capture from
a 105-device IoT testbed carrying 33 attacks in seven families plus benign
traffic. Official page:
https://www.unb.ca/cic/datasets/iotdataset-2023.html

    Neto, E., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., Ghorbani, A.A.
    "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in
    IoT environment", Sensors, 2023. doi:10.3390/s23135941

*** WHAT IS THE PUBLISHER'S, AND WHAT IS OURS ***
This is the only loader in the tree that *derives its rows from packets*
instead of reading rows the publisher already produced, so the split between
measured and constructed has to be stated rather than assumed.

The publisher's: the packet captures themselves, and the family each capture
belongs to — `PCAP/DDoS-UDP_Flood/` holds floods, `PCAP/Benign_Final/` holds
benign, and that folder is the only label that exists anywhere in this release.
Also the publisher's: every byte on the wire, the addresses, the ports, the
inter-arrival times.

Ours: the *grouping* of those packets into flows. Each flow here is the set of
packets sharing a bidirectional 5-tuple within an idle timeout — a definition
this repository chose, matching what CICFlowMeter-style exporters do, not
something the release specifies. Its consequences are disclosed in the metrics
(`flow_definition`) and counted (`non_initial_fragments`,
`unsupported_linktype_frames`) rather than hidden.

No host is reconstructed and no address is invented, which is the line
twin/unsw_nb15.py draws for the opposite case: the addresses here are the
captures' own. `synthetic_hosts` is false.

*** WHY THE PUBLISHED CSVs ARE NOT USED ***
Both variants under CSV/ are 39 aggregate flow statistics — Header_Length,
Rate, IAT, Tot sum, Std, Variance, flag counts, protocol indicators — and
neither carries a source IP, a destination IP, a port, or a timestamp. Four of
the Bouncer's six features (event_rate, dest_port_entropy, unique_dest_count,
same_dest_ratio) are port- or clock-derived, so training on those files would
mean inventing the features and reporting the resulting score as a detection
result. Experiment E hit the same wall and answered `verdict:
NO_VALID_DETECTION_METRIC`.

The captures carry what the CSVs do not, which is why the downloader
(scripts/download_ciciot2023.py) fetches those instead. Read that module's
docstring for the fetch; this one reads what it produced.

*** WHICH FAMILIES BOTH LANES CAN USE ***
Of the 34 families in the release, 25 fall in three groups both of KRONUS's
lanes have a contract for:

    Benign_Final                 -> BENIGN   (Bouncer negative, Detective BENIGN)
    DDoS-*/DoS-*/Mirai-*         -> FLOOD    (Bouncer positive)
    Recon-*/VulnerabilityScan    -> PORT_SCAN (Detective positive)

Everything else is out of scope and is *dropped and counted*, never guessed at
— backdoors, browser hijacking, injection, XSS, spoofing and brute force are
none of them volumetric and none of them a scan, so neither lane has a shape to
learn for them. `resolve_family` returns None for those and the report carries
the drop, the same discipline as `resolve_class` in twin/cicddos2019.py.

*** FLOWS, AND THE TWO TIMEOUTS ***
A flow is emitted when a packet arrives for its 5-tuple that is more than
`FLOW_IDLE_TIMEOUT_S` after the flow's last packet, or when the flow has been
open for more than `FLOW_ACTIVE_TIMEOUT_S` — the idle/active pair a NetFlow
exporter uses. Both are choices of ours and both are reported. A scan is
short-lived and lands in one flow; a sustained flood is cut into a sequence of
`FLOW_ACTIVE_TIMEOUT_S`-long flows, which is what an exporter on that link
would have emitted too.

The 5-tuple is bidirectional: it is keyed on the two endpoints sorted, so a
request and its reply are the same flow. `source_ip`/`source_port` are
whichever endpoint sent the flow's first packet, which is what makes the
Bouncer's `source_ip`-keyed 2-second window mean "this host's traffic" rather
than an arbitrary half of it.

*** THE CLOCK IS REAL ***
Each row carries its flow's first-packet time, so the runner replays at true
inter-arrival spacing. As everywhere else the clock is excluded from
`features` — an attack capture is a different capture from a benign one, so a
timestamp would hand over the label.

*** READING PCAP WITHOUT A NEW DEPENDENCY ***
The captures are read by a small reader in this module rather than by scapy:
the parse needed is one Ethernet/SLL header, one IPv4 header and two ports, and
keeping it here means the experiment adds no install to a laptop that is short
of space, and that every drop it makes is visible in this file. Anything it
cannot parse with confidence — a non-IPv4 frame, a non-initial fragment (whose
ports are not on the wire), a link type it does not know — is counted and
skipped, never guessed at.

Output type is the same `NSLKDDRow` shape every other loader here produces, so
the converters, the Bouncer featurizer and the graph builder are unchanged.
"""

from __future__ import annotations

import struct
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from libs.constants import DataOrigin, Label, Protocol
from twin.nsl_kdd import NSLKDDRow

# --- family resolution ------------------------------------------------------

# folder-name prefix -> (category, KRONUS label). Checked against the real
# folder names printed by `download_ciciot2023.py --list`, so the mapping is
# keyed on what the publisher actually ships rather than on a guess about it.
_FAMILY_RULES: tuple[tuple[tuple[str, ...], str, Label], ...] = (
    (("benign",), "normal", Label.BENIGN),
    (("recon", "vulnerabilityscan"), "probe", Label.PORT_SCAN),
    (("ddos", "dos-", "mirai"), "dos", Label.FLOOD),
)


def resolve_family(name: object) -> tuple[str, Label] | None:
    """Map a capture's family folder onto (category, KRONUS label), or None.

    None rather than a default, so an out-of-scope family is a counted drop
    instead of a silently mislabelled row.
    """
    text = str(name).strip().casefold()
    if not text:
        return None
    for prefixes, category, label in _FAMILY_RULES:
        for prefix in prefixes:
            if text.startswith(prefix):
                return (category, label)
    return None


# --- pcap reading -----------------------------------------------------------

# magic -> (byte order, nanosecond timestamps). All four are the same format
# with a different byte order and a different tick, so one reader covers them.
# The order is named rather than spelled "<"/">" because `int.from_bytes` and
# `struct` spell it differently, and the two must not drift apart.
_PCAP_MAGICS: dict[bytes, tuple[str, bool]] = {
    b"\xa1\xb2\xc3\xd4": ("little", False),
    b"\xd4\xc3\xb2\xa1": ("big", False),
    b"\xa1\xb2\x3c\x4d": ("little", True),
    b"\x4d\x3c\xb2\xa1": ("big", True),
}

_STRUCT_PREFIX = {"little": "<", "big": ">"}

LINKTYPE_NULL = 0
LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
LINKTYPE_LINUX_SLL = 113

ETHERTYPE_IPV4 = 0x0800
ETHERTYPE_VLAN = 0x8100
ETHERTYPE_IPV6 = 0x86DD

PROTOCOL_NUMBER_MAP: dict[int, Protocol] = {
    6: Protocol.TCP,
    17: Protocol.UDP,
    1: Protocol.ICMP,
}

_PCAP_GLOBAL_HEADER = 24
_PCAP_RECORD_HEADER = 16


class PcapFormatError(ValueError):
    """Raised when a file is not a pcap this reader understands."""


def _link_payload(frame: bytes, linktype: int) -> bytes | None:
    """Strip the link layer, returning the IPv4 datagram, or None if the frame
    is something other than IPv4 (counted by the caller, never guessed at)."""
    if linktype == LINKTYPE_ETHERNET:
        if len(frame) < 14:
            return None
        ethertype = int.from_bytes(frame[12:14], "big")
        offset = 14
        # 802.1Q: the real ethertype sits after the 4-byte tag.
        if ethertype == ETHERTYPE_VLAN:
            if len(frame) < 18:
                return None
            ethertype = int.from_bytes(frame[16:18], "big")
            offset = 18
        if ethertype != ETHERTYPE_IPV4:
            return None
        return frame[offset:]
    if linktype == LINKTYPE_LINUX_SLL:
        if len(frame) < 16:
            return None
        if int.from_bytes(frame[14:16], "big") != ETHERTYPE_IPV4:
            return None
        return frame[16:]
    if linktype == LINKTYPE_RAW:
        if not frame or (frame[0] >> 4) != 4:
            return None
        return frame
    if linktype == LINKTYPE_NULL:
        if len(frame) < 4:
            return None
        # BSD loopback: host-endian address family, 2 = AF_INET.
        family = int.from_bytes(frame[:4], "little")
        if family != 2:
            return None
        return frame[4:]
    return None


@dataclass(frozen=True)
class _Packet:
    ts: float
    src_ip: str
    dst_ip: str
    src_port: int | None
    dst_port: int | None
    protocol: Protocol
    length: int


def _parse_ipv4(datagram: bytes, ts: float) -> _Packet | None:
    """Parse one IPv4 datagram. Returns None for anything without usable
    5-tuple information (a non-initial fragment carries no ports)."""
    if len(datagram) < 20 or (datagram[0] >> 4) != 4:
        return None
    ihl = (datagram[0] & 0x0F) * 4
    if ihl < 20 or len(datagram) < ihl:
        return None
    total_length = int.from_bytes(datagram[2:4], "big")
    frag_field = int.from_bytes(datagram[6:8], "big")
    # A non-zero fragment offset means this is not the first fragment, so the
    # transport header (and therefore the ports) is not in this frame. A first
    # fragment has offset 0 and does carry them, so it is parsed normally.
    if frag_field & 0x1FFF:
        return None
    proto_num = datagram[9]
    src_ip = ".".join(str(b) for b in datagram[12:16])
    dst_ip = ".".join(str(b) for b in datagram[16:20])

    # Count what was captured, not what was claimed: a capture taken with a
    # snaplen smaller than the frame really only saw these bytes.
    length = min(total_length, len(datagram)) if total_length else len(datagram)

    protocol = PROTOCOL_NUMBER_MAP.get(proto_num, Protocol.OTHER)
    src_port = dst_port = None
    if proto_num in (6, 17) and len(datagram) >= ihl + 4:
        src_port = int.from_bytes(datagram[ihl:ihl + 2], "big")
        dst_port = int.from_bytes(datagram[ihl + 2:ihl + 4], "big")
    return _Packet(ts, src_ip, dst_ip, src_port, dst_port, protocol, length)


def iter_packets(path: Path, drops: Counter) -> Iterator[_Packet]:
    """Stream the IPv4 packets of one capture, counting what it cannot use.

    Streaming rather than reading the file into memory keeps peak use at one
    frame, which is what makes a multi-hundred-MB capture workable here.
    """
    with open(path, "rb") as handle:
        header = handle.read(_PCAP_GLOBAL_HEADER)
        if len(header) < _PCAP_GLOBAL_HEADER:
            raise PcapFormatError(f"{path.name}: shorter than a pcap header")
        magic = header[:4]
        if magic not in _PCAP_MAGICS:
            # pcapng starts with 0a0d0d0a and is a different container.
            if magic == b"\x0a\x0d\x0d\x0a":
                raise PcapFormatError(
                    f"{path.name}: this is pcapng, not classic pcap")
            raise PcapFormatError(f"{path.name}: not a pcap (magic {magic!r})")
        endian, nanosecond = _PCAP_MAGICS[magic]
        prefix = _STRUCT_PREFIX[endian]
        linktype = int.from_bytes(header[20:24], endian)

        if linktype not in (LINKTYPE_ETHERNET, LINKTYPE_LINUX_SLL,
                            LINKTYPE_RAW, LINKTYPE_NULL):
            drops[f"linktype_{linktype}"] += 1
            return
        drops[f"linktype_used_{linktype}"] += 1

        divisor = 1_000_000_000 if nanosecond else 1_000_000
        while True:
            record = handle.read(_PCAP_RECORD_HEADER)
            if len(record) < _PCAP_RECORD_HEADER:
                break
            ts_sec, ts_frac, incl_len, _orig_len = struct.unpack(
                f"{prefix}IIII", record)
            frame = handle.read(incl_len)
            if len(frame) < incl_len:
                drops["truncated_final_record"] += 1
                break
            ts = ts_sec + ts_frac / divisor

            datagram = _link_payload(frame, linktype)
            if datagram is None:
                ethertype = (int.from_bytes(frame[12:14], "big")
                             if linktype == LINKTYPE_ETHERNET and len(frame) >= 14
                             else None)
                if ethertype == ETHERTYPE_IPV6:
                    drops["ipv6_frames"] += 1
                elif ethertype != ETHERTYPE_IPV4:
                    drops["non_ip_frames"] += 1
                else:
                    drops["unusable_frames"] += 1
                continue

            packet = _parse_ipv4(datagram, ts)
            if packet is None:
                drops["fragments_and_malformed"] += 1
                continue
            yield packet


# --- flow assembly ----------------------------------------------------------

# An exporter's idle/active pair. Both are ours, not the release's, and both go
# into the metrics: a scan fits inside one flow, a sustained flood is cut into
# active-timeout-long flows exactly as an exporter on that link would cut it.
FLOW_IDLE_TIMEOUT_S = 30.0
FLOW_ACTIVE_TIMEOUT_S = 120.0

# Backstop against an unbounded capture exhausting a laptop's memory. Reaching
# it raises rather than truncating, because a silently short list of flows
# would look like a smaller capture rather than a failed one.
DEFAULT_MAX_FLOWS = 2_000_000


@dataclass
class _FlowState:
    """One open 5-tuple. `fwd_*` is the direction of the first packet seen."""

    src_ip: str
    dst_ip: str
    src_port: int | None
    dst_port: int | None
    protocol: Protocol
    first_ts: float
    last_ts: float
    fwd_packets: int = 0
    fwd_bytes: int = 0
    bwd_packets: int = 0
    bwd_bytes: int = 0

    def add(self, packet: _Packet) -> None:
        if packet.src_ip == self.src_ip and packet.src_port == self.src_port:
            self.fwd_packets += 1
            self.fwd_bytes += packet.length
        else:
            self.bwd_packets += 1
            self.bwd_bytes += packet.length
        self.last_ts = packet.ts

    def duration_ms(self) -> int:
        return max(0, round((self.last_ts - self.first_ts) * 1000))


def _flow_key(packet: _Packet) -> tuple:
    """Bidirectional key: the endpoints sorted, so both directions match.

    Ports are None for ICMP, which still keys correctly — the tuple is
    positional, so a None port is a value like any other. The protocol is part
    of the key, so a TCP and a UDP conversation between the same ports stay
    two flows.
    """
    a = (packet.src_ip, packet.src_port if packet.src_port is not None else -1)
    b = (packet.dst_ip, packet.dst_port if packet.dst_port is not None else -1)
    if b < a:
        a, b = b, a
    return (a, b, packet.protocol.value)


def _flow_to_row(flow: _FlowState, raw_label: str, category: str,
                 label: Label) -> NSLKDDRow:
    total_bytes = flow.fwd_bytes + flow.bwd_bytes
    total_packets = flow.fwd_packets + flow.bwd_packets
    duration_ms = flow.duration_ms()
    # These are *our* measurements of the publisher's packets, not columns the
    # release ships. They are the flow's own statistics, nothing inferred.
    features = {
        "flow_packets": float(total_packets),
        "flow_bytes": float(total_bytes),
        "packets_fwd": float(flow.fwd_packets),
        "packets_bwd": float(flow.bwd_packets),
        "bytes_fwd": float(flow.fwd_bytes),
        "bytes_bwd": float(flow.bwd_bytes),
        "duration_ms": float(duration_ms),
        "mean_packet_bytes": float(total_bytes / total_packets) if total_packets else 0.0,
        "packet_rate_hz": (
            total_packets / (duration_ms / 1000.0) if duration_ms > 0 else 0.0
        ),
    }
    return NSLKDDRow(
        source_ip=flow.src_ip,
        dest_ip=flow.dst_ip,
        source_port=flow.src_port,
        dest_port=flow.dst_port,
        protocol=flow.protocol,
        total_bytes=total_bytes,
        duration_ms=duration_ms,
        raw_label=raw_label,
        category=category,
        kronus_label=label,
        difficulty=0,
        features=features,
        origin=DataOrigin.REAL,
    )


def extract_flows(path: Path, max_flows: int = DEFAULT_MAX_FLOWS
                  ) -> tuple[list[NSLKDDRow], list[float], dict]:
    """Extract every flow in one capture: (rows, start times, report)."""
    drops: Counter = Counter()
    open_flows: dict[tuple, _FlowState] = {}
    finished: list[_FlowState] = []
    packets = 0

    for packet in iter_packets(path, drops):
        packets += 1
        key = _flow_key(packet)
        flow = open_flows.get(key)
        if flow is not None:
            # Close on an idle gap or on the active timeout before folding the
            # packet in, so no flow outlives either bound.
            if (packet.ts - flow.last_ts > FLOW_IDLE_TIMEOUT_S
                    or packet.ts - flow.first_ts > FLOW_ACTIVE_TIMEOUT_S):
                finished.append(flow)
                flow = None
        if flow is None:
            if len(open_flows) >= max_flows:
                raise RuntimeError(
                    f"{path.name}: more than {max_flows:,} concurrent flows; "
                    "lower the per-capture cap rather than let this truncate")
            flow = _FlowState(
                src_ip=packet.src_ip, dst_ip=packet.dst_ip,
                src_port=packet.src_port, dst_port=packet.dst_port,
                protocol=packet.protocol, first_ts=packet.ts, last_ts=packet.ts,
            )
            open_flows[key] = flow
        flow.add(packet)

    finished.extend(open_flows.values())
    finished.sort(key=lambda f: f.first_ts)

    category, label = _family_of(path)
    rows = [_flow_to_row(f, _family_name(path), category, label) for f in finished]
    times = [f.first_ts for f in finished]
    report = {
        "capture": path.name,
        "family": _family_name(path),
        "packets_parsed": packets,
        "flows": len(rows),
        "drops": dict(sorted(drops.items())),
    }
    return rows, times, report


def _family_name(path: Path) -> str:
    """The capture's family folder, or its own stem when it sits at the root."""
    parent = path.parent.name
    if parent and parent not in (".", ""):
        return parent
    return path.stem


def _family_of(path: Path) -> tuple[str, Label]:
    resolved = resolve_family(_family_name(path))
    if resolved is None:
        raise ValueError(
            f"{path.name}: family {_family_name(path)!r} is not one both lanes "
            "have a contract for; see _FAMILY_RULES")
    return resolved


# --- capture discovery ------------------------------------------------------

_CAPTURE_SUFFIXES = (".pcap", ".cap", ".pcapng")


def _capture_paths(path: Path) -> list[Path]:
    """Every capture under a file, a directory of captures, or a directory of
    family folders (the shape the downloader produces)."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    found = sorted(p for p in path.rglob("*")
                   if p.is_file() and p.suffix in _CAPTURE_SUFFIXES)
    return found


def _evenly_spaced(total: int, target: int | None) -> list[int]:
    """Indices for an evenly spaced sample across the whole capture.

    A stride rather than a prefix: the first N flows of a capture are all from
    its first moments, which for a flood is a different regime from its middle.
    """
    if target is None or target >= total or total == 0:
        return list(range(total))
    step = total / target
    return [min(total - 1, int(i * step)) for i in range(target)]


# --- public API -------------------------------------------------------------

def _load_one_capture(capture: Path, sample_per_capture: int | None
                      ) -> tuple[list[NSLKDDRow], list[float], dict]:
    rows, times, report = extract_flows(capture)
    report["flows_kept"] = len(rows)
    if sample_per_capture is not None and len(rows) > sample_per_capture:
        keep = _evenly_spaced(len(rows), sample_per_capture)
        rows = [rows[i] for i in keep]
        times = [times[i] for i in keep]
    report["sampled_to"] = len(rows)
    return rows, times, report


def _read_rows(
    path: str | Path,
    limit: int | None,
    sample_per_capture: int | None,
    timed: bool,
) -> tuple[list[NSLKDDRow], list[float] | None, dict]:
    root = Path(path)
    captures = _capture_paths(root)
    if not captures:
        return [], ([] if timed else None), {
            "captures": [], "rows_loaded": 0, "skipped_captures": [],
            "families": {},
            "reason": f"no captures under {root}",
        }

    rows: list[NSLKDDRow] = []
    times: list[float] = []
    per_capture: list[dict] = []
    skipped: list[dict] = []
    family_counts: Counter = Counter()
    dropped_families: Counter = Counter()

    for capture in captures:
        if resolve_family(_family_name(capture)) is None:
            # Out of scope, counted rather than loaded under a guessed label.
            dropped_families[_family_name(capture)] += 1
            continue
        capture_rows, capture_times, report = _load_one_capture(
            capture, sample_per_capture)
        per_capture.append(report)
        family_counts[report["family"]] += len(capture_rows)
        rows.extend(capture_rows)
        times.extend(capture_times)

    if rows:
        order = sorted(range(len(rows)), key=lambda i: (times[i], i))
        rows = [rows[i] for i in order]
        times = [times[i] for i in order]

    if limit is not None and limit < len(rows):
        keep = _evenly_spaced(len(rows), limit)
        rows = [rows[i] for i in keep]
        times = [times[i] for i in keep]

    report = {
        "captures": [p.name for p in captures],
        "captures_loaded": len(per_capture),
        "rows_loaded": len(rows),
        "per_capture": per_capture,
        "skipped_captures": skipped,
        "dropped_families": dict(sorted(dropped_families.items())),
        "rows_by_family": dict(sorted(family_counts.items())),
        "synthetic_hosts": False,
        "extraction": ("flows extracted by us from the publisher's packet "
                       "captures; addresses, ports and times are the capture's"),
        "flow_definition": {
            "key": "bidirectional 5-tuple (endpoints sorted)",
            "idle_timeout_s": FLOW_IDLE_TIMEOUT_S,
            "active_timeout_s": FLOW_ACTIVE_TIMEOUT_S,
            "source_ip": "endpoint that sent the flow's first packet",
            "timestamp": "flow's first-packet time",
        },
    }
    return rows, (times if timed else None), report


def load_ciciot2023(
    path: str | Path,
    limit: int | None = None,
    sample_per_capture: int | None = None,
    with_features: bool = True,
) -> list[NSLKDDRow]:
    """Load flows from a capture, a directory of captures, or a directory of
    the downloader's per-family folders."""
    rows, _, _ = _read_rows(path, limit, sample_per_capture, False)
    return rows


def load_ciciot2023_timed(
    path: str | Path,
    limit: int | None = None,
    sample_per_capture: int | None = None,
    with_features: bool = True,
) -> tuple[list[NSLKDDRow], list[float]]:
    """Load flows with their first-packet times, aligned index-for-index."""
    rows, times, _ = _read_rows(path, limit, sample_per_capture, True)
    assert times is not None
    return rows, times


def load_ciciot2023_report(
    path: str | Path,
    limit: int | None = None,
    sample_per_capture: int | None = None,
) -> tuple[list[NSLKDDRow], dict]:
    """Load flows and the extraction report in one pass."""
    rows, _, report = _read_rows(path, limit, sample_per_capture, False)
    return rows, report


def load_ciciot2023_by_capture(
    path: str | Path,
    sample_per_capture: int | None = None,
) -> dict[str, tuple[list[NSLKDDRow], list[float], dict]]:
    """Load per capture: {capture name: (rows, times, report)}.

    The runner needs the capture a row came from, because the split unit has to
    be the capture rather than the flow. Every flow in one capture shares that
    capture's burst — the same attacker, the same victim, the same moments — so
    a random split over pooled flows would put half of one burst in train and
    half in test and report the near-duplicate as generalization. Keys are
    capture file names, which are unique in the downloader's flat-per-family
    layout; a genuine collision is renamed rather than silently overwritten.
    """
    by_capture: dict[str, tuple[list[NSLKDDRow], list[float], dict]] = {}
    for capture in _capture_paths(Path(path)):
        if resolve_family(_family_name(capture)) is None:
            continue
        key = capture.name
        if key in by_capture:
            key = f"{_family_name(capture)}/{capture.name}"
        rows, times, report = _load_one_capture(capture, sample_per_capture)
        by_capture[key] = (rows, times, report)
    return by_capture
