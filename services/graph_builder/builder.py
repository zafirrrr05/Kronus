"""features.txt component 3 (Graph Builder): "every couple of seconds,
draws a map from recent events on the belt — a dot per host, a line per
conversation, each line carrying notes: bytes, port, duration." This is
the literal translation step that makes the fan-out shape a port scan
produces (twin.nsl_kdd's scan rows collapsing onto few sources, many
destinations) visible as graph structure rather than just a pile of
independent rows.

Split deliberately in two: `build_snapshot` is a pure function (a window
of events in, a GraphSnapshot out) — trivial to unit-test with a hand-built
event list. `WindowedGraphBuilder` is the thin async wrapper that actually
reads the Event Stream and calls it on a tumbling-window cadence — that's
where the I/O and the Observability wiring live, kept separate on purpose
so the graph-construction logic itself never needs a running event bus to
test.
"""

from __future__ import annotations

import math
import uuid
from collections import Counter, defaultdict
from datetime import datetime

from libs.constants import SNAPSHOT_WINDOW_SECONDS
from libs.observability import observe
from libs.schemas import GraphEdge, GraphNode, GraphSnapshot, TelemetryEvent


def _port_entropy(ports: list[int]) -> float:
    if not ports:
        return 0.0
    counts = Counter(ports)
    total = len(ports)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def build_snapshot(
    events: list[TelemetryEvent], window_start: datetime, window_end: datetime,
    window_id: str | None = None,
) -> GraphSnapshot:
    """Pure: no I/O, no clock reads (window bounds are passed in) — the
    same batch of events always produces the same snapshot.
    """
    window_id = window_id or str(uuid.uuid4())

    out_edges: dict[tuple[str, str], list[TelemetryEvent]] = defaultdict(list)
    for event in events:
        out_edges[(event.source_ip, event.dest_ip)].append(event)

    degree_out: Counter[str] = Counter()
    degree_in: Counter[str] = Counter()
    bytes_total: Counter[str] = Counter()
    ports_by_source: dict[str, set[int]] = defaultdict(set)

    for event in events:
        degree_out[event.source_ip] += 1
        degree_in[event.dest_ip] += 1
        bytes_total[event.source_ip] += event.bytes
        bytes_total[event.dest_ip] += event.bytes
        if event.dest_port is not None:
            ports_by_source[event.source_ip].add(event.dest_port)

    host_ids = set(degree_out) | set(degree_in)
    nodes = [
        GraphNode(
            node_id=host,
            degree_in=float(degree_in.get(host, 0)),
            degree_out=float(degree_out.get(host, 0)),
            bytes_total=float(bytes_total.get(host, 0)),
            unique_ports_contacted=len(ports_by_source.get(host, ())),
        )
        for host in sorted(host_ids)
    ]

    edges = []
    for (src, dst), flows in sorted(out_edges.items()):
        ports = [f.dest_port for f in flows if f.dest_port is not None]
        edges.append(GraphEdge(
            src=src, dst=dst,
            bytes=float(sum(f.bytes for f in flows)),
            flow_count=len(flows),
            port_entropy=_port_entropy(ports),
            duration_mean_ms=sum(f.duration_ms for f in flows) / len(flows),
        ))

    return GraphSnapshot(
        window_id=window_id, window_start=window_start, window_end=window_end,
        nodes=nodes, edges=edges,
    )


class WindowedGraphBuilder:
    """Consumes the live Event Stream and emits one GraphSnapshot per
    tumbling window — the async wrapper around build_snapshot above.
    """

    def __init__(self, window_seconds: float = SNAPSHOT_WINDOW_SECONDS) -> None:
        self._window_seconds = window_seconds
        self._buffer: list[TelemetryEvent] = []
        self._window_start: datetime | None = None

    def ingest(self, event: TelemetryEvent) -> GraphSnapshot | None:
        """Feed one event; returns a completed GraphSnapshot exactly when
        this event closes out a window, else None. Kept synchronous and
        clock-driven by the event's own timestamp (not wall-clock) so it
        replays deterministically over a historical batch — the demo and
        the flow-case integration tests both rely on this.
        """
        if self._window_start is None:
            self._window_start = event.ts

        elapsed = (event.ts - self._window_start).total_seconds()
        if elapsed >= self._window_seconds and self._buffer:
            with observe("graph_builder", "build_snapshot", trigger="window_elapsed"):
                snapshot = build_snapshot(
                    self._buffer, window_start=self._window_start, window_end=event.ts,
                )
            self._buffer = [event]
            self._window_start = event.ts
            return snapshot

        self._buffer.append(event)
        return None

    def flush(self, end_time: datetime) -> GraphSnapshot | None:
        """Force-close the current window (end of a replay batch, or a
        graceful shutdown) rather than losing buffered events."""
        if not self._buffer or self._window_start is None:
            return None
        with observe("graph_builder", "build_snapshot", trigger="flush"):
            snapshot = build_snapshot(self._buffer, window_start=self._window_start, window_end=end_time)
        self._buffer = []
        self._window_start = None
        return snapshot
