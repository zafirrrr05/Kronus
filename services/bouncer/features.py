"""features.txt component 4 (Bouncer): "a statistical first-pass check
running on raw per-flow features... reads the Event Stream directly."

This computes those features from a sliding window of raw TelemetryEvents,
not from NSL-KDD's own precomputed aggregate columns (count, srv_count,
serror_rate, ...) — a live network sensor would never hand the Bouncer
those, only raw per-connection records. Using the same featurizer at
training time (fed a time-ordered replay of NSL-KDD-derived events) and at
serving time (fed live events one at a time) is what keeps training and
serving consistent; computing features two different ways for "replay" vs
"live" is exactly the kind of skew that quietly breaks fast-lane detectors
in production.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque

from libs.schemas import TelemetryEvent

FEATURE_NAMES = [
    "event_rate", "byte_rate", "dest_port_entropy",
    "unique_dest_count", "avg_duration_ms", "same_dest_ratio",
]

WINDOW_SECONDS = 2.0  # matches the ~2s cadence used across the system


class FlowFeaturizer:
    """Stateful: keyed by source_ip, keeps only the last WINDOW_SECONDS of
    each source's events (a deque, so eviction is O(1) amortized). Call
    `features_for` once per incoming event, in timestamp order.
    """

    def __init__(self, window_seconds: float = WINDOW_SECONDS) -> None:
        self._window_seconds = window_seconds
        self._history: dict[str, deque[TelemetryEvent]] = defaultdict(deque)

    def _evict_stale(self, source_ip: str, now) -> None:
        history = self._history[source_ip]
        while history and (now - history[0].ts).total_seconds() > self._window_seconds:
            history.popleft()

    def features_for(self, event: TelemetryEvent) -> dict[str, float]:
        """Records `event` into its source's window, then returns features
        computed over that window (including `event` itself).
        """
        history = self._history[event.source_ip]
        history.append(event)
        self._evict_stale(event.source_ip, event.ts)

        n = len(history)
        total_bytes = sum(e.bytes for e in history)
        dest_ports = [e.dest_port for e in history if e.dest_port is not None]
        dest_ips = [e.dest_ip for e in history]
        same_dest = sum(1 for e in history if e.dest_ip == event.dest_ip)

        return {
            "event_rate": float(n) / self._window_seconds,
            "byte_rate": float(total_bytes) / self._window_seconds,
            "dest_port_entropy": _entropy(dest_ports),
            "unique_dest_count": float(len(set(dest_ips))),
            "avg_duration_ms": sum(e.duration_ms for e in history) / n,
            "same_dest_ratio": same_dest / n,
        }

    def reset(self) -> None:
        self._history.clear()


def _entropy(values: list) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())
