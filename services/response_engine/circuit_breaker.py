"""spec.md NFR-10 / SR-5: "<=20 actions/min/replica... fails to alert-only
beyond the cap, for every signal source." Lives inside the Response Engine
itself (features.txt: "the failure mode it guards against — too many
actions in too little time — is a property of the actor taking the
action, not of any signal upstream, deterministic or otherwise"), not as
a separate service.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

from libs.constants import CIRCUIT_BREAKER_MAX_ACTIONS_PER_MIN
from libs.observability import ACTIVE_GAUGE


class CircuitBreaker:
    def __init__(self, max_actions_per_minute: int = CIRCUIT_BREAKER_MAX_ACTIONS_PER_MIN) -> None:
        self._max = max_actions_per_minute
        self._action_times: deque[datetime] = deque()

    def record_action(self, ts: datetime) -> None:
        self._action_times.append(ts)

    def is_tripped(self, now: datetime) -> bool:
        self._evict_stale(now)
        tripped = len(self._action_times) >= self._max
        ACTIVE_GAUGE.labels(component="response_engine", kind="actions_in_window").set(
            len(self._action_times)
        )
        return tripped

    def _evict_stale(self, now: datetime) -> None:
        while self._action_times and (now - self._action_times[0]) > timedelta(minutes=1):
            self._action_times.popleft()
