"""spec.md FR-12: "The Response Engine SHALL act on at most one verdict
per detection window (keyed by window_id)... a later verdict on the same
window SHALL be logged but not re-trigger enforcement."

Only decisions that actually *act* (block, allowlist_exempt, throttled)
consume a window's slot — a dry_run doesn't represent "this window has
been handled," it's just a log entry (features.txt: gray-zone traffic
still gets a human review path, which a second, later, higher-confidence
verdict on the same window shouldn't be blocked from reaching).
"""

from __future__ import annotations

from libs.constants import PolicyAction

_ACTING_ACTIONS = {PolicyAction.BLOCK, PolicyAction.ALLOWLIST_EXEMPT, PolicyAction.THROTTLED}


class WindowCorrelator:
    def __init__(self) -> None:
        self._resolved: dict[str, str] = {}

    def prior_action_for(self, window_id: str) -> str | None:
        return self._resolved.get(window_id)

    def record(self, window_id: str, action: str) -> None:
        if action in _ACTING_ACTIONS and window_id not in self._resolved:
            self._resolved[window_id] = action
