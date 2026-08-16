"""spec.md FR-9: "Response Engine SHALL consult OPA for every policy
decision; hard-coded fallback logic is prohibited." This is that
consultation — a real subprocess call to the real `opa` binary (verified
working: `opa eval -d policy/response.rego -I --format json
data.kronus.response.decision`), not a Python re-implementation of the
Rego rules. `-I`/--stdin-input is the flag that actually works for piping
input in this OPA version; `-i -` (the file-argument form with a dash)
does not, despite looking like it should — confirmed by testing directly
against the real binary before writing this client, not assumed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from libs.schemas import DetectionVerdict, PolicyDecision

DECISION_QUERY = "data.kronus.response.decision"


class OPAEvaluationError(Exception):
    """Raised when opa itself fails (bad binary, syntax error in the
    policy, timeout) — distinct from a *defined* reject, which
    PolicyClient.evaluate signals by returning None (fail closed).
    """


class PolicyClient:
    def __init__(self, binary_path: str = "opa", policy_dir: str = "policy") -> None:
        self._binary_path = binary_path
        self._policy_dir = policy_dir
        self._policy_version = self._compute_policy_version()

    def _compute_policy_version(self) -> str:
        policy_file = Path(self._policy_dir) / "response.rego"
        digest = hashlib.sha256(policy_file.read_bytes()).hexdigest()
        return digest[:12]

    @property
    def policy_version(self) -> str:
        return self._policy_version

    def evaluate(
        self, verdict: DetectionVerdict, on_allowlist: bool, breaker_tripped: bool,
        prior_action_for_window: str | None,
    ) -> PolicyDecision | None:
        """Returns None when the policy has no defined decision for this
        input (opa's `decision` rule stays undefined — see
        policy/response.rego's valid_input gate) — the fail-closed case
        for malformed input (spec.md §6 required test case 5). Callers
        must never substitute a default action when this returns None.
        """
        opa_input = {
            "verdict": {
                "tier": verdict.tier,
                "label": verdict.label,
                "confidence": verdict.confidence,
                "window_id": verdict.window_id,
                "verdict_id": verdict.verdict_id,
            },
            "on_allowlist": on_allowlist,
            "breaker_tripped": breaker_tripped,
            "prior_action_for_window": prior_action_for_window,
        }

        try:
            result = subprocess.run(
                [self._binary_path, "eval", "-d", self._policy_dir, "-I", "--format", "json",
                 DECISION_QUERY],
                input=json.dumps(opa_input), capture_output=True, text=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            raise OPAEvaluationError(f"opa invocation failed: {exc}") from exc

        if result.returncode != 0:
            raise OPAEvaluationError(f"opa eval exited {result.returncode}: {result.stderr}")

        parsed = json.loads(result.stdout)
        results = parsed.get("result")
        if not results:
            return None  # decision was undefined -> fail closed

        value = results[0]["expressions"][0]["value"]
        return PolicyDecision(
            verdict_id=verdict.verdict_id,
            action=value["action"],
            reason_codes=value["reason_codes"],
            ttl_seconds=value["ttl_seconds"],
            policy_version=self._policy_version,
        )
