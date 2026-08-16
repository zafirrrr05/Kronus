# spec.md §6 — authoritative. This file, not the doc's decision table, wins
# on any disagreement (the doc says so itself).
#
# Design choice worth stating up front: the rules below do NOT branch on
# `tier` anywhere. The Decoy's verdict always arrives with
# confidence == 1.0 and label == "decoy_interaction" (enforced by
# libs/schemas.py's DetectionVerdict validator, not trusted here either —
# see valid_input), which naturally satisfies the ">= block_threshold"
# branch through the exact same rule a maximum-confidence Bouncer or
# Detective verdict would hit. That's what makes "the decoy gets no
# special-case bypass" a structural fact instead of a comment — see
# response_test.rego's decoy_no_bypass tests, which check this by
# comparing decoy and non-decoy inputs against the identical rule set.
package kronus.response

import rego.v1

# Kept in sync with libs/constants.py by hand — Rego can't import Python.
# Both are covered by tests (test_constants_match_policy in response_test.rego
# would need OPA to read Python, which it can't; the two are cross-checked
# instead by tests/unit/test_response_engine.py asserting the Python-side
# constants against values this file returns for known boundary inputs).
block_threshold := 0.85

gray_zone_low := 0.50

block_ttl_seconds := 900 # NFR-9: single IP, <=15 min, reversible

known_tiers := {"bouncer", "detective", "decoy"}

known_labels := {"flood", "port_scan", "lateral_movement", "benign", "uncertain", "decoy_interaction"}

# --- fail-closed input validation (opa test case 5) -------------------------
# `decision` below is defined ONLY when valid_input holds. Malformed input
# leaves `decision` undefined; response_engine/opa_client.py treats an
# undefined result as a reject, never as an implicit allow or block.

default valid_input := false

valid_input if {
	input.verdict.tier in known_tiers
	input.verdict.label in known_labels
	is_number(input.verdict.confidence)
	input.verdict.confidence >= 0
	input.verdict.confidence <= 1
	is_string(input.verdict.window_id)
	input.verdict.window_id != ""
	is_string(input.verdict.verdict_id)
	input.verdict.verdict_id != ""
	is_boolean(input.on_allowlist)
	is_boolean(input.breaker_tripped)
}

# --- FR-12: at most one acting decision per window_id ------------------------

already_resolved if input.prior_action_for_window != null

decision := d if {
	valid_input
	already_resolved
	d := {
		"action": "dry_run",
		"reason_codes": ["window_already_resolved"],
		"ttl_seconds": null,
	}
}

# --- everything below requires the window not already resolved --------------

eligible if {
	valid_input
	not already_resolved
}

# label == "uncertain": gray zone, unconditional on confidence, allowlist,
# breaker, or tier (FR-3; opa test case 3).

decision := d if {
	eligible
	input.verdict.label == "uncertain"
	d := {"action": "dry_run", "reason_codes": ["gray_zone"], "ttl_seconds": null}
}

# label == "benign": nothing to act on, logged only, never notified.

decision := d if {
	eligible
	input.verdict.label == "benign"
	d := {"action": "dry_run", "reason_codes": ["below_gray_zone"], "ttl_seconds": null}
}

# a concrete attack label (flood / port_scan / lateral_movement /
# decoy_interaction), confidence below the gray-zone floor: too weak to
# even flag as ambiguous.

decision := d if {
	eligible
	is_concrete_attack_label(input.verdict.label)
	input.verdict.confidence < gray_zone_low
	d := {"action": "dry_run", "reason_codes": ["below_gray_zone"], "ttl_seconds": null}
}

# same, but inside the gray band: ambiguous, notify a human.

decision := d if {
	eligible
	is_concrete_attack_label(input.verdict.label)
	input.verdict.confidence >= gray_zone_low
	input.verdict.confidence < block_threshold
	d := {"action": "dry_run", "reason_codes": ["gray_zone"], "ttl_seconds": null}
}

# >= block_threshold, allowlisted: exempt regardless of breaker state.

decision := d if {
	eligible
	is_concrete_attack_label(input.verdict.label)
	input.verdict.confidence >= block_threshold
	input.on_allowlist
	d := {"action": "allowlist_exempt", "reason_codes": ["on_allowlist"], "ttl_seconds": null}
}

# >= block_threshold, not allowlisted, breaker tripped: throttled.

decision := d if {
	eligible
	is_concrete_attack_label(input.verdict.label)
	input.verdict.confidence >= block_threshold
	not input.on_allowlist
	input.breaker_tripped
	d := {"action": "throttled", "reason_codes": ["circuit_breaker_exceeded"], "ttl_seconds": null}
}

# >= block_threshold, not allowlisted, breaker not tripped: block.

decision := d if {
	eligible
	is_concrete_attack_label(input.verdict.label)
	input.verdict.confidence >= block_threshold
	not input.on_allowlist
	not input.breaker_tripped
	d := {
		"action": "block",
		"reason_codes": ["confidence_above_threshold"],
		"ttl_seconds": block_ttl_seconds,
	}
}

is_concrete_attack_label(label) if {
	label != "uncertain"
	label != "benign"
}
