# spec.md §6 "Required policy unit tests (opa test, non-exhaustive minimum
# set)" — all seven, numbered to match the spec, plus a few extra boundary
# cases that fell out of implementing the table.
package kronus.response_test

import rego.v1

import data.kronus.response.decision

base_verdict(tier, label, confidence) := {
	"tier": tier,
	"label": label,
	"confidence": confidence,
	"window_id": "w-1",
	"verdict_id": "v-1",
}

base_input(tier, label, confidence, allowlist, breaker) := {
	"verdict": base_verdict(tier, label, confidence),
	"on_allowlist": allowlist,
	"breaker_tripped": breaker,
	"prior_action_for_window": null,
}

# --- 1. confidence exactly at the block threshold resolves to block (boundary) ---

test_case1_confidence_exactly_at_threshold_blocks if {
	d := decision with input as base_input("bouncer", "flood", 0.85, false, false)
	d.action == "block"
}

test_case1_confidence_one_cent_below_threshold_does_not_block if {
	d := decision with input as base_input("bouncer", "flood", 0.849999, false, false)
	d.action == "dry_run"
}

# --- 2. allowlisted source never blocks regardless of confidence ---

test_case2_allowlisted_at_max_confidence_is_exempt_not_blocked if {
	d := decision with input as base_input("detective", "port_scan", 1.0, true, false)
	d.action == "allowlist_exempt"
	d.action != "block"
}

test_case2_allowlisted_at_block_threshold_is_exempt if {
	d := decision with input as base_input("bouncer", "flood", 0.85, true, false)
	d.action == "allowlist_exempt"
}

# --- 3. label "uncertain" never produces block, at any confidence, any mode ---

test_case3_uncertain_at_max_confidence_never_blocks if {
	d := decision with input as base_input("detective", "uncertain", 1.0, false, false)
	d.action == "dry_run"
	d.reason_codes[_] == "gray_zone"
}

test_case3_uncertain_at_zero_confidence_never_blocks if {
	d := decision with input as base_input("detective", "uncertain", 0.0, false, false)
	d.action == "dry_run"
}

test_case3_uncertain_allowlisted_and_breaker_tripped_still_dry_run if {
	# "any mode" — allowlist and breaker state shouldn't matter either
	d := decision with input as base_input("detective", "uncertain", 0.99, true, true)
	d.action == "dry_run"
}

# --- 4. breaker-tripped forces throttled at max confidence, for all three tiers ---

test_case4_breaker_tripped_forces_throttled_bouncer if {
	d := decision with input as base_input("bouncer", "flood", 1.0, false, true)
	d.action == "throttled"
}

test_case4_breaker_tripped_forces_throttled_detective if {
	d := decision with input as base_input("detective", "lateral_movement", 1.0, false, true)
	d.action == "throttled"
}

test_case4_breaker_tripped_forces_throttled_decoy if {
	d := decision with input as base_input("decoy", "decoy_interaction", 1.0, false, true)
	d.action == "throttled"
}

# --- 5. malformed / missing field fails closed (never defaults to block or allow) ---

test_case5_missing_confidence_field_fails_closed if {
	malformed := {
		"verdict": {"tier": "bouncer", "label": "flood", "window_id": "w-1", "verdict_id": "v-1"},
		"on_allowlist": false,
		"breaker_tripped": false,
		"prior_action_for_window": null,
	}
	not decision with input as malformed
}

test_case5_out_of_range_confidence_fails_closed if {
	d := base_input("bouncer", "flood", 1.5, false, false)
	not decision with input as d
}

test_case5_unknown_tier_fails_closed if {
	d := base_input("rogue_tier", "flood", 0.9, false, false)
	not decision with input as d
}

test_case5_unknown_label_fails_closed if {
	d := base_input("bouncer", "not_a_real_label", 0.9, false, false)
	not decision with input as d
}

test_case5_empty_window_id_fails_closed if {
	malformed := object.union(base_input("bouncer", "flood", 0.9, false, false), {"verdict": {
		"tier": "bouncer",
		"label": "flood",
		"confidence": 0.9,
		"window_id": "",
		"verdict_id": "v-1",
	}})
	not decision with input as malformed
}

# --- 6. two verdicts sharing a window_id resolve to exactly one action (FR-12) ---

test_case6_second_verdict_on_resolved_window_does_not_reblock if {
	first_input := base_input("bouncer", "flood", 0.9, false, false)
	first := decision with input as first_input
	first.action == "block"

	second_raw := base_input("detective", "port_scan", 0.95, false, false)
	second_input := object.union(second_raw, {"prior_action_for_window": first.action})
	second := decision with input as second_input
	second.action == "dry_run"
	second.reason_codes[_] == "window_already_resolved"
}

test_case6_applies_regardless_of_which_tier_acted_first if {
	# decoy acts first, bouncer's later verdict on the same window is logged,
	# not re-acted on
	decoy_input := base_input("decoy", "decoy_interaction", 1.0, false, false)
	first := decision with input as decoy_input
	first.action == "block"

	bouncer_raw := base_input("bouncer", "flood", 0.99, false, false)
	bouncer_input := object.union(bouncer_raw, {"prior_action_for_window": first.action})
	second := decision with input as bouncer_input
	second.action == "dry_run"
}

# --- 7. tier == decoy receives no special-case bypass ---

test_case7_decoy_allowlisted_is_exempt_same_as_any_tier if {
	decoy_d := decision with input as base_input("decoy", "decoy_interaction", 1.0, true, false)
	bouncer_d := decision with input as base_input("bouncer", "flood", 1.0, true, false)
	decoy_d.action == bouncer_d.action
	decoy_d.action == "allowlist_exempt"
}

test_case7_decoy_breaker_tripped_throttles_same_as_any_tier if {
	decoy_d := decision with input as base_input("decoy", "decoy_interaction", 1.0, false, true)
	detective_d := decision with input as base_input("detective", "port_scan", 1.0, false, true)
	decoy_d.action == detective_d.action
	decoy_d.action == "throttled"
}

test_case7_decoy_clean_case_blocks_like_max_confidence_bouncer if {
	decoy_d := decision with input as base_input("decoy", "decoy_interaction", 1.0, false, false)
	bouncer_d := decision with input as base_input("bouncer", "flood", 1.0, false, false)
	decoy_d.action == bouncer_d.action
	decoy_d.action == "block"
	decoy_d.ttl_seconds == bouncer_d.ttl_seconds
}

# --- extra: benign traffic never blocks, at any confidence ---

test_benign_never_blocks_even_at_max_confidence if {
	d := decision with input as base_input("bouncer", "benign", 1.0, false, false)
	d.action == "dry_run"
	d.reason_codes[_] == "below_gray_zone"
}

# --- extra: below-gray-zone attack-labeled traffic is logged, not notified ---

test_low_confidence_attack_label_is_below_gray_zone if {
	d := decision with input as base_input("detective", "port_scan", 0.2, false, false)
	d.action == "dry_run"
	d.reason_codes[_] == "below_gray_zone"
}
