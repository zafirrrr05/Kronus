"""spec.md §6's decision table assumes a verdict's `label` already reflects
its confidence band by the time policy sees it (a concrete attack class at
high confidence, "uncertain" in the middle band, unchanged at the low end
where policy/response.rego's below_gray_zone branch takes over). That
derivation is model-calibration logic, not policy — it belongs with the
Bouncer and Detective's own verdict construction, not inside Rego (see
policy/response.rego's top comment). Both lanes need the identical
derivation, so it lives here once rather than twice.
"""

from __future__ import annotations

from libs.constants import BLOCK_CONFIDENCE_THRESHOLD, GRAY_ZONE_LOW_THRESHOLD, Label


def derive_label(predicted_label: Label, confidence: float) -> Label:
    """predicted_label is the model's raw argmax (e.g. Label.FLOOD,
    Label.PORT_SCAN, Label.BENIGN — never UNCERTAIN, which only exists as
    a derived, not predicted, value). Returns the label actually placed on
    the DetectionVerdict.
    """
    if predicted_label == Label.BENIGN:
        return Label.BENIGN
    if confidence >= BLOCK_CONFIDENCE_THRESHOLD:
        return predicted_label
    if confidence >= GRAY_ZONE_LOW_THRESHOLD:
        return Label.UNCERTAIN
    return predicted_label  # below gray zone; policy routes this via label + confidence together
