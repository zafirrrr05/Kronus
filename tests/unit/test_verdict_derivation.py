from libs.constants import Label
from libs.verdict import derive_label


def test_benign_prediction_stays_benign_regardless_of_confidence():
    assert derive_label(Label.BENIGN, 0.99) == Label.BENIGN
    assert derive_label(Label.BENIGN, 0.1) == Label.BENIGN


def test_attack_at_or_above_block_threshold_keeps_concrete_label():
    assert derive_label(Label.FLOOD, 0.85) == Label.FLOOD
    assert derive_label(Label.PORT_SCAN, 1.0) == Label.PORT_SCAN


def test_attack_in_gray_band_becomes_uncertain():
    assert derive_label(Label.FLOOD, 0.5) == Label.UNCERTAIN
    assert derive_label(Label.PORT_SCAN, 0.849) == Label.UNCERTAIN


def test_attack_below_gray_zone_keeps_concrete_label_for_policy_to_route():
    assert derive_label(Label.FLOOD, 0.49) == Label.FLOOD
    assert derive_label(Label.PORT_SCAN, 0.0) == Label.PORT_SCAN
