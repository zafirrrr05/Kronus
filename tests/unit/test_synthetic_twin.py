import ipaddress

from libs.constants import DataOrigin, Label
from twin.nsl_kdd import NSL_KDD_COLUMNS
from twin.synthetic import (
    generate_benign_scenario,
    generate_flood_scenario,
    generate_lateral_movement_scenario,
    generate_mixed_probe_set,
    generate_port_scan_scenario,
)
from twin.topology import build_default_topology

NUMERIC_FEATURE_KEYS = set(NSL_KDD_COLUMNS[:41]) - {"protocol_type", "service", "flag"}


def test_mixed_probe_set_is_all_synthetic_origin():
    rows = generate_mixed_probe_set(seed=1)
    assert rows
    assert all(r.origin == DataOrigin.SYNTHETIC for r in rows)


def test_mixed_probe_set_covers_all_four_labels():
    rows = generate_mixed_probe_set(seed=1)
    labels = {r.kronus_label for r in rows}
    assert labels == {Label.BENIGN, Label.FLOOD, Label.PORT_SCAN, Label.LATERAL_MOVEMENT}


def test_synthetic_ips_are_well_formed():
    for row in generate_mixed_probe_set(seed=2):
        ipaddress.ip_address(row.source_ip)
        ipaddress.ip_address(row.dest_ip)


def test_synthetic_features_use_the_same_keys_as_real_data():
    # spec.md §5.4 sim-to-real discipline: every feature must have a
    # plausible real-network equivalent. Enforced structurally here — the
    # synthetic feature dict must be drawn from exactly the real numeric
    # NSL-KDD feature set, never a twin-only key.
    topology = build_default_topology()
    for row in generate_flood_scenario(topology, n_events=5):
        assert set(row.features.keys()) == NUMERIC_FEATURE_KEYS


def test_port_scan_scenario_touches_every_host_from_one_source():
    topology = build_default_topology(host_count=10)
    rows = generate_port_scan_scenario(topology)
    sources = {r.source_ip for r in rows}
    dests = {r.dest_ip for r in rows}
    assert len(sources) == 1
    assert len(dests) == len(topology.hosts)


def test_lateral_movement_forms_a_connected_hop_chain():
    topology = build_default_topology(host_count=10)
    rows = generate_lateral_movement_scenario(topology, hop_count=4)
    assert len(rows) == 4
    # each hop's destination is the next hop's source (a real walk, not
    # four independent unrelated connections)
    for a, b in zip(rows, rows[1:], strict=False):
        assert a.dest_ip == b.source_ip
    assert all(r.kronus_label == Label.LATERAL_MOVEMENT for r in rows)


def test_benign_scenario_never_connects_a_host_to_itself():
    topology = build_default_topology()
    for row in generate_benign_scenario(topology, n_events=200):
        assert row.source_ip != row.dest_ip
