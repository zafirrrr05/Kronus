import ipaddress
import os

import pytest

from libs.constants import Label
from twin.nsl_kdd import NSL_KDD_COLUMNS, load_nsl_kdd, load_nsl_kdd_dataframe

REAL_TEST_FILE = "data/real/KDDTest+.txt"


@pytest.fixture(scope="module")
def rows():
    if not os.path.exists(REAL_TEST_FILE):
        pytest.skip(f"{REAL_TEST_FILE} not present — run scripts/download_data.py")
    return load_nsl_kdd(REAL_TEST_FILE)


def test_loads_expected_row_count(rows):
    assert len(rows) == 22544  # KDDTest+ known row count


def test_source_and_dest_ips_are_well_formed_ipv4(rows):
    # Regression test for a real bug: an early version of _synthetic_ip
    # produced 5-octet strings ("10.10.114.104.175") because the 2-octet
    # prefix and the hash suffix both contributed octets independently.
    for row in rows[:500]:
        ipaddress.IPv4Address(row.source_ip)  # raises if malformed
        ipaddress.IPv4Address(row.dest_ip)


def test_dos_rows_map_to_flood_label(rows):
    dos_rows = [r for r in rows if r.category == "dos"]
    assert dos_rows, "expected at least one DoS-category row in KDDTest+"
    assert all(r.kronus_label == Label.FLOOD for r in dos_rows)


def test_probe_rows_map_to_port_scan_label(rows):
    probe_rows = [r for r in rows if r.category == "probe"]
    assert probe_rows
    assert all(r.kronus_label == Label.PORT_SCAN for r in probe_rows)


def test_normal_rows_map_to_benign_label(rows):
    normal_rows = [r for r in rows if r.category == "normal"]
    assert normal_rows
    assert all(r.kronus_label == Label.BENIGN for r in normal_rows)


def test_r2l_and_u2r_rows_have_no_kronus_label(rows):
    # Documented, deliberate scope limit — see module docstring. These rows
    # still carry a category and features (used as generic non-benign
    # training signal) but never claim a KRONUS Label the schema doesn't
    # define for them.
    r2l_u2r = [r for r in rows if r.category in ("r2l", "u2r")]
    assert r2l_u2r
    assert all(r.kronus_label is None for r in r2l_u2r)


def test_scan_rows_cluster_onto_fewer_source_ips_than_dest_ips(rows):
    # The actual graph-shape claim this module exists to support: a probe
    # (port-scan) run should show many fewer distinct sources than
    # destinations — one attacker, many targets — which is what makes the
    # fan-out visible to the Graph Builder/Detective at all.
    probe_rows = [r for r in rows if r.category == "probe"]
    sources = {r.source_ip for r in probe_rows}
    dests = {r.dest_ip for r in probe_rows}
    assert len(sources) < len(dests)


def test_normal_traffic_spread_across_the_dataset_does_not_collapse_onto_one_source(rows):
    # Regression test for a real bug: without a temporal-locality bound,
    # every normal-category row sharing (protocol, service, flag, bucket)
    # anywhere in a 20,000+ row replay collapsed onto the same handful of
    # synthetic sources — turning ordinary normal traffic into apparent
    # fan-out scans once graph windows were built (see
    # services/detective/train.py's module history: 20/20 training windows
    # came back labeled port_scan before this fix). Normal traffic spread
    # across many distinct positions in the file must show meaningfully
    # more source diversity than a single sustained scan does.
    normal_rows = [r for r in rows if r.category == "normal"]
    sources = {r.source_ip for r in normal_rows}
    # category-aware clustering (see twin/nsl_kdd.py's _row_to_nslkdd_row)
    # over ~9700 normal rows in KDDTest+ should yield near-one-source-per-row
    assert len(sources) > len(normal_rows) * 0.9


def test_dataframe_loader_matches_row_count():
    if not os.path.exists(REAL_TEST_FILE):
        pytest.skip(f"{REAL_TEST_FILE} not present — run scripts/download_data.py")
    df = load_nsl_kdd_dataframe(REAL_TEST_FILE)
    assert len(df) == 22544
    assert list(df.columns[:41]) == NSL_KDD_COLUMNS[:41]
    assert "category" in df.columns
