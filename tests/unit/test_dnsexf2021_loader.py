"""Unit tests for the CIC-Bell-DNS-EXF-2021 loader (twin/dnsexf2021.py).

These do NOT require the real ~55MB download: they build a small, DNS-EXF-2021
-shaped tree in a tmp dir using the real 16-column header and the real
per-class directory layout, then assert the loader resolves classes, drops
ambiguous vectors, de-duplicates, reconstructs a graph, and converts the rows.

Two things here are genuinely unlike the other loaders' tests and get extra
attention:

1. THE CLASS COMES FROM THE DIRECTORY, NOT THE FILENAME. The released benign
   corpus contains `stateless_features-light_benign.csv`, whose name carries
   both "light" and "benign" — so a first-token-wins filename scan would file
   benign traffic under an attack class. The tests pin down that a
   contradictory name resolves to nothing rather than to a guess.
2. THE LABELS ARE NOT RECOVERABLE FROM THE FEATURES. This dataset's label is a
   capture-level annotation, so the loader's ambiguous-vector drop and
   de-duplication are correctness features, not optimisations. The tests here
   assert both, and assert that `ambiguity_report` measures the ceiling it
   claims to.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path

import pandas as pd
import pytest

from libs.constants import DataOrigin, Label, Protocol
from libs.schemas import TelemetryEvent
from services.telemetry_exporter.converters import flow_row_to_event
from twin.dnsexf2021 import (
    DNS_EXF_CLASSES,
    ambiguity_report,
    load_dnsexf2021,
    load_dnsexf2021_dataframe,
    resolve_class,
    summarize,
)
from twin.nsl_kdd import NSLKDDRow

# The real 16-column header of a labelled per-class file.
_HEADER = [
    "timestamp", "FQDN_count", "subdomain_length", "upper", "lower", "numeric",
    "entropy", "special", "labels", "labels_max", "labels_average",
    "longest_word", "sld", "len", "subdomain", "Label",
]

_OVERHEAD = 44  # the loader's documented DNS packet overhead


def _row(ts, sld, label, entropy=2.0, fqdn_count=1, length=11, longest_word=4,
         subdomain=1):
    return {
        "timestamp": ts, "FQDN_count": fqdn_count, "subdomain_length": 7,
        "upper": 0, "lower": 10, "numeric": 8, "entropy": entropy,
        "special": 6, "labels": 6, "labels_max": 7, "labels_average": 3.1667,
        "longest_word": longest_word, "sld": sld, "len": length,
        "subdomain": subdomain, "Label": label,
    }


def _write(path: Path, rows: list[dict], with_label: bool = True) -> None:
    df = pd.DataFrame(rows, columns=_HEADER)
    if not with_label:
        df = df.drop(columns=["Label"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@pytest.fixture
def dnsexf_dir(tmp_path: Path) -> Path:
    """The real three-directory layout, with the real per-file names."""
    _write(tmp_path / "benign_labeled" / "stateless_features-benign_heavy_1.csv", [
        _row("2021-01-01 00:00:00", "example", "benign", entropy=2.0),
        _row("2021-01-01 00:00:01", "windowsupdate", "benign", entropy=3.0),
        _row("2021-01-01 00:00:02", "googleapis", "benign", entropy=3.5),
    ])
    # The filename trap: this file is BENIGN but its name says "light" first.
    _write(tmp_path / "benign_labeled" / "stateless_features-light_benign.csv", [
        _row("2021-01-01 00:00:10", "microsoft", "benign", entropy=3.8),
        _row("2021-01-01 00:00:11", "local", "benign", entropy=3.9),
    ])
    _write(tmp_path / "heavy_attack_labeled" / "stateless_features-heavy_text.csv", [
        _row("2021-01-01 00:01:00", "FHEPFCELEHFCEPFFFACACACACACACABN",
             "heavy_attack", entropy=4.2),
        _row("2021-01-01 00:01:01", "224", "heavy_attack", entropy=2.5),
    ])
    _write(tmp_path / "light_attack_labeled" / "stateless_features-light_image.csv", [
        _row("2021-01-01 00:02:00", "192", "light_attack", entropy=2.6),
    ])
    return tmp_path


# --- class resolution -------------------------------------------------------

def test_resolve_class_from_each_directory():
    assert resolve_class("benign_labeled") == ("normal", Label.BENIGN)
    assert resolve_class("heavy_attack_labeled") == ("lateral_movement", Label.LATERAL_MOVEMENT)
    assert resolve_class("light_attack_labeled") == ("lateral_movement", Label.LATERAL_MOVEMENT)


def test_contradictory_filename_resolves_to_nothing():
    # The trap this loader is built around: this name holds BOTH "light" and
    # "benign". First-token-wins would map benign traffic to an attack class,
    # so it must return None and let the caller fall through to the directory.
    assert resolve_class("stateless_features-light_benign") is None
    assert resolve_class("stateless_features-benign_heavy_1") is None
    # Unambiguous names still resolve.
    assert resolve_class("stateless_features-heavy_text") == (
        "lateral_movement", Label.LATERAL_MOVEMENT,
    )
    assert resolve_class("stateless_features-light_image") == (
        "lateral_movement", Label.LATERAL_MOVEMENT,
    )


def test_label_column_values_are_recognised():
    # The fallback that makes the mirror's single combined CSV usable.
    assert resolve_class("", "benign") == ("normal", Label.BENIGN)
    assert resolve_class("", "heavy_attack") == ("lateral_movement", Label.LATERAL_MOVEMENT)
    assert resolve_class("", "light_attack") == ("lateral_movement", Label.LATERAL_MOVEMENT)
    assert resolve_class("", "something_else") is None


def test_every_class_in_the_map_is_used_by_a_test():
    assert set(DNS_EXF_CLASSES) == {"benign", "heavy", "light"}


# --- loading ----------------------------------------------------------------

def test_loads_all_three_directories_and_maps_categories(dnsexf_dir):
    rows = load_dnsexf2021(dnsexf_dir)
    assert len(rows) == 8  # 3 + 2 benign, 2 heavy, 1 light
    assert summarize(rows) == {"normal": 5, "lateral_movement": 3}


def test_directory_beats_the_misleading_filename(dnsexf_dir):
    # Every row from `stateless_features-light_benign.csv` sits in
    # benign_labeled/ and must therefore be benign, despite the name.
    rows = load_dnsexf2021(dnsexf_dir)
    from_that_file = [r for r in rows if "light_benign" in r.raw_label]
    assert from_that_file
    assert all(r.category == "normal" for r in from_that_file)
    assert all(r.kronus_label == Label.BENIGN for r in from_that_file)


def test_category_to_kronus_label(dnsexf_dir):
    for r in load_dnsexf2021(dnsexf_dir):
        if r.category == "normal":
            assert r.kronus_label == Label.BENIGN
        else:
            assert r.category == "lateral_movement"
            assert r.kronus_label == Label.LATERAL_MOVEMENT


# --- reconstruction (the disclosed part) ------------------------------------

def test_destination_comes_from_the_real_queried_domain(dnsexf_dir):
    rows = load_dnsexf2021(dnsexf_dir)
    for r in rows:
        ipaddress.IPv4Address(r.dest_ip)
        assert r.dest_ip.startswith("10.40.")
    # A pure function of `sld`: the fixture has eight distinct queried domains,
    # so it must yield eight distinct addresses — which is what makes a
    # window's distinct-destination count its distinct-domain count.
    slds = {"example", "windowsupdate", "googleapis", "microsoft", "local",
            "FHEPFCELEHFCEPFFFACACACACACACABN", "224", "192"}
    assert len(slds) == 8
    assert len({r.dest_ip for r in rows}) == 8


def test_same_domain_maps_to_the_same_address(tmp_path):
    _write(tmp_path / "benign_labeled" / "a.csv", [
        _row("2021-01-01 00:00:00", "example", "benign", entropy=1.0),
        _row("2021-01-01 00:00:01", "example", "benign", entropy=2.0),
    ])
    rows = load_dnsexf2021(tmp_path)
    assert len(rows) == 2
    assert rows[0].dest_ip == rows[1].dest_ip


def test_source_is_constant_so_it_cannot_leak_the_label(dnsexf_dir):
    # The dataset records no client identity. A class-varying source would hand
    # the model a perfect label proxy, so the reconstruction is deliberately
    # class-independent — every row shares one monitored client.
    rows = load_dnsexf2021(dnsexf_dir)
    assert {r.source_ip for r in rows} == {"10.30.0.1"}


def test_dns_is_the_subject_matter_not_an_invention(dnsexf_dir):
    for r in load_dnsexf2021(dnsexf_dir):
        assert r.dest_port == 53
        assert r.protocol == Protocol.UDP
        assert r.source_port is None  # genuinely absent, never a stand-in


def test_total_bytes_is_derived_from_the_real_name_length(tmp_path):
    _write(tmp_path / "benign_labeled" / "a.csv", [
        _row("2021-01-01 00:00:00", "example", "benign", length=30),
    ])
    assert load_dnsexf2021(tmp_path)[0].total_bytes == _OVERHEAD + 30


def test_duration_is_zero_because_none_is_recorded(dnsexf_dir):
    # Left at zero rather than fabricated; the Bouncer (which reads durations)
    # does not train on this dataset.
    assert all(r.duration_ms == 0 for r in load_dnsexf2021(dnsexf_dir))


def test_origin_is_real_and_shape_is_the_shared_row(dnsexf_dir):
    rows = load_dnsexf2021(dnsexf_dir)
    assert all(isinstance(r, NSLKDDRow) for r in rows)
    assert all(r.origin == DataOrigin.REAL for r in rows)
    assert all(r.difficulty == 0 for r in rows)


# --- features ---------------------------------------------------------------

def test_features_are_numeric_and_exclude_identity_and_label(dnsexf_dir):
    for r in load_dnsexf2021(dnsexf_dir):
        for banned in ("Label", "timestamp", "sld", "longest_word"):
            assert banned not in r.features
        assert all(isinstance(v, float) for v in r.features.values())
        assert all(v == v for v in r.features.values())  # no NaN


def test_features_expose_the_real_query_statistics(dnsexf_dir):
    r = load_dnsexf2021(dnsexf_dir)[0]
    assert "entropy" in r.features
    assert "FQDN_count" in r.features
    assert "len" in r.features


# --- the label-quality machinery --------------------------------------------

def test_identical_vectors_are_de_duplicated(tmp_path):
    # Two rows differing only in timestamp and Label are ONE observation;
    # keeping both would weight it twice and put it in both splits.
    _write(tmp_path / "benign_labeled" / "a.csv", [
        _row("2021-01-01 00:00:00", "example", "benign", entropy=2.0),
        _row("2021-01-01 00:00:05", "example", "benign", entropy=2.0),
    ])
    assert len(load_dnsexf2021(tmp_path)) == 1
    assert len(load_dnsexf2021(tmp_path, dedupe=False)) == 2


def test_cross_class_vectors_are_dropped_by_default(tmp_path):
    # The measured defect this loader exists to contain: one feature vector
    # carrying two classes. Left in, the score would partly measure the label
    # noise itself.
    shared = dict(sld="collide", entropy=3.3)
    _write(tmp_path / "benign_labeled" / "a.csv", [
        _row("2021-01-01 00:00:00", "keepme", "benign", entropy=1.0),
        _row("2021-01-01 00:00:01", **shared, label="benign"),
    ])
    _write(tmp_path / "heavy_attack_labeled" / "b.csv", [
        _row("2021-01-01 00:01:00", **shared, label="heavy_attack"),
    ])
    kept = load_dnsexf2021(tmp_path)
    assert len(kept) == 1  # only the unambiguous "keepme" row survives

    raw = load_dnsexf2021(tmp_path, drop_ambiguous=False, dedupe=False)
    assert len(raw) == 3  # the collision is visible when both guards are off


def test_ambiguity_report_measures_the_ceiling(tmp_path):
    # V1: 3 benign + 1 heavy, byte-identical features.
    # V2: 1 benign.  V3: 1 heavy.
    # Ceiling = (3 + 1 + 1) / 6.
    collide = dict(sld="collide", entropy=3.3)
    _write(tmp_path / "benign_labeled" / "a.csv", [
        _row("2021-01-01 00:00:00", **collide, label="benign"),
        _row("2021-01-01 00:00:01", **collide, label="benign"),
        _row("2021-01-01 00:00:02", **collide, label="benign"),
        _row("2021-01-01 00:00:03", "solo", "benign", entropy=1.0),
    ])
    _write(tmp_path / "heavy_attack_labeled" / "b.csv", [
        _row("2021-01-01 00:01:00", **collide, label="heavy_attack"),
        _row("2021-01-01 00:01:01", "other", "heavy_attack", entropy=1.5),
    ])
    report = ambiguity_report(tmp_path)
    assert report["rows"] == 6
    assert report["distinct_feature_vectors"] == 3
    assert report["ambiguous_vectors"] == 1
    assert report["rows_in_ambiguous_vectors"] == 4
    assert report["ambiguous_row_fraction"] == round(4 / 6, 4)
    assert report["deterministic_ceiling"] == round(5 / 6, 4)


def test_ambiguity_report_is_one_when_nothing_collides(dnsexf_dir):
    report = ambiguity_report(dnsexf_dir)
    assert report["ambiguous_vectors"] == 0
    assert report["deterministic_ceiling"] == 1.0
    assert report["rows"] == 8
    assert report["distinct_feature_vectors"] == 8
    assert report["n_files"] == 4


# --- ordering, limits, conversion -------------------------------------------

def test_rows_are_sorted_by_capture_time(tmp_path):
    _write(tmp_path / "benign_labeled" / "a.csv", [
        _row("2021-01-01 00:00:03", "c", "benign", entropy=3.0),
        _row("2021-01-01 00:00:01", "a", "benign", entropy=1.0),
        _row("2021-01-01 00:00:02", "b", "benign", entropy=2.0),
    ])
    rows = load_dnsexf2021(tmp_path)
    assert [r.features["entropy"] for r in rows] == [1.0, 2.0, 3.0]


def test_limit_caps_total_rows(dnsexf_dir):
    assert len(load_dnsexf2021(dnsexf_dir, limit=5)) == 5


def test_rows_convert_to_valid_telemetry_events(dnsexf_dir):
    for r in load_dnsexf2021(dnsexf_dir):
        event = flow_row_to_event(r)
        assert isinstance(event, TelemetryEvent)
        assert event.bytes >= 0
        assert event.duration_ms >= 0
        assert event.protocol in (Protocol.TCP, Protocol.UDP, Protocol.ICMP, Protocol.OTHER)


def test_dataframe_loader_tags_category_and_vector_key(dnsexf_dir):
    df = load_dnsexf2021_dataframe(dnsexf_dir)
    assert len(df) == 8
    assert set(df["category"].dropna().unique()) == {"normal", "lateral_movement"}
    assert df["vector_key"].nunique() == 8


# --- flattened / aggregate input --------------------------------------------

def test_flattened_combined_file_resolves_class_per_row(tmp_path):
    # The mirror's single combined CSV: no class directories, and its `Label`
    # column varies per row. Resolving the file once from its first row would
    # mislabel it, so the class must come from each row's own label.
    _write(tmp_path / "CIC-Bell-DNS-EXF-2021_stateless.csv", [
        _row("2021-01-01 00:00:00", "example", "benign", entropy=1.0),
        _row("2021-01-01 00:00:01", "evil", "heavy_attack", entropy=2.0),
        _row("2021-01-01 00:00:02", "sneaky", "light_attack", entropy=3.0),
    ])
    rows = load_dnsexf2021(tmp_path)
    assert summarize(rows) == {"normal": 1, "lateral_movement": 2}


def test_unlabeled_flattened_file_contributes_nothing(tmp_path):
    # No directory and no label column: the class is unknowable, so the file is
    # skipped rather than guessed at — including for its ambiguous name.
    _write(tmp_path / "stateless_features-light_benign.csv", [
        _row("2021-01-01 00:00:00", "example", "benign"),
    ], with_label=False)
    assert load_dnsexf2021(tmp_path) == []


def test_missing_path_raises_with_pointer(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_dnsexf2021"):
        load_dnsexf2021(tmp_path / "does_not_exist")


# --- real-data test: skips gracefully if the download isn't present ---------

REAL_DATA_DIR = "data/external/dnsexf2021"


def test_real_dnsexf2021_if_present():
    data_dir = Path(REAL_DATA_DIR)
    if not data_dir.exists() or not list(data_dir.glob("**/*.csv")):
        pytest.skip(f"{REAL_DATA_DIR} not populated — run scripts/download_dnsexf2021.py")
    rows = load_dnsexf2021(data_dir, limit=5_000)
    assert rows
    # Only the two classes this dataset can express, and no flood rows.
    assert {r.category for r in rows} <= {"normal", "lateral_movement"}
    assert all(r.dest_ip.startswith("10.40.") for r in rows)
    assert all(r.source_ip == "10.30.0.1" for r in rows)
