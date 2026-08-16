from datetime import datetime, timedelta, timezone

import numpy as np

from libs.constants import Label, Tier
from libs.schemas import GraphEdge, GraphNode, GraphSnapshot
from services.detective.model import (
    RAW_CLASSES,
    DetectiveModel,
    _loss_fn,
    _segment_softmax,
    count_params,
    forward_with_attention,
    init_params,
    loss_grad,
    snapshot_to_arrays,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _toy_graph_inputs(rng, n_nodes=5, n_edges=8):
    node_feats = rng.normal(size=(n_nodes, 4))
    edge_feats = rng.normal(size=(n_edges, 4))
    src_idx = rng.integers(0, n_nodes, size=n_edges)
    dst_idx = rng.integers(0, n_nodes, size=n_edges)
    return node_feats, src_idx, dst_idx, edge_feats


# --- 1. parameter budget -----------------------------------------------------

def test_parameter_count_is_well_under_the_200k_spec_budget():
    params = init_params(np.random.default_rng(0))
    assert count_params(params) < 200_000


# --- 2. segment softmax invariants -------------------------------------------

def test_segment_softmax_sums_to_one_per_destination_group():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=10)
    dst_idx = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    attn = _segment_softmax(scores, dst_idx, num_nodes=4)
    for node in range(4):
        mask = dst_idx == node
        assert np.isclose(attn[mask].sum(), 1.0)


def test_segment_softmax_single_incoming_edge_gives_full_weight():
    scores = np.array([5.0])
    attn = _segment_softmax(scores, np.array([0]), num_nodes=1)
    assert np.isclose(attn[0], 1.0)


# --- 3. THE critical test: hand-derived backprop matches finite differences --

def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    params = init_params(rng)
    node_feats, src_idx, dst_idx, edge_feats = _toy_graph_inputs(rng)
    label_idx = 1

    analytic = loss_grad(params, node_feats, src_idx, dst_idx, edge_feats, label_idx)

    # Spot-check a handful of scalar entries across different parameter
    # tensors (first layer's W, last layer's a, the classifier) with
    # central finite differences. This is the test that would have caught
    # a wrong sign, a wrong axis in the segment-softmax matmul, or a
    # mis-wired residual — the failure modes hand-rolled backprop actually
    # has, that "training loss goes down" would not reliably surface.
    eps = 1e-5
    checks = [
        ("layers", 0, "W", (0, 0)),
        ("layers", 0, "a", (3,)),
        ("layers", -1, "W", (5, 5)),
        ("layers", -1, "a", (0,)),
        ("classifier", None, None, (10, 1)),
    ]

    def loss_at(p):
        return _loss_fn(p, node_feats, src_idx, dst_idx, edge_feats, label_idx)

    for group, layer_idx, key, coord in checks:
        import copy
        p_plus = copy.deepcopy(params)
        p_minus = copy.deepcopy(params)
        if group == "layers":
            p_plus["layers"][layer_idx][key][coord] += eps
            p_minus["layers"][layer_idx][key][coord] -= eps
            analytic_val = analytic["layers"][layer_idx][key][coord]
        else:
            p_plus["classifier"][coord] += eps
            p_minus["classifier"][coord] -= eps
            analytic_val = analytic["classifier"][coord]

        numeric_val = (loss_at(p_plus) - loss_at(p_minus)) / (2 * eps)
        assert np.isclose(analytic_val, numeric_val, atol=1e-3, rtol=1e-2), (
            f"gradient mismatch at {group}/{layer_idx}/{key}{coord}: "
            f"analytic={analytic_val}, numeric={numeric_val}"
        )


# --- 4. forward shapes --------------------------------------------------------

def test_forward_with_attention_output_shapes():
    rng = np.random.default_rng(3)
    params = init_params(rng)
    node_feats, src_idx, dst_idx, edge_feats = _toy_graph_inputs(rng, n_nodes=6, n_edges=9)
    logits, rollout = forward_with_attention(params, node_feats, src_idx, dst_idx, edge_feats)
    assert logits.shape == (len(RAW_CLASSES),)
    assert rollout.shape == (9,)
    assert np.all(np.asarray(rollout) >= 0)  # summed softmax outputs, never negative


# --- 5. end-to-end learning on a toy, separable task -------------------------

def _scan_snapshot() -> GraphSnapshot:
    # one attacker fanning out to 5 hosts -> high degree_out/unique_ports on
    # one node, distinguishable from a single benign edge
    nodes = [GraphNode(node_id="attacker", degree_out=5, degree_in=0, bytes_total=500,
                        unique_ports_contacted=5)]
    nodes += [GraphNode(node_id=f"v{i}", degree_in=1, degree_out=0, bytes_total=100,
                         unique_ports_contacted=0) for i in range(5)]
    edges = [GraphEdge(src="attacker", dst=f"v{i}", bytes=100, flow_count=1,
                        port_entropy=2.3, duration_mean_ms=2) for i in range(5)]
    return GraphSnapshot(window_id="scan", window_start=T0, window_end=T0 + timedelta(seconds=2),
                          nodes=nodes, edges=edges)


def _benign_snapshot() -> GraphSnapshot:
    nodes = [GraphNode(node_id="a", degree_out=1, degree_in=0, bytes_total=1000,
                        unique_ports_contacted=1),
             GraphNode(node_id="b", degree_in=1, degree_out=0, bytes_total=1000,
                        unique_ports_contacted=0)]
    edges = [GraphEdge(src="a", dst="b", bytes=1000, flow_count=3, port_entropy=0.0,
                        duration_mean_ms=200)]
    return GraphSnapshot(window_id="benign", window_start=T0, window_end=T0 + timedelta(seconds=2),
                          nodes=nodes, edges=edges)


def test_training_reduces_loss_on_a_toy_separable_task():
    model = DetectiveModel(rng=np.random.default_rng(11))
    scan, benign = _scan_snapshot(), _benign_snapshot()
    batch = [(scan, Label.PORT_SCAN), (benign, Label.BENIGN)]

    first_loss = model.train_batch(batch, learning_rate=0.05)
    for _ in range(150):
        model.train_batch(batch, learning_rate=0.05)
    last_loss = model.train_batch(batch, learning_rate=0.0)  # lr=0: measure, don't update

    assert last_loss < first_loss


def test_predict_verdict_schema_compliance_on_scan_shape():
    model = DetectiveModel(rng=np.random.default_rng(12))
    batch = [(_scan_snapshot(), Label.PORT_SCAN), (_benign_snapshot(), Label.BENIGN)]
    for _ in range(150):
        model.train_batch(batch, learning_rate=0.05)

    verdict = model.predict_verdict(_scan_snapshot(), window_id="scan")
    assert verdict.tier == Tier.DETECTIVE
    assert verdict.label in (Label.PORT_SCAN, Label.UNCERTAIN)
    if verdict.label != Label.BENIGN:
        assert verdict.evidence.node_ids
        assert verdict.evidence.edge_ids
        assert "attacker" in verdict.evidence.node_ids


def test_predict_verdict_on_empty_graph_is_benign_no_evidence():
    model = DetectiveModel()
    empty = GraphSnapshot(window_id="empty", window_start=T0, window_end=T0, nodes=[], edges=[])
    verdict = model.predict_verdict(empty, window_id="empty")
    assert verdict.label == Label.BENIGN
    assert verdict.evidence.node_ids == []
    assert verdict.evidence.edge_ids == []


def test_snapshot_to_arrays_preserves_node_and_edge_ordering():
    snap = _scan_snapshot()
    node_feats, src_idx, dst_idx, edge_feats, node_ids, edge_id_order = snapshot_to_arrays(snap)
    assert node_feats.shape == (6, 4)
    assert edge_feats.shape == (5, 4)
    assert node_ids[0] == "attacker"
    assert edge_id_order[0] == ("attacker", "v0")


def test_save_and_load_round_trip_produces_identical_predictions(tmp_path):
    model = DetectiveModel(rng=np.random.default_rng(20))
    batch = [(_scan_snapshot(), Label.PORT_SCAN), (_benign_snapshot(), Label.BENIGN)]
    for _ in range(30):
        model.train_batch(batch, learning_rate=0.05)

    before = model.predict_verdict(_scan_snapshot(), window_id="w")

    model.save(tmp_path)
    assert (tmp_path / "detective.npz").exists()
    assert (tmp_path / "detective.onnx").exists()

    reloaded = DetectiveModel.load(tmp_path)
    after = reloaded.predict_verdict(_scan_snapshot(), window_id="w")

    assert before.label == after.label
    assert abs(before.confidence - after.confidence) < 1e-6
