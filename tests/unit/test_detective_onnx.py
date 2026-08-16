import numpy as np

from services.detective.model import forward_with_attention, init_params, snapshot_to_arrays
from services.detective.onnx_export import (
    OnnxDetectiveRuntime,
    build_onnx_model,
    onnx_inputs_from_arrays,
)
from tests.unit.test_detective_model import _benign_snapshot, _scan_snapshot


def test_onnx_graph_passes_the_onnx_checker():
    params = init_params(np.random.default_rng(0))
    model = build_onnx_model(params)  # raises via onnx.checker.check_model if malformed
    assert model.graph.name == "kronus_detective_gat"


def test_onnx_logits_match_numpy_forward_pass_on_scan_graph():
    params = init_params(np.random.default_rng(1))
    node_feats, src_idx, dst_idx, edge_feats, _, _ = snapshot_to_arrays(_scan_snapshot())

    numpy_logits, _ = forward_with_attention(params, node_feats, src_idx, dst_idx, edge_feats)

    runtime = OnnxDetectiveRuntime(build_onnx_model(params))
    onnx_logits = runtime.logits(node_feats, src_idx, dst_idx, edge_feats)

    assert np.allclose(np.asarray(numpy_logits), onnx_logits, atol=1e-4, rtol=1e-3)


def test_onnx_logits_match_numpy_forward_pass_on_benign_graph():
    params = init_params(np.random.default_rng(2))
    node_feats, src_idx, dst_idx, edge_feats, _, _ = snapshot_to_arrays(_benign_snapshot())

    numpy_logits, _ = forward_with_attention(params, node_feats, src_idx, dst_idx, edge_feats)

    runtime = OnnxDetectiveRuntime(build_onnx_model(params))
    onnx_logits = runtime.logits(node_feats, src_idx, dst_idx, edge_feats)

    assert np.allclose(np.asarray(numpy_logits), onnx_logits, atol=1e-4, rtol=1e-3)


def test_onnx_matches_numpy_after_training_not_just_at_random_init():
    # Parity at a random initialization is necessary but not sufficient —
    # confirm it still holds after weights have actually moved via training.
    from libs.constants import Label
    from services.detective.model import DetectiveModel

    model = DetectiveModel(rng=np.random.default_rng(3))
    batch = [(_scan_snapshot(), Label.PORT_SCAN), (_benign_snapshot(), Label.BENIGN)]
    for _ in range(50):
        model.train_batch(batch, learning_rate=0.05)

    node_feats, src_idx, dst_idx, edge_feats, _, _ = snapshot_to_arrays(_scan_snapshot())
    numpy_logits, _ = forward_with_attention(model.params, node_feats, src_idx, dst_idx, edge_feats)

    runtime = OnnxDetectiveRuntime(build_onnx_model(model.params))
    onnx_logits = runtime.logits(node_feats, src_idx, dst_idx, edge_feats)

    assert np.allclose(np.asarray(numpy_logits), onnx_logits, atol=1e-4, rtol=1e-3)


def test_onnx_inputs_helper_shapes_are_consistent():
    node_feats, src_idx, dst_idx, edge_feats, _, _ = snapshot_to_arrays(_scan_snapshot())
    inputs = onnx_inputs_from_arrays(node_feats, src_idx, dst_idx, edge_feats)
    n, m = node_feats.shape[0], src_idx.shape[0]
    assert inputs["sym_group"].shape == (n, 2 * m)
    assert inputs["sym_src_idx"].shape == (2 * m,)
