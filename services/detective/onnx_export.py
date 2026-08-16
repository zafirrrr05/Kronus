"""spec.md §5: "Serving: ONNX Runtime" for the Detective. Since the model
is hand-rolled (see model.py's docstring on why there's no torch here),
there's no `torch.onnx.export` to lean on — the graph is built directly
with `onnx.helper`, one node at a time, mirroring model.py's NumPy forward
pass operation-for-operation so there's minimal room for the two to
diverge. Trust comes from the numerical parity test in
tests/unit/test_detective_onnx.py (same inputs, same params, ONNX Runtime
output compared against the NumPy forward pass), not from construction
alone.

Design choice: the segment-softmax/aggregation grouping matrices (which
depend on the specific graph's edge topology, not on learned parameters)
are computed in plain NumPy *outside* the graph and passed in as regular
input tensors, rather than built with ONNX ops from src_idx/dst_idx. ONNX's
dynamic-shape support handles the varying node/edge counts across
different windows fine either way; doing the topology-dependent bookkeeping
in Python and leaving the graph itself pure linear algebra (MatMul, Gather,
Exp, LeakyRelu, Div — all standard ops) is the simpler, lower-risk split.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from services.detective.model import (
    LEAKY_SLOPE,
    RAW_CLASSES,
)


def _grouping_matrix(dst_idx: np.ndarray, num_nodes: int) -> np.ndarray:
    """Same construction as model.py's _segment_softmax — kept here as a
    standalone function since the ONNX graph takes this as an input rather
    than computing it internally (see module docstring)."""
    return np.equal.outer(np.arange(num_nodes), dst_idx).astype(np.float32)


def build_onnx_model(params: dict) -> onnx.ModelProto:
    """One graph, `NUM_LAYERS` repeated attention+aggregation blocks,
    weights embedded as initializers. Inputs are the graph's dense arrays
    plus its precomputed grouping matrices (see module docstring); every
    dimension that varies per-window (N nodes, M edges) is a symbolic ONNX
    dimension, so one exported graph serves any window size.
    """
    initializers = []
    nodes = []

    def const(name: str, arr: np.ndarray) -> str:
        arr = arr.astype(np.float32)
        initializers.append(helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.flatten()))
        return name

    def axes_const(name: str, axes: list[int]) -> str:
        # Squeeze/Unsqueeze take `axes` as an int64 input tensor from
        # opset 13 onward, not a node attribute — an early version used
        # the pre-13 attribute style and onnx.checker.check_model rejected
        # it outright ("Unrecognized attribute: axes for operator Squeeze").
        arr = np.array(axes, dtype=np.int64)
        initializers.append(helper.make_tensor(name, TensorProto.INT64, arr.shape, arr))
        return name

    node_feats_in = helper.make_tensor_value_info("node_feats", TensorProto.FLOAT, ["N", 4])
    edge_feats_in = helper.make_tensor_value_info("edge_feats", TensorProto.FLOAT, ["M", 4])
    src_idx_in = helper.make_tensor_value_info("src_idx", TensorProto.INT64, ["M"])
    dst_idx_in = helper.make_tensor_value_info("dst_idx", TensorProto.INT64, ["M"])
    sym_src_idx_in = helper.make_tensor_value_info("sym_src_idx", TensorProto.INT64, ["M2"])
    sym_dst_idx_in = helper.make_tensor_value_info("sym_dst_idx", TensorProto.INT64, ["M2"])
    sym_group_in = helper.make_tensor_value_info("sym_group", TensorProto.FLOAT, ["N", "M2"])

    cur = "node_feats"
    for i, layer in enumerate(params["layers"]):
        p = f"l{i}_"
        const(p + "W", layer["W"])
        const(p + "We", layer["We"])
        const(p + "a", layer["a"].reshape(-1, 1))

        nodes.append(helper.make_node("MatMul", [cur, p + "W"], [p + "Wh"]))
        nodes.append(helper.make_node("MatMul", ["edge_feats", p + "We"], [p + "We_out"]))
        # doubled edge features for the symmetric edge set: concat(We, We)
        nodes.append(helper.make_node("Concat", [p + "We_out", p + "We_out"], [p + "We_sym"], axis=0))

        # symmetric attention scores: LeakyRelu(concat(Wh[sym_src], Wh[sym_dst], We_sym) @ a)
        nodes.append(helper.make_node("Gather", [p + "Wh", "sym_src_idx"], [p + "Wh_src"], axis=0))
        nodes.append(helper.make_node("Gather", [p + "Wh", "sym_dst_idx"], [p + "Wh_dst"], axis=0))
        nodes.append(helper.make_node(
            "Concat", [p + "Wh_src", p + "Wh_dst", p + "We_sym"], [p + "concat"], axis=1
        ))
        nodes.append(helper.make_node("MatMul", [p + "concat", p + "a"], [p + "raw_scores_2d"]))
        nodes.append(helper.make_node(
            "Squeeze", [p + "raw_scores_2d", axes_const(p + "sq_ax1", [1])], [p + "raw_scores"]
        ))
        nodes.append(helper.make_node("LeakyRelu", [p + "raw_scores"], [p + "scores"], alpha=LEAKY_SLOPE))

        # segment softmax over sym_dst: normalize by each edge's
        # destination's total incoming weight, looked up directly via
        # Gather (seg_sums[sym_dst_idx]) rather than a Transpose+MatMul —
        # the latter was tried first and reproducibly returned wrong
        # values for this shape (an onnxruntime fusion edge case with a
        # matrix-vector MatMul immediately after Transpose; confirmed by
        # isolating the two nodes alone and comparing against plain NumPy).
        # Gather also states the actual intent — "look up this edge's
        # destination's total" — more directly than routing it through a
        # matmul trick.
        nodes.append(helper.make_node("ReduceMax", [p + "scores"], [p + "max"], keepdims=0))
        nodes.append(helper.make_node("Sub", [p + "scores", p + "max"], [p + "shifted"]))
        nodes.append(helper.make_node("Exp", [p + "shifted"], [p + "exp_scores"]))
        nodes.append(helper.make_node("MatMul", ["sym_group", p + "exp_scores"], [p + "seg_sums"]))
        nodes.append(helper.make_node(
            "Gather", [p + "seg_sums", "sym_dst_idx"], [p + "denom"], axis=0
        ))
        nodes.append(helper.make_node("Add", [p + "denom", const(p + "eps", np.array(1e-12))], [p + "denom_safe"]))
        nodes.append(helper.make_node("Div", [p + "exp_scores", p + "denom_safe"], [p + "attn"]))

        # weighted messages, aggregated back to nodes via sym_group
        nodes.append(helper.make_node(
            "Unsqueeze", [p + "attn", axes_const(p + "unsq_ax1", [1])], [p + "attn_col"]
        ))
        nodes.append(helper.make_node("Mul", [p + "attn_col", p + "Wh_src"], [p + "weighted"]))
        nodes.append(helper.make_node("MatMul", ["sym_group", p + "weighted"], [p + "aggregated"]))

        nodes.append(helper.make_node("Add", [p + "aggregated", p + "Wh"], [p + "pre_act"]))
        nodes.append(helper.make_node("LeakyRelu", [p + "pre_act"], [p + "out"], alpha=LEAKY_SLOPE))
        cur = p + "out"

    const("classifier", params["classifier"])
    nodes.append(helper.make_node("ReduceMean", [cur], ["graph_embedding"], axes=[0], keepdims=0))
    nodes.append(helper.make_node(
        "Unsqueeze", ["graph_embedding", axes_const("out_unsq_ax0", [0])], ["graph_embedding_2d"]
    ))
    nodes.append(helper.make_node("MatMul", ["graph_embedding_2d", "classifier"], ["logits_2d"]))
    nodes.append(helper.make_node(
        "Squeeze", ["logits_2d", axes_const("out_sq_ax0", [0])], ["logits"]
    ))

    graph = helper.make_graph(
        nodes,
        "kronus_detective_gat",
        [node_feats_in, edge_feats_in, src_idx_in, dst_idx_in, sym_src_idx_in, sym_dst_idx_in,
         sym_group_in],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [len(RAW_CLASSES)])],
        initializer=initializers,
    )
    model = helper.make_model(graph, producer_name="kronus-detective")
    model.opset_import[0].version = 17
    onnx.checker.check_model(model)
    return model


def onnx_inputs_from_arrays(
    node_feats: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray, edge_feats: np.ndarray,
) -> dict[str, np.ndarray]:
    """Builds the full input dict (including the topology-derived grouping
    matrices) an ONNX Runtime session needs — the Python-side half of the
    split described in this module's docstring.
    """
    num_nodes = node_feats.shape[0]
    sym_src = np.concatenate([src_idx, dst_idx])
    sym_dst = np.concatenate([dst_idx, src_idx])
    return {
        "node_feats": node_feats.astype(np.float32),
        "edge_feats": edge_feats.astype(np.float32),
        "src_idx": src_idx.astype(np.int64),
        "dst_idx": dst_idx.astype(np.int64),
        "sym_src_idx": sym_src.astype(np.int64),
        "sym_dst_idx": sym_dst.astype(np.int64),
        "sym_group": _grouping_matrix(sym_dst, num_nodes),
    }


class OnnxDetectiveRuntime:
    """Thin onnxruntime.InferenceSession wrapper — the actual serving path
    (spec.md §5's "Serving: ONNX Runtime"), separate from the NumPy/autograd
    path model.py uses for training.
    """

    def __init__(self, model: onnx.ModelProto) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(model.SerializeToString())

    def logits(self, node_feats: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray, edge_feats: np.ndarray) -> np.ndarray:
        inputs = onnx_inputs_from_arrays(node_feats, src_idx, dst_idx, edge_feats)
        (logits,) = self._session.run(["logits"], inputs)
        return logits

    @property
    def n_params_in_graph(self) -> int:
        # Every layer contributes 3 arrays (l{i}_W, l{i}_We, l{i}_a) plus
        # a shared "eps" constant per layer plus the classifier — used by
        # the layer-count assertion in tests/unit/test_detective_onnx.py.
        return sum(1 for i in self._session.get_inputs())
