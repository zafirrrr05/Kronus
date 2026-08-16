"""features.txt component 5 (Detective): "the graph-native model that
reads the shape of relationships between hosts — with every verdict
shipped alongside the specific nodes and edges that produced it."

Hand-rolled GAT (see BUILD_PLAN discussion: PyPI's `torch` pulls a
500MB+ wheel plus several more GB of CUDA dependencies neither needed nor
usable for a <200k-parameter CPU model — disproportionate for what this
is). `autograd` supplies real automatic differentiation over NumPy code;
onnx_export.py in this same package builds the ONNX graph by hand for
serving. Verified by a numerical gradient check in
tests/unit/test_detective_model.py, not just "training loss went down" —
that's the standard way to catch a subtly-wrong backprop before it trains
a model on a bug.

One forward computation (`_gat_layer`, `_forward_layers`), two entry
points: `forward_with_attention` (inference/explainability — attention
weights are the "evidence") and `_loss_fn` (training — wrapped in
autograd.grad). Both call the identical per-layer function, so there is no
second implementation to drift out of sync with the first.
"""

from __future__ import annotations

from pathlib import Path

import autograd.numpy as anp
import numpy as np
from autograd import grad

from libs.constants import AttributionMethod, Label, Tier
from libs.observability import observe
from libs.schemas import DetectionVerdict, GraphSnapshot, VerdictEvidence
from libs.verdict import derive_label

NODE_FEATURE_NAMES = ["degree_in", "degree_out", "bytes_total", "unique_ports_contacted"]
EDGE_FEATURE_NAMES = ["bytes", "flow_count", "port_entropy", "duration_mean_ms"]
RAW_CLASSES = [Label.BENIGN, Label.PORT_SCAN, Label.LATERAL_MOVEMENT]

HIDDEN_DIM = 32
EDGE_EMBED_DIM = 8
NUM_LAYERS = 3
LEAKY_SLOPE = 0.2


def init_params(rng: np.random.Generator) -> dict:
    """Xavier-ish scaling. Total params well under the <200k budget — see
    the count asserted in tests/unit/test_detective_model.py.
    """

    def layer_params(in_dim: int) -> dict:
        attn_in = 2 * HIDDEN_DIM + EDGE_EMBED_DIM
        return {
            "W": rng.normal(0, np.sqrt(2.0 / (in_dim + HIDDEN_DIM)), size=(in_dim, HIDDEN_DIM)),
            "We": rng.normal(0, np.sqrt(2.0 / (len(EDGE_FEATURE_NAMES) + EDGE_EMBED_DIM)),
                              size=(len(EDGE_FEATURE_NAMES), EDGE_EMBED_DIM)),
            "a": rng.normal(0, np.sqrt(2.0 / attn_in), size=(attn_in,)),
        }

    layers = [layer_params(len(NODE_FEATURE_NAMES))]
    layers += [layer_params(HIDDEN_DIM) for _ in range(NUM_LAYERS - 1)]
    classifier = rng.normal(0, np.sqrt(2.0 / (HIDDEN_DIM + len(RAW_CLASSES))),
                             size=(HIDDEN_DIM, len(RAW_CLASSES)))
    return {"layers": layers, "classifier": classifier}


def count_params(params: dict) -> int:
    total = sum(p.size for layer in params["layers"] for p in layer.values())
    return total + params["classifier"].size


def _zeros_like_params(params: dict) -> dict:
    return {
        "layers": [{k: np.zeros_like(v) for k, v in layer.items()} for layer in params["layers"]],
        "classifier": np.zeros_like(params["classifier"]),
    }


def _add_params(a: dict, b: dict) -> dict:
    return {
        "layers": [
            {k: a["layers"][i][k] + b["layers"][i][k] for k in a["layers"][i]}
            for i in range(len(a["layers"]))
        ],
        "classifier": a["classifier"] + b["classifier"],
    }


def _scale_params(params: dict, factor: float) -> dict:
    return {
        "layers": [{k: v * factor for k, v in layer.items()} for layer in params["layers"]],
        "classifier": params["classifier"] * factor,
    }


def _leaky_relu(x):
    return anp.where(x > 0, x, LEAKY_SLOPE * x)


def _segment_softmax(scores, dst_idx: np.ndarray, num_nodes: int):
    # A global (not per-segment) max-shift is mathematically equivalent to
    # a per-segment one after normalization (the constant cancels in the
    # ratio) and is far simpler to vectorize — see model.py's module note.
    shifted = scores - anp.max(scores)
    exp_scores = anp.exp(shifted)
    # (num_nodes, M) one-hot grouping matrix — structural (built from the
    # graph topology only), not a differentiable quantity itself.
    group = np.equal.outer(np.arange(num_nodes), dst_idx).astype(np.float64)
    segment_sums = group @ exp_scores  # (num_nodes,)
    denom_per_edge = group.T @ segment_sums  # (M,) broadcast back per-edge
    return exp_scores / (denom_per_edge + 1e-12)


def _gat_layer(layer_params: dict, node_feats, src_idx: np.ndarray, dst_idx: np.ndarray, edge_feats):
    """One GAT layer. Returns (new_node_feats (N, hidden), attention (M,)
    for the *original* M directed edges — evidence extraction reports
    real flow directions, even though aggregation below is symmetric).

    Aggregation is symmetric (each edge contributes a message to both its
    source and its destination), not destination-only. A destination-only
    GAT never lets a pure-source node (out-degree > 0, in-degree == 0)
    receive any message at all — exactly a port-scanning host's shape.
    Caught for real: an early version left the scanning "attacker" node's
    embedding as a function of its raw input features alone, with no
    message passing ever touching it, while its five victims each picked
    up one incoming message and dominated the pooled graph readout — the
    model could not learn to separate a scan from benign traffic no matter
    how long it trained (see tests/unit/test_detective_model.py history).
    Symmetric aggregation is also the standard choice for flow graphs
    specifically (e.g. E-GraphSAGE treats an edge as informing both
    endpoints), not a special-cased patch for this one shape.
    """
    num_nodes = node_feats.shape[0]
    Wh = node_feats @ layer_params["W"]  # (N, hidden)
    We = edge_feats @ layer_params["We"]  # (M, edge_embed)

    concat = anp.concatenate([Wh[src_idx], Wh[dst_idx], We], axis=1)  # (M, attn_in)
    raw_scores = _leaky_relu(concat @ layer_params["a"])  # (M,)
    attention = _segment_softmax(raw_scores, dst_idx, num_nodes)  # (M,), for evidence only

    # Symmetric message set for aggregation: every edge contributes once
    # to its destination's incoming messages and once to its source's.
    # Attention is recomputed over this doubled, symmetrized edge set so
    # normalization (each node's total incoming weight == 1) still holds
    # in the direction messages actually flow for aggregation.
    sym_src = anp.concatenate([src_idx, dst_idx])
    sym_dst = anp.concatenate([dst_idx, src_idx])
    sym_concat = anp.concatenate([Wh[sym_src], Wh[sym_dst], anp.concatenate([We, We])], axis=1)
    sym_scores = _leaky_relu(sym_concat @ layer_params["a"])
    sym_attention = _segment_softmax(sym_scores, sym_dst, num_nodes)

    weighted_messages = sym_attention[:, None] * Wh[sym_src]  # (2M, hidden)
    group = np.equal.outer(np.arange(num_nodes), sym_dst).astype(np.float64)
    aggregated = group @ weighted_messages  # (N, hidden)

    new_feats = _leaky_relu(aggregated + Wh)  # residual: self-signal survives zero in-degree
    return new_feats, attention


def _forward_layers(params: dict, node_feats, src_idx, dst_idx, edge_feats):
    """Runs all GAT layers; returns (final_node_feats, [attention_per_layer])."""
    feats = node_feats
    attentions = []
    for layer_params in params["layers"]:
        feats, attn = _gat_layer(layer_params, feats, src_idx, dst_idx, edge_feats)
        attentions.append(attn)
    return feats, attentions


def _logits_from_feats(params: dict, final_node_feats):
    graph_embedding = anp.mean(final_node_feats, axis=0)  # mean-pool readout
    return graph_embedding @ params["classifier"]


def forward_with_attention(params: dict, node_feats, src_idx, dst_idx, edge_feats):
    """Inference path: logits + an attention-rollout importance score per
    edge (summed across layers — see model.py's module docstring on why
    this is a reasonable, defensible simplification of full rollout for a
    2-3 layer network).
    """
    final_feats, attentions = _forward_layers(params, node_feats, src_idx, dst_idx, edge_feats)
    logits = _logits_from_feats(params, final_feats)
    rollout = anp.sum(anp.stack(attentions), axis=0)  # (M,)
    return logits, rollout


def _loss_fn(params: dict, node_feats, src_idx, dst_idx, edge_feats, label_idx: int):
    final_feats, _ = _forward_layers(params, node_feats, src_idx, dst_idx, edge_feats)
    logits = _logits_from_feats(params, final_feats)
    log_probs = logits - anp.log(anp.sum(anp.exp(logits - anp.max(logits)))) - anp.max(logits)
    return -log_probs[label_idx]


loss_grad = grad(_loss_fn)


def _sgd_update(
    params: dict, grads: dict, lr: float, velocity: dict, momentum: float,
    max_grad_norm: float = 5.0,
) -> tuple[dict, dict]:
    # Gradient clipping: residual connections with no normalization layer
    # (deliberately omitted — LayerNorm would blow the <200k param budget
    # on a model this small for no real benefit) mean per-step gradients
    # can compound across many training iterations. Clipping the global
    # gradient norm is the standard, minimal fix for exactly this failure
    # mode — caught for real by extended-training tests overflowing to NaN
    # (see tests/unit/test_detective_model.py), not a hypothetical.
    flat = np.concatenate(
        [g.ravel() for layer in grads["layers"] for g in layer.values()]
        + [grads["classifier"].ravel()]
    )
    norm = float(np.linalg.norm(flat))
    scale = min(1.0, max_grad_norm / (norm + 1e-12))
    scaled_grads = _scale_params(grads, scale)

    new_velocity = _add_params(_scale_params(velocity, momentum), _scale_params(scaled_grads, -lr))
    new_params = _add_params(params, new_velocity)
    return new_params, new_velocity


def snapshot_to_arrays(
    snapshot: GraphSnapshot,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[tuple[str, str]]]:
    """GraphSnapshot (Pydantic, string node ids) -> dense arrays a GAT layer
    can consume. Returns (node_feats, src_idx, dst_idx, edge_feats,
    node_id_order, edge_id_order) — the last two let evidence extraction
    map back from array indices to the original node/edge identifiers.

    Byte-count features get log1p'd before anything else touches them.
    Caught for real on actual NSL-KDD-derived windows: bytes_total ranged
    from 0 to ~3.8e8 while degree features stayed in the tens — seven to
    eight orders of magnitude apart — so the very first linear layer's
    output (and everything built on it through the residual connections)
    was dominated entirely by whichever node happened to have the most
    traffic, producing huge, unstable activations independent of learning
    rate or gradient clipping (both operate downstream of this). log1p is
    the standard treatment for network byte counts specifically, which are
    close to log-normally distributed; degree/port-count features are
    already small and bounded and are left as-is.
    """
    node_ids = [n.node_id for n in snapshot.nodes]
    index_of = {node_id: i for i, node_id in enumerate(node_ids)}
    node_feats = np.array([
        [np.log1p(n.bytes_total) if f == "bytes_total" else getattr(n, f) for f in NODE_FEATURE_NAMES]
        for n in snapshot.nodes
    ])

    src_idx = np.array([index_of[e.src] for e in snapshot.edges], dtype=np.int64)
    dst_idx = np.array([index_of[e.dst] for e in snapshot.edges], dtype=np.int64)
    edge_feats = np.array([
        [np.log1p(e.bytes) if f == "bytes" else getattr(e, f) for f in EDGE_FEATURE_NAMES]
        for e in snapshot.edges
    ])
    edge_id_order = [(e.src, e.dst) for e in snapshot.edges]

    return node_feats, src_idx, dst_idx, edge_feats, node_ids, edge_id_order


class DetectiveModel:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.params = init_params(rng or np.random.default_rng(7))
        self._velocity = _zeros_like_params(self.params)

    def train_batch(
        self, batch: list[tuple[GraphSnapshot, Label]], learning_rate: float = 0.01,
        momentum: float = 0.9,
    ) -> float:
        """Averages gradients across the batch before applying one update.
        Preferred over single-example steps: alternating single-example
        SGD on a handful of very different graphs (the common case here —
        a scan-shaped window looks nothing like a benign one) oscillates
        instead of converging once momentum is added, since each step's
        velocity gets built from only the last example's direction. This
        was caught for real (see BUILD_PLAN discussion): momentum alone
        broke a stuck plateau but never stabilized; averaging fixed both.
        """
        if not batch:
            raise ValueError("train_batch requires at least one example")

        total_loss = 0.0
        summed_grads = _zeros_like_params(self.params)
        for snapshot, label in batch:
            node_feats, src_idx, dst_idx, edge_feats, _, _ = snapshot_to_arrays(snapshot)
            label_idx = RAW_CLASSES.index(label)
            total_loss += float(
                _loss_fn(self.params, node_feats, src_idx, dst_idx, edge_feats, label_idx)
            )
            grads = loss_grad(self.params, node_feats, src_idx, dst_idx, edge_feats, label_idx)
            summed_grads = _add_params(summed_grads, grads)

        avg_grads = _scale_params(summed_grads, 1.0 / len(batch))
        self.params, self._velocity = _sgd_update(
            self.params, avg_grads, learning_rate, self._velocity, momentum
        )
        return total_loss / len(batch)

    def train_step(self, snapshot: GraphSnapshot, label: Label, learning_rate: float = 0.01) -> float:
        """Convenience single-example wrapper around train_batch (e.g. for
        a live online-update path) — see train_batch's docstring for why
        the real training loop (train.py) uses batches instead.
        """
        return self.train_batch([(snapshot, label)], learning_rate=learning_rate)

    def predict_verdict(
        self, snapshot: GraphSnapshot, window_id: str, top_k_evidence: int = 3
    ) -> DetectionVerdict:
        with observe("detective", "predict_verdict", window_id=window_id):
            if not snapshot.nodes or not snapshot.edges:
                return DetectionVerdict(
                    window_id=window_id, tier=Tier.DETECTIVE, label=Label.BENIGN, confidence=1.0,
                    evidence=VerdictEvidence(attribution_method=AttributionMethod.ATTENTION_ROLLOUT),
                )
            node_feats, src_idx, dst_idx, edge_feats, node_ids, edge_id_order = snapshot_to_arrays(snapshot)
            logits, rollout = forward_with_attention(self.params, node_feats, src_idx, dst_idx, edge_feats)

            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            predicted_idx = int(np.argmax(probs))
            predicted_label = RAW_CLASSES[predicted_idx]
            confidence = float(probs[predicted_idx])

            top_edges = np.argsort(np.asarray(rollout))[::-1][:top_k_evidence]
            evidence_edges = [f"{edge_id_order[i][0]}->{edge_id_order[i][1]}" for i in top_edges]
            evidence_nodes = sorted({node for i in top_edges for node in edge_id_order[i]})

            label = derive_label(predicted_label, confidence)
            return DetectionVerdict(
                window_id=window_id, tier=Tier.DETECTIVE, label=label, confidence=confidence,
                evidence=VerdictEvidence(
                    node_ids=evidence_nodes if label != Label.BENIGN else [],
                    edge_ids=evidence_edges if label != Label.BENIGN else [],
                    attribution_method=AttributionMethod.ATTENTION_ROLLOUT,
                ),
            )

    def save(self, directory: str | Path) -> None:
        """Persists params as a flat .npz (numpy's native multi-array
        format) plus the ONNX export used for serving — see
        onnx_export.py. Two artifacts, one training-format source of
        truth (.npz) and one serving-format derivative (.onnx), rather
        than trying to make ONNX itself round-trip back into trainable
        params (ONNX is a serving format here, not a checkpoint format).
        """
        import onnx as onnx_module

        from services.detective.onnx_export import build_onnx_model

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        flat = {}
        for i, layer in enumerate(self.params["layers"]):
            for key, arr in layer.items():
                flat[f"layer{i}_{key}"] = arr
        flat["classifier"] = self.params["classifier"]
        flat["n_layers"] = np.array(len(self.params["layers"]))
        np.savez(directory / "detective.npz", **flat)
        onnx_module.save(build_onnx_model(self.params), str(directory / "detective.onnx"))

    @classmethod
    def load(cls, directory: str | Path) -> "DetectiveModel":
        directory = Path(directory)
        data = np.load(directory / "detective.npz")
        n_layers = int(data["n_layers"])
        layers = [
            {"W": data[f"layer{i}_W"], "We": data[f"layer{i}_We"], "a": data[f"layer{i}_a"]}
            for i in range(n_layers)
        ]
        model = cls()
        model.params = {"layers": layers, "classifier": data["classifier"]}
        model._velocity = _zeros_like_params(model.params)
        return model
