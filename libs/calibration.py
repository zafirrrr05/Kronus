"""spec.md §5.5: "temperature scaling minimum" — both Bouncer and Detective
are required to calibrate raw model confidence before it's allowed to drive
a block decision (an uncalibrated 0.95 from a tree ensemble doesn't mean
"95% of the time this is right"). One scalar T, fit by minimizing negative
log-likelihood on a held-out split — the standard, minimal version of this
technique (Guo et al.), and exactly what "minimum" calls for.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


class TemperatureScaler:
    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """logits: raw pre-sigmoid/pre-softmax scores, shape (n,) for binary
        or (n, k) for multiclass. labels: int class indices, shape (n,).
        """

        def nll(t: float) -> float:
            if t <= 0:
                return np.inf
            scaled = logits / t
            probs = _softmax(scaled) if scaled.ndim == 2 else _sigmoid(scaled)
            return float(_neg_log_likelihood(probs, labels))

        result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(result.x)
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        return _softmax(scaled) if scaled.ndim == 2 else _sigmoid(scaled)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _neg_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-12
    if probs.ndim == 1:
        p = np.clip(np.where(labels == 1, probs, 1 - probs), eps, 1.0)
    else:
        p = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    return float(-np.mean(np.log(p)))
