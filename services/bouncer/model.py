"""features.txt component 4 (Bouncer): "XGBoost/LightGBM... serve as the
honest baseline that proves the GNN earns its complexity on everything
else." Trained binary: flood vs not-flood (not-flood absorbs benign,
probe, and r2l/u2r rows — Case 2's "the Bouncer correctly stays quiet" on
a port scan is a property this training target has to produce, not just
assert, so scan-shaped rows are deliberately in the negative class).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from libs.calibration import TemperatureScaler
from libs.constants import AttributionMethod, Label, Tier
from libs.observability import observe
from libs.schemas import DetectionVerdict, VerdictEvidence
from libs.verdict import derive_label
from services.bouncer.features import FEATURE_NAMES


class BouncerModel:
    def __init__(self) -> None:
        self._booster: xgb.Booster | None = None
        self._scaler = TemperatureScaler()

    def fit(self, X: np.ndarray, y: np.ndarray, calibration_split: float = 0.2) -> "BouncerModel":
        n_calib = int(len(X) * calibration_split)
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X))
        calib_idx, train_idx = idx[:n_calib], idx[n_calib:]

        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=FEATURE_NAMES)
        self._booster = xgb.train(
            params={
                "objective": "binary:logistic",
                "max_depth": 5,
                "eta": 0.1,
                "eval_metric": "logloss",
            },
            dtrain=dtrain,
            num_boost_round=100,
        )

        calib_logits = self._raw_margin(X[calib_idx])
        self._scaler.fit(calib_logits, y[calib_idx])
        return self

    def _raw_margin(self, X: np.ndarray) -> np.ndarray:
        dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
        return self._booster.predict(dmat, output_margin=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated P(flood) for each row."""
        return self._scaler.calibrate(self._raw_margin(X))

    def predict_verdict(self, features: dict[str, float], window_id: str) -> DetectionVerdict:
        with observe("bouncer", "predict_verdict", window_id=window_id):
            x = np.array([[features[name] for name in FEATURE_NAMES]])
            p_flood = float(self.predict_proba(x)[0])

            if p_flood >= 0.5:
                predicted_label, confidence = Label.FLOOD, p_flood
            else:
                predicted_label, confidence = Label.BENIGN, 1.0 - p_flood

            label = derive_label(predicted_label, confidence)
            return DetectionVerdict(
                window_id=window_id,
                tier=Tier.BOUNCER,
                label=label,
                confidence=confidence,
                evidence=VerdictEvidence(attribution_method=AttributionMethod.RATE_THRESHOLD),
            )

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(directory / "bouncer.json"))
        (directory / "calibration.json").write_text(
            json.dumps({"temperature": self._scaler.temperature})
        )

    @classmethod
    def load(cls, directory: str | Path) -> "BouncerModel":
        directory = Path(directory)
        model = cls()
        model._booster = xgb.Booster()
        model._booster.load_model(str(directory / "bouncer.json"))
        calib = json.loads((directory / "calibration.json").read_text())
        model._scaler.temperature = calib["temperature"]
        return model
