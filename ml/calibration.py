# QP-ed93b9f4-dff 2026-05-09 03:26:52
"""
ml/calibration.py — Shared calibration class.

Must be in a standalone importable module so both train.py and server/app.py
can load pickled CalibratedModel instances without AttributeError.
"""
import numpy as np


class CalibratedModel:
    """
    Manual isotonic calibration wrapper around any sklearn-compatible model.
    Maps raw XGBoost probabilities to true calibrated probabilities using
    IsotonicRegression — no sklearn version dependency.

    Compatible with joblib.dump/load from any module (train.py or server).
    """
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = np.clip(self.calibrator.predict(np.clip(raw, 0, 1)), 0, 1)
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_booster(self):
        """Expose XGBoost booster for feature name extraction."""
        return self.base_model.get_booster()

    def get_params(self, deep=True):
        return self.base_model.get_params(deep=deep)

    def __getattr__(self, name):
        # Delegate unknown attributes to base model
        return getattr(self.base_model, name)
